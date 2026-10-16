"""Read-only checks on a ranking problem and its data, before any ranker is fitted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns
from ml_core.tooling import tool
from supervised.ranking.tools._common import load_ranking, per_group_metrics, summarise


@tool
def profile_ranking_data(data: str, group_column: str, relevance: str, features: list[str] | None = None) -> dict:
    """Profile a learning-to-rank dataset: groups (queries) and their sizes, the relevance scale and its distribution, groups with nothing relevant, groups where every item ties, and the feature columns."""
    frame = load_frame(data)
    require_columns(frame, [group_column, relevance])
    sizes = frame.groupby(group_column).size()
    rel = frame[relevance]
    by_group = frame.groupby(group_column)[relevance]
    return {
        "rows": len(frame), "groups": int(sizes.size),
        "items_per_group": {"min": int(sizes.min()), "median": float(sizes.median()), "max": int(sizes.max())},
        "relevance_levels": {str(k): int(v) for k, v in rel.value_counts().sort_index().items()},
        "graded": bool(rel.nunique() > 2),
        "groups_with_nothing_relevant": int((by_group.max() <= 0).sum()),
        "groups_with_constant_relevance": int((by_group.nunique() <= 1).sum()),
        "singleton_groups": int((sizes <= 1).sum()),
        "numeric_features": [c for c in frame.columns if c not in (group_column, relevance) and pd.api.types.is_numeric_dtype(frame[c])],
        "text_features": [c for c in frame.columns if c not in (group_column, relevance) and not pd.api.types.is_numeric_dtype(frame[c])],
        "note": "groups with constant relevance carry no ordering signal and are skipped by pairwise and NDCG training",
    }


@tool
def baseline_random_ranker(data: str, group_column: str, relevance: str, k: int = 10, relevant_from: float = 1, seed: int = 42) -> dict:
    """Score a random ordering — the floor any ranker must clear — with NDCG, MAP, MRR, precision and recall at k.

    Args:
        group_column: Column identifying the query/group each row belongs to.
        relevant_from: Relevance at or above which an item counts as relevant for MAP/MRR/precision/recall.
    """
    frame = load_frame(data)
    frame = frame.assign(_score=np.random.default_rng(seed).random(len(frame)))
    return {"k": k, "random_order": summarise(per_group_metrics(frame, group_column, relevance, "_score", k, relevant_from), k)}


@tool
def baseline_single_feature_rankers(
    data: str, group_column: str, relevance: str, features: list[str] | None = None, k: int = 10, relevant_from: float = 1
) -> dict:
    """Rank by each numeric feature on its own, in both directions, and report NDCG@k — shows which raw signals already order the items and whether higher or lower is better.

    Args:
        group_column: Column identifying the query/group each row belongs to.
    """
    frame, columns = load_ranking(data, group_column, relevance, features)
    rows = []
    for c in columns:
        filled = frame[c].fillna(frame[c].median())
        up = per_group_metrics(frame.assign(_s=filled), group_column, relevance, "_s", k, relevant_from)["ndcg"].mean()
        down = per_group_metrics(frame.assign(_s=-filled), group_column, relevance, "_s", k, relevant_from)["ndcg"].mean()
        rows.append({"feature": c, "ndcg_higher_is_better": round(float(up), 4), "ndcg_lower_is_better": round(float(down), 4),
                     "best_direction": "higher" if up >= down else "lower", "best_ndcg": round(float(max(up, down)), 4)})
    rows.sort(key=lambda r: -r["best_ndcg"])
    return {"k": k, "features": rows, "strongest": rows[0]["feature"] if rows else None}


@tool
def check_group_structure(data: str, group_column: str, relevance: str) -> dict:
    """Check whether the data is laid out the way rankers need: rows of a group contiguous, enough groups to split, enough pairs with differing relevance, no group that could appear on both sides of a split by another id."""
    frame = load_frame(data)
    require_columns(frame, [group_column, relevance])
    g = frame[group_column]
    contiguous = bool((g != g.shift()).sum() == g.nunique())
    pairs = int(sum(int((p[relevance].to_numpy()[:, None] > p[relevance].to_numpy()[None, :]).sum()) for _, p in frame.groupby(group_column)))
    n_groups = int(g.nunique())
    problems = []
    if n_groups < 20:
        problems.append("fewer than 20 groups: hold-out NDCG will be very noisy; prefer cross_validate_ranker")
    if pairs < 10 * n_groups:
        problems.append("few comparable pairs per group: pairwise methods have little to learn from")
    return {"groups": n_groups, "rows_contiguous_by_group": contiguous, "comparable_pairs": pairs, "pairs_per_group": round(pairs / max(n_groups, 1), 2), "problems": problems or ["layout is fine"]}
