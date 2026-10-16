"""Judge a ranking honestly: list metrics at several cut-offs, per-query failures, and scoring new rows with a saved ranker."""

from __future__ import annotations

import numpy as np

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.tooling import tool
from supervised.ranking.tools._common import per_group_metrics, summarise


@tool
def evaluate_ranking(
    data: str, group_column: str, relevance: str, score_column: str, k: int = 10, relevant_from: float = 1
) -> dict:
    """Score any ranking you already have: NDCG, MAP, MRR, precision and recall at k, averaged over groups, from a column of scores (higher = ranked first).

    Args:
        group_column: Column identifying the query/group each row belongs to.
        score_column: Column holding the score each item was ranked by.
        k: Cut-off for NDCG/MAP/precision/recall.
        relevant_from: Relevance at or above which an item counts as relevant.
    """
    frame = load_frame(data)
    require_columns(frame, [group_column, relevance, score_column])
    return summarise(per_group_metrics(frame, group_column, relevance, score_column, k, relevant_from), k)


@tool
def ndcg_at_multiple_k(
    data: str, group_column: str, relevance: str, score_column: str, ks: list[int] | None = None, relevant_from: float = 1
) -> dict:
    """NDCG, precision and recall at several cut-offs at once (default 1, 3, 5, 10, 20) — shows whether the ranking is good at the very top or only deeper in the list.

    Args:
        group_column: Column identifying the query/group each row belongs to.
        score_column: Column holding the score each item was ranked by.
        ks: Cut-offs to report.
    """
    frame = load_frame(data)
    require_columns(frame, [group_column, relevance, score_column])
    table = {}
    for k in [int(x) for x in (ks or [1, 3, 5, 10, 20])]:
        s = summarise(per_group_metrics(frame, group_column, relevance, score_column, k, relevant_from), k)
        table[k] = {key.split("@")[0]: v for key, v in s.items() if "@" in key}
    return {"by_k": table}


@tool
def worst_ranked_groups(
    data: str, group_column: str, relevance: str, score_column: str, k: int = 10, n: int = 10,
    relevant_from: float = 1, output_path: str | None = None,
) -> dict:
    """List the groups (queries) whose NDCG@k is lowest, with the best relevance the ranker put at the top — where to look for missing features or label problems.

    Args:
        group_column: Column identifying the query/group each row belongs to.
        score_column: Column holding the score each item was ranked by.
        k: Cut-off for NDCG.
        n: How many groups to list.
    """
    frame = load_frame(data)
    require_columns(frame, [group_column, relevance, score_column])
    table = per_group_metrics(frame, group_column, relevance, score_column, k, relevant_from).dropna(subset=["ndcg"]).sort_values("ndcg")
    top_rel = {g: float(p.sort_values(score_column, ascending=False)[relevance].iloc[0]) for g, p in frame.groupby(group_column)}
    worst = table.head(n).assign(top_ranked_relevance=lambda d: d[group_column].map(top_rel))
    return {"k": k, "mean_ndcg": round(float(table["ndcg"].mean()), 4), "worst_groups": worst.round(4),
            "written_to": save_frame(worst, output_path, "worst_ranked_groups.csv") if output_path else None}


@tool
def position_bias_report(
    data: str, group_column: str, relevance: str, score_column: str, max_position: int = 10
) -> dict:
    """Mean relevance by rank position under a given scoring — a healthy ranker's curve falls steadily; a flat or bumpy one means the scores do not order items by relevance.

    Args:
        group_column: Column identifying the query/group each row belongs to.
        score_column: Column holding the score each item was ranked by.
        max_position: How many positions to report.
    """
    frame = load_frame(data)
    require_columns(frame, [group_column, relevance, score_column])
    ranked = frame.sort_values([group_column, score_column], ascending=[True, False])
    ranked["_pos"] = ranked.groupby(group_column).cumcount() + 1
    curve = ranked[ranked["_pos"] <= max_position].groupby("_pos")[relevance].mean()
    monotone = bool(np.all(np.diff(curve.to_numpy()) <= 1e-9))
    return {"mean_relevance_by_position": {int(p): round(float(v), 4) for p, v in curve.items()},
            "monotone_decreasing": monotone, "verdict": "scores order items by relevance" if monotone else "the curve is not monotone — the ranking is noisy at some positions"}


@tool
def score_with_saved_ranker(
    data: str, model_path: str, group_column: str, output_path: str | None = None, top_k: int | None = None
) -> dict:
    """Score new rows with a ranker saved by one of the fit tools; writes each row's score and its rank within its group.

    Args:
        model_path: Path of the saved ranker (joblib).
        group_column: Column identifying the query/group each row belongs to.
        top_k: Keep only the top k rows per group in the output.
    """
    import joblib

    frame = load_frame(data)
    ranker = joblib.load(model_path)
    require_columns(frame, [group_column] + ranker.columns)
    out = frame.assign(score=ranker.scores(frame))
    out["rank"] = out.groupby(group_column)["score"].rank(ascending=False, method="first").astype(int)
    out = out.sort_values([group_column, "rank"])
    if top_k:
        out = out[out["rank"] <= top_k]
    return {"rows": len(out), "groups": int(out[group_column].nunique()), "written_to": save_frame(out, output_path, "ranked.csv"), "sample": out.head(5)}
