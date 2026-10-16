"""Query strategies: rank the unlabelled pool by how much a label would teach the model, and write the batch to send to annotators."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ml_core.data import load_frame, save_frame
from ml_core.semi import base_clf, full_proba
from ml_core.tooling import tool
from semisupervised.active_learning.tools._common import build_env, select

Base = Literal["logistic_regression", "random_forest", "hist_gradient_boosting", "svm", "knn", "naive_bayes"]


def queue(strategy: str, data: Any, label_column: str, unlabeled_value: Any, features: Any, drop: Any, id_column: str | None, base: str, batch_size: int, seed: int,
          output_path: str | None, **kw: Any) -> dict[str, Any]:
    """Run one strategy on the real pool and return / write the rows to label."""
    env = build_env(data, label_column, features, drop, unlabeled_value, None, seed)
    if len(env.pool) == 0:
        raise ValueError("the pool is empty: there are no unlabelled rows to query")
    if len(set(env.y[env.labelled])) < 2 and strategy not in ("random", "core_set"):
        raise ValueError("fewer than two classes among the labelled rows — a model-based strategy cannot start; use query_random or query_core_set first")
    rng = np.random.default_rng(seed)
    picks, scores = select(strategy, env, base, min(batch_size, len(env.pool)), env.labelled, env.pool, rng, **kw)
    frame = load_frame(data)
    rows = env.pool[picks]
    model = base_clf(base, seed).fit(env.Z[env.labelled], env.y[env.labelled]) if len(set(env.y[env.labelled])) > 1 else None
    out = frame.iloc[rows].copy()
    out.insert(0, "row", rows)
    out["query_score"] = scores[picks]
    out["query_rank"] = np.arange(1, len(rows) + 1)
    if model is not None:
        p = full_proba(model, env.Z[rows], env.K)
        out["model_guess"] = [env.classes[i] for i in p.argmax(axis=1)]
        out["model_confidence"] = p.max(axis=1)
    if id_column and id_column in out.columns:  # keep the id right after `row` so annotators and merge scripts find it first
        out = out[["row", id_column] + [c for c in out.columns if c not in ("row", id_column)]]
    return {"strategy": strategy, "labelled_now": int(len(env.labelled)), "pool": int(len(env.pool)), "batch": int(len(rows)), "rows_to_label": out.head(25),
            "mean_model_confidence_of_batch": round(float(out["model_confidence"].mean()), 4) if "model_confidence" in out else None,
            "written_to": save_frame(out, output_path, f"label_queue_{strategy}.csv"), "next": "send the file to annotators, then call ingest_new_labels with their answers"}


@tool
def query_uncertainty_sampling(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, id_column: str | None = None,
    base: Base = "logistic_regression", batch_size: int = 20, measure: Literal["entropy", "margin", "least_confident"] = "entropy", seed: int = 42, output_path: str | None = None,
) -> dict:
    """Uncertainty sampling: ask for the labels the current model is least sure about. Entropy looks at the whole class distribution, margin at the gap between the top two classes (best for multiclass boundaries), least-confident at the top probability alone. The standard first choice.

    Args:
        batch_size: Rows to send to annotators.
        measure: entropy, margin or least_confident.
        output_path: Where to write the labelling queue.
    """
    return queue(measure, data, label_column, unlabeled_value, features, drop, id_column, base, batch_size, seed, output_path)


@tool
def query_by_committee(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, id_column: str | None = None,
    base: Base = "logistic_regression", batch_size: int = 20, n_members: int = 5, measure: Literal["vote_entropy", "kl"] = "vote_entropy", seed: int = 42, output_path: str | None = None,
) -> dict:
    """Query by committee: train several models on bootstrap resamples of the labels and ask about the rows they disagree on most (vote entropy, or KL divergence from the consensus). Disagreement catches uncertainty in the *model*, not just in one model's probabilities, and works even with learners that give poor probabilities.

    Args:
        batch_size: Rows to send to annotators.
        n_members: Committee size.
        measure: vote_entropy or kl.
        output_path: Where to write the labelling queue.
    """
    return queue(measure, data, label_column, unlabeled_value, features, drop, id_column, base, batch_size, seed, output_path, n_members=n_members)


@tool
def query_bald(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, id_column: str | None = None,
    base: Base = "random_forest", batch_size: int = 20, n_members: int = 8, seed: int = 42, output_path: str | None = None,
) -> dict:
    """BALD (Bayesian active learning by disagreement): the mutual information between the label and the model — high when each ensemble member is confident but they contradict each other, low when every member is simply unsure. It separates *learnable* uncertainty from irreducible noise, so it does not waste labels on rows that are inherently ambiguous.

    Args:
        batch_size: Rows to send to annotators.
        n_members: Ensemble size.
        output_path: Where to write the labelling queue.
    """
    return queue("bald", data, label_column, unlabeled_value, features, drop, id_column, base, batch_size, seed, output_path, n_members=n_members)


@tool
def query_density_weighted(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, id_column: str | None = None,
    base: Base = "logistic_regression", batch_size: int = 20, beta: float = 1.0, seed: int = 42, output_path: str | None = None,
) -> dict:
    """Density-weighted uncertainty: entropy multiplied by how representative the row is of the pool (mean similarity to other rows). Plain uncertainty sampling likes outliers; this prefers uncertain rows in dense regions, where a label teaches the model about many rows at once.

    Args:
        batch_size: Rows to send to annotators.
        beta: Weight of density (0 is plain entropy; larger favours dense regions).
        output_path: Where to write the labelling queue.
    """
    return queue("density_weighted", data, label_column, unlabeled_value, features, drop, id_column, base, batch_size, seed, output_path, beta=beta)


@tool
def query_core_set(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, id_column: str | None = None,
    base: Base = "logistic_regression", batch_size: int = 20, seed: int = 42, output_path: str | None = None,
) -> dict:
    """Core-set (k-center greedy): pick rows that cover the feature space — each new row is the one farthest from everything already labelled or chosen. Needs no model at all, so it is the right way to build or widen a seed set (cold start) and to guard against a model's blind spots.

    Args:
        batch_size: Rows to send to annotators.
        output_path: Where to write the labelling queue.
    """
    return queue("core_set", data, label_column, unlabeled_value, features, drop, id_column, base, batch_size, seed, output_path)


@tool
def query_uncertainty_diversity_batch(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, id_column: str | None = None,
    base: Base = "logistic_regression", batch_size: int = 20, seed: int = 42, output_path: str | None = None,
) -> dict:
    """Batch-aware selection: take the most uncertain rows, cluster them, and send the best of each cluster — so a batch of 20 is 20 different questions rather than 20 near-duplicates of the same uncertain region. Use whenever labels come back in batches.

    Args:
        batch_size: Rows to send to annotators.
        output_path: Where to write the labelling queue.
    """
    return queue("uncertainty_diversity", data, label_column, unlabeled_value, features, drop, id_column, base, batch_size, seed, output_path)


@tool
def query_expected_error_reduction(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, id_column: str | None = None,
    base: Base = "logistic_regression", batch_size: int = 10, n_candidates: int = 25, seed: int = 42, output_path: str | None = None,
) -> dict:
    """Expected error reduction: for the most uncertain candidates, imagine each possible label, retrain, and measure the expected loss left on the pool — choose the rows whose answer would help most. Directly targets accuracy, but retrains many models, so it is run on a small candidate set.

    Args:
        batch_size: Rows to send to annotators.
        n_candidates: Uncertain candidates evaluated in full.
        output_path: Where to write the labelling queue.
    """
    return queue("expected_error_reduction", data, label_column, unlabeled_value, features, drop, id_column, base, batch_size, seed, output_path, n_candidates=n_candidates)


@tool
def query_random(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, id_column: str | None = None,
    base: Base = "logistic_regression", batch_size: int = 20, seed: int = 42, output_path: str | None = None,
) -> dict:
    """Random sampling: the baseline every active strategy must beat, and a safe way to also collect an unbiased evaluation set — mix some random rows into every batch so accuracy can be measured honestly.

    Args:
        batch_size: Rows to send to annotators.
        output_path: Where to write the labelling queue.
    """
    return queue("random", data, label_column, unlabeled_value, features, drop, id_column, base, batch_size, seed, output_path)
