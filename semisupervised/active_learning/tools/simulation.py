"""Replay active learning against known answers before spending on labels: learning curves, strategy comparison, batch size, labels-to-target, and stopping rules."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ml_core.semi import base_clf, metrics
from ml_core.tooling import tool
from semisupervised.active_learning.tools._common import build_env, curve_area, proba_of, run_simulation, select

Base = Literal["logistic_regression", "random_forest", "hist_gradient_boosting", "svm", "knn", "naive_bayes"]
Strategy = Literal["random", "least_confident", "margin", "entropy", "vote_entropy", "kl", "bald", "density_weighted", "core_set", "uncertainty_diversity"]


def _env(data, label_column, unlabeled_value, features, drop, truth_column, seed, initial_size, test_size=0.3):
    return build_env(data, label_column, features, drop, unlabeled_value, truth_column, seed, test_size, initial_size, simulate=True)


@tool
def simulate_active_learning(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    strategy: Strategy = "entropy", base: Base = "logistic_regression", initial_size: int = 10, batch_size: int = 10, rounds: int = 10, seed: int = 42,
) -> dict:
    """Simulate one strategy end to end: start from a small labelled seed, repeatedly query a batch, reveal the true labels, retrain, and record held-out accuracy — the learning curve a real labelling campaign would follow. Needs fully known labels (a labelled dataset, or a truth column).

    Args:
        strategy: Query strategy to simulate.
        initial_size: Labelled rows to start from.
        batch_size: Rows queried per round.
        rounds: Number of query rounds.
    """
    env = _env(data, label_column, unlabeled_value, features, drop, truth_column, seed, initial_size)
    curve = run_simulation(env, strategy, base, batch_size, rounds, np.random.default_rng(seed))
    return {"strategy": strategy, "base": base, "test_rows": int(len(env.test)), "pool_at_start": int(len(env.pool)), "curve": curve, "final_accuracy": curve[-1]["accuracy"], "area_under_curve": round(curve_area(curve), 4)}


@tool
def compare_query_strategies(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    strategies: list[str] | None = None, base: Base = "logistic_regression", initial_size: int = 10, batch_size: int = 10, rounds: int = 8, repeats: int = 3, seed: int = 42,
) -> dict:
    """The decision tool: replay several query strategies (and random sampling) from the same seeds and compare area under the learning curve, final accuracy, and how many labels each needed to match random sampling's final accuracy. Averaged over repeats, because one run of active learning is noisy.

    Args:
        strategies: Strategies to compare; defaults to random, entropy, margin, vote_entropy, density_weighted, uncertainty_diversity and core_set.
        initial_size: Labelled rows to start from.
        batch_size: Rows queried per round.
        rounds: Number of query rounds.
        repeats: Independent repeats with different seeds.
    """
    names = strategies or ["random", "entropy", "margin", "vote_entropy", "density_weighted", "uncertainty_diversity", "core_set"]
    curves: dict[str, list[list[dict]]] = {s: [] for s in names}
    for rep in range(repeats):
        env = _env(data, label_column, unlabeled_value, features, drop, truth_column, seed + rep, initial_size)
        for s in names:
            curves[s].append(run_simulation(env, s, base, batch_size, rounds, np.random.default_rng(seed + rep)))
    mean_curve = {s: [float(np.mean([c[i]["accuracy"] for c in cs])) for i in range(len(cs[0]))] for s, cs in curves.items()}
    labels = [c["labels"] for c in curves[names[0]][0]]
    rand = mean_curve.get("random")
    rows = []
    for s in names:
        row = {"strategy": s, "final_accuracy": round(mean_curve[s][-1], 4), "area_under_curve": round(float(np.mean([curve_area(c) for c in curves[s]])), 4)}
        if rand:
            target = rand[-1]
            hit = next((labels[i] for i, a in enumerate(mean_curve[s]) if a >= target - 1e-9), None)
            row["labels_to_match_random_final"] = hit
            row["labels_saved_vs_random"] = (labels[-1] - hit) if hit else 0
        rows.append(row)
    rows.sort(key=lambda r: -r["area_under_curve"])
    return {"base": base, "repeats": repeats, "labels_at_each_round": labels, "strategies": rows, "best": rows[0]["strategy"],
            "mean_accuracy_curves": {s: [round(a, 4) for a in c] for s, c in mean_curve.items()},
            "reading": "if no strategy beats random, active learning is not worth its operational cost on this data"}


@tool
def simulate_batch_size_effect(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    strategy: Strategy = "entropy", base: Base = "logistic_regression", batch_sizes: list[int] | None = None, total_labels: int = 60, initial_size: int = 10, seed: int = 42,
) -> dict:
    """With the same total label budget, how much worse is it to label in big batches than one at a time? Compares final accuracy for several batch sizes — the cost of annotators returning labels in bulk.

    Args:
        strategy: Query strategy.
        batch_sizes: Batch sizes to try; defaults to 1, 5, 10, 20.
        total_labels: Labels to acquire in total on top of the seed set.
    """
    env = _env(data, label_column, unlabeled_value, features, drop, truth_column, seed, initial_size)
    rows = []
    for b in batch_sizes or [1, 5, 10, 20]:
        curve = run_simulation(env, strategy, base, b, max(1, total_labels // b), np.random.default_rng(seed))
        rows.append({"batch_size": b, "rounds": len(curve) - 1, "final_labels": curve[-1]["labels"], "final_accuracy": curve[-1]["accuracy"]})
    return {"strategy": strategy, "sweep": rows, "reading": "a small drop from the smallest to the largest batch means batching is cheap; a large drop means near-duplicates are wasting labels — use query_uncertainty_diversity_batch"}


@tool
def simulate_labels_to_target(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    target_accuracy: float = 0.85, strategy: Strategy = "margin", base: Base = "logistic_regression", batch_size: int = 10, max_rounds: int = 30, initial_size: int = 10, seed: int = 42,
) -> dict:
    """How many labels does a strategy need to reach a target accuracy, versus random sampling? Stops as soon as the target is hit (or the budget runs out) — the number to put in a labelling budget.

    Args:
        target_accuracy: Held-out accuracy to reach.
        strategy: Query strategy.
        batch_size: Rows queried per round.
        max_rounds: Give up after this many rounds.
    """
    env = _env(data, label_column, unlabeled_value, features, drop, truth_column, seed, initial_size)
    out = {}
    for s in dict.fromkeys([strategy, "random"]):
        curve = run_simulation(env, s, base, batch_size, max_rounds, np.random.default_rng(seed))
        hit = next((c["labels"] for c in curve if c["accuracy"] >= target_accuracy), None)
        out[s] = {"labels_needed": hit, "reached": hit is not None, "best_accuracy_seen": max(c["accuracy"] for c in curve)}
    saved = (out["random"]["labels_needed"] - out[strategy]["labels_needed"]) if out["random"]["reached"] and out[strategy]["reached"] and strategy != "random" else None
    return {"target_accuracy": target_accuracy, "results": out, "labels_saved_vs_random": saved}


@tool
def check_stopping_criteria(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    strategy: Strategy = "margin", base: Base = "logistic_regression", batch_size: int = 10, rounds: int = 12, initial_size: int = 10, seed: int = 42,
) -> dict:
    """Test stopping rules on a simulated campaign: after every round, the agreement of the model's pool predictions with the previous round, the pool's remaining uncertainty and the held-out accuracy gain — shows when more labels stop being worth it and which cheap signal (no test set needed) predicted that point.

    Args:
        strategy: Query strategy.
        batch_size: Rows queried per round.
        rounds: Rounds to simulate.
    """
    env = _env(data, label_column, unlabeled_value, features, drop, truth_column, seed, initial_size)
    rng = np.random.default_rng(seed)
    lab, pool = env.labelled.copy(), env.pool.copy()
    prev = None
    rows = []
    for r in range(rounds + 1):
        model = base_clf(base, seed).fit(env.Z[lab], env.y[lab])
        p = proba_of(model, env.Z[pool], env.K) if len(pool) else np.zeros((0, env.K))
        pred = p.argmax(axis=1)
        acc = metrics(env.y[env.test], proba_of(model, env.Z[env.test], env.K))["accuracy"]
        rows.append({"round": r, "labels": int(len(lab)), "holdout_accuracy": acc, "mean_pool_uncertainty": round(float((1 - p.max(axis=1)).mean()), 4) if len(p) else 0.0,
                     "agreement_with_previous_round": round(float((pred == prev).mean()), 4) if prev is not None and len(prev) == len(pred) + batch_size - batch_size else None})
        prev = pred
        if r == rounds or len(pool) == 0:
            break
        picks, _ = select(strategy, env, base, min(batch_size, len(pool)), lab, pool, rng)
        lab = np.concatenate([lab, pool[picks]])
        keep = np.setdiff1d(np.arange(len(pool)), picks)
        pool = pool[keep]
        prev = pred[keep]
    plateau = next((i for i in range(2, len(rows)) if rows[i]["holdout_accuracy"] - rows[i - 2]["holdout_accuracy"] < 0.005), None)
    signal = next((i for i in range(1, len(rows)) if (rows[i]["agreement_with_previous_round"] or 0) >= 0.98), None)
    return {"curve": rows, "accuracy_plateau_round": plateau, "prediction_stability_round": signal,
            "rule_of_thumb": "stop when the pool predictions agree with the previous round on at least 98% of rows (no test set needed) — compare that round with the accuracy plateau above"}
