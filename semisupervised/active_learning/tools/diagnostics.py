"""Read-only checks before active learning: the seed set, whether the pool resembles it, how labels pay off, and how many more labels a target accuracy would take."""

from __future__ import annotations

from ml_core.semi_diag import (
    confidence_reliability,
    coverage_profile,
    scarcity_curve,
    shift_check,
    supervised_baseline,
    threshold_for_precision,
)
from ml_core.tooling import tool


@tool
def profile_label_coverage(data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None) -> dict:
    """Profile a partially labelled dataset: labelled vs unlabelled rows, the class balance among the labelled ones, labels per class and per feature, and whether there are enough labels for a base learner to start from at all."""
    return coverage_profile(data, label_column, unlabeled_value, features, drop)


@tool
def check_labeled_unlabeled_shift(data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, k: int = 8, seed: int = 42) -> dict:
    """Adversarial validation: train a classifier to tell labelled rows from unlabelled ones. An AUC near 0.5 means they come from the same distribution (semi-supervised learning is safe); a high AUC means the labels were collected on a biased subset and pseudo or propagated labels will extrapolate badly.

    Args:
        k: How many most-shifted features to list.
    """
    return shift_check(data, label_column, unlabeled_value, features, drop, k, seed)


@tool
def baseline_supervised_only(data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "logistic_regression", test_size: float = 0.3, seed: int = 42) -> dict:
    """Fit the base learner on the labelled rows alone and score it on held-out labelled rows — the number every semi-supervised method has to beat.

    Args:
        base: logistic_regression, random_forest, hist_gradient_boosting, svm, knn or naive_bayes.
    """
    return supervised_baseline(data, label_column, unlabeled_value, features, drop, base, test_size, seed)


@tool
def label_scarcity_curve(data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "logistic_regression", test_size: float = 0.3, seed: int = 42) -> dict:
    """Supervised-only accuracy as the number of labels grows (10%, 25%, 50%, 100% of what is available): a steep curve means more labels — or pseudo-labels — would help a lot; a flat one means the labels are not the bottleneck.

    Args:
        base: Base learner.
    """
    return scarcity_curve(data, label_column, unlabeled_value, features, drop, base, test_size, seed)


@tool
def check_base_learner_confidence(data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "logistic_regression", seed: int = 42) -> dict:
    """Reliability of the base learner's confidence, from cross-validation on the labelled rows: in each confidence band, how often is the top prediction actually right? Methods that trust confidence poison themselves if it is over-confident.

    Args:
        base: Base learner.
    """
    return confidence_reliability(data, label_column, unlabeled_value, features, drop, base, seed)


@tool
def suggest_confidence_threshold(data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "logistic_regression", target_precision: float = 0.95, seed: int = 42) -> dict:
    """Pick the confidence threshold at which the base learner's pseudo-labels would be right at least `target_precision` of the time, measured by cross-validation on the labelled rows, and how many unlabelled rows would clear it.

    Args:
        base: Base learner.
        target_precision: Wanted share of correct pseudo-labels, e.g. 0.95.
    """
    return threshold_for_precision(data, label_column, unlabeled_value, features, drop, base, target_precision, seed)


@tool
def check_cold_start(data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "logistic_regression", seed: int = 42) -> dict:
    """Is the seed set big enough to start querying? Uncertainty sampling needs a model that is at least better than chance: reports labelled rows per class, the cross-validated accuracy of the seed-set model, and whether to start with random or diversity (core-set) sampling instead.

    Args:
        base: Base learner.
    """
    import numpy as np

    from ml_core.semi import base_clf
    from semisupervised.active_learning.tools._common import build_env

    env = build_env(data, label_column, features, drop, unlabeled_value, None, seed)
    y = env.y[env.labelled]
    counts = np.bincount(y, minlength=env.K)
    from sklearn.model_selection import cross_val_score

    acc = float(cross_val_score(base_clf(base, seed), env.Z[env.labelled], y, cv=max(2, min(5, int(counts[counts > 0].min())))).mean()) if counts[counts > 0].min() >= 2 else float("nan")
    chance = float(counts.max() / counts.sum())
    return {"labelled": int(len(y)), "pool": int(len(env.pool)), "per_class": dict(zip(env.classes, counts.astype(int).tolist())), "seed_model_cv_accuracy": round(acc, 4) if acc == acc else None, "majority_class_rate": round(chance, 4),
            "recommendation": "too few labels per class to train anything: label a random or core-set batch first (query_core_set)" if counts.min() < 3 or acc != acc else
            "the seed model is no better than the majority rate: start with core-set or random sampling, switch to uncertainty once it learns something" if acc < chance + 0.05 else
            "the seed model is useful: uncertainty-based queries are safe to start with"}


@tool
def estimate_labeling_budget(data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "logistic_regression", target_accuracy: float = 0.9, seed: int = 42) -> dict:
    """Extrapolate how many labels a target accuracy needs: fits a power-law learning curve (error ≈ a·n^−b) to the accuracy of the base learner at growing numbers of labels, and reads off the label count for the target. A rough planning number, not a promise — and what active learning is trying to shrink.

    Args:
        base: Base learner.
        target_accuracy: Accuracy to plan for.
    """
    import numpy as np

    from ml_core.semi import base_clf, full_proba, metrics
    from semisupervised.active_learning.tools._common import build_env

    env = build_env(data, label_column, features, drop, unlabeled_value, None, seed, simulate=True, initial_size=10)
    full = np.concatenate([env.labelled, env.pool])
    rng = np.random.default_rng(seed)
    pts = []
    for n in sorted({int(x) for x in np.linspace(max(2 * env.K, 10), len(full), 7)}):
        accs = []
        for rep in range(3):
            idx = rng.choice(full, size=n, replace=False)
            if len(set(env.y[idx])) < 2:
                continue
            accs.append(metrics(env.y[env.test], full_proba(base_clf(base, seed).fit(env.Z[idx], env.y[idx]), env.Z[env.test], env.K))["accuracy"])
        if accs:
            pts.append((n, float(np.mean(accs))))
    n_arr, err = np.array([p[0] for p in pts], float), np.maximum(1 - np.array([p[1] for p in pts]), 1e-3)
    b, loga = np.polyfit(np.log(n_arr), np.log(err), 1)
    need = float(np.exp((np.log(max(1 - target_accuracy, 1e-3)) - loga) / b)) if b < 0 else None
    return {"curve": [{"labels": int(n), "accuracy": round(a, 4)} for n, a in pts], "fitted_exponent": round(float(b), 3), "target_accuracy": target_accuracy,
            "labels_needed_with_random_sampling": int(need) if need and need < 1e7 else None,
            "note": "random-sampling estimate; run compare_query_strategies to see how many fewer labels an active strategy needs. If the exponent is near zero, labels are not what limits accuracy."}


@tool
def profile_pool_uncertainty(data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "logistic_regression", seed: int = 42) -> dict:
    """How uncertain is the current model about the pool? Distribution of entropy and margin over the unlabelled rows, the share that is already confident (nothing to learn there), and the share near the decision boundary — shows how much active querying has to work with.

    Args:
        base: Base learner.
    """
    import numpy as np

    from ml_core.semi import base_clf
    from semisupervised.active_learning.tools._common import build_env, proba_of, uncertainty

    env = build_env(data, label_column, features, drop, unlabeled_value, None, seed)
    p = proba_of(base_clf(base, seed).fit(env.Z[env.labelled], env.y[env.labelled]), env.Z[env.pool], env.K)
    ent, marg = uncertainty(p, "entropy"), 1 - uncertainty(p, "margin")
    return {"pool": int(len(p)), "mean_entropy": round(float(ent.mean()), 4), "max_entropy": round(float(np.log(env.K)), 4), "confident_share_over_0.9": round(float((p.max(axis=1) > 0.9).mean()), 4),
            "near_boundary_share_margin_under_0.2": round(float((marg < 0.2).mean()), 4), "predicted_class_share": dict(zip(env.classes, np.round(np.bincount(p.argmax(axis=1), minlength=env.K) / len(p), 4).tolist())),
            "reading": "a large near-boundary share means uncertainty sampling has useful rows to pick; if almost everything is confident the model is either right or confidently wrong — check with random samples too"}
