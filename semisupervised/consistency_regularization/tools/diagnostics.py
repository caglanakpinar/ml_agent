"""Read-only checks before consistency-regularised training: how much is labelled, whether the unlabelled rows look like the labelled ones, how far the labels alone go, and whether the base learner's confidence can be trusted."""

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
def measure_prediction_consistency(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "random_forest",
    augmentation: str = "gaussian", strength: float = 0.3, n_augmentations: int = 10, seed: int = 42,
) -> dict:
    """Train the base learner on the labelled rows and measure how stable its predictions on the *unlabelled* rows are under perturbation: the share of rows whose predicted class never changes, and how many are fragile. Fragile rows sit near the decision boundary — exactly where consistency regularisation has something to teach.

    Args:
        base: Base learner.
        augmentation: gaussian, dropout, swap_noise, mixup or mixed.
        strength: Perturbation size (noise std in standardised units, or the drop/swap probability).
        n_augmentations: Perturbed copies per row.
    """
    from ml_core.semi import base_clf, make_context
    from semisupervised.consistency_regularization.tools._common import prediction_stability

    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, None, 0.3, seed, base)
    model = base_clf(base, seed).fit(ctx.Zl, ctx.yl)
    s = prediction_stability(ctx, model, augmentation, strength, n_augmentations)
    return {"base": base, "augmentation": augmentation, "strength": strength, "unlabelled_rows": int(len(s["per_row"])), "mean_stability": round(s["mean_stability"], 4),
            "fully_stable_share": round(s["fully_stable_share"], 4), "fragile_share": round(float((s["per_row"] < 0.8).mean()), 4),
            "verdict": "almost every prediction is already stable: consistency regularisation has little to add" if s["fully_stable_share"] > 0.95
            else "a sizeable share of unlabelled rows is fragile: consistency regularisation can sharpen the boundary there"}


@tool
def check_augmentation_label_preservation(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "random_forest",
    strengths: list[float] | None = None, seed: int = 42,
) -> dict:
    """For each augmentation type and strength, how often a perturbed *labelled* row is still predicted as its true class by a model trained on clean labels — an augmentation must keep the label, or the consistency target is wrong. Pick the strongest one that still preserves the label.

    Args:
        base: Base learner.
        strengths: Perturbation strengths to try; defaults to 0.1, 0.3, 0.6, 1.0.
    """
    import numpy as np

    from ml_core.semi import base_clf, full_proba, make_context
    from semisupervised.consistency_regularization.tools._common import augment

    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, None, 0.3, seed, base)
    model = base_clf(base, seed).fit(ctx.Zl, ctx.yl)
    Z, y = ctx.Z[ctx.split.l_test], ctx.sd.y[ctx.split.l_test]
    rng = np.random.default_rng(seed)
    clean = float((full_proba(model, Z, ctx.K).argmax(axis=1) == y).mean())
    rows = []
    for kind in ("gaussian", "dropout", "swap_noise", "mixup", "mixed"):
        for s in strengths or [0.1, 0.3, 0.6, 1.0]:
            kept = float(np.mean([(full_proba(model, augment(Z, kind, s, rng), ctx.K).argmax(axis=1) == y).mean() for _ in range(5)]))
            rows.append({"augmentation": kind, "strength": s, "accuracy_on_perturbed": round(kept, 4), "label_preserved": round(kept / max(clean, 1e-9), 4)})
    safe = [r for r in rows if r["label_preserved"] >= 0.95]
    best = max(safe, key=lambda r: r["strength"]) if safe else None
    return {"accuracy_on_clean": round(clean, 4), "table": rows, "recommended": best, "note": "pick the strongest setting that still preserves at least 95% of the clean accuracy"}
