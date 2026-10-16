"""Read-only checks before self-training: how much is labelled, whether the unlabelled rows look like the labelled ones, how far the labels alone go, and whether the base learner's confidence can be trusted."""

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
