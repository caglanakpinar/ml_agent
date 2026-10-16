"""Generate pseudo-labels for the unlabelled rows and retrain on labelled + pseudo-labelled data. Each tool can also write the relabelled dataset (every row with its label, where the label came from and its confidence)."""

from __future__ import annotations

from typing import Literal

from ml_core.semi import run_semi
from ml_core.tooling import tool
from semisupervised.pseudo_labeling.tools._common import (
    gen_cluster_consistent,
    gen_ensemble,
    gen_soft,
    gen_threshold,
    gen_top_per_class,
    pseudo_trainer,
)

Base = Literal["logistic_regression", "random_forest", "hist_gradient_boosting", "svm", "knn", "naive_bayes"]


@tool
def generate_pseudo_labels_threshold(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", threshold: float = 0.9, calibrated: bool = False, test_size: float = 0.3, seed: int = 42, model_path: str | None = None, output_path: str | None = None,
) -> dict:
    """One-shot pseudo-labelling: train on the labelled rows, label every unlabelled row the model predicts above the confidence threshold, retrain on both, and (optionally) write the relabelled dataset. The simplest and most common recipe.

    Args:
        calibrated: Calibrate the teacher's probabilities first, so the threshold means what it says.
        output_path: Where to write the relabelled dataset (every row, its label, label_source, pseudo_confidence).
    """
    trainer = pseudo_trainer(gen_threshold, {"threshold": threshold, "calibrated": calibrated}, output_path=output_path)
    return run_semi(f"pseudo_labels(threshold={threshold},{base})", trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, base)


@tool
def generate_pseudo_labels_top_per_class(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", fraction: float = 0.3, min_confidence: float = 0.5, test_size: float = 0.3, seed: int = 42, model_path: str | None = None, output_path: str | None = None,
) -> dict:
    """Pseudo-label the most confident fraction of the unlabelled rows *within each predicted class* — guarantees every class gets pseudo-labels in proportion, which a global threshold does not when one class is easier than another.

    Args:
        fraction: Share of each class's candidates to keep, most confident first.
        min_confidence: Floor below which a row is never labelled.
        output_path: Where to write the relabelled dataset.
    """
    trainer = pseudo_trainer(gen_top_per_class, {"fraction": fraction, "min_conf": min_confidence}, output_path=output_path)
    return run_semi(f"pseudo_labels(top_{fraction}_per_class,{base})", trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, base)


@tool
def generate_pseudo_labels_ensemble(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", threshold: float = 0.85, test_size: float = 0.3, seed: int = 42, model_path: str | None = None, output_path: str | None = None,
) -> dict:
    """Pseudo-label only where several different learners (the base, a random forest, a logistic regression and boosted trees) all predict the same class and their mean confidence clears the threshold — fewer labels, but much cleaner ones, because different models rarely share the same mistake.

    Args:
        output_path: Where to write the relabelled dataset.
    """
    trainer = pseudo_trainer(gen_ensemble, {"threshold": threshold}, output_path=output_path)
    return run_semi(f"pseudo_labels(ensemble_agreement,{base})", trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, base)


@tool
def generate_pseudo_labels_cluster_consistent(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", threshold: float = 0.7, n_clusters: int | None = None, purity: float = 0.8, test_size: float = 0.3, seed: int = 42, model_path: str | None = None, output_path: str | None = None,
) -> dict:
    """Use the cluster assumption: cluster all rows, and pseudo-label an unlabelled row only when the classifier's prediction agrees with the majority label of the labelled rows in its cluster (and that cluster is pure enough). Guards against classifier mistakes that cut across natural groups.

    Args:
        n_clusters: Number of k-means clusters; chosen from the data size when omitted.
        purity: Share of labelled rows in a cluster that must agree for it to vouch for a label.
        output_path: Where to write the relabelled dataset.
    """
    trainer = pseudo_trainer(gen_cluster_consistent, {"threshold": threshold, "n_clusters": n_clusters, "purity": purity}, output_path=output_path)
    return run_semi(f"pseudo_labels(cluster_consistent,{base})", trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, base)


@tool
def generate_soft_pseudo_labels(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", min_probability: float = 0.2, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Soft pseudo-labels: every unlabelled row is used, once for each class the teacher gives at least `min_probability`, weighted by that probability — keeps the teacher's uncertainty instead of forcing a hard choice, so no row is thrown away and no doubtful label is trusted fully.

    Args:
        min_probability: Smallest class probability that still earns a (weighted) copy of the row.
    """
    trainer = pseudo_trainer(gen_soft, {"min_prob": min_probability})
    return run_semi(f"soft_pseudo_labels(min_p={min_probability},{base})", trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, base)
