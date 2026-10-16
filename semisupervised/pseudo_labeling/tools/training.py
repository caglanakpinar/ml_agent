"""Iterative and noise-robust pseudo-label training: noisy student, curriculum, loss-filtered labels, confidence weighting."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ml_core.semi import Fitted, run_semi
from ml_core.tooling import tool
from semisupervised.pseudo_labeling.tools._common import (
    gen_threshold,
    noisy_student,
    pseudo_trainer,
    result,
    teacher_for,
)

Base = Literal["logistic_regression", "random_forest", "hist_gradient_boosting", "svm", "knn", "naive_bayes"]
Student = Literal["random_forest", "hist_gradient_boosting", "logistic_regression", "svm"]


@tool
def fit_noisy_student(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", student: Student = "random_forest", threshold: float = 0.8, rounds: int = 3, noise: float = 0.3,
    test_size: float = 0.3, seed: int = 42, model_path: str | None = None, output_path: str | None = None,
) -> dict:
    """Noisy Student: a teacher pseudo-labels the pool, a student (often a stronger model) is trained on labelled + pseudo rows *with input noise and feature dropout*, and becomes the next round's teacher. The noise stops the student from merely copying the teacher, so it can end up better than it.

    Args:
        student: Learner used for the student models.
        rounds: Teacher → student generations.
        noise: Standard deviation of Gaussian jitter on the (standardised) features.
        output_path: Where to write the relabelled dataset.
    """
    return run_semi(f"noisy_student({base}->{student}x{rounds})", lambda ctx: noisy_student(ctx, threshold, rounds, noise, student, output_path), data, label_column, features, drop,
                    unlabeled_value, truth_column, test_size, seed, model_path, base)


@tool
def fit_pseudo_labeling_curriculum(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", start_threshold: float = 0.97, end_threshold: float = 0.75, rounds: int = 5, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Curriculum pseudo-labelling: the first round trusts only the most confident rows, each later round relaxes the threshold and re-labels the whole pool with the improved model (labels are recomputed, not frozen, so early mistakes can be corrected).

    Args:
        start_threshold: Confidence needed in the first round.
        end_threshold: Confidence needed in the last round.
        rounds: Number of relabelling rounds.
    """
    def trainer(ctx):
        from ml_core.semi import base_clf, full_proba

        thresholds = np.linspace(start_threshold, end_threshold, rounds)
        model, history, gen = teacher_for(ctx), [], {"pos": np.array([], int), "label": np.array([], int), "conf": np.array([])}
        for r, t in enumerate(thresholds):
            proba = full_proba(model, ctx.Zu, ctx.K)
            gen = result(np.flatnonzero(proba.max(axis=1) >= t), proba)
            Z, y = np.vstack([ctx.Zl, ctx.Zu[gen["pos"]]]), np.concatenate([ctx.yl, gen["label"]])
            model = base_clf(ctx.base, ctx.seed).fit(Z, y)
            history.append({"round": r, "threshold": round(float(t), 3), "pseudo_labelled": int(len(gen["pos"]))})
        return Fitted(model, ctx.K), {"rounds": history, "pseudo": {"index": ctx.split.unl[gen["pos"]], "label": gen["label"], "confidence": gen["conf"]}}

    return run_semi(f"pseudo_curriculum({start_threshold}->{end_threshold},{base})", trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, base)


@tool
def fit_pseudo_labeling_with_loss_filter(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", threshold: float = 0.75, drop_fraction: float = 0.2, test_size: float = 0.3, seed: int = 42, model_path: str | None = None, output_path: str | None = None,
) -> dict:
    """Pseudo-label generously, then clean up with the small-loss trick: cross-validate on labelled + pseudo-labelled rows and discard the pseudo-labels the cross-validated model finds most surprising — noisy labels have high loss before the model memorises them.

    Args:
        threshold: Confidence needed to be a candidate pseudo-label.
        drop_fraction: Share of candidate pseudo-labels removed as likely wrong.
        output_path: Where to write the relabelled dataset.
    """
    trainer = pseudo_trainer(gen_threshold, {"threshold": threshold}, drop_fraction=drop_fraction, output_path=output_path)
    return run_semi(f"pseudo_labels(threshold={threshold},loss_filter={drop_fraction},{base})", trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, base)


@tool
def fit_pseudo_labeling_confidence_weighted(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", threshold: float = 0.6, test_size: float = 0.3, seed: int = 42, model_path: str | None = None, output_path: str | None = None,
) -> dict:
    """Pseudo-label at a low threshold but weight each pseudo-labelled row by its confidence when retraining (real labels weigh 1) — more rows, with doubtful ones counting for less.

    Args:
        threshold: Confidence needed to be a candidate pseudo-label.
        output_path: Where to write the relabelled dataset.
    """
    trainer = pseudo_trainer(gen_threshold, {"threshold": threshold}, weight_by_confidence=True, output_path=output_path)
    return run_semi(f"pseudo_labels(weighted,threshold={threshold},{base})", trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, base)


@tool
def fit_pseudo_labeling_with_stronger_final_model(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", final: Student = "hist_gradient_boosting", threshold: float = 0.9, test_size: float = 0.3, seed: int = 42, model_path: str | None = None, output_path: str | None = None,
) -> dict:
    """Use a simple, well-calibrated teacher to produce the pseudo-labels, then train a *different, stronger* model on labelled + pseudo-labelled rows — the pseudo-labels transfer the teacher's knowledge to a model that could not have been trained on so few labels.

    Args:
        final: Learner trained on the combined data.
        output_path: Where to write the relabelled dataset.
    """
    trainer = pseudo_trainer(gen_threshold, {"threshold": threshold, "calibrated": True}, output_path=output_path, final_base=final)
    return run_semi(f"pseudo_labels({base}->{final})", trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, base)
