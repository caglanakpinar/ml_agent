"""Fit a self-training model: the base learner labels the unlabelled rows it is sure about, learns from them, and repeats. Each variant differs in how it decides what to trust."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ml_core.ml import sk
from ml_core.semi import base_clf, full_proba, run_semi
from ml_core.tooling import tool
from semisupervised.self_training.tools._common import Fitted, self_train

Base = Literal["logistic_regression", "random_forest", "hist_gradient_boosting", "svm", "knn", "naive_bayes"]


def _run(label, rule, data, label_column, unlabeled_value, features, drop, truth_column, base, test_size, seed, model_path, **kw):
    return run_semi(label, lambda ctx: self_train(ctx, rule=rule, **kw), data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, base)


@tool
def fit_self_training_threshold(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", threshold: float = 0.9, max_iter: int = 10, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Classic self-training: each round, every unlabelled row the model predicts with confidence above the threshold is added with its predicted label, then the model is refitted. Compares itself with the same learner trained on the real labels alone.

    Args:
        max_iter: Most rounds of label-and-refit.
    """
    return _run(f"self_training(threshold={threshold},{base})", "threshold", data, label_column, unlabeled_value, features, drop, truth_column, base, test_size, seed, model_path, threshold=threshold, max_iter=max_iter)


@tool
def fit_self_training_k_best(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", k_best: int | None = None, max_iter: int = 10, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Self-training that adds a fixed number of the most confident rows per round instead of using a threshold — a slower, steadier growth that does not depend on how well calibrated the model's confidence is.

    Args:
        k_best: Rows added per round; defaults to the unlabelled pool divided by the number of rounds.
        max_iter: Most rounds of label-and-refit.
    """
    return _run(f"self_training(k_best={k_best},{base})", "k_best", data, label_column, unlabeled_value, features, drop, truth_column, base, test_size, seed, model_path, k_best=k_best, max_iter=max_iter)


@tool
def fit_self_training_curriculum(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", start_threshold: float = 0.98, end_threshold: float = 0.8, max_iter: int = 10, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Curriculum self-training: begin with a very strict threshold (only the easiest rows) and relax it each round, so the model has grown stronger before it takes on the harder rows.

    Args:
        start_threshold: Confidence needed in the first round.
        end_threshold: Confidence needed in the last round.
        max_iter: Number of rounds over which the threshold relaxes.
    """
    return _run(f"self_training_curriculum({start_threshold}->{end_threshold},{base})", "curriculum", data, label_column, unlabeled_value, features, drop, truth_column, base, test_size, seed, model_path,
                threshold=start_threshold, end_threshold=end_threshold, max_iter=max_iter)


@tool
def fit_self_training_class_balanced(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", threshold: float = 0.8, max_iter: int = 10, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Self-training that adds confident rows per class in proportion to the labelled class balance — stops the majority class from swallowing every pseudo-label and starving the rare ones.

    Args:
        threshold: Minimum confidence for a row to be a candidate.
        max_iter: Most rounds of label-and-refit.
    """
    return _run(f"self_training_balanced({base})", "balanced", data, label_column, unlabeled_value, features, drop, truth_column, base, test_size, seed, model_path, threshold=threshold, max_iter=max_iter)


@tool
def fit_self_training_calibrated(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "random_forest", threshold: float = 0.9, max_iter: int = 10, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Self-training with the base learner wrapped in probability calibration, so 'confidence 0.9' really means right about 90% of the time — the fix for over-confident learners such as boosted trees, SVMs and naive Bayes.

    Args:
        threshold: Calibrated confidence needed to trust a row.
        max_iter: Most rounds of label-and-refit.
    """
    return _run(f"self_training_calibrated({base})", "threshold", data, label_column, unlabeled_value, features, drop, truth_column, base, test_size, seed, model_path, threshold=threshold, max_iter=max_iter, calibrated=True)


@tool
def fit_self_training_ensemble_agreement(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", threshold: float = 0.85, max_iter: int = 10, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Self-training where a row is pseudo-labelled only if three different learners (the chosen base, a random forest and a logistic regression) all predict the same class and their average confidence clears the threshold — much cleaner pseudo-labels at the cost of fewer.

    Args:
        threshold: Minimum average confidence when the learners agree.
        max_iter: Most rounds of label-and-refit.
    """
    return _run(f"self_training_ensemble({base}+rf+logreg)", "threshold", data, label_column, unlabeled_value, features, drop, truth_column, base, test_size, seed, model_path, threshold=threshold, max_iter=max_iter, ensemble=True)


@tool
def fit_self_training_confidence_weighted(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", threshold: float = 0.7, max_iter: int = 10, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Self-training where each pseudo-labelled row is weighted by its confidence when the model is refitted (real labels weigh 1), so a doubtful pseudo-label pulls less than a sure one — lets the threshold be lower without letting noise dominate.

    Args:
        threshold: Minimum confidence for a row to be a candidate.
        max_iter: Most rounds of label-and-refit.
    """
    return _run(f"self_training_weighted({base})", "threshold", data, label_column, unlabeled_value, features, drop, truth_column, base, test_size, seed, model_path, threshold=threshold, max_iter=max_iter, confidence_weights=True)


@tool
def fit_sklearn_self_training(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Literal["logistic_regression", "random_forest", "hist_gradient_boosting", "svm", "naive_bayes"] = "logistic_regression", criterion: Literal["threshold", "k_best"] = "threshold",
    threshold: float = 0.9, k_best: int = 50, max_iter: int = 10, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """scikit-learn's own SelfTrainingClassifier — the reference implementation, useful as a cross-check on the loops above and to see how many rounds it ran before nothing more cleared the bar.

    Args:
        criterion: threshold or k_best selection.
        k_best: Rows added per round when criterion is k_best.
        max_iter: Most rounds.
    """
    sk()
    from sklearn.semi_supervised import SelfTrainingClassifier

    def trainer(ctx):
        st = SelfTrainingClassifier(base_clf(base, seed), threshold=threshold, criterion=criterion, k_best=k_best, max_iter=max_iter)
        Z = np.vstack([ctx.Zl, ctx.Zu])
        y = np.concatenate([ctx.yl, -np.ones(len(ctx.Zu), int)])
        st.fit(Z, y)
        pseudo = st.transduction_[len(ctx.yl):]
        got = np.flatnonzero(st.labeled_iter_[len(ctx.yl):] > 0)
        return Fitted(st, ctx.K), {"rounds_run": int(st.n_iter_), "termination": str(st.termination_condition_),
                                   "pseudo": {"index": ctx.split.unl[got], "label": pseudo[got], "confidence": full_proba(st, ctx.Zu[got], ctx.K).max(axis=1)}}

    return run_semi(f"sklearn_self_training({criterion},{base})", trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, base)
