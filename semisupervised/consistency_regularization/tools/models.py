"""Fit a consistency-regularised model: Pi-model, Mean Teacher, FixMatch-style training, an augmentation-only control, consistency-filtered pseudo-labelling and distillation under perturbation."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ml_core.semi import Fitted, base_clf, full_proba, run_semi
from ml_core.tooling import tool
from semisupervised.consistency_regularization.tools._common import RFFNet, augment, softmax, train_net

Aug = Literal["gaussian", "dropout", "swap_noise", "mixup", "mixed"]
Base = Literal["logistic_regression", "random_forest", "hist_gradient_boosting", "svm", "knn", "naive_bayes"]


def _net(label, mode, data, label_column, unlabeled_value, features, drop, truth_column, augmentation, strength, epochs, lr, consistency_weight, test_size, seed, model_path, **kw):
    return run_semi(label, lambda ctx: train_net(ctx, mode, augmentation, strength, epochs, lr, consistency_weight, **kw), data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, "logistic_regression")


@tool
def fit_pi_model(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    augmentation: Aug = "gaussian", strength: float = 0.3, epochs: int = 300, lr: float = 0.05, consistency_weight: float = 5.0, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Pi-model: the network sees two independently perturbed copies of every unlabelled row and is penalised when its two predictions differ, on top of ordinary cross-entropy on the labelled rows. The simplest consistency regulariser.

    Args:
        augmentation: gaussian, dropout, swap_noise, mixup or mixed (gaussian + dropout).
        strength: Perturbation size.
        epochs: Gradient steps.
        lr: Adam step size.
        consistency_weight: Weight of the unlabelled-data term relative to the labelled loss (ramped up slowly).
    """
    return _net(f"pi_model({augmentation}={strength})", "pi", data, label_column, unlabeled_value, features, drop, truth_column, augmentation, strength, epochs, lr, consistency_weight, test_size, seed, model_path)


@tool
def fit_mean_teacher(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    augmentation: Aug = "gaussian", strength: float = 0.3, epochs: int = 300, lr: float = 0.05, consistency_weight: float = 5.0, ema: float = 0.95,
    test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Mean Teacher: the consistency target comes from a teacher whose weights are an exponential moving average of the student's — steadier targets than the Pi-model's own noisy predictions, and the averaged weights are the final model.

    Args:
        augmentation: gaussian, dropout, swap_noise, mixup or mixed.
        strength: Perturbation size.
        epochs: Gradient steps.
        consistency_weight: Weight of the unlabelled-data term.
        ema: Moving-average decay of the teacher; closer to 1 averages over more steps.
    """
    return _net(f"mean_teacher({augmentation}={strength},ema={ema})", "mean_teacher", data, label_column, unlabeled_value, features, drop, truth_column, augmentation, strength, epochs, lr, consistency_weight, test_size, seed, model_path, ema=ema)


@tool
def fit_fixmatch_tabular(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    augmentation: Aug = "gaussian", strength: float = 0.4, threshold: float = 0.9, epochs: int = 300, lr: float = 0.05, consistency_weight: float = 5.0,
    test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """FixMatch for tabular data: a weakly perturbed copy of an unlabelled row yields a pseudo-label if the network is confident; the network is then trained to predict that label from a strongly perturbed copy. Combines pseudo-labelling with consistency — usually the strongest of the three when the augmentation keeps labels.

    Args:
        augmentation: Perturbation family for the strong copy.
        strength: Strong-perturbation size (the weak copy uses a third of it).
        threshold: Confidence the weak copy needs before its label is used.
        epochs: Gradient steps.
        consistency_weight: Weight of the unlabelled-data term.
    """
    return _net(f"fixmatch_tabular({augmentation}={strength},tau={threshold})", "fixmatch", data, label_column, unlabeled_value, features, drop, truth_column, augmentation, strength, epochs, lr, consistency_weight, test_size, seed, model_path, threshold=threshold)


@tool
def fit_augmentation_only_baseline(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    augmentation: Aug = "gaussian", strength: float = 0.3, epochs: int = 300, lr: float = 0.05, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """The control experiment: the same network and the same augmentation, trained on the labelled rows only, with no unlabelled term. If this matches Pi-model / Mean Teacher / FixMatch, the gain came from the augmentation, not from the unlabelled data.

    Args:
        augmentation: gaussian, dropout, swap_noise, mixup or mixed.
        strength: Perturbation size.
        epochs: Gradient steps.
    """
    return _net(f"augmentation_only({augmentation}={strength})", "supervised", data, label_column, unlabeled_value, features, drop, truth_column, augmentation, strength, epochs, lr, 0.0, test_size, seed, model_path)


@tool
def fit_consistency_filtered_pseudo_labeling(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "random_forest", augmentation: Aug = "gaussian", strength: float = 0.3, n_augmentations: int = 10, min_stability: float = 1.0, threshold: float = 0.8,
    test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Use consistency as a *filter* with any learner: pseudo-label an unlabelled row only if its predicted class survives every perturbation (stability ≥ `min_stability`) and its confidence clears the threshold, then retrain. Stable rows are far from the boundary, so their labels are more often right.

    Args:
        base: Learner (any scikit-learn-style classifier works here).
        augmentation: gaussian, dropout, swap_noise, mixup or mixed.
        strength: Perturbation size.
        n_augmentations: Perturbed copies per row.
        min_stability: Share of perturbed copies that must keep the predicted class (1.0 = all).
        threshold: Confidence a row also needs.
    """
    from semisupervised.consistency_regularization.tools._common import prediction_stability

    def trainer(ctx):
        teacher = base_clf(base, seed).fit(ctx.Zl, ctx.yl)
        proba = full_proba(teacher, ctx.Zu, ctx.K)
        stab = prediction_stability(ctx, teacher, augmentation, strength, n_augmentations)
        keep = np.flatnonzero((stab["per_row"] >= min_stability) & (proba.max(axis=1) >= threshold))
        labels = proba[keep].argmax(axis=1)
        model = base_clf(base, seed).fit(np.vstack([ctx.Zl, ctx.Zu[keep]]), np.concatenate([ctx.yl, labels]))
        return Fitted(model, ctx.K), {"mean_stability_of_unlabelled": round(stab["mean_stability"], 4), "kept_rows": int(len(keep)),
                                      "pseudo": {"index": ctx.split.unl[keep], "label": labels, "confidence": proba[keep].max(axis=1)}}

    return run_semi(f"consistency_filtered_pseudo_labels({base},{augmentation}={strength})", trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, base)


@tool
def fit_perturbation_distillation(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    teacher: Base = "random_forest", augmentation: Aug = "gaussian", strength: float = 0.3, copies: int = 5, epochs: int = 300, lr: float = 0.05, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Distillation under perturbation: a tree-based or other teacher fitted on the labels gives soft probabilities for *perturbed copies* of every unlabelled row, and a smooth network student learns to reproduce them (plus the true labels). The student inherits the teacher's knowledge with a decision surface that is smooth by construction.

    Args:
        teacher: Learner that produces the soft targets.
        augmentation: gaussian, dropout, swap_noise, mixup or mixed.
        strength: Perturbation size.
        copies: Perturbed copies per unlabelled row.
        epochs: Student gradient steps.
    """
    def trainer(ctx):
        rng = np.random.default_rng(seed)
        t = base_clf(teacher, seed).fit(ctx.Zl, ctx.yl)
        Zs = np.vstack([augment(ctx.Zu, augmentation, strength, rng) for _ in range(copies)] + [ctx.Zl])
        T = np.vstack([full_proba(t, Zs[: len(ctx.Zu) * copies], ctx.K), np.eye(ctx.K)[ctx.yl]])
        net = RFFNet(ctx.Z.shape[1], ctx.K, 256, 0.5, seed)
        F = net.phi(Zs)
        m = {k: np.zeros_like(v) for k, v in (("W", net.W), ("c", net.c))}
        v = {k: np.zeros_like(x) for k, x in (("W", net.W), ("c", net.c))}
        for step in range(1, epochs + 1):
            P = softmax(F @ net.W + net.c)
            grads = {"W": F.T @ (P - T) / len(T) + 1e-3 * net.W, "c": (P - T).mean(axis=0)}
            for k, p in (("W", net.W), ("c", net.c)):
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                p -= lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
        q = net.predict_proba(ctx.Zu)
        sure = np.flatnonzero(q.max(axis=1) >= 0.9)
        return net, {"teacher": teacher, "student_training_rows": int(len(T)), "pseudo": {"index": ctx.split.unl[sure], "label": q[sure].argmax(axis=1), "confidence": q[sure].max(axis=1)}}

    return run_semi(f"perturbation_distillation({teacher}->rff_net)", trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, teacher)
