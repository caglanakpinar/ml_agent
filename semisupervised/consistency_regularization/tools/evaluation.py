"""Judge consistency regularisation: methods head to head, augmentation strength sweeps, the training trajectory, and scoring new rows."""

from __future__ import annotations


from ml_core.semi import base_clf, full_proba, make_context, mask_labels, metrics, predict_saved
from ml_core.tooling import tool
from semisupervised.consistency_regularization.tools._common import train_net


@tool
def compare_consistency_methods(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, augmentation: str = "gaussian",
    strength: float = 0.3, epochs: int = 250, test_size: float = 0.3, seed: int = 42,
) -> dict:
    """Train the augmentation-only control, Pi-model, Mean Teacher and FixMatch with the same network and augmentation, and compare held-out accuracy with a supervised-only logistic regression — separates the effect of the augmentation from the effect of the unlabelled data.

    Args:
        augmentation: gaussian, dropout, swap_noise, mixup or mixed.
        strength: Perturbation size.
        epochs: Gradient steps per method.
    """
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, None, test_size, seed, "logistic_regression")
    yt = ctx.sd.y[ctx.split.l_test]
    sup = metrics(yt, full_proba(base_clf("logistic_regression", seed).fit(ctx.Zl, ctx.yl), ctx.Z[ctx.split.l_test], ctx.K))["accuracy"]
    rows = []
    for mode in ("supervised", "pi", "mean_teacher", "fixmatch"):
        net, extra = train_net(ctx, mode, augmentation, strength, epochs, monitor_every=epochs)
        acc = metrics(yt, net.predict_proba(ctx.Z[ctx.split.l_test]))["accuracy"]
        rows.append({"method": "augmentation_only (control)" if mode == "supervised" else mode, "holdout_accuracy": acc, "gain_over_logreg": round(acc - sup, 4), "final_consistency_loss": extra["trajectory"][-1]["consistency_loss"]})
    control = rows[0]["holdout_accuracy"]
    for r in rows[1:]:
        r["gain_over_control"] = round(r["holdout_accuracy"] - control, 4)
    rows.sort(key=lambda r: -r["holdout_accuracy"])
    return {"supervised_logreg_accuracy": sup, "methods": rows, "best": rows[0]["method"], "test_rows": int(len(yt)),
            "reading": "gain_over_control is the part due to the unlabelled data; the control already includes the benefit of the augmentation"}


@tool
def tune_augmentation_strength(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, mode: str = "mean_teacher",
    augmentation: str = "gaussian", strengths: list[float] | None = None, epochs: int = 250, test_size: float = 0.3, seed: int = 42,
) -> dict:
    """Sweep the perturbation strength for one consistency method and report held-out accuracy at each — too weak teaches nothing, too strong destroys the label. Tuned on held-out labelled rows: confirm on fresh labels.

    Args:
        mode: pi, mean_teacher or fixmatch.
        augmentation: gaussian, dropout, swap_noise, mixup or mixed.
        strengths: Values to try; defaults to 0.05, 0.15, 0.3, 0.5, 0.8.
    """
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, None, test_size, seed, "logistic_regression")
    yt = ctx.sd.y[ctx.split.l_test]
    rows = []
    for s in strengths or [0.05, 0.15, 0.3, 0.5, 0.8]:
        net, _ = train_net(ctx, mode, augmentation, s, epochs, monitor_every=epochs)
        rows.append({"strength": s, "holdout_accuracy": metrics(yt, net.predict_proba(ctx.Z[ctx.split.l_test]))["accuracy"]})
    best = max(rows, key=lambda r: r["holdout_accuracy"])
    return {"mode": mode, "augmentation": augmentation, "sweep": rows, "best_strength": best["strength"]}


@tool
def consistency_training_trajectory(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, mode: str = "mean_teacher",
    augmentation: str = "gaussian", strength: float = 0.3, epochs: int = 300, test_size: float = 0.3, seed: int = 42,
) -> dict:
    """Record the consistency loss and held-out accuracy every 25 steps: a consistency loss that falls while accuracy rises is healthy; one that collapses to zero with flat accuracy means the network found the trivial solution of ignoring the perturbation.

    Args:
        mode: pi, mean_teacher or fixmatch.
        augmentation: gaussian, dropout, swap_noise, mixup or mixed.
        strength: Perturbation size.
        epochs: Gradient steps.
    """
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, None, test_size, seed, "logistic_regression")
    _, extra = train_net(ctx, mode, augmentation, strength, epochs)
    t = extra["trajectory"]
    acc = [r["holdout_accuracy"] for r in t]
    return {"mode": mode, "trajectory": t, "first": acc[0], "best": max(acc), "last": acc[-1],
            "verdict": "accuracy fell late in training: reduce epochs or the consistency weight" if acc[-1] < max(acc) - 0.02 else "stable training"}


@tool
def simulate_label_scarcity(data: str, label_column: str, keep_fraction: float = 0.1, seed: int = 42, output_path: str | None = None) -> dict:
    """Turn a fully labelled dataset into a semi-supervised experiment: hide most labels (stratified by class) and keep the true ones in a `<label>__truth` column.

    Args:
        keep_fraction: Share of labels to keep.
    """
    return mask_labels(data, label_column, keep_fraction, seed, output_path)


@tool
def predict_with_semi_model(model_path: str, data: str, id_column: str | None = None, output_path: str | None = None) -> dict:
    """Score new rows with a model saved by a consistency-regularisation fit tool.

    Args:
        model_path: Path of the saved model (joblib).
    """
    return predict_saved(model_path, data, id_column, output_path)
