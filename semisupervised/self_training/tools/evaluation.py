"""Judge self-training: against supervised-only across base learners and label budgets, threshold sweeps, the quality of the pseudo-labels, and simulating label scarcity."""

from __future__ import annotations


import numpy as np

from ml_core.semi import base_clf, full_proba, make_context, mask_labels, metrics, predict_saved
from ml_core.tooling import tool
from semisupervised.self_training.tools._common import self_train



@tool
def compare_semi_vs_supervised(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, bases: list[str] | None = None,
    threshold: float = 0.9, test_size: float = 0.3, seed: int = 42,
) -> dict:
    """For several base learners, score the supervised-only model and its self-trained version on the same held-out labels — shows whether the benefit is general or depends on the learner.

    Args:
        bases: Base learners to try (logistic_regression, random_forest, hist_gradient_boosting, svm, knn, naive_bayes); defaults to four.
    """
    rows = []
    for base in bases or ["logistic_regression", "random_forest", "hist_gradient_boosting", "knn"]:
        ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, None, test_size, seed, base)
        yt = ctx.sd.y[ctx.split.l_test]
        sup = metrics(yt, full_proba(base_clf(base, seed).fit(ctx.Zl, ctx.yl), ctx.Z[ctx.split.l_test], ctx.K))
        model, extra = self_train(ctx, threshold=threshold, monitor=False)
        semi = metrics(yt, model.predict_proba(ctx.Z[ctx.split.l_test]))
        rows.append({"base": base, "supervised_accuracy": sup["accuracy"], "self_training_accuracy": semi["accuracy"], "gain": round(semi["accuracy"] - sup["accuracy"], 4),
                     "supervised_f1_macro": sup["f1_macro"], "self_training_f1_macro": semi["f1_macro"], "pseudo_labelled": int(len(extra["pseudo"]["index"]))})
    rows.sort(key=lambda r: -r["gain"])
    return {"threshold": threshold, "by_base": rows, "helps_for": [r["base"] for r in rows if r["gain"] > 0.01]}


@tool
def self_training_trajectory(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "logistic_regression",
    threshold: float = 0.9, max_iter: int = 10, test_size: float = 0.3, seed: int = 42,
) -> dict:
    """Round-by-round record of a self-training run: labelled rows used, pseudo-labels added in total, and held-out accuracy after each round — to see whether accuracy climbs, plateaus, or turns down as pseudo-labels accumulate (confirmation bias).

    Args:
        base: Base learner.
        max_iter: Most rounds.
    """
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, None, test_size, seed, base)
    _, extra = self_train(ctx, threshold=threshold, max_iter=max_iter)
    traj = extra["trajectory"]
    accs = [r["holdout_accuracy"] for r in traj]
    best = int(np.argmax(accs))
    return {"trajectory": traj, "best_round": best, "accuracy_first": accs[0], "accuracy_best": accs[best], "accuracy_last": accs[-1],
            "verdict": "accuracy declines in later rounds: confirmation bias — stop earlier, raise the threshold, or use the class-balanced / ensemble variants" if accs[-1] < accs[best] - 0.01 else
            "accuracy rises and holds: pseudo-labels are helping" if accs[-1] > accs[0] + 0.005 else "flat: the pseudo-labels add nothing measurable"}


@tool
def tune_self_training_threshold(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "logistic_regression",
    thresholds: list[float] | None = None, test_size: float = 0.3, seed: int = 42,
) -> dict:
    """Sweep the confidence threshold and report, for each value, the pseudo-labels added and the held-out accuracy — the trade-off between many noisy labels and few clean ones. Tuned on held-out labelled rows, so confirm the choice on fresh labels before relying on it.

    Args:
        base: Base learner.
        thresholds: Values to try; defaults to 0.6 to 0.98.
    """
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, None, test_size, seed, base)
    yt = ctx.sd.y[ctx.split.l_test]
    sup = metrics(yt, full_proba(base_clf(base, seed).fit(ctx.Zl, ctx.yl), ctx.Z[ctx.split.l_test], ctx.K))["accuracy"]
    rows = []
    for t in thresholds or [0.6, 0.7, 0.8, 0.9, 0.95, 0.98]:
        model, extra = self_train(ctx, threshold=t, monitor=False)
        rows.append({"threshold": t, "pseudo_labelled": int(len(extra["pseudo"]["index"])), "holdout_accuracy": metrics(yt, model.predict_proba(ctx.Z[ctx.split.l_test]))["accuracy"]})
    best = max(rows, key=lambda r: r["holdout_accuracy"])
    return {"supervised_only_accuracy": sup, "sweep": rows, "best_threshold": best["threshold"], "gain_at_best": round(best["holdout_accuracy"] - sup, 4)}


@tool
def label_fraction_sweep(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "logistic_regression",
    fractions: list[float] | None = None, threshold: float = 0.9, test_size: float = 0.3, seed: int = 42,
) -> dict:
    """Re-run supervised-only and self-training with only a fraction of the available labels (the rest become unlabelled) — semi-supervised learning usually helps most when labels are scarcest, and this finds where the benefit begins.

    Args:
        base: Base learner.
        fractions: Shares of the training labels to keep; defaults to 0.1, 0.25, 0.5, 1.0.
    """
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, None, test_size, seed, base)
    yt = ctx.sd.y[ctx.split.l_test]
    rng = np.random.default_rng(seed)
    rows = []
    for frac in fractions or [0.1, 0.25, 0.5, 1.0]:
        keep = np.sort(rng.choice(len(ctx.yl), size=max(2 * ctx.K, int(len(ctx.yl) * frac)), replace=False))
        if len(set(ctx.yl[keep])) < 2:
            continue
        hidden = np.setdiff1d(np.arange(len(ctx.yl)), keep)
        sub = type(ctx)(ctx.Z, ctx.sd, type(ctx.split)(ctx.split.l_train[keep], ctx.split.l_test, np.concatenate([ctx.split.unl, ctx.split.l_train[hidden]])), base, seed)
        sup = metrics(yt, full_proba(base_clf(base, seed).fit(sub.Zl, sub.yl), ctx.Z[ctx.split.l_test], ctx.K))["accuracy"]
        model, extra = self_train(sub, threshold=threshold, monitor=False)
        semi = metrics(yt, model.predict_proba(ctx.Z[ctx.split.l_test]))["accuracy"]
        rows.append({"labels": int(len(keep)), "fraction": frac, "supervised_accuracy": sup, "self_training_accuracy": semi, "gain": round(semi - sup, 4)})
    return {"sweep": rows, "biggest_gain_at": max(rows, key=lambda r: r["gain"])["labels"] if rows else None}


@tool
def pseudo_label_quality_report(
    data: str, label_column: str, truth_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None,
    base: str = "logistic_regression", threshold: float = 0.9, test_size: float = 0.3, seed: int = 42,
) -> dict:
    """Audit the pseudo-labels against the hidden truth: accuracy overall, by confidence band and by class, and how many wrong labels entered training — shows whether the threshold is safe and which class is being corrupted. Needs `truth_column`, so it only runs on experiments where labels were hidden on purpose.

    Args:
        base: Base learner.
    """
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, base)
    _, extra = self_train(ctx, threshold=threshold, monitor=False)
    p = extra["pseudo"]
    truth = ctx.sd.truth[p["index"]]
    known = truth >= 0
    if not known.any():
        raise ValueError("the truth column has no value for any pseudo-labelled row")
    idx, lab, conf, truth = p["index"][known], p["label"][known], p["confidence"][known], truth[known]
    correct = lab == truth
    bands = [(0.0, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.99), (0.99, 1.01)]
    return {"pseudo_labelled": int(len(idx)), "overall_accuracy": round(float(correct.mean()), 4), "wrong_labels_added": int((~correct).sum()),
            "by_confidence": [{"band": f"{lo}-{min(hi, 1)}", "rows": int(m.sum()), "accuracy": round(float(correct[m].mean()), 4)} for lo, hi in bands if (m := (conf >= lo) & (conf < hi)).any()],
            "by_class": {ctx.sd.classes[k]: {"pseudo_labelled": int((lab == k).sum()), "accuracy": round(float(correct[lab == k].mean()), 4)} for k in range(ctx.K) if (lab == k).any()},
            "true_class_share_in_unlabelled": {ctx.sd.classes[k]: round(float((ctx.sd.truth[ctx.split.unl] == k).mean()), 4) for k in range(ctx.K)},
            "pseudo_class_share": {ctx.sd.classes[k]: round(float((lab == k).mean()), 4) for k in range(ctx.K)}}


@tool
def simulate_label_scarcity(data: str, label_column: str, keep_fraction: float = 0.1, seed: int = 42, output_path: str | None = None) -> dict:
    """Turn a fully labelled dataset into a semi-supervised experiment: hide most labels (stratified by class) and keep the true ones in a `<label>__truth` column, so every fit tool can be scored on how much of the hidden information it recovers.

    Args:
        keep_fraction: Share of labels to keep.
    """
    return mask_labels(data, label_column, keep_fraction, seed, output_path)


@tool
def predict_with_semi_model(model_path: str, data: str, id_column: str | None = None, output_path: str | None = None) -> dict:
    """Score new rows with a model saved by a self-training fit tool: the predicted class, its confidence and every class probability.

    Args:
        model_path: Path of the saved model (joblib).
    """
    return predict_saved(model_path, data, id_column, output_path)
