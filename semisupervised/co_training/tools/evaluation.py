"""Judge co-training: variants head to head, the round-by-round trajectory, a label-fraction sweep, the audit against hidden truth, and scoring new rows."""

from __future__ import annotations

import numpy as np

from ml_core.ml import build_preprocessor
from ml_core.semi import base_clf, full_proba, make_context, mask_labels, metrics, predict_saved, view_indices
from ml_core.tooling import tool
from semisupervised.co_training.tools._common import choose_views, co_train, democratic, tri_train


def _views(ctx, seed, strategy="correlation"):
    prep = build_preprocessor(ctx.sd.X, scale=True).fit(ctx.sd.X.iloc[np.concatenate([ctx.split.l_train, ctx.split.unl])])
    groups = choose_views(ctx, prep, None, strategy, seed)
    return groups, [view_indices(prep, g) for g in groups]


@tool
def compare_co_training_variants(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: str = "logistic_regression", test_size: float = 0.3, seed: int = 42,
) -> dict:
    """Run co-training on correlation-based views, on halves, on random views, tri-training and democratic co-learning on the same split, and rank them by held-out accuracy against the supervised-only model.

    Args:
        base: Base learner.
    """
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, base)
    yt = ctx.sd.y[ctx.split.l_test]
    sup = metrics(yt, full_proba(base_clf(base, seed).fit(ctx.Zl, ctx.yl), ctx.Z[ctx.split.l_test], ctx.K))["accuracy"]
    rows = []

    def record(name, fn):
        try:
            model, extra = fn()
            p = extra["pseudo"]
            row = {"method": name, "pseudo_labelled": int(len(p["index"])), "holdout_accuracy": metrics(yt, model.predict_proba(ctx.Z[ctx.split.l_test]))["accuracy"]}
            row["gain"] = round(row["holdout_accuracy"] - sup, 4)
            if ctx.sd.truth is not None and len(p["index"]):
                row["pseudo_accuracy"] = round(float((ctx.sd.truth[p["index"]] == p["label"]).mean()), 4)
            rows.append(row)
        except Exception as error:
            rows.append({"method": name, "error": str(error)[:100]})

    for strategy in ("correlation", "halves", "random"):
        record(f"co_training[{strategy}]", lambda s=strategy: co_train(ctx, _views(ctx, seed, s)[1], monitor=False))
    record("co_training[random_subspaces]", lambda: co_train(ctx, _views(ctx, seed)[1], redraw=True, monitor=False))
    record("tri_training", lambda: tri_train(ctx, monitor=False))
    record("democratic_co_learning", lambda: democratic(ctx, monitor=False))
    rows.sort(key=lambda r: -r.get("holdout_accuracy", -1))
    return {"supervised_only_accuracy": sup, "methods": rows, "best": rows[0]["method"], "test_rows": int(len(yt))}


@tool
def co_training_trajectory(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "logistic_regression",
    rounds: int = 10, threshold: float = 0.8, test_size: float = 0.3, seed: int = 42,
) -> dict:
    """Round-by-round record of co-training: labelled rows each view trains on, pseudo-labels added, and held-out accuracy of each view and of their combination — shows whether the views improve each other or drift.

    Args:
        base: Base learner.
        rounds: Rounds of mutual teaching.
    """
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, None, test_size, seed, base)
    groups, idx = _views(ctx, seed)
    _, extra = co_train(ctx, idx, rounds, 5, threshold)
    traj = extra["trajectory"]
    acc = [r["holdout_accuracy"] for r in traj]
    return {"views": groups, "trajectory": traj, "first": acc[0], "best": max(acc), "last": acc[-1],
            "verdict": "accuracy declines late: the views started confirming each other's mistakes — fewer rounds or a higher threshold" if acc[-1] < max(acc) - 0.01 else "stable or improving"}


@tool
def pseudo_label_quality_report(
    data: str, label_column: str, truth_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None,
    base: str = "logistic_regression", threshold: float = 0.8, test_size: float = 0.3, seed: int = 42,
) -> dict:
    """Audit the pseudo-labels co-training produced against the hidden truth: overall accuracy, by confidence band and by class — shows whether the views taught each other correctly. Needs `truth_column`.

    Args:
        base: Base learner.
    """
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, base)
    _, idx = _views(ctx, seed)
    _, extra = co_train(ctx, idx, 10, 5, threshold, monitor=False)
    p = extra["pseudo"]
    truth = ctx.sd.truth[p["index"]]
    known = truth >= 0
    if not known.any():
        raise ValueError("the truth column has no value for any pseudo-labelled row")
    lab, conf, truth = p["label"][known], p["confidence"][known], truth[known]
    correct = lab == truth
    return {"pseudo_labelled": int(len(lab)), "accuracy": round(float(correct.mean()), 4), "wrong_labels_added": int((~correct).sum()),
            "by_confidence": [{"band": f"{lo}-{min(hi, 1)}", "rows": int(m.sum()), "accuracy": round(float(correct[m].mean()), 4)} for lo, hi in [(0, 0.85), (0.85, 0.95), (0.95, 1.01)] if (m := (conf >= lo) & (conf < hi)).any()],
            "by_class": {ctx.sd.classes[k]: {"rows": int((lab == k).sum()), "accuracy": round(float(correct[lab == k].mean()), 4)} for k in range(ctx.K) if (lab == k).any()}}


@tool
def simulate_label_scarcity(data: str, label_column: str, keep_fraction: float = 0.1, seed: int = 42, output_path: str | None = None) -> dict:
    """Turn a fully labelled dataset into a semi-supervised experiment: hide most labels (stratified by class) and keep the true ones in a `<label>__truth` column, so each variant can be audited against what it recovers.

    Args:
        keep_fraction: Share of labels to keep.
    """
    return mask_labels(data, label_column, keep_fraction, seed, output_path)


@tool
def predict_with_semi_model(model_path: str, data: str, id_column: str | None = None, output_path: str | None = None) -> dict:
    """Score new rows with a model saved by a co-training fit tool (all columns the model was trained on must be present, since each view needs its own).

    Args:
        model_path: Path of the saved model (joblib).
    """
    return predict_saved(model_path, data, id_column, output_path)
