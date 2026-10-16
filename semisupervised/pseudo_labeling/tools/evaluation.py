"""Judge pseudo-labelling: methods head to head, threshold sweeps, label-noise estimates, the audit against hidden truth, class drift, and exporting a relabelled dataset."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ml_core.semi import base_clf, full_proba, make_context, mask_labels, metrics, predict_saved, run_semi
from ml_core.tooling import tool
from semisupervised.pseudo_labeling.tools._common import (
    gen_cluster_consistent,
    gen_ensemble,
    gen_threshold,
    gen_top_per_class,
    noisy_student,
    pseudo_trainer,
)


def _score(ctx, fitted) -> dict:
    return metrics(ctx.sd.y[ctx.split.l_test], fitted.predict_proba(ctx.Z[ctx.split.l_test]))


@tool
def compare_pseudo_labeling_methods(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: str = "logistic_regression", test_size: float = 0.3, seed: int = 42,
) -> dict:
    """Run the main pseudo-labelling recipes — threshold, top-per-class, ensemble agreement, cluster-consistent, noisy student — on the same split, with the same base learner, and rank them by held-out accuracy against the supervised-only model; also reports each one's pseudo-label count and (with a truth column) accuracy.

    Args:
        base: Base learner used as the teacher.
    """
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, base)
    yt = ctx.sd.y[ctx.split.l_test]
    sup = metrics(yt, full_proba(base_clf(base, seed).fit(ctx.Zl, ctx.yl), ctx.Z[ctx.split.l_test], ctx.K))
    recipes = {"threshold_0.9": (gen_threshold, {"threshold": 0.9}), "threshold_0.7": (gen_threshold, {"threshold": 0.7}), "top_30pct_per_class": (gen_top_per_class, {"fraction": 0.3}),
               "ensemble_agreement": (gen_ensemble, {"threshold": 0.8}), "cluster_consistent": (gen_cluster_consistent, {"threshold": 0.7})}
    rows = []
    for name, (gen, kw) in recipes.items():
        try:
            fitted, extra = pseudo_trainer(gen, kw)(ctx)
            row = {"method": name, "pseudo_labelled": int(len(extra["pseudo"]["index"])), **{k: v for k, v in _score(ctx, fitted).items() if k in ("accuracy", "f1_macro")}}
            if ctx.sd.truth is not None and len(extra["pseudo"]["index"]):
                row["pseudo_accuracy"] = round(float((ctx.sd.truth[extra["pseudo"]["index"]] == extra["pseudo"]["label"]).mean()), 4)
            rows.append(row)
        except Exception as error:
            rows.append({"method": name, "error": str(error)[:100]})
    try:
        fitted, extra = noisy_student(ctx)
        rows.append({"method": "noisy_student", "pseudo_labelled": int(len(extra["pseudo"]["index"])), **{k: v for k, v in _score(ctx, fitted).items() if k in ("accuracy", "f1_macro")}})
    except Exception as error:
        rows.append({"method": "noisy_student", "error": str(error)[:100]})
    for r in rows:
        if "accuracy" in r:
            r["gain_over_supervised"] = round(r["accuracy"] - sup["accuracy"], 4)
    rows.sort(key=lambda r: -r.get("accuracy", -1))
    return {"supervised_only": {k: sup[k] for k in ("accuracy", "f1_macro")}, "methods": rows, "best": rows[0]["method"], "test_rows": int(len(yt))}


@tool
def pseudo_label_threshold_sweep(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: str = "logistic_regression", thresholds: list[float] | None = None, test_size: float = 0.3, seed: int = 42,
) -> dict:
    """For each confidence threshold: how many unlabelled rows get labelled, how accurate those labels are (when truth is known) and the retrained model's held-out accuracy — the trade-off between coverage and label quality.

    Args:
        thresholds: Values to try; defaults to 0.5 to 0.98.
        base: Base learner used as the teacher.
    """
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, base)
    yt = ctx.sd.y[ctx.split.l_test]
    sup = metrics(yt, full_proba(base_clf(base, seed).fit(ctx.Zl, ctx.yl), ctx.Z[ctx.split.l_test], ctx.K))["accuracy"]
    rows = []
    for t in thresholds or [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]:
        fitted, extra = pseudo_trainer(gen_threshold, {"threshold": t})(ctx)
        p = extra["pseudo"]
        row = {"threshold": t, "pseudo_labelled": int(len(p["index"])), "holdout_accuracy": _score(ctx, fitted)["accuracy"]}
        if ctx.sd.truth is not None and len(p["index"]):
            row["pseudo_accuracy"] = round(float((ctx.sd.truth[p["index"]] == p["label"]).mean()), 4)
        rows.append(row)
    best = max(rows, key=lambda r: r["holdout_accuracy"])
    return {"supervised_only_accuracy": sup, "sweep": rows, "best_threshold": best["threshold"], "note": "tuned on held-out labelled rows: confirm the chosen threshold on fresh labels"}


@tool
def estimate_label_noise_rate(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "random_forest", k: int = 15, seed: int = 42,
) -> dict:
    """Cross-validate on the labelled rows and list those whose given label the out-of-fold model confidently disagrees with — likely mislabelled rows. Noise in the real labels is inherited and amplified by every pseudo-labelling method, so clean it first.

    Args:
        base: Learner used for the out-of-fold predictions.
        k: How many suspect rows to list.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, None, 0.3, seed, base)
    Z, y = np.vstack([ctx.Zl, ctx.Z[ctx.split.l_test]]), np.concatenate([ctx.yl, ctx.sd.y[ctx.split.l_test]])
    rows = np.concatenate([ctx.split.l_train, ctx.split.l_test])
    oof = cross_val_predict(base_clf(base, seed), Z, y, cv=StratifiedKFold(min(5, int(np.bincount(y).min())), shuffle=True, random_state=seed), method="predict_proba")
    p_given = oof[np.arange(len(y)), y]
    suspect = np.argsort(p_given)[:k]
    wrong = oof.argmax(axis=1) != y
    return {"labelled_rows": int(len(y)), "out_of_fold_disagreement": round(float(wrong.mean()), 4), "confident_disagreement_rows": int(((oof.max(axis=1) > 0.9) & wrong).sum()),
            "suspect_rows": [{"row_in_data": int(rows[i]), "given_label": ctx.sd.classes[y[i]], "model_says": ctx.sd.classes[int(oof[i].argmax())], "p_given_label": round(float(p_given[i]), 4)} for i in suspect],
            "reading": "a high confident-disagreement count means the real labels are noisy: fix them, or use fit_pseudo_labeling_with_loss_filter"}


@tool
def pseudo_label_class_drift_report(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: str = "logistic_regression", threshold: float = 0.9, test_size: float = 0.3, seed: int = 42,
) -> dict:
    """Compare the class balance of the real labels, the pseudo-labels, and (if known) the true unlabelled rows — pseudo-labelling tends to over-produce the easy class, and a skewed pseudo-label mix biases the retrained model.

    Args:
        base: Base learner used as the teacher.
    """
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, base)
    _, extra = pseudo_trainer(gen_threshold, {"threshold": threshold})(ctx)
    p = extra["pseudo"]
    share = lambda v: {ctx.sd.classes[k]: round(float((v == k).mean()), 4) for k in range(ctx.K)}  # noqa: E731
    out = {"threshold": threshold, "real_labels": share(ctx.yl), "pseudo_labels": share(p["label"]) if len(p["label"]) else {}, "pseudo_labelled": int(len(p["label"])), "unlabelled_rows": int(len(ctx.split.unl))}
    if ctx.sd.truth is not None:
        known = ctx.sd.truth[ctx.split.unl]
        out["true_unlabelled"] = share(known[known >= 0])
        out["coverage_by_class"] = {ctx.sd.classes[k]: round(float(((p["label"] == k)).sum() / max((known == k).sum(), 1)), 4) for k in range(ctx.K)}
    gap = max(abs(out["real_labels"][c] - out["pseudo_labels"].get(c, 0)) for c in out["real_labels"]) if out["pseudo_labels"] else 0
    out["verdict"] = "pseudo-label mix is close to the real label mix" if gap < 0.1 else "the pseudo-label mix is skewed towards one class: use generate_pseudo_labels_top_per_class or a class-balanced variant"
    return out


@tool
def export_pseudo_labeled_dataset(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    method: Literal["threshold", "top_per_class", "ensemble", "cluster_consistent", "noisy_student"] = "ensemble", base: str = "logistic_regression", threshold: float = 0.85,
    output_path: str = "pseudo_labelled.csv", test_size: float = 0.3, seed: int = 42,
) -> dict:
    """Produce the relabelled dataset itself with the chosen method: every row with its final label, `label_source` (human / pseudo / unlabelled) and `pseudo_confidence` — the hand-off file for another model or team.

    Args:
        method: Which pseudo-labelling recipe to use.
        output_path: Where to write the relabelled dataset.
    """
    generators = {"threshold": (gen_threshold, {"threshold": threshold}), "top_per_class": (gen_top_per_class, {"fraction": 0.3}),
                  "ensemble": (gen_ensemble, {"threshold": threshold}), "cluster_consistent": (gen_cluster_consistent, {"threshold": threshold})}
    if method == "noisy_student":
        trainer = lambda ctx: noisy_student(ctx, threshold, 3, 0.3, "random_forest", output_path)  # noqa: E731
    else:
        gen, kw = generators[method]
        trainer = pseudo_trainer(gen, kw, output_path=output_path)
    report = run_semi(f"export({method})", trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, None, base)
    keep = ("model", "labelled_rows", "unlabelled_rows", "relabelled_dataset", "pseudo_labels", "gain_over_supervised", "verdict")
    return {k: report[k] for k in keep if k in report}


@tool
def simulate_label_scarcity(data: str, label_column: str, keep_fraction: float = 0.1, seed: int = 42, output_path: str | None = None) -> dict:
    """Turn a fully labelled dataset into a semi-supervised experiment: hide most labels (stratified by class) and keep the true ones in a `<label>__truth` column, so every method can be audited against what it recovers.

    Args:
        keep_fraction: Share of labels to keep.
    """
    return mask_labels(data, label_column, keep_fraction, seed, output_path)


@tool
def predict_with_semi_model(model_path: str, data: str, id_column: str | None = None, output_path: str | None = None) -> dict:
    """Score new rows with a model saved by a pseudo-labelling fit tool: the predicted class, its confidence and every class probability.

    Args:
        model_path: Path of the saved model (joblib).
    """
    return predict_saved(model_path, data, id_column, output_path)
