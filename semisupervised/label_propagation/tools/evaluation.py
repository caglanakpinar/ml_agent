"""Judge label propagation: graph settings tuned by hiding labels, methods head to head, the transductive protocol, confidence, and scoring new rows."""

from __future__ import annotations

import numpy as np

from ml_core.semi import base_clf, full_proba, make_context, mask_labels, metrics, predict_saved
from ml_core.tooling import tool
from semisupervised.label_propagation.tools._common import propagate


def _hide_and_score(ctx, kind, kernel, k, gamma, alpha, seed, folds=4) -> float:
    """Cross-validate a graph setting: hide a fold of the labelled training rows, propagate from the rest, score the hidden ones."""
    from sklearn.model_selection import StratifiedKFold

    y = ctx.yl
    cv = StratifiedKFold(max(2, min(folds, int(np.bincount(y).min()))), shuffle=True, random_state=seed)
    accs = []
    for keep, hide in cv.split(ctx.Zl, y):
        Zall = np.vstack([ctx.Zl[keep], ctx.Zl[hide], ctx.Zu])
        yy = np.concatenate([y[keep], -np.ones(len(hide) + len(ctx.Zu), int)])
        from semisupervised.label_propagation.tools._common import make_propagator

        model = make_propagator(kind, kernel, k, gamma, alpha, 500).fit(Zall, yy)
        pred = model.transduction_[len(keep) : len(keep) + len(hide)]
        accs.append(float((pred == y[hide]).mean()))
    return float(np.mean(accs))


@tool
def tune_graph_parameters(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, kernel: str = "knn",
    values: list[float] | None = None, kind: str = "spreading", alpha: float = 0.2, test_size: float = 0.3, seed: int = 42,
) -> dict:
    """Choose the graph's size parameter (k for a k-NN graph, gamma for an RBF graph) by hiding folds of the labelled rows and scoring how well propagation recovers them. Uses only the training labels; the held-out test labels stay untouched.

    Args:
        kernel: knn (tunes the number of neighbours) or rbf (tunes gamma).
        values: Values to try; defaults to k = 3, 5, 10, 20, 40 or gamma = 0.05, 0.2, 0.5, 1, 3.
        kind: spreading or propagation.
        alpha: Soft-clamping factor (spreading).
    """
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, None, test_size, seed, "logistic_regression")
    vals = values or ([3, 5, 10, 20, 40] if kernel == "knn" else [0.05, 0.2, 0.5, 1.0, 3.0])
    rows = [{("n_neighbors" if kernel == "knn" else "gamma"): v, "recovered_accuracy": round(_hide_and_score(ctx, kind, kernel, int(v) if kernel == "knn" else 10, float(v), alpha, seed), 4)} for v in vals]
    best = max(rows, key=lambda r: r["recovered_accuracy"])
    return {"kernel": kernel, "kind": kind, "sweep": rows, "best": best}


@tool
def sweep_spreading_alpha(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, n_neighbors: int = 10,
    alphas: list[float] | None = None, test_size: float = 0.3, seed: int = 42,
) -> dict:
    """Sweep label spreading's alpha — how much neighbours may override a given label — and score recovery of hidden labels. Low alpha trusts the labels; high alpha trusts the graph; the right value shows how noisy the labels are.

    Args:
        n_neighbors: Neighbours per row in the graph.
        alphas: Values to try; defaults to 0.01, 0.1, 0.3, 0.6, 0.9.
    """
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, None, test_size, seed, "logistic_regression")
    rows = [{"alpha": a, "recovered_accuracy": round(_hide_and_score(ctx, "spreading", "knn", n_neighbors, 1.0, a, seed), 4)} for a in (alphas or [0.01, 0.1, 0.3, 0.6, 0.9])]
    best = max(rows, key=lambda r: r["recovered_accuracy"])
    return {"sweep": rows, "best_alpha": best["alpha"], "reading": "a best alpha near 0 means the labels are clean; a large alpha means the graph corrects the labels (they may be noisy)"}


@tool
def compare_propagation_methods(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    n_neighbors: int = 10, test_size: float = 0.3, seed: int = 42,
) -> dict:
    """Score label propagation and spreading on k-NN and RBF graphs against a supervised k-NN and logistic regression on the same held-out labels — shows whether the graph adds anything beyond the labelled rows' own neighbours.

    Args:
        n_neighbors: Neighbours per row in the k-NN graphs.
    """
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, "knn")
    yt = ctx.sd.y[ctx.split.l_test]
    rows = []
    for name in ("knn", "logistic_regression"):
        rows.append({"method": f"supervised_{name}", "holdout_accuracy": metrics(yt, full_proba(base_clf(name, seed).fit(ctx.Zl, ctx.yl), ctx.Z[ctx.split.l_test], ctx.K))["accuracy"]})
    for kind, kernel in (("spreading", "knn"), ("propagation", "knn"), ("spreading", "rbf"), ("propagation", "rbf")):
        try:
            model, extra, dist = propagate(ctx, kind, kernel, n_neighbors, 0.5, 0.2, 500)
            raw = model.predict_proba(ctx.Z[ctx.split.l_test])
            full = np.zeros((len(raw), ctx.K))
            full[:, [int(c) for c in model.classes_]] = raw
            row = {"method": f"label_{kind}[{kernel}]", "holdout_accuracy": metrics(yt, full)["accuracy"], "iterations": extra["iterations"], "mean_confidence": round(float(dist.max(axis=1).mean()), 4)}
            if ctx.sd.truth is not None:
                known = ctx.sd.truth[ctx.split.unl] >= 0
                row["accuracy_on_unlabelled_vs_truth"] = round(float((dist.argmax(axis=1)[known] == ctx.sd.truth[ctx.split.unl][known]).mean()), 4)
            rows.append(row)
        except Exception as error:
            rows.append({"method": f"label_{kind}[{kernel}]", "error": str(error)[:100]})
    rows.sort(key=lambda r: -r.get("holdout_accuracy", -1))
    return {"methods": rows, "best": rows[0]["method"], "test_rows": int(len(yt))}


@tool
def transductive_evaluation(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, n_neighbors: int = 10, alpha: float = 0.2,
    test_size: float = 0.3, seed: int = 42,
) -> dict:
    """The transductive protocol that matches how propagation is really used: the held-out labelled rows sit *in the graph as unlabelled nodes*, labels are propagated, and their propagated labels are scored — usually a little higher than the inductive hold-out score, because the test rows are part of the graph.

    Args:
        n_neighbors: Neighbours per row in the graph.
        alpha: Soft-clamping factor.
    """
    from semisupervised.label_propagation.tools._common import make_propagator

    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, None, test_size, seed, "knn")
    Zall = np.vstack([ctx.Zl, ctx.Z[ctx.split.l_test], ctx.Zu])
    y = np.concatenate([ctx.yl, -np.ones(len(ctx.split.l_test) + len(ctx.Zu), int)])
    model = make_propagator("spreading", "knn", n_neighbors, 1.0, alpha, 1000).fit(Zall, y)
    pred = model.transduction_[len(ctx.yl) : len(ctx.yl) + len(ctx.split.l_test)]
    yt = ctx.sd.y[ctx.split.l_test]
    inductive = model.predict_proba(ctx.Z[ctx.split.l_test])
    return {"transductive_accuracy": round(float((pred == yt).mean()), 4), "inductive_accuracy_same_model": round(float((inductive.argmax(axis=1) == yt).mean()), 4),
            "supervised_knn_accuracy": metrics(yt, full_proba(base_clf("knn", seed).fit(ctx.Zl, ctx.yl), ctx.Z[ctx.split.l_test], ctx.K))["accuracy"], "test_rows": int(len(yt))}


@tool
def propagation_confidence_report(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    n_neighbors: int = 10, alpha: float = 0.2, test_size: float = 0.3, seed: int = 42,
) -> dict:
    """How confident the propagated labels are: distribution of the top-class share, how many unlabelled rows stay ambiguous (nearly equal share for two classes), per-class confidence, and (with a truth column) accuracy by confidence band — which propagated labels can be used and which should go to a human.

    Args:
        n_neighbors: Neighbours per row in the graph.
        alpha: Soft-clamping factor.
    """
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, "knn")
    _, _, dist = propagate(ctx, "spreading", "knn", n_neighbors, 1.0, alpha, 1000)
    conf, lab = dist.max(axis=1), dist.argmax(axis=1)
    out = {"unlabelled_rows": int(len(conf)), "confidence_quantiles": {q: round(float(np.quantile(conf, q)), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
           "ambiguous_rows": int((conf < 0.6).sum()), "per_class": {ctx.sd.classes[k]: {"rows": int((lab == k).sum()), "mean_confidence": round(float(conf[lab == k].mean()), 4)} for k in range(ctx.K) if (lab == k).any()}}
    if ctx.sd.truth is not None:
        t = ctx.sd.truth[ctx.split.unl]
        known = t >= 0
        out["accuracy_by_confidence"] = [{"band": f"{lo}-{min(hi, 1)}", "rows": int(m.sum()), "accuracy": round(float((lab[m] == t[m]).mean()), 4)} for lo, hi in [(0, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)] if (m := known & (conf >= lo) & (conf < hi)).any()]
    return out


@tool
def simulate_label_scarcity(data: str, label_column: str, keep_fraction: float = 0.1, seed: int = 42, output_path: str | None = None) -> dict:
    """Turn a fully labelled dataset into a semi-supervised experiment: hide most labels (stratified by class) and keep the true ones in a `<label>__truth` column.

    Args:
        keep_fraction: Share of labels to keep.
    """
    return mask_labels(data, label_column, keep_fraction, seed, output_path)


@tool
def predict_with_semi_model(model_path: str, data: str, id_column: str | None = None, output_path: str | None = None) -> dict:
    """Score new rows with a saved propagation model: new rows are attached to the graph through the kernel and receive the propagated distribution of their neighbours.

    Args:
        model_path: Path of the saved model (joblib).
    """
    return predict_saved(model_path, data, id_column, output_path)
