"""Judge a multi-label model honestly: per-label scores, per-label thresholds, label-count calibration, top-k precision, errors, and scoring new rows. Also stratified splitting for label sets."""

from __future__ import annotations

from typing import Literal

import joblib
import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.ml import sk
from ml_core.tooling import tool
from supervised.multi_label_classification.tools._common import (
    decide,
    fit_multilabel,
    holdout_ml,
    load_multilabel,
    multilabel_metrics,
)

Kind = Literal["binary_relevance", "classifier_chain", "label_powerset", "rakel", "random_forest", "mlknn"]
_BASE = {"binary_relevance": "logistic_regression", "classifier_chain": "logistic_regression", "label_powerset": "random_forest", "rakel": "random_forest"}


def _oof(kind: str, X: pd.DataFrame, Y: np.ndarray, names: list[str], text_column: str | None, cv: int, seed: int) -> np.ndarray:
    from sklearn.model_selection import KFold

    out = np.zeros(Y.shape)
    for tr, te in KFold(cv, shuffle=True, random_state=seed).split(X):
        out[te] = fit_multilabel(kind, X.iloc[tr], Y[tr], names, text_column, base=_BASE.get(kind, "logistic_regression"), seed=seed).proba(X.iloc[te])
    return out


@tool
def per_label_report(
    data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", features: list[str] | None = None, drop: list[str] | None = None,
    text_column: str | None = None, model: Kind = "binary_relevance", cv: int = 4, seed: int = 42,
) -> dict:
    """Out-of-fold ROC-AUC, average precision, F1 at 0.5, support and positive share for every label — shows which labels the model can predict and which it cannot, since micro averages hide the rare ones.

    Args:
        labels: Columns holding one 0/1 indicator per label.
        label_column: Alternatively, one column listing each row's labels joined by `separator`.
        separator: Delimiter inside `label_column` cells.
        text_column: Column of free text to use as features (TF-IDF).
        model: Approach to evaluate.
    """
    sk()
    from sklearn import metrics as m

    X, Y, names = load_multilabel(data, labels, label_column, separator, features, drop, text_column)
    proba = _oof(model, X, Y, names, text_column, cv, seed)
    pred = (proba >= 0.5).astype(int)
    rows = []
    for j, n in enumerate(names):
        ok = 0 < Y[:, j].sum() < len(Y)
        rows.append({"label": n, "support": int(Y[:, j].sum()), "roc_auc": round(float(m.roc_auc_score(Y[:, j], proba[:, j])), 4) if ok else None,
                     "average_precision": round(float(m.average_precision_score(Y[:, j], proba[:, j])), 4) if ok else None,
                     "f1_at_0.5": round(float(m.f1_score(Y[:, j], pred[:, j], zero_division=0)), 4)})
    rows.sort(key=lambda r: (r["average_precision"] is None, r["average_precision"]))
    return {"model": model, "per_label": rows, "weakest": [r["label"] for r in rows[:3]], "overall": multilabel_metrics(Y, decide(proba), proba)}


@tool
def optimise_label_thresholds(
    data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", features: list[str] | None = None, drop: list[str] | None = None,
    text_column: str | None = None, model: Kind = "binary_relevance", objective: Literal["f1", "balanced"] = "f1", cv: int = 4, test_size: float = 0.25, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Choose a separate probability threshold for each label on out-of-fold predictions (maximising F1), then check on held-out rows that it beats the default 0.5 — usually the cheapest large gain for imbalanced labels. Saves the model with the thresholds attached.

    Args:
        labels: Columns holding one 0/1 indicator per label.
        label_column: Alternatively, one column listing each row's labels joined by `separator`.
        separator: Delimiter inside `label_column` cells.
        text_column: Column of free text to use as features (TF-IDF).
        model: Approach to threshold.
        objective: f1 maximises per-label F1; balanced maximises balanced accuracy.
    """
    from ml_core.ml import save_model

    X, Y, names = load_multilabel(data, labels, label_column, separator, features, drop, text_column)
    X_tr, X_te, Y_tr, Y_te = holdout_ml(X, Y, test_size, seed)
    proba = _oof(model, X_tr, Y_tr, names, text_column, cv, seed)
    grid = np.linspace(0.05, 0.95, 37)
    thresholds = np.full(len(names), 0.5)
    for j in range(len(names)):
        if Y_tr[:, j].sum() == 0:
            continue
        best, best_v = 0.5, -1.0
        for t in grid:
            p = proba[:, j] >= t
            tp, fp, fn = (p & (Y_tr[:, j] == 1)).sum(), (p & (Y_tr[:, j] == 0)).sum(), (~p & (Y_tr[:, j] == 1)).sum()
            tn = (~p & (Y_tr[:, j] == 0)).sum()
            v = 2 * tp / max(2 * tp + fp + fn, 1) if objective == "f1" else 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
            if v > best_v:
                best, best_v = float(t), float(v)
        thresholds[j] = best
    final = fit_multilabel(model, X_tr, Y_tr, names, text_column, base=_BASE.get(model, "logistic_regression"), seed=seed)
    p_te = final.proba(X_te)
    final.thresholds = thresholds
    return {"model": model, "thresholds": dict(zip(names, np.round(thresholds, 3).tolist())), "holdout_default_0.5": multilabel_metrics(Y_te, decide(p_te, 0.5), p_te),
            "holdout_tuned": multilabel_metrics(Y_te, decide(p_te, thresholds), p_te), "model_path": save_model(final, model_path)}


@tool
def label_count_calibration(
    data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", features: list[str] | None = None, drop: list[str] | None = None,
    text_column: str | None = None, model: Kind = "binary_relevance", cv: int = 4, seed: int = 42,
) -> dict:
    """Compare the number of labels predicted per row with the true number — a model that predicts too few or too many labels per row is miscalibrated even if per-label scores look fine.

    Args:
        labels: Columns holding one 0/1 indicator per label.
        label_column: Alternatively, one column listing each row's labels joined by `separator`.
        separator: Delimiter inside `label_column` cells.
        text_column: Column of free text to use as features (TF-IDF).
        model: Approach to evaluate.
    """
    X, Y, names = load_multilabel(data, labels, label_column, separator, features, drop, text_column)
    proba = _oof(model, X, Y, names, text_column, cv, seed)
    pred = decide(proba)
    true_n, pred_n = Y.sum(axis=1), pred.sum(axis=1)
    return {"true_labels_per_row": {"mean": round(float(true_n.mean()), 3), "distribution": pd.Series(true_n).value_counts().sort_index().to_dict()},
            "predicted_labels_per_row": {"mean": round(float(pred_n.mean()), 3), "distribution": pd.Series(pred_n).value_counts().sort_index().to_dict()},
            "mean_absolute_count_error": round(float(np.abs(true_n - pred_n).mean()), 3), "correlation": round(float(np.corrcoef(true_n, pred_n)[0, 1]), 3) if pred_n.std() > 0 else None,
            "verdict": "over-predicting labels: raise thresholds" if pred_n.mean() > 1.3 * true_n.mean() else "under-predicting labels: lower thresholds" if pred_n.mean() < 0.7 * true_n.mean() else "label counts are well calibrated"}


@tool
def precision_at_k_report(
    data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", features: list[str] | None = None, drop: list[str] | None = None,
    text_column: str | None = None, model: Kind = "binary_relevance", ks: list[int] | None = None, cv: int = 4, seed: int = 42,
) -> dict:
    """Precision@k, recall@k and nDCG@k when the model is used to suggest its top-k labels per row (tagging, recommendation) rather than a fixed label set — the right view when an interface shows a few suggestions.

    Args:
        labels: Columns holding one 0/1 indicator per label.
        label_column: Alternatively, one column listing each row's labels joined by `separator`.
        separator: Delimiter inside `label_column` cells.
        text_column: Column of free text to use as features (TF-IDF).
        model: Approach to evaluate.
        ks: Cut-offs to report, e.g. [1, 3, 5].
    """
    X, Y, names = load_multilabel(data, labels, label_column, separator, features, drop, text_column)
    proba = _oof(model, X, Y, names, text_column, cv, seed)
    order = np.argsort(-proba, axis=1)
    out = {}
    for k in [min(int(x), len(names)) for x in (ks or [1, 3, 5])]:
        top = np.take_along_axis(Y, order[:, :k], axis=1)
        discount = 1 / np.log2(np.arange(2, k + 2))
        ideal = np.sort(Y, axis=1)[:, ::-1][:, :k]
        ndcg = (top * discount).sum(axis=1) / np.maximum((ideal * discount).sum(axis=1), 1e-9)
        has = Y.sum(axis=1) > 0
        out[k] = {"precision": round(float(top.mean()), 4), "recall": round(float((top.sum(axis=1)[has] / Y.sum(axis=1)[has]).mean()), 4), "ndcg": round(float(ndcg[has].mean()), 4)}
    return {"model": model, "at_k": out}


@tool
def multilabel_error_analysis(
    data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", features: list[str] | None = None, drop: list[str] | None = None,
    text_column: str | None = None, model: Kind = "binary_relevance", k: int = 15, cv: int = 4, seed: int = 42, output_path: str | None = None,
) -> dict:
    """The rows with the most wrong labels out-of-fold — which labels were missed, which were invented — plus the labels most often missed and most often wrongly added.

    Args:
        labels: Columns holding one 0/1 indicator per label.
        label_column: Alternatively, one column listing each row's labels joined by `separator`.
        separator: Delimiter inside `label_column` cells.
        text_column: Column of free text to use as features (TF-IDF).
        model: Approach to evaluate.
        k: How many rows to list.
    """
    X, Y, names = load_multilabel(data, labels, label_column, separator, features, drop, text_column)
    pred = decide(_oof(model, X, Y, names, text_column, cv, seed))
    wrong = (pred != Y).sum(axis=1)
    rows = []
    for i in np.argsort(-wrong)[:k]:
        rows.append({"row": int(i), "wrong_labels": int(wrong[i]), "true": [names[j] for j in np.flatnonzero(Y[i])], "predicted": [names[j] for j in np.flatnonzero(pred[i])],
                     "missed": [names[j] for j in np.flatnonzero((Y[i] == 1) & (pred[i] == 0))], "invented": [names[j] for j in np.flatnonzero((Y[i] == 0) & (pred[i] == 1))]})
    table = pd.DataFrame(rows)
    return {"rows_with_any_error": int((wrong > 0).sum()), "share_of_rows_perfect": round(float((wrong == 0).mean()), 4),
            "most_often_missed": dict(zip(names, ((Y == 1) & (pred == 0)).sum(axis=0).astype(int).tolist())), "most_often_invented": dict(zip(names, ((Y == 0) & (pred == 1)).sum(axis=0).astype(int).tolist())),
            "worst_rows": table, "written_to": save_frame(table, output_path, "multilabel_errors.csv") if output_path else None}


@tool
def split_multilabel_stratified(
    data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", test_size: float = 0.2, seed: int = 42, output_dir: str = "multilabel_split",
) -> dict:
    """Split rows into train and test with iterative stratification: rare labels are distributed first so every label keeps about the same share on both sides — a random split can leave a rare label out of the test set entirely.

    Args:
        labels: Columns holding one 0/1 indicator per label.
        label_column: Alternatively, one column listing each row's labels joined by `separator`.
        separator: Delimiter inside `label_column` cells.
        output_dir: Directory to write train.csv and test.csv into.
    """
    from pathlib import Path

    frame = load_frame(data)
    _, Y, names = load_multilabel(frame, labels, label_column, separator)
    n, rng = len(Y), np.random.default_rng(seed)
    desired = np.array([1 - test_size, test_size])
    remaining = Y.copy()
    assign = -np.ones(n, dtype=int)
    capacity = desired * n
    label_desired = desired[:, None] * Y.sum(axis=0)[None, :]
    unassigned = set(range(n))
    while unassigned:
        counts = remaining[list(unassigned)].sum(axis=0)
        if counts.sum() == 0:
            break
        j = int(np.argmin(np.where(counts > 0, counts, np.inf)))
        for i in rng.permutation([i for i in unassigned if remaining[i, j]]):
            need = label_desired[:, j]
            fold = int(np.argmax(need + 1e-9 * (capacity - (assign == np.arange(2)[:, None]).sum(axis=1))))
            if capacity[fold] <= 0:
                fold = 1 - fold
            assign[i] = fold
            capacity[fold] -= 1
            label_desired[fold] -= remaining[i]
            unassigned.discard(i)
    for i in list(unassigned):
        fold = int(np.argmax(capacity))
        assign[i] = fold
        capacity[fold] -= 1
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame[assign == 0].to_csv(out / "train.csv", index=False)
    frame[assign == 1].to_csv(out / "test.csv", index=False)
    return {"train_rows": int((assign == 0).sum()), "test_rows": int((assign == 1).sum()),
            "test_share_by_label": {nm: round(float(Y[assign == 1, j].sum() / max(Y[:, j].sum(), 1)), 3) for j, nm in enumerate(names)}, "written_to": {"train": str(out / "train.csv"), "test": str(out / "test.csv")}}


@tool
def predict_multilabel(model_path: str, data: str, id_column: str | None = None, threshold: float | None = None, top_k: int | None = None, output_path: str | None = None) -> dict:
    """Score new rows with a saved multi-label model: the predicted label set per row, every label's probability, and optionally only the top-k labels — uses the per-label thresholds if `optimise_label_thresholds` saved them.

    Args:
        model_path: Path of the saved multi-label model (joblib).
        id_column: Column to carry through to the output.
        threshold: Override the saved threshold(s) with one value.
        top_k: Return exactly the k most likely labels per row instead of thresholding.
    """
    model = joblib.load(model_path)
    frame = load_frame(data)
    require_columns(frame, model.columns)
    proba = model.proba(frame)
    if top_k:
        pred = np.zeros_like(proba, dtype=int)
        np.put_along_axis(pred, np.argsort(-proba, axis=1)[:, :top_k], 1, axis=1)
    else:
        pred = decide(proba, model.thresholds if threshold is None else threshold)
    out = pd.DataFrame({"predicted_labels": ["|".join(model.labels[j] for j in np.flatnonzero(r)) for r in pred]}, index=frame.index)
    for j, name in enumerate(model.labels):
        out[f"p_{name}"] = proba[:, j]
    if id_column and id_column in frame.columns:
        out.insert(0, id_column, frame[id_column])
    return {"rows": len(out), "mean_labels_per_row": round(float(pred.sum(axis=1).mean()), 3), "written_to": save_frame(out, output_path, "multilabel_predictions.csv"), "sample": out.head(5)}
