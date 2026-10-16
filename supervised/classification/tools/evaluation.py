"""Judge a classifier honestly: cross-validation, confusion, thresholds, calibration, importance, errors, and scoring new data."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from ml_core.data import load_frame, save_frame
from ml_core.ml import holdout, make_pipeline, sk, split_xy, top_items
from ml_core.tooling import tool
from supervised.classification.tools._common import (
    classification_metrics,
    make_estimator,
    positive_class,
)

Model = Literal["logistic_regression", "random_forest", "extra_trees", "gradient_boosting",
                "hist_gradient_boosting", "decision_tree", "svm", "knn", "gaussian_nb", "mlp", "adaboost", "lda", "sgd"]


def _pipe(model: str, X: pd.DataFrame, seed: int) -> Any:
    fixed = ("knn", "gaussian_nb", "bernoulli_nb", "lda")
    est = make_estimator(model) if model in fixed else make_estimator(model, random_state=seed)
    return make_pipeline(est, X, scale=model in ("logistic_regression", "svm", "knn", "mlp", "lda", "sgd", "gaussian_nb"))


@tool
def evaluate_classifier_cv(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    model: Model = "random_forest", cv: int = 5, seed: int = 42,
) -> dict:
    """Stratified k-fold cross-validation of one algorithm with several metrics at once, mean and spread — a steadier estimate than a single hold-out.

    Args:
        model: Algorithm to evaluate.
    """
    sk()
    from sklearn.model_selection import StratifiedKFold, cross_validate

    X, y = split_xy(data, target, features, drop)
    folds = StratifiedKFold(max(2, min(cv, int(y.value_counts().min()))), shuffle=True, random_state=seed)
    scoring = ["accuracy", "balanced_accuracy", "f1_macro"] + (["roc_auc", "average_precision"] if y.nunique() == 2 else [])
    if y.nunique() == 2 and not set(y.unique()) <= {0, 1}:
        scoring = ["accuracy", "balanced_accuracy", "f1_macro"]
    res = cross_validate(_pipe(model, X, seed), X, y, cv=folds, scoring=scoring, n_jobs=1)
    return {"model": model, "folds": folds.get_n_splits(),
            "metrics": {k[5:]: {"mean": round(float(v.mean()), 5), "std": round(float(v.std()), 5)} for k, v in res.items() if k.startswith("test_")}}


@tool
def confusion_matrix_report(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    model: Model = "random_forest", cv: int = 5, seed: int = 42,
) -> dict:
    """Out-of-fold confusion matrix with per-class precision, recall, F1 and support — shows which classes get mixed up.

    Args:
        model: Algorithm to evaluate.
    """
    sk()
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X, y = split_xy(data, target, features, drop)
    folds = StratifiedKFold(max(2, min(cv, int(y.value_counts().min()))), shuffle=True, random_state=seed)
    pred = cross_val_predict(_pipe(model, X, seed), X, y, cv=folds)
    labels = sorted(y.unique(), key=str)
    return {
        "labels": [str(l) for l in labels],
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "per_class": classification_report(y, pred, labels=labels, output_dict=True, zero_division=0),
    }


@tool
def search_decision_threshold(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    model: Model = "random_forest", metric: Literal["f1", "youden", "cost", "recall_at_precision"] = "f1",
    false_positive_cost: float = 1.0, false_negative_cost: float = 5.0, min_precision: float = 0.8,
    positive_label: str | None = None, cv: int = 5, seed: int = 42,
) -> dict:
    """Find the probability cut-off that optimises F1, Youden's J, a cost matrix, or recall at a minimum precision, on out-of-fold probabilities (binary only).

    Args:
        model: Algorithm that produces the probabilities.
        metric: f1, youden, cost, or recall_at_precision.
        false_positive_cost: Cost of flagging a negative (metric=cost).
        false_negative_cost: Cost of missing a positive (metric=cost).
        min_precision: Precision floor (metric=recall_at_precision).
    """
    sk()
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X, y = split_xy(data, target, features, drop)
    if y.nunique() != 2:
        raise ValueError("threshold search needs a binary target")
    classes = sorted(y.unique(), key=str)
    pos = positive_class(classes, positive_label)
    truth = (y == pos).astype(int).to_numpy()
    folds = StratifiedKFold(max(2, min(cv, int(y.value_counts().min()))), shuffle=True, random_state=seed)
    proba = cross_val_predict(_pipe(model, X, seed), X, y, cv=folds, method="predict_proba")[:, classes.index(pos)]
    grid = np.unique(np.round(np.quantile(proba, np.linspace(0.01, 0.99, 197)), 4))
    rows = []
    for t in grid:
        pred = proba >= t
        tp, fp = int((pred & (truth == 1)).sum()), int((pred & (truth == 0)).sum())
        fn, tn = int((~pred & (truth == 1)).sum()), int((~pred & (truth == 0)).sum())
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append({"threshold": float(t), "precision": precision, "recall": recall, "f1": f1,
                     "youden": recall + tn / max(tn + fp, 1) - 1,
                     "cost": fp * false_positive_cost + fn * false_negative_cost})
    if metric == "cost":
        best = min(rows, key=lambda r: r["cost"])
    elif metric == "recall_at_precision":
        feasible = [r for r in rows if r["precision"] >= min_precision]
        best = max(feasible, key=lambda r: r["recall"]) if feasible else max(rows, key=lambda r: r["precision"])
    else:
        best = max(rows, key=lambda r: r[metric])
    return {"positive_class": str(pos), "metric": metric,
            "best": {k: round(v, 4) for k, v in best.items()}, "default_0.5_f1": round(min(rows, key=lambda r: abs(r["threshold"] - 0.5))["f1"], 4)}


@tool
def calibrate_classifier(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    model: Model = "random_forest", method: Literal["sigmoid", "isotonic"] = "isotonic",
    test_size: float = 0.25, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Calibrate predicted probabilities (Platt or isotonic) and compare Brier score and expected calibration error before and after (binary only).

    Args:
        model: Algorithm to calibrate.
        method: sigmoid (Platt) suits small data; isotonic needs more.
    """
    sk()
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss
    from ml_core.ml import save_model

    X, y = split_xy(data, target, features, drop)
    if y.nunique() != 2:
        raise ValueError("calibration here is binary only")
    classes = sorted(y.unique(), key=str)
    X_tr, X_te, y_tr, y_te = holdout(X, y, test_size, seed, stratify=True)
    raw = _pipe(model, X_tr, seed).fit(X_tr, y_tr)
    cal = CalibratedClassifierCV(_pipe(model, X_tr, seed), method=method, cv=3).fit(X_tr, y_tr)
    truth = (y_te == classes[-1]).astype(int).to_numpy()

    def ece(p: np.ndarray, bins: int = 10) -> float:
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
        return float(sum(abs(truth[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))

    p_raw, p_cal = raw.predict_proba(X_te)[:, 1], cal.predict_proba(X_te)[:, 1]
    return {"model": model, "method": method,
            "before": {"brier": round(brier_score_loss(truth, p_raw), 5), "ece": round(ece(p_raw), 5)},
            "after": {"brier": round(brier_score_loss(truth, p_cal), 5), "ece": round(ece(p_cal), 5)},
            "model_path": save_model(cal, model_path)}


@tool
def permutation_importance_classifier(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    model: Model = "random_forest", scoring: Literal["balanced_accuracy", "f1_macro", "roc_auc", "accuracy"] = "balanced_accuracy",
    n_repeats: int = 10, k: int = 15, test_size: float = 0.25, seed: int = 42,
) -> dict:
    """Permutation importance on held-out rows: how much the score drops when one raw column is shuffled — model-agnostic and not biased towards high-cardinality columns.

    Args:
        model: Algorithm to explain.
        n_repeats: Shuffles per column.
    """
    sk()
    from sklearn.inspection import permutation_importance

    X, y = split_xy(data, target, features, drop)
    X_tr, X_te, y_tr, y_te = holdout(X, y, test_size, seed, stratify=True)
    pipe = _pipe(model, X_tr, seed).fit(X_tr, y_tr)
    metric = scoring if (y.nunique() == 2 or scoring != "roc_auc") else "balanced_accuracy"
    res = permutation_importance(pipe, X_te, y_te, scoring=metric, n_repeats=n_repeats, random_state=seed, n_jobs=1)
    scores = dict(zip(X.columns, res.importances_mean))
    return {"scoring": metric, "importance": top_items(scores, k),
            "no_effect": [c for c, v in scores.items() if v <= 0][:20]}


@tool
def learning_curve_report(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    model: Model = "random_forest", cv: int = 5, seed: int = 42,
) -> dict:
    """Train/validation score at growing training sizes — shows whether more data or a simpler/richer model would help (balanced accuracy).

    Args:
        model: Algorithm to evaluate.
    """
    sk()
    from sklearn.model_selection import StratifiedKFold, learning_curve

    X, y = split_xy(data, target, features, drop)
    folds = StratifiedKFold(max(2, min(cv, int(y.value_counts().min()))), shuffle=True, random_state=seed)
    sizes, train, valid = learning_curve(_pipe(model, X, seed), X, y, cv=folds, scoring="balanced_accuracy",
                                         train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=1)
    gap = float(train[-1].mean() - valid[-1].mean())
    verdict = ("high variance: the model memorises — more data, stronger regularisation or a simpler model" if gap > 0.1
               else "high bias: train and validation are both low — add features or use a richer model" if valid[-1].mean() < 0.7
               else "healthy: small gap and a good validation score")
    return {"train_sizes": sizes.tolist(), "train_score": train.mean(axis=1).round(4).tolist(),
            "validation_score": valid.mean(axis=1).round(4).tolist(), "final_gap": round(gap, 4), "verdict": verdict}


@tool
def error_analysis(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    model: Model = "random_forest", k: int = 20, cv: int = 5, seed: int = 42, output_path: str | None = None,
) -> dict:
    """List the rows the model gets confidently wrong out-of-fold, plus the columns whose values differ most between errors and correct rows — where to look for label noise or missing features.

    Args:
        model: Algorithm to analyse.
        k: How many worst errors to return.
    """
    sk()
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X, y = split_xy(data, target, features, drop)
    folds = StratifiedKFold(max(2, min(cv, int(y.value_counts().min()))), shuffle=True, random_state=seed)
    pipe = _pipe(model, X, seed)
    proba = cross_val_predict(pipe, X, y, cv=folds, method="predict_proba")
    classes = sorted(y.unique(), key=str)
    pred = np.array(classes, dtype=object)[proba.argmax(axis=1)]
    wrong = pred != y.to_numpy()
    confidence = proba.max(axis=1)
    out = X.copy()
    out["actual"], out["predicted"], out["confidence"] = y.to_numpy(), pred, confidence
    worst = out[wrong].sort_values("confidence", ascending=False)
    shifts = {}
    for c in X.select_dtypes("number").columns:
        sd = X[c].std()
        if sd and sd > 0:
            shifts[c] = float((X.loc[wrong, c].mean() - X.loc[~wrong, c].mean()) / sd)
    return {"error_rate": round(float(wrong.mean()), 4), "errors": int(wrong.sum()),
            "worst_errors": worst.head(k), "columns_where_errors_differ": top_items(shifts, 10),
            "written_to": save_frame(worst, output_path, "classification_errors.csv") if output_path else None}


@tool
def tune_classifier(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    model: Literal["logistic_regression", "random_forest", "gradient_boosting", "hist_gradient_boosting", "svm", "knn"] = "random_forest",
    n_iter: int = 20, cv: int = 3, scoring: Literal["f1_macro", "balanced_accuracy", "roc_auc", "average_precision", "accuracy"] = "f1_macro",
    seed: int = 42, model_path: str | None = None,
) -> dict:
    """Randomised hyper-parameter search with cross-validation over a sensible preset space for the chosen algorithm; reports the best settings and the refit model's hold-out score.

    Args:
        model: Algorithm to tune.
        n_iter: Settings sampled.
    """
    sk()
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from ml_core.ml import save_model

    X, y = split_xy(data, target, features, drop)
    spaces = {
        "logistic_regression": {"C": np.logspace(-3, 2, 20)},
        "random_forest": {"n_estimators": [150, 300, 500], "max_depth": [None, 6, 10, 16], "min_samples_leaf": [1, 2, 5, 10], "max_features": ["sqrt", 0.5, 1.0]},
        "gradient_boosting": {"n_estimators": [100, 200, 400], "learning_rate": [0.02, 0.05, 0.1, 0.2], "max_depth": [2, 3, 4, 5], "subsample": [0.7, 0.85, 1.0]},
        "hist_gradient_boosting": {"max_iter": [150, 300, 500], "learning_rate": [0.02, 0.05, 0.1, 0.2], "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0, 0.1, 1.0]},
        "svm": {"C": np.logspace(-2, 2, 10), "gamma": ["scale", 0.01, 0.1, 1.0]},
        "knn": {"n_neighbors": [3, 5, 9, 15, 25, 41], "weights": ["uniform", "distance"]},
    }
    metric = scoring if (y.nunique() == 2 or scoring not in ("roc_auc", "average_precision")) else "f1_macro"
    X_tr, X_te, y_tr, y_te = holdout(X, y, 0.2, seed, stratify=True)
    pipe = _pipe(model, X_tr, seed)
    grid = {f"model__{k}": v for k, v in spaces[model].items()}
    size = int(np.prod([len(v) for v in grid.values()]))
    search = RandomizedSearchCV(pipe, grid, n_iter=min(n_iter, size), cv=StratifiedKFold(max(2, min(cv, int(y_tr.value_counts().min()))), shuffle=True, random_state=seed),
                                scoring=metric, random_state=seed, n_jobs=1).fit(X_tr, y_tr)
    best = search.best_estimator_
    proba = best.predict_proba(X_te) if hasattr(best, "predict_proba") else None
    return {"model": model, "scoring": metric, "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "best_cv_score": round(float(search.best_score_), 5),
            "holdout": classification_metrics(y_te, best.predict(X_te), proba, list(best.classes_)),
            "model_path": save_model(best, model_path)}


@tool
def predict_with_saved_model(data: str, model_path: str, output_path: str | None = None, id_column: str | None = None) -> dict:
    """Score new rows with a model saved by one of the fit tools, writing predictions (and class probabilities when available) to a file.

    Args:
        model_path: Path of the saved joblib model.
        id_column: Column to carry through to the output so predictions can be joined back.
    """
    import joblib

    frame = load_frame(data)
    model = joblib.load(model_path)
    prep = model.named_steps["prep"] if hasattr(model, "named_steps") else None
    needed = list(prep.feature_names_in_) if prep is not None and hasattr(prep, "feature_names_in_") else list(frame.columns)
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise KeyError(f"the model needs columns {missing} that the data does not have")
    out = pd.DataFrame(index=frame.index)
    if id_column and id_column in frame.columns:
        out[id_column] = frame[id_column]
    out["prediction"] = model.predict(frame[needed])
    if hasattr(model, "predict_proba"):
        for cls, col in zip(model.classes_, model.predict_proba(frame[needed]).T):
            out[f"proba_{cls}"] = col
    path = save_frame(out, output_path, "predictions.csv")
    return {"rows": len(out), "written_to": path, "prediction_counts": out["prediction"].value_counts().to_dict(), "sample": out.head(5)}
