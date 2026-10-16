"""Helpers the classification tool modules share. Not a tool module itself (leading underscore)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_core.ml import feature_names, holdout, make_pipeline, save_model, sk, split_xy, svc_with_probabilities, top_items

SCORINGS = ("accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc", "average_precision")


def make_estimator(name: str, **params: Any) -> Any:
    """A fresh estimator by short name, with sensible defaults overridden by `params`."""
    sk()
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import (
        AdaBoostClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.naive_bayes import BernoulliNB, GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.tree import DecisionTreeClassifier

    seed = params.pop("random_state", 42)
    table: dict[str, tuple[Any, dict[str, Any]]] = {
        "logistic_regression": (LogisticRegression, {"max_iter": 1000}),
        "random_forest": (RandomForestClassifier, {"n_estimators": 300, "n_jobs": -1, "random_state": seed}),
        "extra_trees": (ExtraTreesClassifier, {"n_estimators": 300, "n_jobs": -1, "random_state": seed}),
        "gradient_boosting": (GradientBoostingClassifier, {"random_state": seed}),
        "hist_gradient_boosting": (HistGradientBoostingClassifier, {"random_state": seed}),
        "decision_tree": (DecisionTreeClassifier, {"max_depth": 6, "random_state": seed}),
        "svm": (lambda **kw: svc_with_probabilities(**kw), {"random_state": seed}),
        "knn": (KNeighborsClassifier, {"n_neighbors": 15}),
        "gaussian_nb": (GaussianNB, {}),
        "bernoulli_nb": (BernoulliNB, {}),
        "mlp": (MLPClassifier, {"hidden_layer_sizes": (64, 32), "max_iter": 500, "random_state": seed}),
        "adaboost": (AdaBoostClassifier, {"random_state": seed}),
        "lda": (LinearDiscriminantAnalysis, {}),
        "sgd": (SGDClassifier, {"loss": "log_loss", "random_state": seed}),
    }
    if name not in table:
        raise ValueError(f"unknown classifier {name!r}; choose one of {sorted(table)}")
    cls, defaults = table[name]
    return cls(**{**defaults, **params})


MODEL_NAMES = (
    "logistic_regression", "random_forest", "extra_trees", "gradient_boosting", "hist_gradient_boosting",
    "decision_tree", "svm", "knn", "gaussian_nb", "bernoulli_nb", "mlp", "adaboost", "lda", "sgd",
)


def parse_layers(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(text).replace(" ", "").split(",") if part) or (64,)


def positive_class(classes: list[Any], positive_label: Any = None) -> Any:
    """The class treated as positive in binary metrics: the one named, else the rarer-by-convention last class."""
    if positive_label is not None:
        for c in classes:
            if str(c) == str(positive_label):
                return c
        raise ValueError(f"positive_label {positive_label!r} is not one of the classes {classes}")
    return classes[-1]


def classification_metrics(
    y_true: Any, y_pred: Any, proba: np.ndarray | None, classes: list[Any], positive_label: Any = None
) -> dict[str, float]:
    """Hold-out metrics: accuracy family always; AUC / PR-AUC / log-loss when probabilities exist."""
    sk()
    from sklearn import metrics as m

    out: dict[str, float] = {
        "accuracy": m.accuracy_score(y_true, y_pred),
        "balanced_accuracy": m.balanced_accuracy_score(y_true, y_pred),
        "f1_macro": m.f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": m.f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": m.precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": m.recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if len(classes) == 2:
        pos = positive_class(classes, positive_label)
        out["precision_positive"] = m.precision_score(y_true, y_pred, pos_label=pos, zero_division=0)
        out["recall_positive"] = m.recall_score(y_true, y_pred, pos_label=pos, zero_division=0)
        out["f1_positive"] = m.f1_score(y_true, y_pred, pos_label=pos, zero_division=0)
    if proba is not None and len(set(pd.Series(y_true))) > 1:
        try:
            if len(classes) == 2:
                col = classes.index(positive_class(classes, positive_label))
                truth = (pd.Series(y_true).to_numpy() == classes[col]).astype(int)
                out["roc_auc"] = m.roc_auc_score(truth, proba[:, col])
                out["average_precision"] = m.average_precision_score(truth, proba[:, col])
                out["brier"] = m.brier_score_loss(truth, proba[:, col])
            else:
                out["roc_auc_ovr"] = m.roc_auc_score(
                    y_true, proba, multi_class="ovr", average="macro", labels=classes
                )
            out["log_loss"] = m.log_loss(y_true, proba, labels=classes)
        except ValueError:
            pass
    return {k: round(float(v), 5) for k, v in out.items()}


def importances(pipe: Any, X: pd.DataFrame, k: int = 12) -> dict[str, float] | None:
    """Model-native importances (trees) or |coefficients| (linear), keyed by transformed column name."""
    model = pipe.named_steps["model"]
    names = feature_names(pipe, X)
    values = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_)).mean(axis=0)
    if values is None or len(values) != len(names):
        return None
    return top_items(dict(zip(names, values)), k)


def run_classifier(
    estimator: Any,
    label: str,
    data: Any,
    target: str,
    features: Any = None,
    drop: Any = None,
    test_size: float = 0.2,
    seed: int = 42,
    scale: bool = True,
    model_path: str | None = None,
    cv: int = 0,
    positive_label: Any = None,
) -> dict[str, Any]:
    """Hold out a stratified test set, fit `estimator` behind the shared preprocessor, and report."""
    sk()
    from sklearn.model_selection import cross_validate

    X, y = split_xy(data, target, features, drop)
    if y.nunique() < 2:
        raise ValueError(f"target {target!r} has one class; a classifier needs at least two")
    pipe = make_pipeline(estimator, X, scale=scale)
    X_train, X_test, y_train, y_test = holdout(X, y, test_size, seed, stratify=True)
    pipe.fit(X_train, y_train)

    classes = [c for c in pipe.classes_]
    proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
    result: dict[str, Any] = {
        "model": label,
        "rows_train": len(X_train),
        "rows_test": len(X_test),
        "classes": [str(c) for c in classes],
        "holdout": classification_metrics(y_test, pipe.predict(X_test), proba, classes, positive_label),
        "top_features": importances(pipe, X),
    }
    if cv and cv > 1:
        scores = cross_validate(
            make_pipeline(estimator, X, scale=scale), X, y, cv=min(cv, int(y.value_counts().min())),
            scoring=["balanced_accuracy", "f1_macro"], n_jobs=1,
        )
        result["cv"] = {
            k.replace("test_", ""): {"mean": round(float(v.mean()), 5), "std": round(float(v.std()), 5)}
            for k, v in scores.items() if k.startswith("test_")
        }
    result["model_path"] = save_model(pipe, model_path)
    return result
