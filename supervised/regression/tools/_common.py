"""Helpers the regression tool modules share. Not a tool module itself (leading underscore)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_core.ml import feature_names, holdout, make_pipeline, save_model, sk, split_xy, top_items

SCALED = ("linear", "ridge", "lasso", "elastic_net", "svr", "knn", "mlp", "huber", "bayesian_ridge", "sgd")


def make_estimator(name: str, **params: Any) -> Any:
    """A fresh regressor by short name; `params` override the defaults."""
    sk()
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import (
        BayesianRidge,
        ElasticNet,
        HuberRegressor,
        Lasso,
        LinearRegression,
        Ridge,
        SGDRegressor,
    )
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    seed = params.pop("random_state", 42)
    table: dict[str, tuple[Any, dict[str, Any]]] = {
        "linear": (LinearRegression, {}),
        "ridge": (Ridge, {"alpha": 1.0}),
        "lasso": (Lasso, {"alpha": 0.01, "max_iter": 5000}),
        "elastic_net": (ElasticNet, {"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 5000}),
        "random_forest": (RandomForestRegressor, {"n_estimators": 300, "n_jobs": -1, "random_state": seed}),
        "extra_trees": (ExtraTreesRegressor, {"n_estimators": 300, "n_jobs": -1, "random_state": seed}),
        "gradient_boosting": (GradientBoostingRegressor, {"random_state": seed}),
        "hist_gradient_boosting": (HistGradientBoostingRegressor, {"random_state": seed}),
        "decision_tree": (DecisionTreeRegressor, {"max_depth": 6, "random_state": seed}),
        "svr": (SVR, {}),
        "knn": (KNeighborsRegressor, {"n_neighbors": 10}),
        "mlp": (MLPRegressor, {"hidden_layer_sizes": (64, 32), "max_iter": 800, "random_state": seed}),
        "huber": (HuberRegressor, {"max_iter": 500}),
        "bayesian_ridge": (BayesianRidge, {}),
        "sgd": (SGDRegressor, {"random_state": seed}),
    }
    if name not in table:
        raise ValueError(f"unknown regressor {name!r}; choose one of {sorted(table)}")
    cls, defaults = table[name]
    return cls(**{**defaults, **params})


MODEL_NAMES = tuple(["linear", "ridge", "lasso", "elastic_net", "random_forest", "extra_trees", "gradient_boosting",
                     "hist_gradient_boosting", "decision_tree", "svr", "knn", "mlp", "huber", "bayesian_ridge", "sgd"])


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """RMSE, MAE, median AE, R², and MAPE (over rows where the truth is not ~0)."""
    sk()
    from sklearn import metrics as m

    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    out = {
        "rmse": float(np.sqrt(m.mean_squared_error(y_true, y_pred))),
        "mae": float(m.mean_absolute_error(y_true, y_pred)),
        "median_ae": float(m.median_absolute_error(y_true, y_pred)),
        "r2": float(m.r2_score(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }
    nonzero = np.abs(y_true) > 1e-9
    if nonzero.any():
        out["mape"] = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])))
    return {k: round(v, 5) for k, v in out.items()}


def importances(pipe: Any, X: pd.DataFrame, k: int = 12) -> dict[str, float] | None:
    model = pipe.named_steps["model"]
    model = getattr(model, "regressor_", model)
    names = feature_names(pipe, X)
    values = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.ravel(np.asarray(model.coef_))
    if values is None or len(values) != len(names):
        return None
    return top_items(dict(zip(names, values)), k)


def run_regressor(
    estimator: Any, label: str, data: Any, target: str, features: Any = None, drop: Any = None,
    test_size: float = 0.2, seed: int = 42, scale: bool = True, model_path: str | None = None, cv: int = 0,
) -> dict[str, Any]:
    """Hold out rows, fit `estimator` behind the shared preprocessor, and report hold-out (and CV) error."""
    sk()
    from sklearn.model_selection import KFold, cross_validate

    X, y = split_xy(data, target, features, drop)
    y = pd.to_numeric(y, errors="raise")
    pipe = make_pipeline(estimator, X, scale=scale)
    X_tr, X_te, y_tr, y_te = holdout(X, y, test_size, seed)
    pipe.fit(X_tr, y_tr)
    result: dict[str, Any] = {
        "model": label,
        "rows_train": len(X_tr),
        "rows_test": len(X_te),
        "holdout": regression_metrics(y_te, pipe.predict(X_te)),
        "top_features": importances(pipe, X),
        "target_mean": round(float(y.mean()), 5),
        "target_std": round(float(y.std()), 5),
    }
    if cv and cv > 1:
        scores = cross_validate(make_pipeline(estimator, X, scale=scale), X, y, cv=KFold(cv, shuffle=True, random_state=seed),
                                scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"], n_jobs=1)
        result["cv"] = {
            "rmse": {"mean": round(float(-scores["test_neg_root_mean_squared_error"].mean()), 5), "std": round(float(scores["test_neg_root_mean_squared_error"].std()), 5)},
            "mae": {"mean": round(float(-scores["test_neg_mean_absolute_error"].mean()), 5), "std": round(float(scores["test_neg_mean_absolute_error"].std()), 5)},
            "r2": {"mean": round(float(scores["test_r2"].mean()), 5), "std": round(float(scores["test_r2"].std()), 5)},
        }
    result["model_path"] = save_model(pipe, model_path)
    return result
