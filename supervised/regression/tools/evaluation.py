"""Judge a regressor honestly: cross-validation, residuals, intervals, importance, partial dependence, tuning, and scoring new data."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from ml_core.data import load_frame, save_frame
from ml_core.ml import holdout, make_pipeline, save_model, sk, split_xy, top_items
from ml_core.tooling import tool
from supervised.regression.tools._common import SCALED, make_estimator, regression_metrics

Model = Literal["linear", "ridge", "lasso", "elastic_net", "random_forest", "extra_trees", "gradient_boosting",
                "hist_gradient_boosting", "decision_tree", "svr", "knn", "huber", "bayesian_ridge"]


def _pipe(model: str, X: pd.DataFrame, seed: int) -> Any:
    fixed = ("linear", "ridge", "lasso", "elastic_net", "svr", "knn", "huber", "bayesian_ridge")
    est = make_estimator(model) if model in fixed else make_estimator(model, random_state=seed)
    return make_pipeline(est, X, scale=model in SCALED)


@tool
def evaluate_regressor_cv(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    model: Model = "random_forest", cv: int = 5, seed: int = 42,
) -> dict:
    """K-fold cross-validation of one algorithm with RMSE, MAE and R² (mean and spread) — a steadier estimate than one hold-out.

    Args:
        model: Algorithm to evaluate.
    """
    sk()
    from sklearn.model_selection import KFold, cross_validate

    X, y = split_xy(data, target, features, drop)
    res = cross_validate(_pipe(model, X, seed), X, y.astype(float), cv=KFold(cv, shuffle=True, random_state=seed),
                         scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"], n_jobs=1)
    def stat(key: str, sign: int) -> dict:
        v = sign * res[f"test_{key}"]
        return {"mean": round(float(v.mean()), 5), "std": round(float(v.std()), 5)}
    return {"model": model, "folds": cv, "rmse": stat("neg_root_mean_squared_error", -1), "mae": stat("neg_mean_absolute_error", -1), "r2": stat("r2", 1),
            "target_std": round(float(y.std()), 5)}


@tool
def residual_analysis(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    model: Model = "random_forest", cv: int = 5, seed: int = 42,
) -> dict:
    """Out-of-fold residual diagnostics: bias, normality, heteroscedasticity (error growing with the prediction), and the segments where the model is worst.

    Args:
        model: Algorithm to diagnose.
    """
    sk()
    from scipy import stats
    from sklearn.model_selection import KFold, cross_val_predict

    X, y = split_xy(data, target, features, drop)
    y = y.astype(float)
    pred = cross_val_predict(_pipe(model, X, seed), X, y, cv=KFold(cv, shuffle=True, random_state=seed))
    res = y.to_numpy() - pred
    hetero = float(stats.spearmanr(np.abs(res), pred)[0])
    deciles = pd.qcut(pd.Series(pred), 5, duplicates="drop")
    by_band = pd.Series(np.abs(res)).groupby(deciles.values, observed=True).mean()
    notes = []
    if abs(res.mean()) > 0.1 * res.std():
        notes.append("noticeable bias: predictions are systematically off in one direction")
    if abs(hetero) > 0.3:
        notes.append("heteroscedastic: error grows with the prediction — try a log/transformed target or a gamma GLM")
    if abs(float(stats.skew(res))) > 1:
        notes.append("skewed residuals: a few large misses dominate; check outliers or use a robust loss")
    return {"metrics": regression_metrics(y, pred), "residual_mean": round(float(res.mean()), 5), "residual_skew": round(float(stats.skew(res)), 4),
            "error_vs_prediction_spearman": round(hetero, 4), "mean_abs_error_by_prediction_band": {str(k): round(float(v), 4) for k, v in by_band.items()},
            "notes": notes or ["no residual problem stands out"]}


@tool
def prediction_intervals(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    model: Model = "gradient_boosting", coverage: float = 0.9, test_size: float = 0.25, seed: int = 42,
) -> dict:
    """Split-conformal prediction intervals around any model: fits on one part, calibrates the residual quantile on another, and checks the coverage on the rest.

    Args:
        model: Algorithm whose predictions get intervals.
        coverage: Target share of true values inside the interval, e.g. 0.9.
    """
    sk()
    from sklearn.model_selection import train_test_split

    X, y = split_xy(data, target, features, drop)
    y = y.astype(float)
    X_rest, X_te, y_rest, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_tr, X_cal, y_tr, y_cal = train_test_split(X_rest, y_rest, test_size=0.3, random_state=seed)
    pipe = _pipe(model, X_tr, seed).fit(X_tr, y_tr)
    scores = np.abs(y_cal.to_numpy() - pipe.predict(X_cal))
    q = float(np.quantile(scores, min(1.0, np.ceil((len(scores) + 1) * coverage) / len(scores))))
    pred = pipe.predict(X_te)
    inside = (y_te.to_numpy() >= pred - q) & (y_te.to_numpy() <= pred + q)
    return {"model": model, "target_coverage": coverage, "half_width": round(q, 5), "observed_coverage": round(float(inside.mean()), 4),
            "relative_width": round(float(2 * q / (y.max() - y.min())), 4),
            "note": "the interval has the same width everywhere; if errors grow with the prediction, use fit_quantile_regression pairs instead"}


@tool
def permutation_importance_regressor(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    model: Model = "random_forest", n_repeats: int = 10, k: int = 15, test_size: float = 0.25, seed: int = 42,
) -> dict:
    """Permutation importance on held-out rows: how much RMSE rises when one raw column is shuffled — model-agnostic and fair to high-cardinality columns.

    Args:
        model: Algorithm to explain.
        n_repeats: Shuffles per column.
    """
    sk()
    from sklearn.inspection import permutation_importance

    X, y = split_xy(data, target, features, drop)
    X_tr, X_te, y_tr, y_te = holdout(X, y.astype(float), test_size, seed)
    pipe = _pipe(model, X_tr, seed).fit(X_tr, y_tr)
    res = permutation_importance(pipe, X_te, y_te, scoring="neg_root_mean_squared_error", n_repeats=n_repeats, random_state=seed, n_jobs=1)
    scores = dict(zip(X.columns, res.importances_mean))
    return {"unit": "increase in RMSE when the column is shuffled", "importance": top_items(scores, k), "no_effect": [c for c, v in scores.items() if v <= 0][:20]}


@tool
def partial_dependence_report(
    data: str, target: str, feature: str, features: list[str] | None = None, drop: list[str] | None = None,
    model: Model = "gradient_boosting", grid_points: int = 10, seed: int = 42,
) -> dict:
    """Partial dependence of the prediction on one numeric feature — the average predicted target as that feature moves across its range, others held as observed.

    Args:
        feature: The numeric feature to sweep.
        model: Algorithm to explain.
        grid_points: Points along the feature's range.
    """
    sk()
    from sklearn.inspection import partial_dependence

    X, y = split_xy(data, target, features, drop)
    if feature not in X.columns:
        raise KeyError(f"feature {feature!r} is not among {list(X.columns)}")
    pipe = _pipe(model, X, seed).fit(X, y.astype(float))
    pd_res = partial_dependence(pipe, X, [feature], grid_resolution=grid_points, kind="average")
    grid, avg = pd_res["grid_values"][0], pd_res["average"][0]
    slope = float(np.polyfit(grid, avg, 1)[0]) if len(grid) > 1 else 0.0
    return {"feature": feature, "grid": np.round(grid, 4).tolist(), "average_prediction": np.round(avg, 4).tolist(),
            "overall_slope": round(slope, 5), "shape": "increasing" if avg[-1] > avg[0] * 1.02 and slope > 0 else "decreasing" if slope < 0 else "flat"}


@tool
def learning_curve_regression(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    model: Model = "random_forest", cv: int = 5, seed: int = 42,
) -> dict:
    """Train and validation RMSE at growing training sizes — shows whether more data, regularisation or a richer model would help.

    Args:
        model: Algorithm to evaluate.
    """
    sk()
    from sklearn.model_selection import KFold, learning_curve

    X, y = split_xy(data, target, features, drop)
    sizes, train, valid = learning_curve(_pipe(model, X, seed), X, y.astype(float), cv=KFold(cv, shuffle=True, random_state=seed),
                                         scoring="neg_root_mean_squared_error", train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=1)
    tr, va = -train.mean(axis=1), -valid.mean(axis=1)
    gap = float(va[-1] - tr[-1])
    verdict = ("high variance: validation error far above training error — more data or stronger regularisation" if gap > 0.25 * va[-1]
               else "still improving: validation error is falling at the largest size — more data would help" if va[-2] - va[-1] > 0.03 * va[-1]
               else "converged: more data will not help much — improve features or model class")
    return {"train_sizes": sizes.tolist(), "train_rmse": tr.round(4).tolist(), "validation_rmse": va.round(4).tolist(), "verdict": verdict}


@tool
def error_by_segment(
    data: str, target: str, segment: str, features: list[str] | None = None, drop: list[str] | None = None,
    model: Model = "random_forest", cv: int = 5, seed: int = 42,
) -> dict:
    """Out-of-fold error broken down by a categorical (or binned numeric) column — finds the segments the model serves badly.

    Args:
        segment: Column to group errors by; numeric columns are cut into quintiles.
        model: Algorithm to evaluate.
    """
    sk()
    from sklearn.model_selection import KFold, cross_val_predict

    frame = load_frame(data)
    X, y = split_xy(frame, target, features, drop)
    y = y.astype(float)
    pred = cross_val_predict(_pipe(model, X, seed), X, y, cv=KFold(cv, shuffle=True, random_state=seed))
    groups = frame.loc[X.index, segment]
    if pd.api.types.is_numeric_dtype(groups) and groups.nunique() > 8:
        groups = pd.qcut(groups, 5, duplicates="drop")
    err = pd.DataFrame({"g": groups.astype(str).to_numpy(), "abs": np.abs(y.to_numpy() - pred), "signed": pred - y.to_numpy()})
    table = err.groupby("g").agg(rows=("abs", "size"), mae=("abs", "mean"), bias=("signed", "mean")).round(4)
    return {"segment": segment, "overall_mae": round(float(err["abs"].mean()), 4), "segments": table.reset_index().to_dict("records"),
            "worst_segment": str(table["mae"].idxmax())}


@tool
def tune_regressor(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    model: Literal["ridge", "lasso", "random_forest", "gradient_boosting", "hist_gradient_boosting", "svr", "knn"] = "gradient_boosting",
    n_iter: int = 20, cv: int = 3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Randomised hyper-parameter search with cross-validation over a preset space for the chosen algorithm; reports the best settings and the refit model's hold-out error.

    Args:
        model: Algorithm to tune.
        n_iter: Settings sampled.
    """
    sk()
    from sklearn.model_selection import KFold, RandomizedSearchCV

    X, y = split_xy(data, target, features, drop)
    y = y.astype(float)
    spaces = {
        "ridge": {"alpha": np.logspace(-3, 3, 25)},
        "lasso": {"alpha": np.logspace(-4, 0, 25)},
        "random_forest": {"n_estimators": [150, 300, 500], "max_depth": [None, 6, 10, 16], "min_samples_leaf": [1, 2, 5, 10], "max_features": [0.5, 0.75, 1.0]},
        "gradient_boosting": {"n_estimators": [100, 200, 400], "learning_rate": [0.02, 0.05, 0.1], "max_depth": [2, 3, 4, 5], "subsample": [0.7, 0.85, 1.0]},
        "hist_gradient_boosting": {"max_iter": [150, 300, 500], "learning_rate": [0.02, 0.05, 0.1], "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0, 0.1, 1.0]},
        "svr": {"C": np.logspace(-1, 3, 10), "epsilon": [0.01, 0.1, 0.3], "gamma": ["scale", 0.01, 0.1]},
        "knn": {"n_neighbors": [3, 5, 9, 15, 25], "weights": ["uniform", "distance"]},
    }
    X_tr, X_te, y_tr, y_te = holdout(X, y, 0.2, seed)
    grid = {f"model__{k}": v for k, v in spaces[model].items()}
    size = int(np.prod([len(v) for v in grid.values()]))
    search = RandomizedSearchCV(_pipe(model, X_tr, seed), grid, n_iter=min(n_iter, size), cv=KFold(cv, shuffle=True, random_state=seed),
                                scoring="neg_root_mean_squared_error", random_state=seed, n_jobs=1).fit(X_tr, y_tr)
    return {"model": model, "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "best_cv_rmse": round(float(-search.best_score_), 5), "holdout": regression_metrics(y_te, search.best_estimator_.predict(X_te)),
            "model_path": save_model(search.best_estimator_, model_path)}


@tool
def predict_with_saved_regressor(data: str, model_path: str, output_path: str | None = None, id_column: str | None = None) -> dict:
    """Score new rows with a model saved by one of the fit tools and write the predictions to a file.

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
    return {"rows": len(out), "written_to": save_frame(out, output_path, "predictions.csv"), "summary": out["prediction"].describe().round(4).to_dict(), "sample": out.head(5)}
