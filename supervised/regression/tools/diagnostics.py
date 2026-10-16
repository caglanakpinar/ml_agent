"""Read-only checks on a regression problem and its data, before any model is fitted."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ml_core.data import as_list, load_frame
from ml_core.ml import holdout, numeric_and_categorical, sk, split_xy
from ml_core.tooling import tool
from supervised.regression.tools._common import regression_metrics


@tool
def profile_regression_data(data: str, target: str, features: list[str] | None = None) -> dict:
    """Profile a regression dataset: size, target distribution, column kinds, missing values, constant and id-like columns, duplicate rows."""
    frame = load_frame(data)
    y = pd.to_numeric(frame[target], errors="coerce")
    names = as_list(features) or [c for c in frame.columns if c != target]
    X = frame[names]
    numeric, categorical = numeric_and_categorical(X)
    missing = X.isna().mean()
    return {
        "rows": len(frame),
        "target": {"missing": int(y.isna().sum()), "mean": y.mean(), "std": y.std(), "min": y.min(),
                   "p25": y.quantile(.25), "median": y.median(), "p75": y.quantile(.75), "max": y.max(),
                   "zeros": int((y == 0).sum()), "negatives": int((y < 0).sum())},
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "high_cardinality_columns": [c for c in names if c not in numeric and c not in categorical],
        "constant_columns": [c for c in names if X[c].nunique(dropna=False) <= 1],
        "id_like_columns": [c for c in names if len(frame) > 20 and X[c].nunique() >= 0.98 * len(frame)],
        "columns_with_missing": {c: round(float(r), 4) for c, r in missing[missing > 0].items()},
        "duplicate_rows": int(frame.duplicated().sum()),
    }


@tool
def analyze_target_distribution(data: str, target: str) -> dict:
    """Measure the target's skew, kurtosis, zero share and outliers, and say whether to model it on a log scale or with a count/positive-skew GLM."""
    from scipy import stats

    y = pd.to_numeric(load_frame(data)[target], errors="coerce").dropna()
    skew, kurt = float(stats.skew(y)), float(stats.kurtosis(y))
    q1, q3 = y.quantile(.25), y.quantile(.75)
    iqr = q3 - q1
    outliers = int(((y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)).sum())
    is_count = bool((y >= 0).all() and np.allclose(y, np.round(y)))
    advice = []
    if is_count:
        advice.append("non-negative integers: try fit_glm with family='poisson' (or tweedie if there are many zeros)")
    if (y > 0).all() and skew > 1:
        advice.append("positive and right-skewed: fit_transformed_target_regression (log) or fit_glm family='gamma'")
    if outliers / len(y) > 0.05:
        advice.append("many target outliers: prefer fit_huber_regression or a tree model, and report MAE alongside RMSE")
    if not advice:
        advice.append("roughly symmetric: model the target as is")
    return {"n": len(y), "skew": round(skew, 4), "excess_kurtosis": round(kurt, 4), "zero_share": round(float((y == 0).mean()), 4),
            "iqr_outliers": outliers, "looks_like_counts": is_count, "recommendation": advice}


@tool
def check_multicollinearity(data: str, target: str, features: list[str] | None = None, vif_threshold: float = 10.0) -> dict:
    """Variance inflation factors and highly correlated pairs among numeric features — decides whether to regularise (ridge/lasso) or drop columns before a linear model.

    Args:
        vif_threshold: VIF above which a column is flagged; 5–10 is the usual range.
    """
    frame = load_frame(data)
    names = [c for c in (as_list(features) or frame.columns) if c != target]
    X = frame[names].select_dtypes("number")
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 0]
    corr = np.corrcoef(X.to_numpy(float), rowvar=False)
    inverse = np.linalg.pinv(corr)
    vif = dict(zip(X.columns, np.diag(inverse)))
    pairs = [{"a": X.columns[i], "b": X.columns[j], "corr": round(float(corr[i, j]), 4)}
             for i in range(len(X.columns)) for j in range(i + 1, len(X.columns)) if abs(corr[i, j]) >= 0.9]
    flagged = {c: round(float(v), 2) for c, v in vif.items() if v > vif_threshold}
    return {"vif": {c: round(float(v), 2) for c, v in sorted(vif.items(), key=lambda kv: -kv[1])},
            "flagged": flagged, "highly_correlated_pairs": pairs,
            "recommendation": "use ridge/elastic_net or drop one of each flagged pair" if flagged or pairs else "no collinearity problem for linear models"}


@tool
def check_feature_target_relationships(data: str, target: str, features: list[str] | None = None, k: int = 15) -> dict:
    """Per numeric feature: Pearson vs Spearman correlation with the target. A Spearman clearly above Pearson means a monotone but non-linear effect — a hint for trees, splines or polynomials.

    Args:
        k: How many features to return.
    """
    frame = load_frame(data)
    y = pd.to_numeric(frame[target], errors="coerce")
    rows = []
    for c in [c for c in (as_list(features) or frame.columns) if c != target]:
        if pd.api.types.is_numeric_dtype(frame[c]) and frame[c].nunique() > 1:
            pair = pd.concat([frame[c], y], axis=1).dropna()
            p, s = pair.corr().iloc[0, 1], pair.corr(method="spearman").iloc[0, 1]
            rows.append({"feature": c, "pearson": round(float(p), 4), "spearman": round(float(s), 4), "nonlinear_hint": bool(abs(s) - abs(p) > 0.1)})
    rows.sort(key=lambda r: -max(abs(r["pearson"]), abs(r["spearman"])))
    return {"features": rows[:k], "nonlinear": [r["feature"] for r in rows if r["nonlinear_hint"]]}


@tool
def baseline_regressor(data: str, target: str, strategy: Literal["mean", "median"] = "mean", test_size: float = 0.2, seed: int = 42) -> dict:
    """Score a predict-the-average baseline so every real model has a floor to beat (R² of 0 by construction).

    Args:
        strategy: Predict the training mean or median.
    """
    sk()
    from sklearn.dummy import DummyRegressor

    X, y = split_xy(data, target)
    y = pd.to_numeric(y)
    X_tr, X_te, y_tr, y_te = holdout(X, y, test_size, seed)
    dummy = DummyRegressor(strategy=strategy).fit(X_tr, y_tr)
    return {"strategy": strategy, "holdout": regression_metrics(y_te, dummy.predict(X_te))}


@tool
def find_target_outliers(data: str, target: str, method: Literal["iqr", "zscore", "mad"] = "iqr", threshold: float = 3.0, k: int = 20) -> dict:
    """List extreme target values by IQR fences, z-score or median absolute deviation — to decide whether to cap, drop or model them robustly.

    Args:
        method: iqr (1.5×IQR fences), zscore, or mad (robust z).
        threshold: Cut-off for zscore/mad.
        k: How many outlying rows to list.
    """
    frame = load_frame(data)
    y = pd.to_numeric(frame[target], errors="coerce")
    if method == "iqr":
        q1, q3 = y.quantile(.25), y.quantile(.75)
        mask = (y < q1 - 1.5 * (q3 - q1)) | (y > q3 + 1.5 * (q3 - q1))
    elif method == "zscore":
        mask = ((y - y.mean()) / y.std()).abs() > threshold
    else:
        med = y.median()
        mad = (y - med).abs().median() or 1e-9
        mask = (0.6745 * (y - med) / mad).abs() > threshold
    idx = y[mask].abs().sort_values(ascending=False).index[:k]
    return {"method": method, "outliers": int(mask.sum()), "share": round(float(mask.mean()), 4),
            "rows": [{"row": int(i), "value": float(y[i])} for i in idx]}
