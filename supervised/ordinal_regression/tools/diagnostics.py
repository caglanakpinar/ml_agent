"""Read-only checks on an ordinal problem and its data, before any model is fitted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_core.data import as_list, load_frame
from ml_core.ml import build_preprocessor, numeric_and_categorical, sk
from ml_core.tooling import tool
from supervised.ordinal_regression.tools._common import encode_target, load_ordinal, ordinal_metrics


@tool
def profile_ordinal_target(data: str, target: str, order: list[str] | None = None, features: list[str] | None = None) -> dict:
    """Profile an ordered target: its levels in order, their counts and shares, how the order was determined (a wrong alphabetical order is the most common silent error), imbalance across levels, and the usable feature columns.

    Args:
        order: The levels from lowest to highest, e.g. ["low", "medium", "high"].
    """
    frame = load_frame(data)
    y = frame[target].dropna()
    codes, levels, how = encode_target(y, order)
    counts = np.bincount(codes, minlength=len(levels))
    names = as_list(features) or [c for c in frame.columns if c != target]
    numeric, categorical = numeric_and_categorical(frame[names])
    return {"rows": len(frame), "levels": levels, "n_levels": len(levels), "order_source": how, "counts": dict(zip(levels, counts.astype(int).tolist())),
            "shares": dict(zip(levels, (counts / counts.sum()).round(4).tolist())), "imbalance_ratio": round(float(counts.max() / max(counts.min(), 1)), 2),
            "numeric_columns": numeric, "categorical_columns": categorical, "target_missing": int(frame[target].isna().sum()),
            "warnings": [w for w in ["levels are in ALPHABETICAL order — confirm it matches the real ranking, or pass `order`" if "ALPHABETICAL" in how else None,
                                     "very few levels (3): consider whether plain classification is enough" if len(levels) == 3 else None,
                                     "more than 10 levels: treating the target as a number (regression) loses little" if len(levels) > 10 else None,
                                     "a level has under 20 rows: its cut-point will be unstable" if counts.min() < 20 else None] if w]}


@tool
def check_proportional_odds(data: str, target: str, order: list[str] | None = None, features: list[str] | None = None, drop: list[str] | None = None) -> dict:
    """Test the parallel-lines assumption behind the ordered-logit model: fit one binary logistic model per cut-point (y > k) and compare their coefficients. If they differ a lot across cut-points, a single ordered logit is wrong — use Frank–Hall or a tree model.

    Args:
        order: The levels from lowest to highest.
    """
    sk()
    from sklearn.linear_model import LogisticRegression

    X, y, levels, _ = load_ordinal(data, target, features, drop, order)
    Z = np.asarray(build_preprocessor(X).fit_transform(X), float)
    coefs = []
    for k in range(len(levels) - 1):
        t = (y > k).astype(int)
        if t.min() != t.max() and t.sum() >= 10 and (1 - t).sum() >= 10:
            coefs.append(LogisticRegression(C=1.0, max_iter=1000).fit(Z, t).coef_[0])
    if len(coefs) < 2:
        return {"verdict": "not testable: too few rows on each side of the cut-points"}
    coefs = np.asarray(coefs)
    spread = np.std(coefs, axis=0) / (np.abs(coefs).mean(axis=0) + 1e-9)
    correlation = float(np.mean([np.corrcoef(coefs[i], coefs[j])[0, 1] for i in range(len(coefs)) for j in range(i + 1, len(coefs))]))
    relative = float(np.mean(spread))
    return {"cutpoints_tested": len(coefs), "mean_coefficient_correlation": round(correlation, 4), "mean_relative_spread": round(relative, 4),
            "verdict": "parallel lines hold: an ordered logit is appropriate" if correlation > 0.9 and relative < 0.5 else
            "parallel lines are doubtful: prefer fit_frank_hall_ordinal or a tree-based base model; an ordered logit will average over different effects"}


@tool
def check_feature_ordinal_association(data: str, target: str, order: list[str] | None = None, features: list[str] | None = None, drop: list[str] | None = None, k: int = 15) -> dict:
    """Rank features by Spearman correlation with the ordered target (numeric) and Kruskal–Wallis (categorical) — a model-free look at which columns move with the level and in which direction.

    Args:
        order: The levels from lowest to highest.
        k: How many features to return.
    """
    from scipy import stats

    X, y, levels, _ = load_ordinal(data, target, features, drop, order)
    rows = []
    for c in X.columns:
        col = X[c]
        if pd.api.types.is_numeric_dtype(col):
            ok = col.notna()
            if col[ok].nunique() > 1:
                rho, p = stats.spearmanr(col[ok], y[ok.to_numpy()])
                rows.append({"feature": c, "kind": "numeric", "spearman": round(float(rho), 4), "p_value": round(float(p), 6), "direction": "higher level as it rises" if rho > 0 else "lower level as it rises"})
        elif col.nunique() <= 30:
            groups = [y[(col == v).to_numpy()] for v in col.dropna().unique() if (col == v).sum() > 1]
            if len(groups) > 1:
                h, p = stats.kruskal(*groups)
                rows.append({"feature": c, "kind": "categorical", "kruskal_h": round(float(h), 3), "p_value": round(float(p), 6)})
    rows.sort(key=lambda r: r["p_value"])
    return {"features": rows[:k], "no_signal": [r["feature"] for r in rows if r["p_value"] > 0.2]}


@tool
def baseline_ordinal(data: str, target: str, order: list[str] | None = None, test_size: float = 0.2, seed: int = 42) -> dict:
    """Score two do-nothing baselines on held-out rows — always the most common level, and always the median level — with every ordinal metric, so a model has a floor to beat (the median baseline usually wins on MAE).

    Args:
        order: The levels from lowest to highest.
    """
    from ml_core.ml import holdout

    X, y, levels, _ = load_ordinal(data, target, None, None, order)
    _, _, y_tr, y_te = holdout(X, pd.Series(y), test_size, seed, stratify=True)
    mode, median = int(np.bincount(y_tr).argmax()), int(np.median(y_tr))
    return {"levels": levels, "mode_level": levels[mode], "median_level": levels[median],
            "always_mode": ordinal_metrics(y_te.to_numpy(), np.full(len(y_te), mode), len(levels)),
            "always_median": ordinal_metrics(y_te.to_numpy(), np.full(len(y_te), median), len(levels))}


@tool
def check_level_spacing(data: str, target: str, order: list[str] | None = None, features: list[str] | None = None, drop: list[str] | None = None) -> dict:
    """Fit a regression on the level index and compare the average prediction at each level: roughly equal steps mean the levels behave like equally spaced numbers (plain regression is fine); uneven steps mean the gaps differ and a proper ordinal model is worth it.

    Args:
        order: The levels from lowest to highest.
    """
    from supervised.ordinal_regression.tools._common import base_regressor

    X, y, levels, _ = load_ordinal(data, target, features, drop, order)
    Z = np.asarray(build_preprocessor(X).fit_transform(X), float)
    from sklearn.model_selection import cross_val_predict

    pred = cross_val_predict(base_regressor("ridge"), Z, y.astype(float), cv=5)
    means = [float(pred[y == k].mean()) for k in range(len(levels)) if (y == k).any()]
    steps = np.diff(means)
    cv_ = float(np.std(steps) / (np.mean(steps) + 1e-9)) if len(steps) > 1 else 0.0
    return {"mean_prediction_by_level": dict(zip(levels, np.round(means, 3).tolist())), "step_sizes": np.round(steps, 3).tolist(), "step_variation": round(cv_, 3),
            "verdict": "levels look equally spaced: regression with rounding is a fair option" if cv_ < 0.35 else "uneven gaps between levels: prefer an ordinal model (ordered logit / Frank–Hall) or learned cut-points"}
