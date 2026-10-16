"""Fit one ordinal model and score it on held-out rows against a median-level baseline: ordered logit, Frank–Hall, regression with rounded or learned cut-points, and plain multiclass for comparison."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ml_core.ml import holdout, sk
from ml_core.tooling import tool
from supervised.ordinal_regression.tools._common import fit_ordinal, load_ordinal, ordinal_metrics, run_ordinal

Decode = Literal["median", "argmax", "expected"]
ClassBase = Literal["logistic_regression", "random_forest", "hist_gradient_boosting"]
RegBase = Literal["ridge", "random_forest", "hist_gradient_boosting"]


@tool
def fit_ordered_logit(
    data: str, target: str, order: list[str] | None = None, features: list[str] | None = None, drop: list[str] | None = None,
    l2: float = 0.0, link: Literal["logit", "probit"] = "logit", decode_with: Decode = "median", test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit the proportional-odds model (ordered logit / probit): one coefficient per feature and K−1 ordered thresholds, so the effect of a feature is the same at every cut-point. The classic interpretable ordinal model — check `check_proportional_odds` first.

    Args:
        order: The levels from lowest to highest; numeric targets are ordered numerically, strings alphabetically unless given.
        l2: Ridge penalty on the coefficients.
        link: logit (odds ratios) or probit.
        decode_with: How probabilities become a level — median (minimises absolute error), argmax, or expected.
    """
    return run_ordinal("ordered_logit", f"ordered_{link}", data, target, features, drop, order, test_size, seed, model_path, l2=l2, link=link, decode_with=decode_with)


@tool(requires=("statsmodels",))
def fit_ordered_model_statsmodels(
    data: str, target: str, order: list[str] | None = None, features: list[str] | None = None, drop: list[str] | None = None,
    distribution: Literal["logit", "probit"] = "logit", test_size: float = 0.2, seed: int = 42,
) -> dict:
    """Fit the proportional-odds model with statsmodels' OrderedModel — the reference implementation, giving z-statistics, p-values and confidence intervals for each coefficient (numeric and one-hot features).

    Args:
        order: The levels from lowest to highest.
        distribution: logit or probit.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    X, y, levels, how = load_ordinal(data, target, features, drop, order)
    X = pd.get_dummies(X, drop_first=True).astype(float)
    X = X.fillna(X.median())
    X = (X - X.mean()) / X.std().replace(0, 1)
    X_tr, X_te, y_tr, y_te = holdout(X, pd.Series(y), test_size, seed, stratify=True)
    fit = OrderedModel(y_tr.to_numpy(), X_tr, distr=distribution).fit(method="bfgs", disp=False)
    proba = np.asarray(fit.model.predict(fit.params, exog=X_te))
    pred = (np.cumsum(proba, axis=1) >= 0.5).argmax(axis=1)
    table = pd.DataFrame({"coefficient": fit.params[: X.shape[1]], "std_error": fit.bse[: X.shape[1]], "p_value": fit.pvalues[: X.shape[1]]}, index=X.columns).round(5)
    return {"model": f"statsmodels_ordered_{distribution}", "levels": levels, "order_source": how, "log_likelihood": round(float(fit.llf), 3), "aic": round(float(fit.aic), 3),
            "coefficients": table.reset_index(names="feature").sort_values("p_value").to_dict("records"), "holdout": ordinal_metrics(y_te.to_numpy(), pred, len(levels))}


@tool
def fit_frank_hall_ordinal(
    data: str, target: str, order: list[str] | None = None, features: list[str] | None = None, drop: list[str] | None = None,
    base: ClassBase = "logistic_regression", decode_with: Decode = "median", test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit the Frank–Hall decomposition: one binary classifier per cut-point (is the level above k?), combined into level probabilities. Works with any classifier, so effects may differ by cut-point and a tree model can capture non-linearity — the flexible ordinal choice.

    Args:
        order: The levels from lowest to highest.
        base: Binary classifier used at every cut-point.
        decode_with: median, argmax or expected.
    """
    return run_ordinal("frank_hall", f"frank_hall({base})", data, target, features, drop, order, test_size, seed, model_path, base=base, decode_with=decode_with)


@tool
def fit_regression_with_rounding(
    data: str, target: str, order: list[str] | None = None, features: list[str] | None = None, drop: list[str] | None = None,
    base: RegBase = "ridge", test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Treat the levels as numbers 0…K−1, fit a regression, and round to the nearest level. The simplest ordinal approach — fine when the gaps between levels are about equal, and the baseline to beat.

    Args:
        order: The levels from lowest to highest.
        base: Regressor fitted on the level index.
    """
    return run_ordinal("regression_round", f"regression_rounding({base})", data, target, features, drop, order, test_size, seed, model_path, base=base)


@tool
def fit_regression_with_learned_thresholds(
    data: str, target: str, order: list[str] | None = None, features: list[str] | None = None, drop: list[str] | None = None,
    base: RegBase = "hist_gradient_boosting", test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit a regression on the level index, then learn the cut-points that turn its score into a level by maximising quadratic weighted kappa on out-of-fold scores — the standard winning recipe for rating-prediction competitions.

    Args:
        order: The levels from lowest to highest.
        base: Regressor fitted on the level index.
    """
    return run_ordinal("regression_thresholds", f"regression_learned_thresholds({base})", data, target, features, drop, order, test_size, seed, model_path, base=base)


@tool
def fit_cost_sensitive_classifier(
    data: str, target: str, order: list[str] | None = None, features: list[str] | None = None, drop: list[str] | None = None,
    base: ClassBase = "random_forest", decode_with: Decode = "median", test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit an ordinary multiclass classifier and decode its probabilities with an order-aware rule (the median of the predicted distribution minimises expected absolute error) — turns any classifier into a decent ordinal one, so a one-step miss is preferred to a three-step miss.

    Args:
        order: The levels from lowest to highest.
        base: Multiclass classifier.
        decode_with: median (order-aware), argmax (ignores order), or expected.
    """
    return run_ordinal("multiclass", f"multiclass({base})+{decode_with}", data, target, features, drop, order, test_size, seed, model_path, base=base, decode_with=decode_with)


@tool
def fit_multiclass_ignoring_order(
    data: str, target: str, order: list[str] | None = None, features: list[str] | None = None, drop: list[str] | None = None,
    base: ClassBase = "random_forest", test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit a plain multiclass classifier that ignores the ordering (argmax decoding) — the comparison that shows what the ordinal structure is worth.

    Args:
        order: The levels from lowest to highest.
        base: Multiclass classifier.
    """
    return run_ordinal("multiclass", f"multiclass({base})+argmax", data, target, features, drop, order, test_size, seed, model_path, base=base, decode_with="argmax")


@tool
def compare_ordinal_models(
    data: str, target: str, order: list[str] | None = None, features: list[str] | None = None, drop: list[str] | None = None,
    cv: int = 5, seed: int = 42, metric: Literal["quadratic_weighted_kappa", "mae", "macro_mae", "within_one", "accuracy"] = "quadratic_weighted_kappa",
) -> dict:
    """Cross-validate the main ordinal approaches on identical folds — ordered logit, Frank–Hall (logistic and boosted), rounded and threshold-tuned regression, order-aware and order-blind classifiers — and rank them on an ordinal metric.

    Args:
        order: The levels from lowest to highest.
        metric: Ranking metric; quadratic_weighted_kappa and macro_mae respect the ordering and the imbalance.
    """
    sk()
    from sklearn.model_selection import StratifiedKFold

    X, y, levels, how = load_ordinal(data, target, features, drop, order)
    specs = [("ordered_logit", "ordered_logit", {}), ("frank_hall_logistic", "frank_hall", {"base": "logistic_regression"}), ("frank_hall_boosting", "frank_hall", {"base": "hist_gradient_boosting"}),
             ("regression_rounding", "regression_round", {"base": "ridge"}), ("regression_learned_thresholds", "regression_thresholds", {"base": "hist_gradient_boosting"}),
             ("multiclass_median_decode", "multiclass", {"base": "random_forest", "decode_with": "median"}), ("multiclass_argmax", "multiclass", {"base": "random_forest", "decode_with": "argmax"})]
    scores: dict[str, list[dict]] = {name: [] for name, _, _ in specs}
    folds = StratifiedKFold(max(2, min(cv, int(np.bincount(y).min()))), shuffle=True, random_state=seed)
    for tr, te in folds.split(X, y):
        for name, kind, params in specs:
            try:
                model = fit_ordinal(kind, X.iloc[tr], y[tr], levels, seed=seed, **params)
                scores[name].append(ordinal_metrics(y[te], model.predict_codes(X.iloc[te]), len(levels)))
            except Exception:
                pass
    board = [{"model": n, **{m: round(float(np.mean([s[m] for s in v])), 5) for m in ("quadratic_weighted_kappa", "mae", "macro_mae", "within_one", "accuracy")}} for n, v in scores.items() if v]
    board.sort(key=lambda r: r[metric], reverse=metric not in ("mae", "macro_mae"))
    return {"levels": levels, "order_source": how, "folds": folds.get_n_splits(), "ranked_by": metric, "leaderboard": board, "best": board[0]["model"] if board else None}
