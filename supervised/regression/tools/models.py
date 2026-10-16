"""Fit one regression algorithm and score it on held-out rows — one function per algorithm family."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ml_core.ml import make_pipeline, sk, split_xy
from ml_core.tooling import tool
from supervised.regression.tools._common import MODEL_NAMES, SCALED, make_estimator, run_regressor


@tool
def fit_linear_regression(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    test_size: float = 0.2, seed: int = 42, cv: int = 0, model_path: str | None = None,
) -> dict:
    """Fit ordinary least squares — the interpretable baseline: each coefficient is the change in the target per unit of a (standardised) feature."""
    return run_regressor(make_estimator("linear"), "linear", data, target, features, drop, test_size, seed, True, model_path, cv)


@tool
def fit_ridge(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    alpha: float = 1.0, test_size: float = 0.2, seed: int = 42, cv: int = 0, model_path: str | None = None,
) -> dict:
    """Fit ridge regression (L2) — shrinks coefficients; the fix for correlated features and for more columns than rows.

    Args:
        alpha: Penalty strength; larger shrinks more.
    """
    return run_regressor(make_estimator("ridge", alpha=alpha), f"ridge(alpha={alpha})", data, target, features, drop, test_size, seed, True, model_path, cv)


@tool
def fit_lasso(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    alpha: float = 0.01, test_size: float = 0.2, seed: int = 42, cv: int = 0, model_path: str | None = None,
) -> dict:
    """Fit lasso (L1) — drives weak coefficients to exactly zero, so it selects features while it fits.

    Args:
        alpha: Penalty strength; larger zeroes more features.
    """
    out = run_regressor(make_estimator("lasso", alpha=alpha), f"lasso(alpha={alpha})", data, target, features, drop, test_size, seed, True, model_path, cv)
    return out


@tool
def fit_elastic_net(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    alpha: float = 0.01, l1_ratio: float = 0.5, test_size: float = 0.2, seed: int = 42, cv: int = 0,
    model_path: str | None = None,
) -> dict:
    """Fit elastic net — a blend of L1 and L2: selects features like lasso but keeps correlated groups together like ridge.

    Args:
        alpha: Penalty strength.
        l1_ratio: 1 is pure lasso, 0 is pure ridge.
    """
    return run_regressor(make_estimator("elastic_net", alpha=alpha, l1_ratio=l1_ratio), f"elastic_net({alpha}, {l1_ratio})", data, target, features, drop, test_size, seed, True, model_path, cv)


@tool
def fit_bayesian_ridge(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    test_size: float = 0.2, seed: int = 42, cv: int = 0, model_path: str | None = None,
) -> dict:
    """Fit Bayesian ridge — a linear model that learns its own regularisation strength from the data, with no alpha to tune."""
    return run_regressor(make_estimator("bayesian_ridge"), "bayesian_ridge", data, target, features, drop, test_size, seed, True, model_path, cv)


@tool
def fit_huber_regression(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    epsilon: float = 1.35, test_size: float = 0.2, seed: int = 42, cv: int = 0, model_path: str | None = None,
) -> dict:
    """Fit Huber regression — a linear model that stops outliers dragging the fit: squared loss for small errors, absolute loss for large ones.

    Args:
        epsilon: Error size (in robust SDs) beyond which a row counts as an outlier; smaller is more robust.
    """
    return run_regressor(make_estimator("huber", epsilon=epsilon), f"huber(eps={epsilon})", data, target, features, drop, test_size, seed, True, model_path, cv)


@tool
def fit_quantile_regression(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    quantile: float = 0.5, alpha: float = 0.0, test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit a linear quantile regression — predicts a chosen quantile (0.5 = median, 0.9 = 'high but plausible') instead of the mean; use pairs of quantiles for intervals.

    Args:
        quantile: Which quantile to predict, between 0 and 1.
        alpha: L1 penalty.
    """
    sk()
    from sklearn.linear_model import QuantileRegressor

    out = run_regressor(QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs"), f"quantile({quantile})", data, target, features, drop, test_size, seed, True, model_path, 0)
    out["pinball_note"] = "judge a quantile model by coverage (share of rows below the prediction ≈ the quantile), not by RMSE"
    return out


@tool
def fit_polynomial_regression(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    degree: int = 2, alpha: float = 1.0, interaction_only: bool = False, test_size: float = 0.2, seed: int = 42,
    cv: int = 0, model_path: str | None = None,
) -> dict:
    """Fit ridge regression on polynomial and interaction terms of the numeric features — captures curvature while staying a linear, explainable model.

    Args:
        degree: Highest power of the expansion; 2–3 is plenty.
        alpha: Ridge penalty that keeps the expanded terms in check.
        interaction_only: Keep only products of different features, no squares.
    """
    sk()
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    est = Pipeline([("poly", PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)),
                    ("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    return run_regressor(est, f"polynomial(degree={degree})", data, target, features, drop, test_size, seed, True, model_path, cv)


@tool
def fit_glm(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    family: Literal["poisson", "gamma", "tweedie"] = "poisson", alpha: float = 1.0, power: float = 1.5,
    test_size: float = 0.2, seed: int = 42, cv: int = 0, model_path: str | None = None,
) -> dict:
    """Fit a generalised linear model with a log link — the right likelihood for counts (poisson), positive skewed amounts (gamma), or amounts with many zeros (tweedie).

    Args:
        family: poisson for counts, gamma for strictly positive amounts, tweedie for zero-inflated amounts.
        alpha: L2 penalty.
        power: Tweedie power between 1 and 2 (family=tweedie only).
    """
    sk()
    from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor

    est = {"poisson": lambda: PoissonRegressor(alpha=alpha, max_iter=1000),
           "gamma": lambda: GammaRegressor(alpha=alpha, max_iter=1000),
           "tweedie": lambda: TweedieRegressor(power=power, alpha=alpha, link="log", max_iter=1000)}[family]()
    X, y = split_xy(data, target, features, drop)
    if family in ("poisson", "tweedie") and (y < 0).any():
        raise ValueError(f"{family} needs a non-negative target")
    if family == "gamma" and (y <= 0).any():
        raise ValueError("gamma needs a strictly positive target")
    return run_regressor(est, f"glm({family})", data, target, features, drop, test_size, seed, True, model_path, cv)


@tool
def fit_transformed_target_regression(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    transform: Literal["log1p", "sqrt", "yeo-johnson"] = "log1p", base: Literal["ridge", "random_forest", "hist_gradient_boosting"] = "ridge",
    test_size: float = 0.2, seed: int = 42, cv: int = 0, model_path: str | None = None,
) -> dict:
    """Fit a model on a transformed target (log, sqrt, Yeo-Johnson) and back-transform its predictions — the standard fix for a right-skewed target; errors are reported on the original scale.

    Args:
        transform: Target transformation; log1p and sqrt need a non-negative target.
        base: Algorithm fitted on the transformed target.
    """
    sk()
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import FunctionTransformer, PowerTransformer

    funcs = {"log1p": FunctionTransformer(np.log1p, np.expm1, check_inverse=False),
             "sqrt": FunctionTransformer(np.sqrt, np.square, check_inverse=False),
             "yeo-johnson": PowerTransformer(method="yeo-johnson")}
    X, y = split_xy(data, target, features, drop)
    if transform in ("log1p", "sqrt") and (y < 0).any():
        raise ValueError(f"{transform} needs a non-negative target; use yeo-johnson")
    est = TransformedTargetRegressor(regressor=make_estimator(base, random_state=seed) if base != "ridge" else make_estimator("ridge"),
                                     transformer=funcs[transform])
    return run_regressor(est, f"{base}+{transform}(target)", data, target, features, drop, test_size, seed, base == "ridge", model_path, cv)


@tool
def fit_random_forest_regressor(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    n_estimators: int = 300, max_depth: int | None = None, min_samples_leaf: int = 2, max_features: float = 1.0,
    test_size: float = 0.2, seed: int = 42, cv: int = 0, model_path: str | None = None,
) -> dict:
    """Fit a random forest regressor — a strong default for non-linear tabular data; robust to outliers and needs little tuning, but cannot extrapolate beyond the training range.

    Args:
        n_estimators: Number of trees.
        max_depth: Depth limit; unset grows fully.
        min_samples_leaf: Larger values smooth predictions.
        max_features: Share of features tried per split.
    """
    est = make_estimator("random_forest", n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=seed)
    return run_regressor(est, "random_forest", data, target, features, drop, test_size, seed, False, model_path, cv)


@tool
def fit_extra_trees_regressor(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    n_estimators: int = 300, max_depth: int | None = None, test_size: float = 0.2, seed: int = 42, cv: int = 0,
    model_path: str | None = None,
) -> dict:
    """Fit extremely randomised trees — faster than a random forest and often smoother, with slightly more bias.

    Args:
        n_estimators: Number of trees.
        max_depth: Depth limit.
    """
    est = make_estimator("extra_trees", n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    return run_regressor(est, "extra_trees", data, target, features, drop, test_size, seed, False, model_path, cv)


@tool
def fit_gradient_boosting_regressor(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    n_estimators: int = 300, learning_rate: float = 0.05, max_depth: int = 3, subsample: float = 0.85,
    loss: Literal["squared_error", "absolute_error", "huber"] = "squared_error",
    test_size: float = 0.2, seed: int = 42, cv: int = 0, model_path: str | None = None,
) -> dict:
    """Fit gradient-boosted trees — usually the most accurate on tabular data; choose a robust loss when the target has outliers.

    Args:
        n_estimators: Boosting rounds.
        learning_rate: Shrinkage per round.
        max_depth: Depth of each tree.
        subsample: Fraction of rows per round.
        loss: squared_error, absolute_error (median-targeting) or huber.
    """
    est = make_estimator("gradient_boosting", n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, loss=loss, random_state=seed)
    return run_regressor(est, f"gradient_boosting({loss})", data, target, features, drop, test_size, seed, False, model_path, cv)


@tool
def fit_hist_gradient_boosting_regressor(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    max_iter: int = 400, learning_rate: float = 0.05, max_leaf_nodes: int = 31, l2_regularization: float = 0.0,
    loss: Literal["squared_error", "absolute_error", "poisson", "gamma"] = "squared_error",
    test_size: float = 0.2, seed: int = 42, cv: int = 0, model_path: str | None = None,
) -> dict:
    """Fit histogram gradient boosting — boosted trees fast enough for 100k+ rows, with count (poisson) and skewed-positive (gamma) losses.

    Args:
        max_iter: Boosting rounds.
        learning_rate: Shrinkage per round.
        max_leaf_nodes: Leaves per tree.
        l2_regularization: L2 penalty on leaf values.
        loss: squared_error, absolute_error, poisson or gamma.
    """
    est = make_estimator("hist_gradient_boosting", max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, l2_regularization=l2_regularization, loss=loss, random_state=seed)
    return run_regressor(est, f"hist_gradient_boosting({loss})", data, target, features, drop, test_size, seed, False, model_path, cv)


@tool(requires=("lightgbm",))
def fit_lightgbm_regressor(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    n_estimators: int = 500, learning_rate: float = 0.05, num_leaves: int = 31,
    test_size: float = 0.2, seed: int = 42, cv: int = 0, model_path: str | None = None,
) -> dict:
    """Fit LightGBM — the usual choice for large tabular data with many categorical levels.

    Args:
        n_estimators: Boosting rounds.
        learning_rate: Shrinkage per round.
        num_leaves: Leaves per tree.
    """
    from lightgbm import LGBMRegressor

    est = LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, random_state=seed, verbose=-1)
    return run_regressor(est, "lightgbm", data, target, features, drop, test_size, seed, False, model_path, cv)


@tool
def fit_decision_tree_regressor(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    max_depth: int = 5, min_samples_leaf: int = 10, test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit one shallow regression tree — a piecewise-constant model readable as if/then rules.

    Args:
        max_depth: Tree depth; small keeps the rules readable.
        min_samples_leaf: Smallest leaf allowed.
    """
    est = make_estimator("decision_tree", max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed)
    return run_regressor(est, "decision_tree", data, target, features, drop, test_size, seed, False, model_path, 0)


@tool
def fit_svr(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    kernel: Literal["rbf", "linear", "poly"] = "rbf", C: float = 10.0, epsilon: float = 0.1,
    test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit support-vector regression — smooth non-linear fits on small/medium data; slow past ~50k rows. The target is standardised internally.

    Args:
        kernel: rbf, linear or poly.
        C: Fit tightness; larger fits training rows harder.
        epsilon: Width of the no-penalty tube (in standardised target units).
    """
    sk()
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import StandardScaler

    est = TransformedTargetRegressor(regressor=make_estimator("svr", kernel=kernel, C=C, epsilon=epsilon), transformer=StandardScaler())
    return run_regressor(est, f"svr({kernel})", data, target, features, drop, test_size, seed, True, model_path, 0)


@tool
def fit_knn_regressor(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    n_neighbors: int = 10, weights: Literal["uniform", "distance"] = "distance",
    test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit k-nearest-neighbours regression — predicts the average target of the most similar rows; a quick non-parametric check on whether similarity predicts the target.

    Args:
        n_neighbors: Neighbours averaged.
        weights: distance weights closer neighbours more.
    """
    return run_regressor(make_estimator("knn", n_neighbors=n_neighbors, weights=weights), f"knn(k={n_neighbors})", data, target, features, drop, test_size, seed, True, model_path, 0)


@tool
def fit_mlp_regressor(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    hidden_layers: str = "64,32", alpha: float = 0.0001, max_iter: int = 800,
    test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit a small feed-forward neural network — worth trying with many rows and rich interactions. The target is standardised internally.

    Args:
        hidden_layers: Comma-separated layer widths, e.g. "64,32".
        alpha: L2 penalty.
        max_iter: Training epochs.
    """
    sk()
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import StandardScaler

    layers = tuple(int(p) for p in hidden_layers.replace(" ", "").split(",") if p) or (64,)
    est = TransformedTargetRegressor(regressor=make_estimator("mlp", hidden_layer_sizes=layers, alpha=alpha, max_iter=max_iter, random_state=seed), transformer=StandardScaler())
    return run_regressor(est, f"mlp({hidden_layers})", data, target, features, drop, test_size, seed, True, model_path, 0)


@tool
def fit_sgd_regressor(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    penalty: Literal["l2", "l1", "elasticnet"] = "l2", alpha: float = 0.0001, test_size: float = 0.2, seed: int = 42,
    model_path: str | None = None,
) -> dict:
    """Fit a linear model by stochastic gradient descent — for data too large for the closed-form solvers. The target is standardised internally.

    Args:
        penalty: l2, l1 or elasticnet.
        alpha: Regularisation strength.
    """
    sk()
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import StandardScaler

    est = TransformedTargetRegressor(regressor=make_estimator("sgd", penalty=penalty, alpha=alpha, random_state=seed), transformer=StandardScaler())
    return run_regressor(est, f"sgd({penalty})", data, target, features, drop, test_size, seed, True, model_path, 0)


@tool
def compare_regressors(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    models: list[str] | None = None, cv: int = 5,
    scoring: Literal["rmse", "mae", "r2"] = "rmse", seed: int = 42,
) -> dict:
    """Cross-validate several regressors on identical folds and rank them — the fastest way to see which family suits the data.

    Args:
        models: Which to compare (linear, ridge, lasso, elastic_net, random_forest, extra_trees, gradient_boosting, hist_gradient_boosting, decision_tree, svr, knn, mlp, huber, bayesian_ridge, sgd). Defaults to a representative six.
        scoring: rmse, mae or r2.
    """
    sk()
    from sklearn.model_selection import KFold, cross_val_score

    X, y = split_xy(data, target, features, drop)
    y = y.astype(float)
    chosen = models or ["ridge", "lasso", "random_forest", "gradient_boosting", "hist_gradient_boosting", "knn"]
    unknown = [m for m in chosen if m not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"unknown model(s) {unknown}; choose from {list(MODEL_NAMES)}")
    metric = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}[scoring]
    folds = KFold(max(2, cv), shuffle=True, random_state=seed)
    board = []
    for name in chosen:
        try:
            est = make_estimator(name) if name in ("linear", "ridge", "lasso", "elastic_net", "svr", "knn", "huber", "bayesian_ridge", "decision_tree") else make_estimator(name, random_state=seed)
            scores = cross_val_score(make_pipeline(est, X, scale=name in SCALED), X, y, cv=folds, scoring=metric, n_jobs=1)
            sign = 1 if scoring == "r2" else -1
            board.append({"model": name, "mean": round(float(sign * scores.mean()), 5), "std": round(float(scores.std()), 5)})
        except Exception as error:
            board.append({"model": name, "error": str(error)[:120]})
    board.sort(key=lambda r: r.get("mean", np.inf if scoring != "r2" else -np.inf), reverse=(scoring == "r2"))
    return {"scoring": scoring, "folds": cv, "leaderboard": board, "best": board[0]["model"] if board and "mean" in board[0] else None}
