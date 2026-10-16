"""Ordinal-regression machinery the tool modules share: reading an ordered target, the ordinal metrics,
and the models themselves — a proportional-odds (ordered logit) fit, the Frank–Hall decomposition into
cumulative binary problems, regression with learned cut-points, and a plain classifier for comparison.
All of them end up as an `OrdinalModel` that predicts class probabilities and decodes them to a level."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_core.data import as_list
from ml_core.ml import build_preprocessor, holdout, save_model, sk, split_xy


# ---- target -------------------------------------------------------------------------------------
def encode_target(y: pd.Series, order: Any = None) -> tuple[np.ndarray, list[str], str]:
    """Map the target to 0..K-1 along its natural order. Returns codes, level labels, and how the order was found."""
    given = as_list(order)
    if given:
        levels = given
        mapping = {str(l): i for i, l in enumerate(levels)}
        unknown = sorted(set(y.astype(str)) - set(mapping))
        if unknown:
            raise ValueError(f"target values {unknown[:5]} are not in `order` {levels}")
        return y.astype(str).map(mapping).to_numpy(int), [str(l) for l in levels], "given by `order`"
    if pd.api.types.is_numeric_dtype(y):
        vals = sorted(y.unique())
        mapping = {v: i for i, v in enumerate(vals)}
        return y.map(mapping).to_numpy(int), [str(v) for v in vals], "numeric order"
    levels = sorted(y.astype(str).unique())
    mapping = {l: i for i, l in enumerate(levels)}
    return y.astype(str).map(mapping).to_numpy(int), levels, "ALPHABETICAL order — pass `order` if that is not the real ranking"


def load_ordinal(data: Any, target: str, features: Any, drop: Any, order: Any) -> tuple[pd.DataFrame, np.ndarray, list[str], str]:
    X, y = split_xy(data, target, features, drop)
    codes, levels, how = encode_target(y, order)
    if len(levels) < 3:
        raise ValueError(f"the target has {len(levels)} level(s); ordinal regression needs at least 3 (use classification for two)")
    return X, codes, levels, how


# ---- metrics ------------------------------------------------------------------------------------
def ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_levels: int) -> dict[str, float]:
    sk()
    from scipy import stats
    from sklearn.metrics import cohen_kappa_score

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    err = np.abs(y_true - y_pred)
    per_level_mae = [err[y_true == k].mean() for k in range(n_levels) if (y_true == k).any()]
    out = {
        "accuracy": float((err == 0).mean()),
        "within_one": float((err <= 1).mean()),
        "mae": float(err.mean()),
        "macro_mae": float(np.mean(per_level_mae)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "linear_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="linear")),
    }
    if len(set(y_pred)) > 1 and len(set(y_true)) > 1:
        out["spearman"] = float(stats.spearmanr(y_true, y_pred)[0])
        out["kendall_tau"] = float(stats.kendalltau(y_true, y_pred)[0])
    return {k: round(v, 5) for k, v in out.items()}


def decode(proba: np.ndarray, how: str = "median") -> np.ndarray:
    """Turn class probabilities into levels: argmax (the mode), median (minimises expected absolute error), or expected (rounded mean)."""
    if how == "argmax":
        return proba.argmax(axis=1)
    if how == "expected":
        return np.clip(np.rint(proba @ np.arange(proba.shape[1])), 0, proba.shape[1] - 1).astype(int)
    return (np.cumsum(proba, axis=1) >= 0.5).argmax(axis=1)


def optimise_cutpoints(scores: np.ndarray, y: np.ndarray, n_levels: int, passes: int = 4) -> np.ndarray:
    """Cut-points on a continuous score that maximise quadratic weighted kappa, by coordinate search."""
    from sklearn.metrics import cohen_kappa_score

    cuts = np.arange(n_levels - 1) + 0.5
    grid = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 120)))
    for _ in range(passes):
        for i in range(len(cuts)):
            lo = cuts[i - 1] if i else -np.inf
            hi = cuts[i + 1] if i < len(cuts) - 1 else np.inf
            best, best_val = cuts[i], -2.0
            for c in grid[(grid > lo) & (grid < hi)]:
                trial = cuts.copy()
                trial[i] = c
                k = cohen_kappa_score(y, np.searchsorted(trial, scores), weights="quadratic")
                if k > best_val:
                    best, best_val = c, k
            cuts[i] = best
    return cuts


# ---- cores --------------------------------------------------------------------------------------
class OrderedLogit:
    """Proportional-odds model: P(y <= k | x) = sigmoid(theta_k - x·beta), fitted by penalised maximum likelihood."""

    def __init__(self, n_levels: int, l2: float = 0.0, link: str = "logit"):
        # one-hot columns are collinear with each other; a whisper of ridge keeps the fit and its standard errors well defined
        self.K, self.l2, self.link = n_levels, max(l2, 1e-3), link

    def _cdf(self, z: np.ndarray) -> np.ndarray:
        if self.link == "probit":
            from scipy.stats import norm

            return norm.cdf(z)
        return 1 / (1 + np.exp(-np.clip(z, -35, 35)))

    def _thetas(self, a: np.ndarray) -> np.ndarray:
        return np.cumsum(np.concatenate([a[:1], np.exp(a[1:])]))

    def _proba(self, Z: np.ndarray, beta: np.ndarray, theta: np.ndarray) -> np.ndarray:
        eta = Z @ beta
        cum = np.hstack([np.zeros((len(Z), 1)), self._cdf(theta[None, :] - eta[:, None]), np.ones((len(Z), 1))])
        return np.clip(np.diff(cum, axis=1), 1e-12, 1)

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "OrderedLogit":
        from scipy.optimize import minimize

        p = Z.shape[1]
        counts = np.bincount(y, minlength=self.K) + 0.5
        start_theta = np.log(np.cumsum(counts)[:-1] / (counts.sum() - np.cumsum(counts)[:-1]))
        a0 = np.concatenate([start_theta[:1], np.log(np.maximum(np.diff(start_theta), 1e-3))])

        def nll(x: np.ndarray) -> float:
            beta, theta = x[:p], self._thetas(x[p:])
            prob = self._proba(Z, beta, theta)
            return -float(np.log(prob[np.arange(len(y)), y]).sum()) + 0.5 * self.l2 * float(beta @ beta)

        res = minimize(nll, np.concatenate([np.zeros(p), a0]), method="L-BFGS-B")
        self.beta, self.theta, self.converged, self.loglik = res.x[:p], self._thetas(res.x[p:]), bool(res.success), -float(res.fun)
        # Wald standard errors from a numerical Hessian
        h, k = 1e-4, len(res.x)
        H = np.zeros((k, k))
        f0 = nll(res.x)
        for i in range(k):
            for j in range(i, k):
                ei, ej = np.eye(k)[i] * h, np.eye(k)[j] * h
                H[i, j] = H[j, i] = (nll(res.x + ei + ej) - nll(res.x + ei) - nll(res.x + ej) + f0) / h**2
        self.se = np.sqrt(np.clip(np.diag(np.linalg.pinv(H + 1e-9 * np.eye(k))), 0, None))[:p]
        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        return self._proba(Z, self.beta, self.theta)


class FrankHall:
    """K-1 binary classifiers for P(y > k); class probabilities are the differences of consecutive cumulative probabilities."""

    def __init__(self, n_levels: int, base: str, seed: int = 0):
        self.K, self.base, self.seed = n_levels, base, seed
        self.models: list[Any] = []

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "FrankHall":
        self.models = []
        for k in range(self.K - 1):
            target = (y > k).astype(int)
            if target.min() == target.max():
                self.models.append(("constant", float(target[0])))
            else:
                self.models.append(("model", base_classifier(self.base, self.seed).fit(Z, target)))
        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        gt = np.column_stack([np.full(len(Z), m[1]) if m[0] == "constant" else m[1].predict_proba(Z)[:, 1] for m in self.models])
        gt = np.minimum.accumulate(gt, axis=1)  # P(y > k) must not increase with k
        cum = np.hstack([np.ones((len(Z), 1)), gt, np.zeros((len(Z), 1))])
        return np.clip(-np.diff(cum, axis=1), 1e-12, 1)


class RegressionThresholds:
    """A regressor on the level index, with cut-points (rounded or optimised) turning its output into a level."""

    def __init__(self, n_levels: int, regressor: Any, cutpoints: np.ndarray | None = None):
        self.K, self.regressor, self.cuts = n_levels, regressor, cutpoints if cutpoints is not None else np.arange(n_levels - 1) + 0.5

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "RegressionThresholds":
        self.regressor.fit(Z, y.astype(float))
        return self

    def score(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(self.regressor.predict(Z), float)

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        level = np.searchsorted(self.cuts, self.score(Z))
        proba = np.full((len(Z), self.K), 0.0)
        proba[np.arange(len(Z)), level] = 1.0
        return proba


class MulticlassBaseline:
    def __init__(self, n_levels: int, estimator: Any):
        self.K, self.est = n_levels, estimator

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "MulticlassBaseline":
        self.est.fit(Z, y)
        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        raw = self.est.predict_proba(Z)
        out = np.zeros((len(Z), self.K))
        out[:, list(self.est.classes_)] = raw
        return out


class OrdinalModel:
    """Preprocessor + a core model + the level names; `predict` decodes probabilities to the chosen level."""

    def __init__(self, kind: str, prep: Any, core: Any, levels: list[str], columns: list[str], decode_with: str = "median"):
        self.kind, self.prep, self.core, self.levels, self.columns, self.decode_with = kind, prep, core, levels, columns, decode_with

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.core.predict_proba(np.asarray(self.prep.transform(X[self.columns]), float))

    def predict_codes(self, X: pd.DataFrame, how: str | None = None) -> np.ndarray:
        if isinstance(self.core, RegressionThresholds):
            return np.searchsorted(self.core.cuts, self.core.score(np.asarray(self.prep.transform(X[self.columns]), float)))
        return decode(self.proba(X), how or self.decode_with)


def base_classifier(name: str, seed: int = 0) -> Any:
    sk()
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    return {"logistic_regression": lambda: LogisticRegression(max_iter=1000), "random_forest": lambda: RandomForestClassifier(200, min_samples_leaf=3, n_jobs=-1, random_state=seed),
            "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=seed)}[name]()


def base_regressor(name: str, seed: int = 0) -> Any:
    sk()
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge

    return {"ridge": lambda: Ridge(alpha=1.0), "random_forest": lambda: RandomForestRegressor(200, min_samples_leaf=3, n_jobs=-1, random_state=seed),
            "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=seed)}[name]()


def build_core(kind: str, n_levels: int, Z: np.ndarray, y: np.ndarray, base: str = "logistic_regression", l2: float = 0.0, link: str = "logit", seed: int = 0, optimise: bool = False) -> Any:
    if kind == "ordered_logit":
        return OrderedLogit(n_levels, l2, link).fit(Z, y)
    if kind == "frank_hall":
        return FrankHall(n_levels, base, seed).fit(Z, y)
    if kind in ("regression_round", "regression_thresholds"):
        model = RegressionThresholds(n_levels, base_regressor(base, seed)).fit(Z, y)
        if kind == "regression_thresholds":
            from sklearn.model_selection import KFold

            oof = np.zeros(len(y))
            for tr, te in KFold(4, shuffle=True, random_state=seed).split(Z):
                oof[te] = RegressionThresholds(n_levels, base_regressor(base, seed)).fit(Z[tr], y[tr]).score(Z[te])
            model.cuts = optimise_cutpoints(oof, y, n_levels)
        return model
    if kind == "multiclass":
        return MulticlassBaseline(n_levels, base_classifier(base, seed)).fit(Z, y)
    raise ValueError(f"unknown ordinal model {kind!r}")


def fit_ordinal(kind: str, X: pd.DataFrame, y: np.ndarray, levels: list[str], base: str = "logistic_regression", l2: float = 0.0, link: str = "logit", seed: int = 0, decode_with: str = "median") -> OrdinalModel:
    prep = build_preprocessor(X, scale=True)
    Z = np.asarray(prep.fit_transform(X), float)
    return OrdinalModel(kind, prep, build_core(kind, len(levels), Z, y, base, l2, link, seed), levels, list(X.columns), decode_with)


def run_ordinal(kind: str, label: str, data: Any, target: str, features: Any, drop: Any, order: Any, test_size: float, seed: int, model_path: str | None,
                base: str = "logistic_regression", l2: float = 0.0, link: str = "logit", decode_with: str = "median") -> dict[str, Any]:
    X, y, levels, how = load_ordinal(data, target, features, drop, order)
    X_tr, X_te, y_tr, y_te = holdout(X, pd.Series(y), test_size, seed, stratify=True)
    model = fit_ordinal(kind, X_tr, y_tr.to_numpy(), levels, base, l2, link, seed, decode_with)
    pred = model.predict_codes(X_te)
    median_baseline = np.full(len(y_te), int(np.median(y_tr)))
    out: dict[str, Any] = {"model": label, "levels": levels, "order_source": how, "rows_train": len(X_tr), "rows_test": len(X_te),
                           "holdout": ordinal_metrics(y_te.to_numpy(), pred, len(levels)),
                           "median_baseline": ordinal_metrics(y_te.to_numpy(), median_baseline, len(levels)),
                           "predicted_distribution": {levels[k]: int((pred == k).sum()) for k in range(len(levels))},
                           "actual_distribution": {levels[k]: int((y_te.to_numpy() == k).sum()) for k in range(len(levels))}}
    core = model.core
    if isinstance(core, OrderedLogit):
        names = [str(n) for n in model.prep.get_feature_names_out()]
        out["coefficients"] = sorted([{"feature": n, "log_odds_per_sd": round(float(b), 4), "odds_ratio_per_sd": round(float(np.exp(b)), 4), "std_error": round(float(s), 4)}
                                      for n, b, s in zip(names, core.beta, core.se)], key=lambda r: -abs(r["log_odds_per_sd"]))
        out["thresholds"] = [round(float(t), 4) for t in core.theta]
        out["converged"] = core.converged
        out["reading"] = "a positive coefficient moves rows towards HIGHER levels; the odds ratio applies at every cut-point (the proportional-odds assumption)"
    if isinstance(core, RegressionThresholds):
        out["cutpoints"] = [round(float(c), 4) for c in core.cuts]
    out["model_path"] = save_model(model, model_path)
    return out
