"""Density-estimation machinery the tool modules share: loading one or several numeric columns, a picklable `DensityModel`
with `log_density` and `sample`, held-out log-likelihood (the honest score of a density), and the curves written by the 1-D tools."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_core.data import as_list, load_frame, require_columns, save_frame
from ml_core.ml import sk


def numeric_columns(data: Any, columns: Any = None, drop: Any = None, max_rows: int | None = None, seed: int = 42, minimum: int = 1) -> tuple[pd.DataFrame, list[str]]:
    frame = load_frame(data)
    names = as_list(columns) or [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and c not in set(as_list(drop) or [])]
    require_columns(frame, names)
    names = [c for c in names if pd.api.types.is_numeric_dtype(frame[c])]
    if len(names) < minimum:
        raise ValueError(f"need at least {minimum} numeric column(s); found {names}")
    X = frame[names].dropna()
    if max_rows and len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
    return X.reset_index(drop=True), names


def column_values(data: Any, column: str, max_rows: int | None = None, seed: int = 42) -> np.ndarray:
    frame = load_frame(data)
    require_columns(frame, [column])
    x = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(float)
    if len(x) < 10:
        raise ValueError(f"only {len(x)} numeric values in {column!r}; a density needs at least 10")
    if max_rows and len(x) > max_rows:
        x = np.random.default_rng(seed).choice(x, max_rows, replace=False)
    return x


def grid_for(x: np.ndarray, n: int = 200, pad: float = 0.05, lower: float | None = None, upper: float | None = None) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    span = hi - lo or 1.0
    return np.linspace(lo - pad * span if lower is None else lower, hi + pad * span if upper is None else upper, n)


def curve_summary(grid: np.ndarray, dens: np.ndarray, points: int = 25) -> dict[str, list[float]]:
    idx = np.linspace(0, len(grid) - 1, points).astype(int)
    return {"x": [round(float(v), 4) for v in grid[idx]], "density": [round(float(v), 6) for v in dens[idx]]}


def write_curve(grid: np.ndarray, dens: np.ndarray, path: str | None, default: str) -> str | None:
    return save_frame(pd.DataFrame({"x": grid, "density": dens}), path, default) if path is not None else None


def modes_of(grid: np.ndarray, dens: np.ndarray, prominence: float = 0.05) -> list[dict[str, float]]:
    from scipy.signal import find_peaks

    peaks, props = find_peaks(dens, prominence=prominence * dens.max())
    return [{"x": round(float(grid[p]), 4), "density": round(float(dens[p]), 6)} for p in peaks]


# ---- multivariate density models ----------------------------------------------------------------
class KNNDensity:
    """k-NN density: p(x) ≈ k / (n · V_k(x)), the volume of the ball reaching the k-th neighbour."""

    def __init__(self, k: int = 10):
        self.k = k

    def fit(self, Z: np.ndarray) -> "KNNDensity":
        from sklearn.neighbors import NearestNeighbors

        self.nn, self.n, self.d = NearestNeighbors(n_neighbors=self.k + 1).fit(Z), len(Z), Z.shape[1]
        return self

    def score_samples(self, Z: np.ndarray) -> np.ndarray:
        from scipy.special import gammaln

        d = self.nn.kneighbors(Z, n_neighbors=self.k + 1)[0]
        r = np.maximum(d[:, -1] if np.mean(d[:, 0]) > 1e-9 else d[:, -1], 1e-12)
        log_vol = (self.d / 2) * np.log(np.pi) - gammaln(self.d / 2 + 1) + self.d * np.log(r)
        return np.log(self.k) - np.log(self.n) - log_vol


class GaussianCopula:
    """Marginals by smoothed ECDFs, dependence by a Gaussian copula: p(x) = c(F(x)) · Π f_j(x_j). Models skewed, differently shaped marginals that a Gaussian mixture handles poorly, and samples exactly."""

    def fit(self, Z: np.ndarray) -> "GaussianCopula":
        from scipy.stats import gaussian_kde, norm, rankdata

        self.sorted = np.sort(Z, axis=0)
        n = len(Z)
        U = np.column_stack([rankdata(Z[:, j]) / (n + 1) for j in range(Z.shape[1])])
        N = norm.ppf(U)
        self.corr = np.corrcoef(N, rowvar=False) if Z.shape[1] > 1 else np.ones((1, 1))
        self.kdes = [gaussian_kde(Z[:, j]) for j in range(Z.shape[1])]
        self.inv = np.linalg.pinv(self.corr)
        self.logdet = np.linalg.slogdet(self.corr)[1]
        return self

    def _u(self, Z: np.ndarray) -> np.ndarray:
        n = len(self.sorted)
        return np.column_stack([np.clip((np.searchsorted(self.sorted[:, j], Z[:, j], side="right") + 0.5) / (n + 1), 1e-4, 1 - 1e-4) for j in range(Z.shape[1])])

    def score_samples(self, Z: np.ndarray) -> np.ndarray:
        from scipy.stats import norm

        N = norm.ppf(self._u(Z))
        log_c = -0.5 * self.logdet - 0.5 * np.einsum("ij,jk,ik->i", N, self.inv - np.eye(len(self.inv)), N)
        return log_c + sum(np.log(np.maximum(k(Z[:, j]), 1e-300)) for j, k in enumerate(self.kdes))

    def sample(self, n_samples: int, random_state: int | None = None) -> np.ndarray:
        from scipy.stats import norm

        rng = np.random.default_rng(random_state)
        U = norm.cdf(rng.multivariate_normal(np.zeros(len(self.corr)), self.corr, size=n_samples))
        return np.column_stack([np.quantile(self.sorted[:, j], U[:, j]) for j in range(U.shape[1])])


def build_density(kind: str, d: int, params: dict[str, Any], seed: int) -> Any:
    sk()
    from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
    from sklearn.neighbors import KernelDensity

    if kind == "kde":
        return KernelDensity(kernel=params.get("kernel", "gaussian"), bandwidth=params.get("bandwidth", 0.5))
    if kind == "gmm":
        return GaussianMixture(params.get("n_components", 3), covariance_type=params.get("covariance_type", "full"), n_init=2, random_state=seed)
    if kind == "bayesian_gmm":
        return BayesianGaussianMixture(n_components=params.get("n_components", 8), weight_concentration_prior_type="dirichlet_process", weight_concentration_prior=0.05, max_iter=500, random_state=seed)
    if kind == "knn":
        return KNNDensity(params.get("k", 10))
    if kind == "copula":
        return GaussianCopula()
    raise ValueError(f"unknown density estimator {kind!r}")


class DensityModel:
    """Standardiser + a fitted estimator: `log_density(X)` and `sample(n)` on the original scale."""

    def __init__(self, kind: str, columns: list[str], mean: np.ndarray, std: np.ndarray, core: Any, params: dict[str, Any]):
        self.kind, self.columns, self.mean, self.std, self.core, self.params = kind, columns, mean, std, core, params

    def log_density(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        A = X[self.columns].to_numpy(float) if isinstance(X, pd.DataFrame) else np.asarray(X, float)
        Z = (A - self.mean) / self.std
        return np.asarray(self.core.score_samples(Z)) - np.log(self.std).sum()  # change of variables back to the original scale

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        if not hasattr(self.core, "sample"):
            raise ValueError(f"{self.kind} cannot generate samples; use kde, gmm, bayesian_gmm or copula")
        s = self.core.sample(n, random_state=seed) if self.kind == "copula" else self.core.sample(n)[0] if self.kind != "kde" else self.core.sample(n, random_state=seed)
        return np.asarray(s) * self.std + self.mean


def fit_density(X: pd.DataFrame, kind: str, params: dict[str, Any], seed: int, bandwidth_rule: str | None = None) -> DensityModel:
    A = X.to_numpy(float)
    mean, std = A.mean(axis=0), np.where(A.std(axis=0) > 0, A.std(axis=0), 1.0)
    Z = (A - mean) / std
    p = dict(params)
    if kind == "kde" and bandwidth_rule:
        n, d = Z.shape
        p["bandwidth"] = n ** (-1 / (d + 4)) * (1.0 if bandwidth_rule == "scott" else (4 / (d + 2)) ** (1 / (d + 4)))
    core = build_density(kind, Z.shape[1], p, seed).fit(Z)
    return DensityModel(kind, list(X.columns), mean, std, core, p)


def heldout_loglik(X: pd.DataFrame, kind: str, params: dict[str, Any], seed: int, folds: int = 5, bandwidth_rule: str | None = None) -> tuple[float, float]:
    """Mean (and standard error of the) log-density of held-out rows under models fitted on the rest — higher is better; the only fair score of a density."""
    from sklearn.model_selection import KFold

    scores = []
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        m = fit_density(X.iloc[tr], kind, params, seed, bandwidth_rule)
        scores.append(m.log_density(X.iloc[te]))
    allv = np.concatenate(scores)
    allv = allv[np.isfinite(allv)] if np.isfinite(allv).any() else allv
    return float(allv.mean()), float(allv.std() / np.sqrt(len(allv)))
