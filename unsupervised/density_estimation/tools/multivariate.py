"""Densities over several columns: kernel density (bandwidth by cross-validation), Gaussian and Bayesian mixtures, k-nearest-neighbour density, a Gaussian copula, a 2-D histogram, and conditional densities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_core.ml import save_model
from ml_core.tooling import tool
from unsupervised.density_estimation.tools._common import (
    fit_density,
    heldout_loglik,
    numeric_columns,
)


def _fit(kind, label, data, columns, drop, params, seed, model_path, max_rows, bandwidth_rule=None):
    X, names = numeric_columns(data, columns, drop, max_rows, seed, minimum=1)
    model = fit_density(X, kind, params, seed, bandwidth_rule)
    mean, se = heldout_loglik(X, kind, params, seed, 5, bandwidth_rule)
    own = model.log_density(X)
    out = {"estimator": label, "rows": int(len(X)), "columns": names, "heldout_mean_log_density": round(mean, 4), "heldout_standard_error": round(se, 4), "train_mean_log_density": round(float(own.mean()), 4),
           "overfit_gap": round(float(own.mean() - mean), 4), "log_density_quantiles": {q: round(float(np.quantile(own, q)), 3) for q in (0.01, 0.1, 0.5, 0.9)}, "can_sample": hasattr(model.core, "sample")}
    out["model_path"] = save_model(model, model_path)
    return out, model, X


@tool
def fit_kde_multivariate(data: str, columns: list[str] | None = None, drop: list[str] | None = None, kernel: str = "gaussian", bandwidth: float | None = None, tune_bandwidth: bool = True, max_rows: int = 20000, seed: int = 42, model_path: str | None = None) -> dict:
    """Multivariate kernel density estimate on standardised columns, with the bandwidth chosen by cross-validated likelihood (or fixed by you). Flexible — no shape assumed — but it degrades quickly beyond 4–6 columns and needs many rows; it can also generate samples.

    Args:
        kernel: gaussian, tophat, epanechnikov, exponential, linear or cosine.
        bandwidth: Bandwidth in standardised units; tuned when omitted.
        tune_bandwidth: Search a bandwidth grid by held-out likelihood.
        max_rows: Rows sampled.
    """
    X, names = numeric_columns(data, columns, drop, max_rows, seed)
    best = bandwidth
    grid_rows = []
    if bandwidth is None and tune_bandwidth:
        n, d = X.shape
        base = n ** (-1 / (d + 4))
        for f in (0.3, 0.5, 0.75, 1.0, 1.5, 2.0):
            ll, _ = heldout_loglik(X, "kde", {"kernel": kernel, "bandwidth": base * f}, seed, 4)
            grid_rows.append({"bandwidth": round(base * f, 4), "heldout_log_density": round(ll, 4)})
        best = max(grid_rows, key=lambda r: r["heldout_log_density"])["bandwidth"]
    out, _, _ = _fit("kde", f"kde({kernel})", data, columns, drop, {"kernel": kernel, "bandwidth": best or 0.5}, seed, model_path, max_rows, None if best else "scott")
    out["bandwidth"] = best
    out["bandwidth_search"] = grid_rows
    out["warning"] = "KDE needs many rows per dimension: with more than ~5 columns the held-out log-density will fall below a Gaussian mixture's" if len(names) > 5 else None
    return out


@tool
def fit_gaussian_mixture_density(data: str, columns: list[str] | None = None, drop: list[str] | None = None, n_components: int | None = None, max_components: int = 8, covariance_type: str = "full", max_rows: int = 20000, seed: int = 42, model_path: str | None = None) -> dict:
    """Gaussian-mixture density over several columns, with the number of components chosen by held-out likelihood when not given: a compact, smooth, sampleable model that scales to dozens of columns, with each component readable as a regime (its mean and weight).

    Args:
        n_components: Fixed number of components; chosen by cross-validated BIC when omitted.
        max_components: Largest number tried when choosing.
        covariance_type: full, tied, diag or spherical.
        max_rows: Rows sampled.
    """
    from sklearn.mixture import GaussianMixture

    X, names = numeric_columns(data, columns, drop, max_rows, seed)
    Z = (X.to_numpy(float) - X.mean().to_numpy()) / np.where(X.std().to_numpy() > 0, X.std().to_numpy(), 1)
    bics = {}
    if n_components is None:
        for k in range(1, max_components + 1):
            bics[k] = float(GaussianMixture(k, covariance_type=covariance_type, n_init=2, random_state=seed).fit(Z).bic(Z))
        n_components = min(bics, key=bics.get)
    out, model, _ = _fit("gmm", f"gmm(k={n_components},{covariance_type})", data, columns, drop, {"n_components": n_components, "covariance_type": covariance_type}, seed, model_path, max_rows)
    out["components_chosen"] = int(n_components)
    out["bic_by_components"] = {k: round(v, 1) for k, v in bics.items()}
    out["components"] = [{"weight": round(float(w), 4), **{c: round(float(m), 3) for c, m in zip(names, model.core.means_[i] * model.std + model.mean)}} for i, w in enumerate(model.core.weights_)]
    return out


@tool
def fit_bayesian_mixture_density(data: str, columns: list[str] | None = None, drop: list[str] | None = None, max_components: int = 10, max_rows: int = 20000, seed: int = 42, model_path: str | None = None) -> dict:
    """Dirichlet-process Gaussian mixture: give an upper bound on the components and the model switches off the surplus, so the number of regimes is learned and the density is less prone to over-fitting than a plain mixture with too many components.

    Args:
        max_components: Upper bound on the components.
        max_rows: Rows sampled.
    """
    out, model, _ = _fit("bayesian_gmm", f"bayesian_gmm(max={max_components})", data, columns, drop, {"n_components": max_components}, seed, model_path, max_rows)
    w = model.core.weights_
    out["effective_components"] = int((w > 0.01).sum())
    out["component_weights"] = np.sort(np.round(w, 4))[::-1][:10].tolist()
    return out


@tool
def fit_knn_density(data: str, columns: list[str] | None = None, drop: list[str] | None = None, k: int | None = None, max_rows: int = 20000, seed: int = 42, model_path: str | None = None) -> dict:
    """k-nearest-neighbour density: the density at a point is inversely proportional to the volume of the ball needed to reach its k-th neighbour. Adapts bandwidth to local density (wide in sparse regions, narrow in dense ones), handles unequal densities well, but is not a proper (normalised) density and cannot sample.

    Args:
        k: Neighbours; chosen from held-out likelihood among 5, 10, 20, 40 when omitted.
    """
    X, names = numeric_columns(data, columns, drop, max_rows, seed)
    if k is None:
        scores = {kk: heldout_loglik(X, "knn", {"k": kk}, seed, 4)[0] for kk in (5, 10, 20, 40) if kk < len(X) // 5}
        k = max(scores, key=scores.get)
    out, _, _ = _fit("knn", f"knn_density(k={k})", data, columns, drop, {"k": k}, seed, model_path, max_rows)
    out["k"] = int(k)
    return out


@tool
def fit_gaussian_copula(data: str, columns: list[str] | None = None, drop: list[str] | None = None, max_rows: int = 20000, seed: int = 42, model_path: str | None = None) -> dict:
    """Gaussian copula: each column keeps its own, arbitrarily shaped marginal distribution (smoothed from the data), and a correlation matrix on normal scores ties them together. The right model for skewed, differently shaped columns with monotone dependence, and it samples realistic synthetic rows; the dependence matrix reads like a correlation matrix.

    Args:
        max_rows: Rows sampled.
    """

    X, names = numeric_columns(data, columns, drop, max_rows, seed, minimum=2)
    out, model, _ = _fit("copula", "gaussian_copula", data, columns, drop, {}, seed, model_path, max_rows)
    out["dependence_matrix"] = pd.DataFrame(model.core.corr, index=names, columns=names).round(3).reset_index(names="column").to_dict("records")
    return out


@tool
def fit_histogram_2d(data: str, x: str, y: str, bins: int = 30, max_rows: int = 100000, seed: int = 42, output_path: str | None = None) -> dict:
    """Two-dimensional histogram density of a pair of columns: the density in each cell, the highest-density cell, the mutual information of the pair (how much knowing one narrows the other, in nats), and the share of cells that are empty — the simplest way to *see* a joint distribution and check a fitted model against it.

    Args:
        x: First column.
        y: Second column.
        bins: Bins per axis.
        output_path: Where to write the cell table.
    """
    from ml_core.data import save_frame

    X, _ = numeric_columns(data, [x, y], None, max_rows, seed, minimum=2)
    h, xe, ye = np.histogram2d(X[x], X[y], bins=bins, density=True)
    counts, _, _ = np.histogram2d(X[x], X[y], bins=bins)
    p = counts / counts.sum()
    px, py = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True)
    mask = p > 0
    mi = float(np.sum(p[mask] * np.log(p[mask] / (px @ py)[mask])))
    i, j = np.unravel_index(h.argmax(), h.shape)
    table = pd.DataFrame([{"x_low": xe[a], "x_high": xe[a + 1], "y_low": ye[b], "y_high": ye[b + 1], "density": h[a, b], "count": counts[a, b]} for a in range(bins) for b in range(bins) if counts[a, b] > 0])
    return {"rows": int(len(X)), "bins_per_axis": bins, "mutual_information_nats": round(mi, 4), "correlation": round(float(np.corrcoef(X[x], X[y])[0, 1]), 4),
            "peak_cell": {"x": [round(float(xe[i]), 4), round(float(xe[i + 1]), 4)], "y": [round(float(ye[j]), 4), round(float(ye[j + 1]), 4)], "density": round(float(h.max()), 6)}, "empty_cell_share": round(float((counts == 0).mean()), 4),
            "written_to": save_frame(table, output_path, "histogram_2d.csv") if output_path else None}


@tool
def estimate_conditional_density(data: str, target: str, given: str, at: list[float] | None = None, bandwidth_x: float | None = None, bandwidth_y: float | None = None, max_rows: int = 10000, seed: int = 42) -> dict:
    """The density of `target` *given* a value of another column, by kernel weighting (Nadaraya–Watson conditional KDE): for each requested value of `given`, the conditional mean, spread, quantiles and mode of `target`. Shows how a whole distribution — not only its average — moves with another variable, and where the spread changes (heteroscedasticity).

    Args:
        target: The column whose distribution is wanted.
        given: The column that is conditioned on.
        at: Values of `given` to condition at; defaults to its quartiles.
        bandwidth_x: Kernel width on `given` (silverman when omitted).
        bandwidth_y: Kernel width on `target` (silverman when omitted).
    """
    X, _ = numeric_columns(data, [given, target], None, max_rows, seed, minimum=2)
    gx, gy = X[given].to_numpy(float), X[target].to_numpy(float)
    n = len(gx)
    hx = bandwidth_x or 1.06 * gx.std() * n ** (-0.2)
    hy = bandwidth_y or 1.06 * gy.std() * n ** (-0.2)
    grid = np.linspace(gy.min() - 2 * hy, gy.max() + 2 * hy, 300)
    rows = []
    for v in (at if at else np.quantile(gx, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()):
        w = np.exp(-0.5 * ((gx - v) / hx) ** 2)
        if w.sum() < 1e-9:
            rows.append({"given": v, "error": "no data near this value"})
            continue
        w = w / w.sum()
        dens = (w[:, None] * np.exp(-0.5 * ((grid[None, :] - gy[:, None]) / hy) ** 2) / (hy * np.sqrt(2 * np.pi))).sum(axis=0)
        cdf = np.cumsum(dens) / dens.sum()
        mean = float((w * gy).sum())
        rows.append({"given": round(float(v), 4), "effective_rows": round(float(1 / (w**2).sum()), 1), "mean": round(mean, 4), "std": round(float(np.sqrt((w * (gy - mean) ** 2).sum())), 4), "mode": round(float(grid[dens.argmax()]), 4),
                     "quantiles": {q: round(float(np.interp(q, cdf, grid)), 4) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}})
    stds = [r["std"] for r in rows if "std" in r]
    return {"target": target, "given": given, "bandwidth_given": round(float(hx), 4), "bandwidth_target": round(float(hy), 4), "conditional": rows,
            "spread_changes_with_given": bool(stds and max(stds) > 1.5 * min(stds)), "reading": "if the spread changes a lot across the rows, a model that assumes constant error variance will mis-state uncertainty"}
