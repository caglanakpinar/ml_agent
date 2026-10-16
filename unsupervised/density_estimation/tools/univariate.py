"""One-dimensional density estimation: histograms with bin rules, kernel density with bandwidth rules and boundary handling, parametric families, mixtures, ECDF and quantiles, and tail models."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ml_core.tooling import tool
from unsupervised.density_estimation.tools._common import (
    column_values,
    curve_summary,
    grid_for,
    modes_of,
    write_curve,
)

FAMILIES = ["norm", "lognorm", "gamma", "expon", "weibull_min", "beta", "t", "laplace", "logistic", "pareto", "gumbel_r", "skewnorm"]


@tool
def estimate_histogram_density(data: str, column: str, rule: Literal["freedman_diaconis", "scott", "sturges", "sqrt", "cross_validated"] = "freedman_diaconis", bins: int | None = None, output_path: str | None = None) -> dict:
    """Histogram density estimate with a principled bin count: Freedman–Diaconis (robust, from the IQR), Scott (from the SD), Sturges, square-root, or the number of bins that maximises leave-one-out likelihood. Reports the bin edges, densities and the mode — the plainest honest picture of a distribution.

    Args:
        rule: How to choose the number of bins when `bins` is not given.
        bins: Fix the number of bins instead.
        output_path: Where to write the curve (bin centres and densities).
    """
    x = column_values(data, column)
    n = len(x)
    if bins is None:
        if rule == "cross_validated":
            best, best_v = 10, -np.inf
            for b in range(5, min(120, n // 5)):
                h, e = np.histogram(x, bins=b)
                w = np.diff(e)
                loo = np.sum(np.where(h > 1, h * np.log(np.maximum((h - 1) / (n - 1) / w, 1e-12)), 0))
                if loo > best_v:
                    best, best_v = b, loo
            bins = best
        else:
            bins = int(len(np.histogram_bin_edges(x, bins={"freedman_diaconis": "fd", "scott": "scott", "sturges": "sturges", "sqrt": "sqrt"}[rule])) - 1)
    h, edges = np.histogram(x, bins=bins, density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    return {"n": n, "bins": int(bins), "bin_width": round(float(edges[1] - edges[0]), 5), "mode": round(float(centres[h.argmax()]), 4), "peak_density": round(float(h.max()), 6), "curve": curve_summary(centres, h, min(25, len(h))),
            "written_to": write_curve(centres, h, output_path, "histogram_density.csv") if output_path else None}


@tool
def estimate_kde(data: str, column: str, kernel: Literal["gaussian", "epanechnikov", "tophat", "exponential", "cosine", "linear"] = "gaussian", bandwidth_rule: Literal["scott", "silverman", "cross_validated", "value"] = "silverman",
                 bandwidth: float | None = None, bounded_below: float | None = None, bounded_above: float | None = None, max_rows: int = 50000, output_path: str | None = None, seed: int = 42) -> dict:
    """Kernel density estimate: a smooth bump on every observation. Bandwidth by Silverman's or Scott's rule, or the leave-one-out-likelihood optimum (better for multimodal data), or a value you give. For data with a hard limit (a count, a price) set a bound and the density is reflected at it instead of leaking past it.

    Args:
        kernel: gaussian, epanechnikov, tophat, exponential, cosine or linear.
        bandwidth_rule: How to choose the bandwidth.
        bandwidth: Bandwidth in the units of the data (used when the rule is `value`, or to override).
        bounded_below: Hard lower limit of the support (e.g. 0); the density is reflected there.
        bounded_above: Hard upper limit of the support.
        max_rows: Values sampled when the column is large.
        output_path: Where to write the curve.
    """
    from sklearn.neighbors import KernelDensity

    x = column_values(data, column, max_rows, seed)
    n, sd = len(x), x.std() or 1.0
    iqr = np.subtract(*np.quantile(x, [0.75, 0.25]))
    if bandwidth is not None or bandwidth_rule == "value":
        h = float(bandwidth or 0.3 * sd)
    elif bandwidth_rule == "scott":
        h = 1.06 * sd * n ** (-0.2)
    elif bandwidth_rule == "cross_validated":
        hs = np.linspace(0.05, 1.5, 60) * sd * n ** (-0.2)
        h = float(hs[int(np.argmax([_loo(x[:3000], hh) for hh in hs]))])
    else:
        h = 0.9 * min(sd, iqr / 1.34 or sd) * n ** (-0.2)
    lo, hi = bounded_below, bounded_above
    base = x
    if lo is not None:
        base = np.concatenate([base, 2 * lo - x])
    if hi is not None:
        base = np.concatenate([base, 2 * hi - x])
    kde = KernelDensity(kernel=kernel, bandwidth=h).fit(base[:, None])
    grid = grid_for(x, 300, lower=lo, upper=hi)
    scale = 1 + (lo is not None) + (hi is not None)
    dens = np.exp(kde.score_samples(grid[:, None])) * scale
    if lo is not None:
        dens[grid < lo] = 0
    if hi is not None:
        dens[grid > hi] = 0
    area = float(np.trapezoid(dens, grid))
    return {"n": n, "kernel": kernel, "bandwidth": round(h, 5), "bandwidth_rule": "given" if bandwidth is not None else bandwidth_rule, "modes": modes_of(grid, dens), "area_under_curve": round(area, 4),
            "curve": curve_summary(grid, dens), "written_to": write_curve(grid, dens, output_path, "kde_density.csv") if output_path else None}


def _loo(x: np.ndarray, h: float) -> float:
    """Leave-one-out log-likelihood of a Gaussian KDE with bandwidth h (the value a bandwidth is judged by)."""
    d = (x[:, None] - x[None, :]) / h
    k = np.exp(-0.5 * d**2) / (h * np.sqrt(2 * np.pi))
    np.fill_diagonal(k, 0)
    return float(np.mean(np.log(np.maximum(k.sum(axis=1) / (len(x) - 1), 1e-300))))


@tool
def tune_kde_bandwidth(data: str, column: str, n_values: int = 25, max_rows: int = 3000, seed: int = 42) -> dict:
    """Choose the KDE bandwidth by leave-one-out likelihood on a log-spaced grid, and compare it with the Silverman and Scott rules — the likelihood-optimal bandwidth is the one that predicts held-out values best, which a rule of thumb does not guarantee for skewed or multimodal data.

    Args:
        n_values: Bandwidths tried.
        max_rows: Values sampled (the leave-one-out computation is quadratic).
    """
    x = column_values(data, column, max_rows, seed)
    n, sd = len(x), x.std() or 1.0
    iqr = np.subtract(*np.quantile(x, [0.75, 0.25]))
    rules = {"silverman": 0.9 * min(sd, iqr / 1.34 or sd) * n ** (-0.2), "scott": 1.06 * sd * n ** (-0.2)}
    hs = np.geomspace(0.05, 2.5, n_values) * rules["silverman"]
    ll = np.array([_loo(x, h) for h in hs])
    best = float(hs[int(ll.argmax())])
    return {"best_bandwidth": round(best, 5), "best_loo_loglik": round(float(ll.max()), 4), "rules": {k: {"bandwidth": round(float(v), 5), "loo_loglik": round(_loo(x, v), 4)} for k, v in rules.items()},
            "grid": [{"bandwidth": round(float(h), 5), "loo_loglik": round(float(v), 4)} for h, v in zip(hs[::3], ll[::3])],
            "reading": "best well below Silverman means the data is multimodal or has sharp features the rule over-smooths; best above means noisy tails the rule under-smooths"}


@tool
def fit_parametric_distributions(data: str, column: str, families: list[str] | None = None, max_rows: int = 20000, seed: int = 42) -> dict:
    """Fit candidate distribution families by maximum likelihood (normal, lognormal, gamma, exponential, Weibull, beta, Student t, Laplace, logistic, Pareto, Gumbel, skew-normal) and rank them by AIC, BIC and the Kolmogorov–Smirnov statistic — with the fitted parameters. A parametric fit is compact, extrapolates into the tails, and says something about the process.

    Args:
        families: scipy.stats distribution names to try; defaults to twelve common ones.
        max_rows: Values sampled for the fits.
    """
    from scipy import stats

    x = column_values(data, column, max_rows, seed)
    rows = []
    for name in families or FAMILIES:
        dist = getattr(stats, name, None)
        if dist is None:
            rows.append({"family": name, "error": "unknown scipy.stats distribution"})
            continue
        try:
            xs, jacobian = x, 0.0  # log|dy/dx| when the data had to be rescaled to fit the family's support, so likelihoods stay comparable on the original scale
            if name in ("lognorm", "gamma", "expon", "weibull_min", "pareto") and x.min() <= 0:
                xs = x - x.min() + 1e-6 if name != "expon" else x - x.min()
            if name == "beta" and (x.min() < 0 or x.max() > 1):
                b = 0.998 / (x.max() - x.min() or 1)
                xs, jacobian = (x - x.min()) * b + 0.001, len(x) * float(np.log(b))
            params = dist.fit(xs)
            ll = float(np.sum(dist.logpdf(xs, *params))) + jacobian
            k = len(params)
            ks = stats.kstest(xs, name, args=params)
            rows.append({"family": name, "log_likelihood": round(ll, 2), "aic": round(2 * k - 2 * ll, 2), "bic": round(k * np.log(len(xs)) - 2 * ll, 2), "ks_statistic": round(float(ks.statistic), 4),
                         "ks_p_value": round(float(ks.pvalue), 4), "parameters": [round(float(p), 5) for p in params], "transformed_to_fit_support": bool(xs is not x)})
        except Exception as error:
            rows.append({"family": name, "error": str(error)[:80]})
    ok = sorted([r for r in rows if "aic" in r], key=lambda r: r["aic"])
    return {"n": len(x), "ranked_by_aic": ok, "failed": [r for r in rows if "error" in r], "best": ok[0]["family"] if ok else None,
            "reading": "an AIC gap under ~2 between families means the data cannot tell them apart; a tiny KS p-value for the best family means none of them fits — use a mixture or a KDE"}


@tool
def fit_mixture_density_1d(data: str, column: str, max_components: int = 6, criterion: Literal["bic", "aic"] = "bic", max_rows: int = 20000, seed: int = 42, output_path: str | None = None) -> dict:
    """Gaussian-mixture density of one column with the number of components chosen by BIC (or AIC): means, standard deviations and weights of each component — the model for multimodal data, and a way to read off the regimes hidden in a variable.

    Args:
        max_components: Largest number of components tried.
        criterion: bic (prefers fewer components) or aic.
        output_path: Where to write the curve.
    """
    from sklearn.mixture import GaussianMixture

    x = column_values(data, column, max_rows, seed)
    fits = [GaussianMixture(k, n_init=3, random_state=seed).fit(x[:, None]) for k in range(1, max_components + 1)]
    scores = [f.bic(x[:, None]) if criterion == "bic" else f.aic(x[:, None]) for f in fits]
    best = fits[int(np.argmin(scores))]
    grid = grid_for(x, 300)
    dens = np.exp(best.score_samples(grid[:, None]))
    order = np.argsort(best.means_.ravel())
    return {"n": len(x), "components_chosen": int(best.n_components), f"{criterion}_by_components": [round(float(s), 1) for s in scores],
            "components": [{"weight": round(float(best.weights_[i]), 4), "mean": round(float(best.means_[i, 0]), 4), "std": round(float(np.sqrt(best.covariances_[i].ravel()[0])), 4)} for i in order],
            "curve": curve_summary(grid, dens), "written_to": write_curve(grid, dens, output_path, "mixture_density.csv") if output_path else None}


@tool
def estimate_ecdf_and_quantiles(data: str, column: str, quantiles: list[float] | None = None, n_bootstrap: int = 500, confidence: float = 0.95, seed: int = 42) -> dict:
    """The empirical CDF and its quantiles with bootstrap confidence intervals, plus the Dvoretzky–Kiefer–Wolfowitz band around the whole CDF — a density-free description of a distribution, and the honest statement of how well a quantile is known at this sample size.

    Args:
        quantiles: Levels between 0 and 1; defaults to 1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%.
        n_bootstrap: Bootstrap resamples.
        confidence: Interval level.
    """
    x = column_values(data, column)
    qs = np.asarray(quantiles or [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
    rng = np.random.default_rng(seed)
    boots = np.array([np.quantile(rng.choice(x, len(x), replace=True), qs) for _ in range(n_bootstrap)])
    lo, hi = np.quantile(boots, [(1 - confidence) / 2, 1 - (1 - confidence) / 2], axis=0)
    eps = float(np.sqrt(np.log(2 / (1 - confidence)) / (2 * len(x))))
    return {"n": len(x), "quantiles": [{"level": float(q), "value": round(float(np.quantile(x, q)), 5), "ci_low": round(float(l), 5), "ci_high": round(float(h), 5)} for q, l, h in zip(qs, lo, hi)],
            "dkw_band_half_width": round(eps, 4), "reading": "extreme quantiles (1%, 99%) have wide intervals in small samples — a parametric tail model (fit_tail_distribution) is steadier"}


@tool
def fit_tail_distribution(data: str, column: str, tail: Literal["upper", "lower"] = "upper", threshold_quantile: float = 0.9, return_periods: list[float] | None = None) -> dict:
    """Peaks-over-threshold: fit a generalised Pareto distribution to the exceedances over a high threshold, and report the shape (heavy-tailed if positive), the exceedance probability beyond chosen levels, and return levels (the value exceeded on average once in N observations) — the right way to extrapolate to extremes the sample has barely seen.

    Args:
        tail: upper or lower.
        threshold_quantile: Quantile above which values count as the tail.
        return_periods: Observation counts N for return levels; defaults to 100, 1000, 10000.
    """
    from scipy.stats import genpareto

    x = column_values(data, column)
    v = x if tail == "upper" else -x
    u = float(np.quantile(v, threshold_quantile))
    exc = v[v > u] - u
    if len(exc) < 20:
        raise ValueError(f"only {len(exc)} exceedances over the threshold; lower `threshold_quantile`")
    c, _, scale = genpareto.fit(exc, floc=0)
    p_u = len(exc) / len(v)
    levels = {}
    for N in return_periods or [100, 1000, 10000]:
        m = N * p_u
        levels[int(N)] = round(float((u + (scale / c) * (m**c - 1) if abs(c) > 1e-6 else u + scale * np.log(m)) * (1 if tail == "upper" else -1)), 4)
    return {"tail": tail, "threshold": round(u * (1 if tail == "upper" else -1), 4), "exceedances": int(len(exc)), "shape": round(float(c), 4), "scale": round(float(scale), 4),
            "tail_type": "heavy (power-law-like)" if c > 0.1 else "light (bounded)" if c < -0.1 else "exponential-like", "return_levels": levels, "observed_max": round(float(x.max() if tail == "upper" else x.min()), 4)}
