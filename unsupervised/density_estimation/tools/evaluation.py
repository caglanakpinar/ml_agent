"""Judge a density and put it to work: estimators compared by held-out likelihood, goodness of fit, sampling and synthetic-vs-real checks, how typical a row is, probabilities of regions, highest-density regions, and drift between two datasets."""

from __future__ import annotations

from typing import Literal

import joblib
import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.tooling import tool
from unsupervised.density_estimation.tools._common import (
    column_values,
    heldout_loglik,
    numeric_columns,
)


@tool
def compare_density_estimators(data: str, columns: list[str] | None = None, drop: list[str] | None = None, estimators: list[str] | None = None, cv: int = 5, max_rows: int = 6000, seed: int = 42) -> dict:
    """The decision tool: held-out mean log-density (higher = better) of a kernel density estimate, Gaussian mixtures with 2–6 components, a Bayesian mixture, k-NN density and a Gaussian copula, by cross-validation on the same folds, with standard errors. An estimator that wins on training likelihood but not here is over-fitting.

    Args:
        estimators: Any of kde, gmm2, gmm3, gmm4, gmm6, bayesian_gmm, knn, copula; defaults to all.
        cv: Folds.
    """
    X, names = numeric_columns(data, columns, drop, max_rows, seed)
    specs = {"kde": ("kde", {"bandwidth": 0.4}, "scott"), "gmm2": ("gmm", {"n_components": 2}, None), "gmm3": ("gmm", {"n_components": 3}, None), "gmm4": ("gmm", {"n_components": 4}, None),
             "gmm6": ("gmm", {"n_components": 6}, None), "bayesian_gmm": ("bayesian_gmm", {"n_components": 8}, None), "knn": ("knn", {"k": 15}, None), "copula": ("copula", {}, None)}
    rows = []
    for name in estimators or list(specs):
        kind, params, rule = specs[name]
        if kind == "copula" and len(names) < 2:
            continue
        try:
            m, se = heldout_loglik(X, kind, params, seed, cv, rule)
            rows.append({"estimator": name, "heldout_mean_log_density": round(m, 4), "standard_error": round(se, 4), "can_sample": kind != "knn"})
        except Exception as error:
            rows.append({"estimator": name, "error": str(error)[:100]})
    ok = sorted([r for r in rows if "heldout_mean_log_density" in r], key=lambda r: -r["heldout_mean_log_density"])
    best = ok[0]
    tied = [r["estimator"] for r in ok if r["heldout_mean_log_density"] >= best["heldout_mean_log_density"] - 2 * best["standard_error"]]
    return {"columns": names, "rows": len(X), "ranked": ok, "failed": [r for r in rows if "error" in r], "best": best["estimator"], "statistically_tied_with_best": tied,
            "note": "log-densities are on the original scale and are comparable only between estimators for the same columns; pick the simplest of the tied ones"}


@tool
def evaluate_density_fit(data: str, column: str, distribution: str = "norm", params: list[float] | None = None, n_bins: int = 20) -> dict:
    """Goodness of fit of a distribution to one column: Kolmogorov–Smirnov, Anderson–Darling-style tail check, a chi-square on equal-probability bins, and the probability-integral-transform (PIT) uniformity — if the fitted density is right, F(x) of the data is uniform. With no parameters given the family is fitted first.

    Args:
        distribution: scipy.stats distribution name (norm, lognorm, gamma, expon, weibull_min, t, ...).
        params: Parameters of the distribution; fitted by maximum likelihood when omitted.
        n_bins: Bins for the chi-square test.
    """
    from scipy import stats

    x = column_values(data, column)
    dist = getattr(stats, distribution)
    p = tuple(params) if params else dist.fit(x)
    u = dist.cdf(x, *p)
    ks = stats.kstest(x, distribution, args=p)
    edges = dist.ppf(np.linspace(0, 1, n_bins + 1)[1:-1], *p)
    obs = np.bincount(np.searchsorted(edges, x), minlength=n_bins)
    chi = stats.chisquare(obs, np.full(n_bins, len(x) / n_bins), ddof=len(p))
    pit_hist = np.histogram(u, bins=10, range=(0, 1))[0] / len(x)
    return {"distribution": distribution, "parameters": [round(float(v), 5) for v in p], "parameters_fitted": params is None, "ks_statistic": round(float(ks.statistic), 4), "ks_p_value": round(float(ks.pvalue), 4),
            "chi_square_p_value": round(float(chi.pvalue), 4), "pit_histogram_deciles": pit_hist.round(3).tolist(), "pit_ideal": 0.1,
            "pit_pattern": "U-shaped: the model's tails are too thin" if pit_hist[0] + pit_hist[-1] > 0.28 else "hump-shaped: the model is too wide" if pit_hist[4] + pit_hist[5] > 0.28 else "roughly uniform: calibrated",
            "note": "p-values are optimistic when the parameters were fitted on the same data; read the PIT histogram alongside them"}


@tool
def sample_from_density(model_path: str, n: int = 1000, seed: int = 42, output_path: str | None = None) -> dict:
    """Generate synthetic rows from a saved density model (KDE, Gaussian mixture, Bayesian mixture or Gaussian copula) — and compare the synthetic means, standard deviations and correlations with those the model was fitted on, so a poor model announces itself.

    Args:
        model_path: Path of the saved density model (joblib).
        n: Rows to generate.
    """
    model = joblib.load(model_path)
    S = pd.DataFrame(model.sample(n, seed), columns=model.columns)
    return {"estimator": model.kind, "rows": n, "synthetic_mean": S.mean().round(4).to_dict(), "synthetic_std": S.std().round(4).to_dict(), "fitted_on_mean": dict(zip(model.columns, np.round(model.mean, 4))), "fitted_on_std": dict(zip(model.columns, np.round(model.std, 4))),
            "synthetic_correlation": S.corr().round(3).reset_index(names="column").to_dict("records") if len(model.columns) > 1 else None, "written_to": save_frame(S, output_path, "synthetic_rows.csv"), "sample": S.head(5)}


@tool
def compare_real_vs_synthetic(real: str, synthetic: str, columns: list[str] | None = None, seed: int = 42) -> dict:
    """How realistic are generated rows? A classifier is trained to tell real from synthetic (cross-validated AUC: 0.5 means indistinguishable), per-column Kolmogorov–Smirnov distances, and the difference of the two correlation matrices — the standard acceptance test for a generative density.

    Args:
        real: Path to the real table.
        synthetic: Path to the synthetic table (e.g. written by `sample_from_density`).
    """
    from scipy import stats
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict, StratifiedKFold

    R, names = numeric_columns(real, columns, None, 5000, seed)
    S = load_frame(synthetic)[names].dropna().sample(min(5000, len(load_frame(synthetic))), random_state=seed)
    Z = np.vstack([R.to_numpy(float), S.to_numpy(float)])
    y = np.concatenate([np.zeros(len(R)), np.ones(len(S))])
    auc = float(roc_auc_score(y, cross_val_predict(RandomForestClassifier(150, min_samples_leaf=3, random_state=seed, n_jobs=-1), Z, y, cv=StratifiedKFold(4, shuffle=True, random_state=seed), method="predict_proba")[:, 1]))
    ks = {c: round(float(stats.ks_2samp(R[c], S[c]).statistic), 4) for c in names}
    cdiff = float(np.abs(R.corr().to_numpy() - S.corr().to_numpy()).max()) if len(names) > 1 else 0.0
    return {"real_vs_synthetic_auc": round(auc, 4), "ks_distance_by_column": ks, "max_correlation_difference": round(cdiff, 4),
            "verdict": "synthetic rows are hard to tell from real ones" if auc < 0.6 else "detectably synthetic — the model misses structure (see which columns have the largest KS distance, and the correlation gap)" if auc < 0.8 else "easily distinguished: the density is a poor model of this data"}


@tool
def score_typicality(model_path: str, data: str, id_column: str | None = None, reference: str | None = None, output_path: str | None = None) -> dict:
    """How typical is each row under a saved density model? The log-density, its percentile among the rows the model was fitted on (when `reference` is given), and a typicality label — the density-based way to find rows that are rare *given the whole joint distribution*, usable as an anomaly score.

    Args:
        model_path: Path of the saved density model (joblib).
        id_column: Column to carry through to the output.
        reference: Table of rows the model was fitted on, used to turn log-densities into percentiles.
    """
    model = joblib.load(model_path)
    frame = load_frame(data)
    require_columns(frame, model.columns)
    ld = model.log_density(frame)
    out = pd.DataFrame({"log_density": ld}, index=frame.index)
    if reference:
        ref = model.log_density(load_frame(reference).dropna(subset=model.columns))
        out["density_percentile"] = [float((ref <= v).mean()) for v in ld]
        out["typicality"] = np.where(out["density_percentile"] < 0.01, "very rare", np.where(out["density_percentile"] < 0.1, "uncommon", "typical"))
    if id_column and id_column in frame.columns:
        out.insert(0, id_column, frame[id_column])
    return {"estimator": model.kind, "rows": len(out), "log_density_quantiles": {q: round(float(np.quantile(ld, q)), 3) for q in (0.01, 0.1, 0.5, 0.9)}, "rarest": out.sort_values("log_density").head(10), "written_to": save_frame(out, output_path, "typicality.csv")}


@tool
def probability_of_region(data: str, column: str, lower: float | None = None, upper: float | None = None, method: Literal["empirical", "kde", "parametric"] = "kde", distribution: str = "norm", n_bootstrap: int = 300, seed: int = 42) -> dict:
    """P(lower ≤ X ≤ upper) for one column, three ways — the plain empirical share, the integral of a KDE, or a fitted parametric family — with a bootstrap interval on the empirical estimate. The answer to 'how likely is a value in this range?' with its uncertainty, and a check that the smooth models agree with the data.

    Args:
        column: Column to evaluate.
        lower: Lower limit (open if omitted).
        upper: Upper limit (open if omitted).
        method: Which estimate to headline.
        distribution: Family for the parametric estimate.
        n_bootstrap: Bootstrap resamples for the interval.
    """
    from scipy import stats

    x = column_values(data, column)
    lo, hi = -np.inf if lower is None else lower, np.inf if upper is None else upper
    emp = float(((x >= lo) & (x <= hi)).mean())
    kde = stats.gaussian_kde(x)
    kde_p = float(kde.integrate_box_1d(lo, hi))
    dist = getattr(stats, distribution)
    par = dist.fit(x)
    par_p = float(dist.cdf(hi, *par) - dist.cdf(lo, *par))
    rng = np.random.default_rng(seed)
    boots = [float(((b >= lo) & (b <= hi)).mean()) for b in (rng.choice(x, len(x)) for _ in range(n_bootstrap))]
    est = {"empirical": emp, "kde": kde_p, "parametric": par_p}
    return {"column": column, "region": [lower, upper], "probability": round(est[method], 5), "method": method, "all_methods": {k: round(v, 5) for k, v in est.items()},
            "empirical_95pct_interval": [round(float(np.quantile(boots, 0.025)), 5), round(float(np.quantile(boots, 0.975)), 5)], "observations_in_region": int(((x >= lo) & (x <= hi)).sum()),
            "reading": "when the three disagree the parametric family is probably wrong for this column; when the region holds fewer than ~10 observations the empirical interval is the honest statement"}


@tool
def highest_density_region(data: str, column: str, mass: float = 0.9, max_rows: int = 50000, seed: int = 42) -> dict:
    """The smallest set of values containing a given share of the probability (the highest-density region), from a KDE — for a skewed or multimodal column this is *not* the central interval: it can be a union of intervals around each mode, and is shorter than the equal-tailed interval.

    Args:
        mass: Probability the region must contain, e.g. 0.9.
        max_rows: Values sampled.
    """
    from scipy import stats

    x = column_values(data, column, max_rows, seed)
    kde = stats.gaussian_kde(x)
    g = np.linspace(x.min() - 0.1 * np.ptp(x), x.max() + 0.1 * np.ptp(x), 2000)
    d = kde(g)
    order = np.argsort(-d)
    cum = np.cumsum(d[order]) * (g[1] - g[0])
    thr = float(d[order][np.searchsorted(cum, mass)])
    inside = d >= thr
    edges = np.flatnonzero(np.diff(inside.astype(int)))
    starts = ([0] if inside[0] else []) + [e + 1 for e in edges if not inside[e]]
    ends = [e for e in edges if inside[e]] + ([len(g) - 1] if inside[-1] else [])
    intervals = [[round(float(g[a]), 4), round(float(g[b]), 4)] for a, b in zip(starts, ends)]
    q = np.quantile(x, [(1 - mass) / 2, 1 - (1 - mass) / 2])
    return {"mass": mass, "intervals": intervals, "total_width": round(float(sum(b - a for a, b in intervals)), 4), "equal_tailed_interval": [round(float(q[0]), 4), round(float(q[1]), 4)], "equal_tailed_width": round(float(q[1] - q[0]), 4), "density_threshold": round(thr, 6),
            "multimodal": len(intervals) > 1}


@tool
def measure_distribution_drift(reference: str, current: str, columns: list[str] | None = None, bins: int = 20) -> dict:
    """Has the distribution moved between two datasets (training vs production, last month vs this)? Per column: Kolmogorov–Smirnov distance, Jensen–Shannon divergence over shared bins, the population stability index and the shift of mean and spread — and an overall verdict by the usual PSI bands (under 0.1 stable, 0.1–0.25 moderate, above 0.25 major).

    Args:
        reference: Path to the baseline table.
        current: Path to the table to compare with it.
        bins: Quantile bins (from the reference) for the divergence measures.
    """
    from scipy import stats
    from scipy.spatial.distance import jensenshannon

    R, names = numeric_columns(reference, columns, None, None, 0)
    C = load_frame(current)
    require_columns(C, names)
    rows = []
    for c in names:
        r, cu = R[c].to_numpy(float), pd.to_numeric(C[c], errors="coerce").dropna().to_numpy(float)
        edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        pr = np.histogram(r, edges)[0] / len(r) + 1e-6
        pc = np.histogram(cu, edges)[0] / len(cu) + 1e-6
        psi = float(np.sum((pc - pr) * np.log(pc / pr)))
        rows.append({"column": c, "ks_distance": round(float(stats.ks_2samp(r, cu).statistic), 4), "js_divergence": round(float(jensenshannon(pr, pc) ** 2), 4), "psi": round(psi, 4),
                     "mean_shift_in_sd": round(float((cu.mean() - r.mean()) / (r.std() or 1)), 3), "spread_ratio": round(float(cu.std() / (r.std() or 1)), 3), "level": "major" if psi > 0.25 else "moderate" if psi > 0.1 else "stable"})
    rows.sort(key=lambda r: -r["psi"])
    return {"columns": rows, "major": [r["column"] for r in rows if r["level"] == "major"], "moderate": [r["column"] for r in rows if r["level"] == "moderate"], "overall": "major drift" if any(r["level"] == "major" for r in rows) else "moderate drift" if any(r["level"] == "moderate" for r in rows) else "stable"}


@tool
def estimate_entropy_and_divergence(data: str, other: str | None = None, columns: list[str] | None = None, drop: list[str] | None = None, n_neighbors: int = 5, max_rows: int = 5000, seed: int = 42) -> dict:
    """Information-theoretic summaries from nearest-neighbour distances (Kozachenko–Leonenko): the differential entropy of the data — a measure of how spread out the joint distribution is — and, given a second table, the KL divergence from the first to the second (estimated with the k-NN method of Wang et al.), a multivariate drift score with no binning.

    Args:
        other: Path to a second table with the same columns; adds the divergence.
        n_neighbors: Neighbours used by the estimators.
        max_rows: Rows sampled.
    """
    from scipy.special import digamma, gammaln
    from sklearn.neighbors import NearestNeighbors

    X, names = numeric_columns(data, columns, drop, max_rows, seed)
    A = ((X - X.mean()) / X.std().replace(0, 1)).to_numpy(float)
    n, d = A.shape
    r = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(A).kneighbors(A)[0][:, -1]
    log_vol = (d / 2) * np.log(np.pi) - gammaln(d / 2 + 1)
    entropy = float(digamma(n) - digamma(n_neighbors) + log_vol + d * np.mean(np.log(np.maximum(r, 1e-12))))
    out = {"rows": n, "columns": names, "differential_entropy_nats_standardised": round(entropy, 4), "entropy_if_independent_gaussian": round(float(0.5 * d * np.log(2 * np.pi * np.e)), 4)}
    if other:
        B = load_frame(other)[names].dropna().sample(min(max_rows, len(load_frame(other))), random_state=seed)
        B = ((B - X.mean()) / X.std().replace(0, 1)).to_numpy(float)
        rho = NearestNeighbors(n_neighbors=n_neighbors).fit(A).kneighbors(B, n_neighbors=n_neighbors)[0][:, -1]
        nu = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(B).kneighbors(B)[0][:, -1]
        kl = float(d * np.mean(np.log(np.maximum(rho, 1e-12) / np.maximum(nu, 1e-12))) + np.log(len(A) / (len(B) - 1)))
        out["kl_divergence_other_to_data"] = round(max(kl, 0.0), 4)
        out["drift"] = "negligible" if kl < 0.05 else "small" if kl < 0.2 else "substantial"
    return out
