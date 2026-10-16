"""Read-only checks before estimating a density: shape, modality, normality, bounds, discreteness, tails and — for several columns — dependence."""

from __future__ import annotations

import numpy as np

from ml_core.tooling import tool
from unsupervised.density_estimation.tools._common import column_values, grid_for, modes_of, numeric_columns


@tool
def profile_distribution(data: str, column: str) -> dict:
    """Profile one numeric column as a distribution: size, mean / median / SD, skewness and kurtosis, quantiles, support (min, max, share of exact zeros and of integers), the share of IQR outliers and tail heaviness — the facts that decide which density estimator and what boundary handling are sensible."""
    from scipy import stats

    x = column_values(data, column)
    q = np.quantile(x, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    iqr = q[4] - q[2]
    ints = float(np.mean(np.isclose(x, np.round(x))))
    return {"n": len(x), "mean": float(x.mean()), "median": float(q[3]), "std": float(x.std()), "skew": round(float(stats.skew(x)), 3), "excess_kurtosis": round(float(stats.kurtosis(x)), 3),
            "quantiles": dict(zip(["1%", "5%", "25%", "50%", "75%", "95%", "99%"], np.round(q, 4).tolist())), "min": float(x.min()), "max": float(x.max()),
            "zero_share": round(float((x == 0).mean()), 4), "integer_share": round(ints, 4), "distinct_values": int(len(np.unique(x))), "iqr_outlier_share": round(float(((x < q[2] - 1.5 * iqr) | (x > q[4] + 1.5 * iqr)).mean()), 4),
            "advice": [a for a in ["all values are non-negative: use a bounded estimator (reflection KDE, gamma / lognormal) so density does not leak below zero" if x.min() >= 0 else None,
                                   "mostly integers with few distinct values: treat it as discrete (a histogram or counts model), not a continuous density" if ints > 0.98 and len(np.unique(x)) < 30 else None,
                                   "many exact zeros: model the zero mass separately from the continuous part (zero-inflated)" if (x == 0).mean() > 0.1 else None,
                                   "heavy right tail: consider a log transform or a tail model (fit_tail_distribution)" if stats.skew(x) > 2 else None] if a]}


@tool
def check_modality(data: str, column: str, max_rows: int = 20000) -> dict:
    """How many peaks (modes) does the distribution have? Counts KDE peaks across a range of bandwidths (a peak that survives wide smoothing is real), and the bimodality coefficient — one mode means a single parametric family may do; several mean a mixture or a non-parametric estimate is needed.

    Args:
        max_rows: Values sampled when the column is large.
    """
    from scipy import stats

    x = column_values(data, column, max_rows)
    z = (x - x.mean()) / (x.std() or 1)
    g = grid_for(z, 400)
    counts = {}
    for h in (0.1, 0.2, 0.3, 0.5, 0.8):
        kde = stats.gaussian_kde(z, bw_method=h / z.std() if z.std() else h)
        counts[h] = len(modes_of(g, kde(g), 0.03))
    sk, ku = stats.skew(x), stats.kurtosis(x)
    bc = (sk**2 + 1) / (ku + 3 * (len(x) - 1) ** 2 / ((len(x) - 2) * (len(x) - 3)))
    robust = max(set(counts.values()), key=list(counts.values()).count)
    return {"peaks_by_bandwidth": counts, "robust_peak_count": int(robust), "bimodality_coefficient": round(float(bc), 4), "bimodal_if_above": 0.555,
            "verdict": "unimodal: a parametric family or a single-bandwidth KDE is adequate" if robust == 1 else f"{robust} modes persist across bandwidths: use a mixture ({robust} components) or an adaptive KDE"}


@tool
def test_normality(data: str, column: str, max_rows: int = 5000, seed: int = 42) -> dict:
    """Is the column normal? Shapiro–Wilk, D'Agostino–Pearson, Anderson–Darling and the Q–Q correlation, on the raw values and on their logarithm — and which of the two is closer to normal. With large samples the tests reject trivial departures, so read the Q–Q correlation (≥ 0.99 is close) alongside the p-values.

    Args:
        max_rows: Values sampled for the tests.
    """
    from scipy import stats

    x = column_values(data, column, max_rows, seed)

    def run(v: np.ndarray) -> dict:
        qq = float(np.corrcoef(np.sort(v), stats.norm.ppf((np.arange(1, len(v) + 1) - 0.5) / len(v)))[0, 1])
        ad = stats.anderson(v, "norm")
        return {"shapiro_p": round(float(stats.shapiro(v[:5000]).pvalue), 5), "dagostino_p": round(float(stats.normaltest(v).pvalue), 5), "anderson_statistic": round(float(ad.statistic), 3),
                "anderson_5pct_critical": round(float(ad.critical_values[2]), 3), "qq_correlation": round(qq, 4)}

    out = {"n": len(x), "raw": run(x)}
    if (x > 0).all():
        out["log_transformed"] = run(np.log(x))
        out["closer_to_normal"] = "log" if out["log_transformed"]["qq_correlation"] > out["raw"]["qq_correlation"] else "raw"
    out["verdict"] = "close to normal: a Gaussian is a fair model" if out["raw"]["qq_correlation"] >= 0.995 else "not normal: use a skewed / heavy-tailed family, a transform, a mixture or a KDE"
    return out


@tool
def check_tails(data: str, column: str, tail_fraction: float = 0.05) -> dict:
    """Tail behaviour: the Hill estimator of the tail index on each side (smaller = heavier, below ~2 means infinite variance), the share of the total carried by the top 1%, and a comparison with the tail a normal fit would give — tells whether a Gaussian will badly under-state extreme events.

    Args:
        tail_fraction: Share of the largest (and smallest) values used for the tail estimates.
    """
    from scipy import stats

    x = column_values(data, column)
    k = max(10, int(len(x) * tail_fraction))

    def hill(v: np.ndarray) -> float | None:
        v = np.sort(v[v > 0])[::-1]
        return float(1 / np.mean(np.log(v[:k] / v[k]))) if len(v) > k + 1 and v[k] > 0 else None

    z = (x - x.mean()) / (x.std() or 1)
    out = {"right_tail_index": hill(x - x.min() + 1e-9), "left_tail_index": hill(x.max() - x + 1e-9), "share_beyond_3_sd": round(float((np.abs(z) > 3).mean()), 5), "share_beyond_3_sd_if_normal": round(float(2 * stats.norm.sf(3)), 5),
           "top_1pct_share_of_total": round(float(np.sort(x)[-max(1, len(x) // 100):].sum() / x.sum()), 4) if x.min() >= 0 and x.sum() > 0 else None}
    ratio = out["share_beyond_3_sd"] / out["share_beyond_3_sd_if_normal"]
    out["tail_heaviness_vs_normal"] = round(float(ratio), 2)
    out["verdict"] = "tails are about as thin as a normal's" if ratio < 1.5 else "tails are clearly heavier than a normal's: a Gaussian will under-state extremes — use t / Laplace / lognormal / a GPD tail model"
    return out


@tool
def check_dependence_structure(data: str, columns: list[str] | None = None, drop: list[str] | None = None, max_rows: int = 5000, seed: int = 42) -> dict:
    """For several columns: Pearson vs Spearman correlation, the mutual information of each pair, and the correlation of the extremes (do large values occur together more than the bulk suggests?) — decides between independent marginals, a Gaussian copula or a full multivariate mixture / KDE.

    Args:
        max_rows: Rows sampled.
    """
    from sklearn.feature_selection import mutual_info_regression

    X, names = numeric_columns(data, columns, drop, max_rows, seed, minimum=2)
    pear, spear = X.corr(), X.corr(method="spearman")
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            mi = float(mutual_info_regression(X[[a]].to_numpy(), X[b].to_numpy(), random_state=seed)[0])
            top = (X[a] > X[a].quantile(0.9)) & (X[b] > X[b].quantile(0.9))
            joint = float(top.mean())
            rows.append({"a": a, "b": b, "pearson": round(float(pear.loc[a, b]), 3), "spearman": round(float(spear.loc[a, b]), 3), "mutual_information": round(mi, 4),
                         "joint_top_decile_vs_independent": round(joint / 0.01, 2), "nonlinear_dependence": bool(mi > 0.1 and abs(pear.loc[a, b]) < 0.3)})
    rows.sort(key=lambda r: -r["mutual_information"])
    return {"pairs": rows[:15], "advice": "dependence is weak: independent marginals are a fair model" if all(r["mutual_information"] < 0.03 for r in rows) else
            "non-linear dependence present: a Gaussian copula or mixture is not enough alone — use a KDE / k-NN density" if any(r["nonlinear_dependence"] for r in rows) else "monotone dependence: a Gaussian copula with flexible marginals models it well"}
