"""Statistical detectors: z-score, robust (MAD) z-score, IQR fences, Grubbs' test, Mahalanobis distance, ECOD, HBOS and a Gaussian-mixture density. Fast, explainable and assumption-driven."""

from __future__ import annotations

import numpy as np

from ml_core.tooling import tool
from unsupervised.anomaly_detection.tools._common import run_detector


def _d(kind, label, data, columns, drop, scale, truth_column, contamination, params, seed, output_path, model_path, id_column, top, max_rows=None):
    return run_detector(kind, label, data, columns, drop, scale, truth_column, contamination, params, seed, output_path, model_path, id_column, top, max_rows)


@tool
def detect_zscore(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, contamination: float = 0.05, id_column: str | None = None, top: int = 15, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Z-score rule: score = the largest |z| across columns (how many standard deviations the most extreme value is from the mean). The simplest detector — fine for one roughly normal column, misled by the very outliers it looks for (they inflate the mean and SD).

    Args:
        truth_column: Column of known anomalies, used only to score the detector (never to fit).
        id_column: Column to show beside the flagged rows.
        top: Rows listed in the reply.
    """
    return _d("zscore", "zscore", data, columns, drop, True, truth_column, contamination, {}, seed, output_path, model_path, id_column, top)


@tool
def detect_robust_zscore(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, contamination: float = 0.05, id_column: str | None = None, top: int = 15, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Robust z-score: the median and the median absolute deviation replace the mean and SD, so a handful of extreme rows cannot hide themselves by widening the yardstick — the better univariate rule whenever outliers exist at all.

    Args:
        truth_column: Column of known anomalies, used only to score the detector.
        id_column: Column to show beside the flagged rows.
        top: Rows listed in the reply.
    """
    return _d("robust_zscore", "robust_zscore(MAD)", data, columns, drop, True, truth_column, contamination, {}, seed, output_path, model_path, id_column, top)


@tool
def detect_iqr_fences(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, contamination: float = 0.05, id_column: str | None = None, top: int = 15, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Tukey's fences: score = how many interquartile ranges a value lies beyond the quartiles, worst column wins (the classic box-plot rule uses 1.5). Distribution-free and robust, but univariate — blind to rows that are unusual only in combination.

    Args:
        truth_column: Column of known anomalies, used only to score the detector.
        id_column: Column to show beside the flagged rows.
        top: Rows listed in the reply.
    """
    return _d("iqr", "iqr_fences", data, columns, drop, False, truth_column, contamination, {}, seed, output_path, model_path, id_column, top)


@tool
def detect_grubbs(data: str, column: str, alpha: float = 0.05, max_outliers: int = 20, truth_column: str | None = None) -> dict:
    """Grubbs' test, applied repeatedly to one numeric column: removes the most extreme value while it is significantly far from the rest of a normal sample, and reports each removal with its test statistic and critical value. A formal univariate test, for small samples that are roughly normal.

    Args:
        column: The numeric column to test.
        alpha: Significance level.
        max_outliers: Most values to remove.
        truth_column: Column of known anomalies, used only to report how many removed rows were true anomalies.
    """
    from scipy import stats

    from ml_core.data import load_frame

    frame = load_frame(data)
    x = frame[column].astype(float).dropna()
    idx, found = list(x.index), []
    vals = x.to_numpy().copy()
    for _ in range(max_outliers):
        n = len(vals)
        if n < 7:
            break
        i = int(np.argmax(np.abs(vals - vals.mean())))
        g = abs(vals[i] - vals.mean()) / vals.std(ddof=1)
        t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        crit = (n - 1) / np.sqrt(n) * np.sqrt(t**2 / (n - 2 + t**2))
        if g <= crit:
            break
        found.append({"row": int(idx[i]), "value": float(vals[i]), "G": round(float(g), 3), "critical": round(float(crit), 3)})
        vals, idx = np.delete(vals, i), idx[:i] + idx[i + 1 :]
    out = {"column": column, "tested": int(len(x)), "outliers_found": len(found), "outliers": found, "assumption": "the remaining values are approximately normal — check a histogram first; skewed data produces false outliers"}
    if truth_column and truth_column in frame:
        t = frame.loc[[f["row"] for f in found], truth_column]
        out["true_anomalies_among_found"] = int((t.astype(str).isin(["1", "True", "true"]) | (pd_numeric(t) > 0)).sum()) if len(found) else 0
    return out


def pd_numeric(s):
    import pandas as pd

    return pd.to_numeric(s, errors="coerce").fillna(0)


@tool
def detect_mahalanobis(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, contamination: float = 0.05, robust: bool = True, id_column: str | None = None, top: int = 15, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Mahalanobis distance: how far a row is from the centre *accounting for the correlations between columns* — flags rows that break the usual relationships (a tall, light person) even when each column alone looks normal. The robust version estimates the centre and covariance from the cleanest subset, so the outliers cannot bend it.

    Args:
        truth_column: Column of known anomalies, used only to score the detector.
        robust: Use the minimum covariance determinant estimate instead of the plain covariance.
        id_column: Column to show beside the flagged rows.
        top: Rows listed in the reply.
    """
    return _d("mahalanobis", f"mahalanobis(robust={robust})", data, columns, drop, True, truth_column, contamination, {"robust": robust}, seed, output_path, model_path, id_column, top, 8000)


@tool
def detect_ecod(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, contamination: float = 0.05, id_column: str | None = None, top: int = 15, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """ECOD (empirical-cumulative-distribution outlier detection): estimates every column's tail probabilities from its empirical distribution and sums their −log — parameter-free, handles skewed columns, scales to many columns, and the per-column terms say exactly which columns made a row unusual.

    Args:
        truth_column: Column of known anomalies, used only to score the detector.
        id_column: Column to show beside the flagged rows.
        top: Rows listed in the reply.
    """
    return _d("ecod", "ecod", data, columns, drop, True, truth_column, contamination, {}, seed, output_path, model_path, id_column, top)


@tool
def detect_hbos(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, contamination: float = 0.05, bins: int | None = None, id_column: str | None = None, top: int = 15, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Histogram-based outlier score: one histogram per column, a row's score is the sum of −log of the density of the bin it falls in. Linear time — the fastest multivariate detector — but it assumes columns are independent.

    Args:
        truth_column: Column of known anomalies, used only to score the detector.
        bins: Histogram bins; the square root of the row count when omitted.
        id_column: Column to show beside the flagged rows.
        top: Rows listed in the reply.
    """
    return _d("hbos", "hbos", data, columns, drop, True, truth_column, contamination, {"bins": bins}, seed, output_path, model_path, id_column, top)


@tool
def detect_gmm_density(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, contamination: float = 0.05, n_components: int = 3, covariance_type: str = "full", id_column: str | None = None, top: int = 15, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Gaussian-mixture density: fit a mixture of Gaussians to the rows and score by negative log-likelihood — rows in low-density regions are anomalies, and several normal 'modes' (regimes, customer types) are allowed, unlike a single Mahalanobis ellipsoid.

    Args:
        truth_column: Column of known anomalies, used only to score the detector.
        n_components: Gaussian components (the number of normal modes).
        covariance_type: full, tied, diag or spherical.
        id_column: Column to show beside the flagged rows.
        top: Rows listed in the reply.
    """
    return _d("gmm", f"gmm_density(k={n_components})", data, columns, drop, True, truth_column, contamination, {"n_components": n_components, "covariance_type": covariance_type}, seed, output_path, model_path, id_column, top)
