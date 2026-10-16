"""Read-only checks on a series before any forecaster is chosen: shape, gaps, seasonality, trend, stationarity, autocorrelation, decomposition and outliers."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ml_core.tooling import tool
from supervised.time_series_forecasting.tools._common import default_period, load_series, seasonal_indices


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = np.asarray(x, float) - np.mean(x)
    denom = float(np.dot(x, x)) or 1.0
    return np.array([np.dot(x[: len(x) - k], x[k:]) / denom for k in range(nlags + 1)])


def _pacf(x: np.ndarray, nlags: int) -> np.ndarray:
    r = _acf(x, nlags)
    out = [1.0]
    for k in range(1, nlags + 1):
        R = np.array([[r[abs(i - j)] for j in range(k)] for i in range(k)])
        try:
            out.append(float(np.linalg.solve(R, r[1 : k + 1])[-1]))
        except np.linalg.LinAlgError:
            out.append(0.0)
    return np.array(out)


@tool
def profile_time_series(data: str, date_column: str, value_column: str, freq: str | None = None) -> dict:
    """Profile a series: length, date range, inferred frequency, duplicate timestamps, gaps and missing values, summary statistics, zeros and negatives, and the default seasonal period for that frequency.

    Args:
        freq: Pandas frequency alias (D, W, MS, h ...); inferred when omitted.
    """
    s, inferred, repairs = load_series(data, date_column, value_column, freq)
    return {"points": len(s), "start": str(s.index[0]), "end": str(s.index[-1]), "frequency": inferred, "default_period": default_period(inferred),
            "repairs_needed": repairs, "mean": s.mean(), "std": s.std(), "min": s.min(), "max": s.max(), "zeros": int((s == 0).sum()), "negatives": int((s < 0).sum()),
            "coefficient_of_variation": round(float(s.std() / s.mean()), 4) if s.mean() else None,
            "enough_for_seasonality": bool(default_period(inferred) and len(s) >= 2 * default_period(inferred)),
            "warnings": [w for w in [
                "fewer than 30 points: prefer naive / smoothing methods; ML models will overfit" if len(s) < 30 else None,
                "many zeros: intermittent demand — ordinary forecasters are biased here" if (s == 0).mean() > 0.3 else None,
                "values were interpolated across gaps — forecasts and errors depend on that guess" if repairs["missing_values_filled"] else None] if w]}


@tool
def detect_seasonality(data: str, date_column: str, value_column: str, max_period: int | None = None, freq: str | None = None) -> dict:
    """Find the seasonal period(s): autocorrelation peaks on the detrended series and the dominant FFT frequencies, compared with the period the frequency implies, and how strong the seasonal pattern is.

    Args:
        max_period: Largest period to consider; defaults to a third of the series.
    """
    s, inferred, _ = load_series(data, date_column, value_column, freq)
    y = s.to_numpy()
    detrended = y - np.polyval(np.polyfit(np.arange(len(y)), y, 1), np.arange(len(y)))
    top = max_period or len(y) // 3
    acf = _acf(detrended, top)
    peaks = [k for k in range(2, top) if acf[k] > acf[k - 1] and acf[k] >= acf[k + 1] and acf[k] > 0.2] if top > 3 else []
    spectrum = np.abs(np.fft.rfft(detrended)) ** 2
    freqs = np.fft.rfftfreq(len(y))
    order = np.argsort(spectrum[1:])[::-1][:5] + 1
    fft_periods = [round(1 / freqs[i], 2) for i in order if freqs[i] > 0 and 2 <= 1 / freqs[i] <= top]
    best = peaks[0] if peaks else None
    strength = None
    if best and len(y) >= 2 * best:
        idx = seasonal_indices(y, best, multiplicative=False)
        strength = round(float(1 - np.var(detrended - idx[np.arange(len(y)) % best]) / np.var(detrended)), 4)
    return {"frequency": inferred, "expected_period": default_period(inferred), "acf_peak_periods": peaks[:6], "fft_periods": fft_periods[:4],
            "recommended_period": best or default_period(inferred), "seasonal_strength": strength,
            "reading": "strength above ~0.6 is a strong seasonal pattern; a seasonal method will clearly help" if strength and strength > 0.6 else "weak or no seasonality — a non-seasonal method is enough" if strength is not None else "no clear seasonal peak"}


@tool
def check_trend_and_variance(data: str, date_column: str, value_column: str, freq: str | None = None) -> dict:
    """Check for a linear trend (slope, p-value, share of variance), a level shift between halves, and whether the spread grows with the level — which says if a log transform or multiplicative seasonality is needed.

    Args:
        freq: Pandas frequency alias; inferred when omitted.
    """
    from scipy import stats

    s, inferred, _ = load_series(data, date_column, value_column, freq)
    y, t = s.to_numpy(), np.arange(len(s))
    fit = stats.linregress(t, y)
    window = max(4, len(y) // 12)
    roll_mean, roll_std = s.rolling(window).mean().dropna(), s.rolling(window).std().dropna()
    corr = float(np.corrcoef(roll_mean.to_numpy()[: len(roll_std)], roll_std.to_numpy())[0, 1]) if len(roll_std) > 3 else 0.0
    half = len(y) // 2
    return {"slope_per_step": round(float(fit.slope), 6), "slope_p_value": round(float(fit.pvalue), 6), "trend_r2": round(float(fit.rvalue**2), 4),
            "first_half_mean": round(float(y[:half].mean()), 4), "second_half_mean": round(float(y[half:].mean()), 4),
            "level_vs_spread_correlation": round(corr, 4),
            "advice": [a for a in [
                "significant trend: use a method with a trend term (holt, holt_winters, arima d=1, harmonic trend_degree>=1) or drift" if fit.pvalue < 0.01 and fit.rvalue**2 > 0.1 else "no strong linear trend",
                "spread grows with the level: model the log of the series or use multiplicative seasonality (seasonal='mul')" if corr > 0.5 and (y > 0).all() else None] if a]}


@tool(requires=("statsmodels",))
def test_stationarity(data: str, date_column: str, value_column: str, freq: str | None = None) -> dict:
    """Augmented Dickey–Fuller and KPSS tests on the series and its first difference, and how many differences ARIMA-style models need.

    Args:
        freq: Pandas frequency alias; inferred when omitted.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    s, _, _ = load_series(data, date_column, value_column, freq)

    def verdict(x: np.ndarray) -> dict:
        adf_p, kpss_p = adfuller(x, autolag="AIC")[1], kpss(x, regression="c", nlags="auto")[1]
        return {"adf_p_value": round(float(adf_p), 5), "kpss_p_value": round(float(kpss_p), 5), "stationary": bool(adf_p < 0.05 and kpss_p > 0.05)}

    levels, diff1 = verdict(s.to_numpy()), verdict(np.diff(s.to_numpy()))
    return {"level": levels, "first_difference": diff1, "suggested_d": 0 if levels["stationary"] else 1 if diff1["stationary"] else 2,
            "reading": "ADF rejects a unit root (p<0.05) and KPSS does not reject stationarity (p>0.05) when the series is stationary"}


@tool
def autocorrelation_report(data: str, date_column: str, value_column: str, nlags: int = 24, difference: int = 0, freq: str | None = None) -> dict:
    """ACF and PACF with significance bounds, the significant lags, and a suggested AR/MA order — the classic reading for choosing ARIMA orders.

    Args:
        nlags: How many lags.
        difference: Differences to apply before computing (0, 1 or 2).
    """
    s, _, _ = load_series(data, date_column, value_column, freq)
    y = s.to_numpy()
    for _ in range(difference):
        y = np.diff(y)
    nlags = min(nlags, len(y) // 2 - 1)
    acf, pacf, bound = _acf(y, nlags), _pacf(y, nlags), 1.96 / np.sqrt(len(y))
    sig_acf = [k for k in range(1, nlags + 1) if abs(acf[k]) > bound]
    sig_pacf = [k for k in range(1, nlags + 1) if abs(pacf[k]) > bound]
    ar = max([k for k in sig_pacf if k <= 5], default=0)
    ma = max([k for k in sig_acf if k <= 5], default=0)
    return {"nlags": nlags, "bound_95": round(float(bound), 4), "acf": acf.round(3).tolist(), "pacf": pacf.round(3).tolist(),
            "significant_acf_lags": sig_acf[:15], "significant_pacf_lags": sig_pacf[:15], "suggested_p": ar, "suggested_q": ma,
            "reading": "PACF cuts off after p lags and ACF decays → AR(p); ACF cuts off after q lags and PACF decays → MA(q); slow-decaying ACF → difference first"}


@tool
def decompose_series(data: str, date_column: str, value_column: str, period: int | None = None, model: Literal["additive", "multiplicative"] = "additive", freq: str | None = None, output_path: str | None = None) -> dict:
    """Classical decomposition into trend, seasonal and remainder, with the strength of each and the seasonal profile — shows what a forecaster has to capture.

    Args:
        model: additive, or multiplicative when the seasonal swing grows with the level.
    """
    from ml_core.data import save_frame

    s, inferred, _ = load_series(data, date_column, value_column, freq)
    m = period or default_period(inferred)
    if not m or m < 2 or len(s) < 2 * m:
        raise ValueError("need a seasonal period and at least two full periods to decompose")
    mult = model == "multiplicative" and (s > 0).all()
    idx = seasonal_indices(s.to_numpy(), m, multiplicative=mult)
    seasonal = pd.Series(idx[np.arange(len(s)) % m], index=s.index)
    deseason = s / seasonal if mult else s - seasonal
    trend = deseason.rolling(m, center=True, min_periods=1).mean()
    resid = s / (trend * seasonal) if mult else s - trend - seasonal
    var = lambda x: float(np.nanvar(x))  # noqa: E731
    strength_season = max(0.0, 1 - var(resid) / var(resid + seasonal - (1 if mult else 0))) if not mult else max(0.0, 1 - var(resid) / var(seasonal * trend))
    strength_trend = max(0.0, 1 - var(resid) / var(resid + trend - (1 if mult else 0))) if not mult else max(0.0, 1 - var(resid) / var(trend * resid))
    table = pd.DataFrame({"observed": s, "trend": trend, "seasonal": seasonal, "remainder": resid})
    return {"period": m, "model": "multiplicative" if mult else "additive", "trend_strength": round(strength_trend, 4), "seasonal_strength": round(strength_season, 4),
            "seasonal_profile": [round(float(v), 4) for v in idx], "remainder_std": round(float(np.nanstd(resid)), 5),
            "written_to": save_frame(table.reset_index(names=date_column), output_path, "decomposition.csv") if output_path else None}


@tool
def detect_series_outliers(data: str, date_column: str, value_column: str, window: int = 14, threshold: float = 3.5, freq: str | None = None, k: int = 20) -> dict:
    """Flag points far from a rolling median, in robust (MAD) units — spikes and drops that will distort a forecaster unless treated.

    Args:
        window: Rolling window length.
        threshold: Robust z-score above which a point is flagged.
        k: How many outliers to list.
    """
    s, _, _ = load_series(data, date_column, value_column, freq)
    median = s.rolling(window, center=True, min_periods=3).median()
    mad = (s - median).abs().rolling(window, center=True, min_periods=3).median().replace(0, np.nan)
    z = 0.6745 * (s - median) / mad
    flagged = z[z.abs() > threshold].sort_values(key=abs, ascending=False)
    return {"window": window, "outliers": int(len(flagged)), "share": round(len(flagged) / len(s), 4),
            "worst": [{"date": str(i), "value": float(s[i]), "robust_z": round(float(v), 2)} for i, v in flagged.head(k).items()]}
