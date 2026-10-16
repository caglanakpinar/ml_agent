"""Anomalies in a series over time: rolling statistics, seasonal-residual outliers, change points and windowed isolation forests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.tooling import tool


def _series(data, date_column, value_column) -> pd.Series:
    frame = load_frame(data)
    require_columns(frame, [date_column, value_column])
    frame = frame.assign(**{date_column: pd.to_datetime(frame[date_column], errors="coerce")}).dropna(subset=[date_column]).sort_values(date_column)
    s = pd.Series(pd.to_numeric(frame[value_column], errors="coerce").to_numpy(), index=pd.DatetimeIndex(frame[date_column]))
    return s.groupby(level=0).mean()


def _report(s: pd.Series, score: pd.Series, threshold: float, label: str, output_path, top: int, extra: dict | None = None) -> dict:
    flagged = score.abs() >= threshold
    worst = score.abs().sort_values(ascending=False).head(top)
    table = pd.DataFrame({"value": s, "score": score, "is_flagged": flagged.astype(int)})
    return {"detector": label, "points": int(len(s)), "threshold": round(float(threshold), 3), "flagged": int(flagged.sum()), "share": round(float(flagged.mean()), 4),
            "worst": [{"time": str(i), "value": float(s[i]), "score": round(float(score[i]), 2)} for i in worst.index], "written_to": save_frame(table.reset_index(names="time"), output_path, f"ts_anomalies_{label.split('(')[0]}.csv") if output_path else None, **(extra or {})}


@tool
def detect_ts_rolling_zscore(data: str, date_column: str, value_column: str, window: int = 24, threshold: float = 3.5, robust: bool = True, top: int = 15, output_path: str | None = None) -> dict:
    """Rolling z-score: each point is compared with the median (or mean) and spread of the *preceding* window only — a causal detector that could have run live, adapting to level shifts and trends, but blind to anomalies longer than the window.

    Args:
        window: Points in the trailing window.
        threshold: Score above which a point is flagged.
        robust: Median / MAD instead of mean / SD.
        top: Points listed.
    """
    s = _series(data, date_column, value_column)
    past = s.shift(1)
    if robust:
        centre = past.rolling(window, min_periods=max(5, window // 3)).median()
        spread = past.rolling(window, min_periods=max(5, window // 3)).apply(lambda x: np.median(np.abs(x - np.median(x))) / 0.6745 or np.nan, raw=True)
    else:
        centre, spread = past.rolling(window, min_periods=max(5, window // 3)).mean(), past.rolling(window, min_periods=max(5, window // 3)).std()
    score = ((s - centre) / spread.replace(0, np.nan)).dropna()
    return _report(s, score, threshold, f"rolling_zscore(window={window})", output_path, top)


@tool
def detect_ts_seasonal_residual(data: str, date_column: str, value_column: str, period: int | None = None, threshold: float = 3.5, top: int = 15, output_path: str | None = None) -> dict:
    """Remove trend and seasonality (classical decomposition, period found from the autocorrelation when omitted), then flag points whose *remainder* is extreme in robust units — the right detector when 'normal' means 'normal for this hour of the week', not 'close to the mean'.

    Args:
        period: Seasonal period in steps; detected when omitted.
        threshold: Robust-z score of the remainder above which a point is flagged.
        top: Points listed.
    """
    s = _series(data, date_column, value_column)
    y = s.to_numpy()
    if not period:
        d = y - np.polyval(np.polyfit(np.arange(len(y)), y, 1), np.arange(len(y)))
        acf = np.array([np.dot(d[: len(d) - k], d[k:]) for k in range(2, min(len(d) // 3, 400))])
        period = int(np.argmax(acf)) + 2 if len(acf) else 1
    trend = s.rolling(period, center=True, min_periods=1).median() if period > 1 else s.rolling(5, center=True, min_periods=1).median()
    detr = s - trend
    seasonal = detr.groupby(np.arange(len(s)) % period).transform("median") if period > 1 else pd.Series(0.0, index=s.index)
    resid = detr - seasonal
    mad = np.median(np.abs(resid - np.median(resid))) / 0.6745 or float(resid.std() or 1)
    score = (resid - np.median(resid)) / mad
    return _report(s, score, threshold, f"seasonal_residual(period={period})", output_path, top, {"period_used": int(period), "remainder_robust_sd": round(float(mad), 4)})


@tool
def detect_ts_change_points(data: str, date_column: str, value_column: str, min_size: int = 20, penalty: float | None = None, max_points: int = 5) -> dict:
    """Change points in the *level* of the series: greedy binary segmentation on the cumulative-sum statistic, accepting a split while it reduces the squared error by more than a BIC-style penalty. Finds where the series moved to a new regime — as opposed to single-point spikes — and gives the mean before and after each.

    Args:
        min_size: Shortest segment allowed.
        penalty: Required reduction in squared error per split; 3·log(n)·variance when omitted.
        max_points: Most change points reported.
    """
    s = _series(data, date_column, value_column)
    y = s.to_numpy()
    pen = penalty if penalty is not None else 3 * np.log(len(y)) * float(np.var(np.diff(y)) / 2 or np.var(y))

    def best_split(lo: int, hi: int):
        seg = y[lo:hi]
        if len(seg) < 2 * min_size:
            return None
        n = len(seg)
        best, gain = None, 0.0
        total = np.sum((seg - seg.mean()) ** 2)
        for k in range(min_size, n - min_size):
            left, right = seg[:k], seg[k:]
            cost = np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2)
            if total - cost > gain:
                best, gain = k, total - cost
        return (lo + best, gain) if best is not None and gain > pen else None

    segments, points = [(0, len(y))], []
    while len(points) < max_points:
        cands = [(sp, seg) for seg in segments if (sp := best_split(*seg))]
        if not cands:
            break
        (k, gain), seg = max(cands, key=lambda c: c[0][1])
        points.append((k, gain))
        segments.remove(seg)
        segments += [(seg[0], k), (k, seg[1])]
    points.sort()
    bounds = [0] + [k for k, _ in points] + [len(y)]
    return {"points": int(len(y)), "change_points": [{"time": str(s.index[k]), "index": int(k), "mean_before": round(float(y[max(0, k - min_size) : k].mean()), 4), "mean_after": round(float(y[k : k + min_size].mean()), 4), "error_reduction": round(float(g), 2)} for k, g in points],
            "segments": [{"from": str(s.index[a]), "to": str(s.index[b - 1]), "mean": round(float(y[a:b].mean()), 4), "std": round(float(y[a:b].std()), 4)} for a, b in zip(bounds[:-1], bounds[1:])], "penalty_used": round(float(pen), 3),
            "note": "a steady trend is reported as a staircase of small level shifts; detrend or difference the series first if only abrupt changes matter"}


@tool
def detect_ts_windowed_isolation_forest(data: str, date_column: str, value_column: str, window: int = 12, contamination: float = 0.02, seed: int = 42, top: int = 15, output_path: str | None = None) -> dict:
    """Turn the series into overlapping windows (the last `window` values, plus their mean, spread, slope and range) and run an isolation forest on them — catches anomalous *shapes* (a flat-line, a ramp, a burst), not only unusual single values.

    Args:
        window: Points per window.
        contamination: Share of windows flagged.
        top: Points listed.
    """
    from sklearn.ensemble import IsolationForest

    s = _series(data, date_column, value_column)
    y = s.to_numpy()
    rows = np.array([y[i - window : i] for i in range(window, len(y) + 1)])
    t = np.arange(window)
    feats = np.column_stack([rows, rows.mean(axis=1), rows.std(axis=1), np.array([np.polyfit(t, r, 1)[0] for r in rows]), rows.max(axis=1) - rows.min(axis=1)])
    feats = (feats - feats.mean(axis=0)) / np.where(feats.std(axis=0) > 0, feats.std(axis=0), 1)
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=seed, n_jobs=-1).fit(feats)
    score = pd.Series(-iso.score_samples(feats), index=s.index[window - 1 :])
    thr = float(np.quantile(score, 1 - contamination))
    return _report(s, score, thr, f"windowed_isolation_forest(window={window})", output_path, top)


@tool
def detect_ts_forecast_residual(data: str, date_column: str, value_column: str, period: int | None = None, lags: int = 24, threshold: float = 3.5, top: int = 15, output_path: str | None = None) -> dict:
    """Predict each point from its own history with a ridge regression on lagged values (and the seasonal lag), walking forward so every prediction uses only the past, and flag points whose forecast error is extreme in robust units. Flags what a good forecaster would not have expected — including anomalies inside a trend or seasonal pattern.

    Args:
        period: Seasonal period; its lag is added as a feature.
        lags: Number of recent values used.
        threshold: Robust-z of the forecast error above which a point is flagged.
        top: Points listed.
    """
    from sklearn.linear_model import Ridge

    s = _series(data, date_column, value_column)
    y = s.to_numpy()
    use = sorted(set(range(1, lags + 1)) | ({period} if period else set()))
    start = max(use)
    X = np.array([[y[t - l] for l in use] for t in range(start, len(y))])
    target = y[start:]
    cut = max(start + 20, len(target) // 3)
    pred = np.full(len(target), np.nan)
    for t in range(cut, len(target), max(1, len(target) // 200)):  # refit periodically, predict forward in between
        m = Ridge(alpha=1.0).fit(X[:t], target[:t])
        pred[t : t + max(1, len(target) // 200)] = m.predict(X[t : t + max(1, len(target) // 200)])
    err = pd.Series(target - pred, index=s.index[start:]).dropna()
    mad = np.median(np.abs(err - err.median())) / 0.6745 or float(err.std() or 1)
    score = (err - err.median()) / mad
    return _report(s, score, threshold, "forecast_residual", output_path, top, {"lags_used": len(use), "error_robust_sd": round(float(mad), 4)})
