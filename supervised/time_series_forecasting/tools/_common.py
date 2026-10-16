"""Forecasting machinery the tool modules share: loading a regular series, the forecaster registry,
rolling-origin backtesting and the error metrics. Every forecaster has the same shape —

    forecaster(train: pd.Series, horizon: int, period: int | None, **params) -> np.ndarray

— so any of them can be run once, backtested, or compared without the caller knowing how it works."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.ml import sk

PERIOD_BY_FREQ = {"D": 7, "B": 5, "H": 24, "h": 24, "W": 52, "M": 12, "ME": 12, "MS": 12, "Q": 4, "QE": 4, "QS": 4, "min": 60, "T": 60, "s": 60, "S": 60}


def load_series(data: Any, date_column: str, value_column: str, freq: str | None = None, fill: str = "interpolate") -> tuple[pd.Series, str, dict[str, Any]]:
    """Load one series onto a regular `DatetimeIndex`. Returns the series, its frequency, and what had to be repaired."""
    frame = load_frame(data)
    require_columns(frame, [date_column, value_column])
    frame = frame.assign(**{date_column: pd.to_datetime(frame[date_column], errors="coerce")})
    frame = frame.dropna(subset=[date_column]).sort_values(date_column)
    duplicates = int(frame[date_column].duplicated().sum())
    s = pd.Series(pd.to_numeric(frame[value_column], errors="coerce").to_numpy(), index=frame[date_column].to_numpy())
    s = s.groupby(level=0).mean()
    s.index = pd.DatetimeIndex(s.index)
    inferred = freq or pd.infer_freq(s.index) if len(s) > 2 else freq
    if not inferred:
        step = pd.Series(s.index).diff().dropna().median()
        inferred = pd.tseries.frequencies.to_offset(step).freqstr
    regular = s.asfreq(inferred)
    gaps = int(regular.isna().sum() - s.isna().sum())
    missing = int(regular.isna().sum())
    if missing:
        regular = regular.interpolate(limit_direction="both") if fill == "interpolate" else regular.ffill().bfill()
    return regular.astype(float), inferred, {"duplicate_timestamps_averaged": duplicates, "gap_steps_filled": max(gaps, 0), "missing_values_filled": missing}


def default_period(freq: str) -> int | None:
    base = freq.lstrip("0123456789")
    for key in (base, base.split("-")[0]):
        if key in PERIOD_BY_FREQ:
            return PERIOD_BY_FREQ[key]
    return None


def future_index(series: pd.Series, freq: str, horizon: int) -> pd.DatetimeIndex:
    offset = pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(series.index[-1] + offset, periods=horizon, freq=freq)


# ---- metrics ------------------------------------------------------------------------------------
def forecast_metrics(actual: np.ndarray, forecast: np.ndarray, train: np.ndarray | None = None, period: int | None = None) -> dict[str, float | None]:
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    err = a - f
    out: dict[str, float | None] = {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err**2))), "bias": float(np.mean(f - a))}
    nz = np.abs(a) > 1e-9
    out["mape"] = float(np.mean(np.abs(err[nz] / a[nz]))) if nz.any() else None
    denom = (np.abs(a) + np.abs(f)) / 2
    out["smape"] = float(np.mean(np.where(denom > 0, np.abs(err) / np.where(denom > 0, denom, 1), 0)))
    if train is not None and len(train) > (period or 1) + 1:
        m = period if period and period < len(train) else 1
        scale = np.mean(np.abs(train[m:] - train[:-m]))
        out["mase"] = float(np.mean(np.abs(err)) / scale) if scale > 0 else None
    return {k: (round(v, 5) if v is not None else None) for k, v in out.items()}


# ---- exponential smoothing (own implementation, no statsmodels needed) --------------------------
def _hw_run(y: np.ndarray, m: int, alpha: float, beta: float, gamma: float, phi: float, trend: bool, seasonal: str | None, init: tuple) -> tuple[float, tuple]:
    level, slope, seas = init[0], init[1], list(init[2])
    sse = 0.0
    for t, obs in enumerate(y):
        s = seas[t % m] if seasonal else (0.0 if seasonal != "mul" else 1.0)
        base = level + (phi * slope if trend else 0.0)
        pred = base * s if seasonal == "mul" else base + s
        sse += (obs - pred) ** 2
        if seasonal == "mul":
            new_level = alpha * (obs / s if s else obs) + (1 - alpha) * base
        else:
            new_level = alpha * (obs - s) + (1 - alpha) * base
        if trend:
            slope = beta * (new_level - level) + (1 - beta) * phi * slope
        if seasonal:
            seas[t % m] = gamma * (obs / new_level if seasonal == "mul" and new_level else obs - new_level) + (1 - gamma) * s
        level = new_level
    return sse, (level, slope, seas)


def exp_smoothing(y: np.ndarray, h: int, period: int | None, trend: bool = False, damped: bool = False, seasonal: str | None = None) -> np.ndarray:
    """Simple / Holt (damped) / Holt–Winters (additive or multiplicative) exponential smoothing, parameters fitted by SSE."""
    from scipy.optimize import minimize

    y = np.asarray(y, float)
    m = period if (seasonal and period and period > 1 and len(y) >= 2 * period) else 1
    seasonal = seasonal if m > 1 else None
    if seasonal == "mul" and (y <= 0).any():
        seasonal = "add"
    if m > 1:
        first = y[:m]
        level0 = first.mean()
        seas0 = (first / level0 if seasonal == "mul" else first - level0)
        slope0 = (y[m : 2 * m].mean() - first.mean()) / m if trend else 0.0
    else:
        level0, seas0, slope0 = y[0], np.zeros(1), (y[1] - y[0] if trend and len(y) > 1 else 0.0)

    def unpack(x: np.ndarray) -> tuple[float, float, float, float]:
        return x[0], (x[1] if trend else 0.0), (x[2] if seasonal else 0.0), (x[3] if damped else 1.0)

    def loss(x: np.ndarray) -> float:
        a, b, g, p = unpack(x)
        return _hw_run(y, m, a, b, g, p, trend, seasonal, (level0, slope0, seas0))[0]

    bounds = [(0.01, 0.99), (0.0, 0.5), (0.0, 0.5), (0.8, 0.98)]
    res = minimize(loss, [0.3, 0.1, 0.1, 0.95], bounds=bounds, method="L-BFGS-B")
    a, b, g, p = unpack(res.x)
    _, (level, slope, seas) = _hw_run(y, m, a, b, g, p, trend, seasonal, (level0, slope0, seas0))
    out = []
    for step in range(1, h + 1):
        damp = sum(p**i for i in range(1, step + 1)) if (trend and damped) else step
        base = level + (damp * slope if trend else 0.0)
        s = seas[(len(y) + step - 1) % m] if seasonal else None
        out.append(base * s if seasonal == "mul" else base + s if seasonal else base)
    return np.asarray(out)


def seasonal_indices(y: np.ndarray, period: int, multiplicative: bool = True) -> np.ndarray:
    """Classical seasonal indices by centred moving average, normalised to mean 1 (or 0 when additive)."""
    s = pd.Series(y)
    trend = s.rolling(period, center=True).mean()
    if period % 2 == 0:
        trend = trend.rolling(2, center=True).mean().shift(-1)
    detr = (s / trend) if multiplicative else (s - trend)
    idx = detr.groupby(np.arange(len(s)) % period).mean().to_numpy()
    return idx / np.nanmean(idx) if multiplicative else idx - np.nanmean(idx)


# ---- the forecasters ----------------------------------------------------------------------------
def f_naive(train: pd.Series, h: int, period: int | None, **_: Any) -> np.ndarray:
    return np.repeat(train.iloc[-1], h)


def f_mean(train: pd.Series, h: int, period: int | None, window: int | None = None, **_: Any) -> np.ndarray:
    return np.repeat(train.iloc[-window:].mean() if window else train.mean(), h)


def f_seasonal_naive(train: pd.Series, h: int, period: int | None, **_: Any) -> np.ndarray:
    if not period or period < 2 or len(train) < period:
        return f_naive(train, h, period)
    last = train.iloc[-period:].to_numpy()
    return np.array([last[i % period] for i in range(h)])


def f_drift(train: pd.Series, h: int, period: int | None, **_: Any) -> np.ndarray:
    slope = (train.iloc[-1] - train.iloc[0]) / max(len(train) - 1, 1)
    return train.iloc[-1] + slope * np.arange(1, h + 1)


def f_ses(train: pd.Series, h: int, period: int | None, **_: Any) -> np.ndarray:
    return exp_smoothing(train.to_numpy(), h, period)


def f_holt(train: pd.Series, h: int, period: int | None, damped: bool = False, **_: Any) -> np.ndarray:
    return exp_smoothing(train.to_numpy(), h, period, trend=True, damped=damped)


def f_holt_winters(train: pd.Series, h: int, period: int | None, trend: bool = True, damped: bool = False, seasonal: str = "add", **_: Any) -> np.ndarray:
    return exp_smoothing(train.to_numpy(), h, period, trend=trend, damped=damped, seasonal=seasonal)


def f_theta(train: pd.Series, h: int, period: int | None, **_: Any) -> np.ndarray:
    y = train.to_numpy(float)
    seasonal = bool(period and period > 1 and len(y) >= 2 * period and (y > 0).all())
    idx = seasonal_indices(y, period) if seasonal else None
    adj = y / idx[np.arange(len(y)) % period] if seasonal else y
    n = len(adj)
    b0 = np.polyfit(np.arange(n), adj, 1)[0]
    from scipy.optimize import minimize_scalar

    def sse(a: float) -> float:
        level, total = adj[0], 0.0
        for obs in adj[1:]:
            total += (obs - level) ** 2
            level = a * obs + (1 - a) * level
        return total

    alpha = float(minimize_scalar(sse, bounds=(0.01, 0.99), method="bounded").x)
    level = adj[0]
    for obs in adj[1:]:
        level = alpha * obs + (1 - alpha) * level
    steps = np.arange(1, h + 1)
    fc = level + 0.5 * b0 * ((steps - 1) + 1 / alpha - ((1 - alpha) ** n) / alpha)
    return fc * idx[(n + steps - 1) % period] if seasonal else fc


def f_harmonic(train: pd.Series, h: int, period: int | None, periods: list[float] | None = None, harmonics: int = 3, trend_degree: int = 1, alpha: float = 1.0, **_: Any) -> np.ndarray:
    sk()
    from sklearn.linear_model import Ridge

    cycles = [float(p) for p in (periods or ([period] if period and period > 1 else []))]
    n = len(train)

    def design(t: np.ndarray) -> np.ndarray:
        cols = [(t / max(n, 1)) ** d for d in range(1, trend_degree + 1)]
        for p in cycles:
            for k in range(1, harmonics + 1):
                cols += [np.sin(2 * np.pi * k * t / p), np.cos(2 * np.pi * k * t / p)]
        return np.column_stack(cols) if cols else np.zeros((len(t), 1))

    t_train = np.arange(n, dtype=float)
    model = Ridge(alpha=alpha).fit(design(t_train), train.to_numpy())
    return model.predict(design(np.arange(n, n + h, dtype=float)))


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"dow": index.dayofweek, "month": index.month, "dom": index.day, "hour": index.hour, "woy": index.isocalendar().week.to_numpy().astype(int)}, index=index)


def _estimator(name: str, seed: int = 0) -> Any:
    sk()
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge

    return {"ridge": lambda: Ridge(alpha=1.0), "random_forest": lambda: RandomForestRegressor(200, min_samples_leaf=2, n_jobs=-1, random_state=seed),
            "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=seed)}[name]()


def lag_matrix(values: np.ndarray, lags: list[int], windows: list[int]) -> tuple[np.ndarray, int]:
    """Rows t = max(lags)..n-1 with lag features and trailing-window means; returns (features, first_row_index)."""
    start = max(lags + windows + [1])
    rows = []
    for t in range(start, len(values)):
        rows.append([values[t - l] for l in lags] + [values[t - w : t].mean() for w in windows])
    return np.asarray(rows), start


def f_lag_regression(train: pd.Series, h: int, period: int | None, model: str = "hist_gradient_boosting", lags: list[int] | None = None, windows: list[int] | None = None,
                     calendar: bool = True, strategy: str = "recursive", **_: Any) -> np.ndarray:
    """Lag / rolling-mean / calendar features fed to a regressor — recursively (one model, feed predictions back) or directly (one model per step)."""
    values = train.to_numpy(float)
    n = len(values)
    lag_list = [l for l in (lags or sorted({1, 2, 3, *([period, 2 * period] if period and period > 1 else [])})) if 0 < l < n // 2]
    win_list = [w for w in (windows if windows is not None else ([period] if period and period > 1 else [])) if 1 < w < n // 2]
    start = max(lag_list + win_list + [1])
    freq = train.index.freqstr or pd.infer_freq(train.index) or "D"
    cal = calendar_features(train.index.append(future_index(train, freq, h))).to_numpy(float) if calendar else None

    def row(vals: np.ndarray, t: int, target_t: int) -> list[float]:
        base = [vals[t - l] for l in lag_list] + [vals[t - w : t].mean() for w in win_list]
        return base + (list(cal[target_t]) if cal is not None else [])

    if strategy == "direct":
        out = []
        for step in range(1, h + 1):
            X = np.asarray([row(values, t, t + step - 1) for t in range(start, n - step + 1)])
            y = values[start + step - 1 :]
            est = _estimator(model).fit(X, y)
            out.append(float(est.predict(np.asarray([row(values, n, n + step - 1)]))[0]))
        return np.asarray(out)

    est = _estimator(model).fit(np.asarray([row(values, t, t) for t in range(start, n)]), values[start:])
    history, out = values.copy(), []
    for step in range(h):
        t = len(history)
        pred = float(est.predict(np.asarray([row(history, t, n + step)]))[0])
        out.append(pred)
        history = np.append(history, pred)
    return np.asarray(out)


def f_arima(train: pd.Series, h: int, period: int | None, order: tuple[int, int, int] = (1, 1, 1), seasonal_order: tuple[int, int, int] | None = None, **_: Any) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    seas = (*seasonal_order, period or 1) if seasonal_order and period and period > 1 else (0, 0, 0, 0)
    fit = SARIMAX(train.to_numpy(), order=tuple(order), seasonal_order=seas, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return np.asarray(fit.forecast(h))


FORECASTERS: dict[str, Callable[..., np.ndarray]] = {
    "naive": f_naive, "mean": f_mean, "seasonal_naive": f_seasonal_naive, "drift": f_drift, "ses": f_ses, "holt": f_holt,
    "holt_winters": f_holt_winters, "theta": f_theta, "harmonic": f_harmonic, "lag_regression": f_lag_regression, "arima": f_arima,
}
NEEDS: dict[str, tuple[str, ...]] = {"arima": ("statsmodels",)}


def run_forecaster(kind: str, label: str, data: Any, date_column: str, value_column: str, horizon: int, period: int | None, holdout: int, freq: str | None,
                   output_path: str | None, **params: Any) -> dict[str, Any]:
    """Forecast `horizon` steps ahead, or — with `holdout` > 0 — hide the last `holdout` points, forecast them, and score."""
    series, inferred, repairs = load_series(data, date_column, value_column, freq)
    period = period or default_period(inferred)
    steps = holdout if holdout else horizon
    train = series.iloc[:-holdout] if holdout else series
    if len(train) < 8:
        raise ValueError(f"only {len(train)} points to train on; need at least 8")
    fc = np.asarray(FORECASTERS[kind](train, steps, period, **params), float)
    index = series.index[-holdout:] if holdout else future_index(series, inferred, steps)
    out: dict[str, Any] = {"model": label, "frequency": inferred, "period": period, "train_points": len(train), "repairs": repairs}
    if holdout:
        out["holdout_points"] = holdout
        out["holdout"] = forecast_metrics(series.iloc[-holdout:].to_numpy(), fc, train.to_numpy(), period)
        out["naive_reference"] = forecast_metrics(series.iloc[-holdout:].to_numpy(), f_naive(train, steps, period), train.to_numpy(), period)
    table = pd.DataFrame({date_column: index, "forecast": fc})
    if holdout:
        table["actual"] = series.iloc[-holdout:].to_numpy()
    out["forecast"] = table.head(12).assign(**{date_column: lambda d: d[date_column].astype(str)})
    out["forecast_points"] = len(table)
    out["written_to"] = save_frame(table, output_path, f"forecast_{kind}.csv") if output_path else None
    return out


def backtest(series: pd.Series, kind: str, horizon: int, n_splits: int, period: int | None, min_train: int | None = None, step: int | None = None, **params: Any) -> list[dict[str, Any]]:
    """Rolling-origin evaluation: grow the training window, forecast `horizon`, score, repeat."""
    n = len(series)
    min_train = min_train or max(int(n * 0.5), 2 * (period or 1) + 2)
    step = step or max(1, (n - min_train - horizon) // max(n_splits - 1, 1))
    origins = list(range(min_train, n - horizon + 1, step))[-n_splits:]
    rows = []
    for origin in origins:
        train, actual = series.iloc[:origin], series.iloc[origin : origin + horizon].to_numpy()
        fc = np.asarray(FORECASTERS[kind](train, horizon, period, **params), float)
        rows.append({"origin": str(series.index[origin - 1].date()) if hasattr(series.index[origin - 1], "date") else str(origin), "forecast": fc, "actual": actual,
                     **forecast_metrics(actual, fc, train.to_numpy(), period)})
    return rows
