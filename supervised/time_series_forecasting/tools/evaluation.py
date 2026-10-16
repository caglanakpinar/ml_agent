"""Judge forecasters honestly: rolling-origin backtests, head-to-head comparison, error by horizon, residual diagnostics, empirical prediction intervals, and scoring an existing forecast file."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns
from ml_core.tooling import tool
from supervised.time_series_forecasting.tools._common import (
    FORECASTERS,
    NEEDS,
    backtest,
    default_period,
    forecast_metrics,
    load_series,
)

Method = Literal["naive", "mean", "seasonal_naive", "drift", "ses", "holt", "holt_winters", "theta", "harmonic", "lag_regression", "arima"]


def _params(method: str, extra: dict[str, Any] | None) -> dict[str, Any]:
    defaults: dict[str, dict[str, Any]] = {"holt_winters": {"seasonal": "add"}, "lag_regression": {"model": "ridge"}, "arima": {"order": (1, 1, 1)}}
    return {**defaults.get(method, {}), **(extra or {})}


def _summary(rows: list[dict]) -> dict[str, float | None]:
    out = {}
    for key in ("mae", "rmse", "mape", "smape", "mase"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = round(float(np.mean(vals)), 5) if vals else None
    return out


def _check_deps(method: str) -> None:
    import importlib.util

    for pkg in NEEDS.get(method, ()):
        if importlib.util.find_spec(pkg) is None:
            raise ImportError(f"{method} needs {pkg}")


@tool
def backtest_forecaster(
    data: str, date_column: str, value_column: str, method: Method = "theta", horizon: int = 12, n_splits: int = 5,
    min_train: int | None = None, period: int | None = None, params: dict | None = None, freq: str | None = None,
) -> dict:
    """Rolling-origin backtest of one forecaster: train on a growing window, forecast `horizon`, score, repeat — the honest estimate of out-of-sample error, with MASE (below 1 beats a naive forecast).

    Args:
        method: Forecaster to evaluate.
        n_splits: Number of forecast origins.
        min_train: Smallest training window; half the series by default.
        params: Extra arguments for the forecaster, e.g. {"model": "ridge"} or {"order": [2, 1, 1]}.
    """
    _check_deps(method)
    s, inferred, _ = load_series(data, date_column, value_column, freq)
    m = period or default_period(inferred)
    rows = backtest(s, method, horizon, n_splits, m, min_train, None, **_params(method, params))
    return {"method": method, "horizon": horizon, "origins": len(rows), "average": _summary(rows),
            "by_origin": [{k: v for k, v in r.items() if k not in ("forecast", "actual")} for r in rows],
            "verdict": "beats a seasonal-naive-scale forecast" if (_summary(rows).get("mase") or 9) < 1 else "no better than a naive forecast — a simpler method may do as well"}


@tool
def compare_forecasters(
    data: str, date_column: str, value_column: str, methods: list[str] | None = None, horizon: int = 12, n_splits: int = 4,
    period: int | None = None, freq: str | None = None, metric: Literal["mase", "smape", "rmse", "mae"] = "mase",
) -> dict:
    """Backtest several forecasters over the same origins and rank them — run this before committing to one method; the winner is often a simple one.

    Args:
        methods: Forecasters to compare (naive, mean, seasonal_naive, drift, ses, holt, holt_winters, theta, harmonic, lag_regression, arima). Defaults to seven that need no extra packages.
        n_splits: Number of forecast origins.
        metric: Metric to rank by; MASE is scale-free and comparable across series.
    """
    s, inferred, _ = load_series(data, date_column, value_column, freq)
    m = period or default_period(inferred)
    chosen = methods or ["naive", "seasonal_naive", "drift", "ses", "holt_winters", "theta", "harmonic"]
    board = []
    for method in chosen:
        if method not in FORECASTERS:
            board.append({"method": method, "error": f"unknown; choose from {sorted(FORECASTERS)}"})
            continue
        try:
            _check_deps(method)
            rows = backtest(s, method, horizon, n_splits, m, None, None, **_params(method, None))
            board.append({"method": method, **_summary(rows)})
        except Exception as error:
            board.append({"method": method, "error": str(error)[:120]})
    board.sort(key=lambda r: r.get(metric) if r.get(metric) is not None else np.inf)
    return {"metric": metric, "horizon": horizon, "origins": n_splits, "leaderboard": board, "best": board[0]["method"] if board and board[0].get(metric) is not None else None}


@tool
def error_by_horizon(
    data: str, date_column: str, value_column: str, method: Method = "theta", horizon: int = 12, n_splits: int = 5,
    period: int | None = None, params: dict | None = None, freq: str | None = None,
) -> dict:
    """Backtest error at each step ahead (1 … horizon) — shows how fast accuracy decays, and the longest horizon at which the forecast is still better than naive.

    Args:
        method: Forecaster to evaluate.
        n_splits: Number of forecast origins.
        params: Extra arguments for the forecaster.
    """
    _check_deps(method)
    s, inferred, _ = load_series(data, date_column, value_column, freq)
    m = period or default_period(inferred)
    rows = backtest(s, method, horizon, n_splits, m, None, None, **_params(method, params))
    naive = backtest(s, "naive", horizon, n_splits, m, None, None)
    err = np.mean([np.abs(r["forecast"] - r["actual"]) for r in rows], axis=0)
    base = np.mean([np.abs(r["forecast"] - r["actual"]) for r in naive], axis=0)
    better = [h + 1 for h in range(horizon) if err[h] < base[h]]
    return {"method": method, "mae_by_step": [round(float(x), 4) for x in err], "naive_mae_by_step": [round(float(x), 4) for x in base],
            "steps_better_than_naive": better, "reading": "the forecast adds value over naive only at the listed steps"}


@tool
def residual_diagnostics(
    data: str, date_column: str, value_column: str, method: Method = "theta", lags: int = 10, period: int | None = None,
    params: dict | None = None, freq: str | None = None,
) -> dict:
    """One-step-ahead residuals of a method: bias, autocorrelation (Ljung–Box) and normality. Autocorrelated residuals mean signal is left on the table — the model can still be improved.

    Args:
        method: Forecaster to diagnose.
        lags: Lags used in the Ljung–Box test.
        params: Extra arguments for the forecaster.
    """
    from scipy import stats

    _check_deps(method)
    s, inferred, _ = load_series(data, date_column, value_column, freq)
    m = period or default_period(inferred)
    start = max(int(len(s) * 0.5), 2 * (m or 1) + 2)
    resid = []
    for t in range(start, len(s) - 1, max(1, (len(s) - start) // 40)):
        resid.append(s.iloc[t] - FORECASTERS[method](s.iloc[:t], 1, m, **_params(method, params))[0])
    r = np.asarray(resid)
    n = len(r)
    nl = min(lags, n // 3)
    x = r - r.mean()
    acf = np.array([np.dot(x[: n - k], x[k:]) / np.dot(x, x) for k in range(1, nl + 1)])
    q = float(n * (n + 2) * np.sum(acf**2 / (n - np.arange(1, nl + 1))))
    p = float(stats.chi2.sf(q, nl))
    return {"method": method, "residuals": n, "mean": round(float(r.mean()), 5), "std": round(float(r.std()), 5), "ljung_box_q": round(q, 3), "ljung_box_p": round(p, 5),
            "skew": round(float(stats.skew(r)), 4), "autocorrelated": bool(p < 0.05),
            "reading": "residuals still autocorrelated: the method misses structure (try adding trend/seasonality or ARIMA terms)" if p < 0.05 else "residuals look like white noise: little structure left to model"}


@tool
def empirical_prediction_intervals(
    data: str, date_column: str, value_column: str, method: Method = "theta", horizon: int = 12, coverage: float = 0.8, n_splits: int = 8,
    period: int | None = None, params: dict | None = None, freq: str | None = None, output_path: str | None = None,
) -> dict:
    """Forecast with prediction intervals taken from backtest errors at each step ahead — distribution-free and valid for any method, unlike model-based intervals that assume normal errors.

    Args:
        method: Forecaster to use.
        coverage: Target share of future values inside the interval, e.g. 0.8.
        n_splits: Backtest origins used to measure the errors.
        params: Extra arguments for the forecaster.
    """
    from ml_core.data import save_frame
    from supervised.time_series_forecasting.tools._common import future_index

    _check_deps(method)
    s, inferred, _ = load_series(data, date_column, value_column, freq)
    m = period or default_period(inferred)
    p = _params(method, params)
    rows = backtest(s, method, horizon, n_splits, m, None, None, **p)
    errors = np.array([r["actual"] - r["forecast"] for r in rows])
    lo_q, hi_q = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    fc = np.asarray(FORECASTERS[method](s, horizon, m, **p), float)
    lower, upper = fc + np.quantile(errors, lo_q, axis=0), fc + np.quantile(errors, hi_q, axis=0)
    table = pd.DataFrame({date_column: future_index(s, inferred, horizon), "forecast": fc, "lower": lower, "upper": upper})
    inside = np.mean((np.array([r["actual"] for r in rows]) >= np.array([r["forecast"] for r in rows]) + np.quantile(errors, lo_q, axis=0)) &
                     (np.array([r["actual"] for r in rows]) <= np.array([r["forecast"] for r in rows]) + np.quantile(errors, hi_q, axis=0)))
    return {"method": method, "target_coverage": coverage, "backtest_coverage": round(float(inside), 3), "origins": len(rows),
            "forecast": table.head(12).assign(**{date_column: lambda d: d[date_column].astype(str)}), "written_to": save_frame(table, output_path, "forecast_intervals.csv") if output_path else None}


@tool
def score_forecast_file(data: str, actual_column: str, forecast_column: str, train_column: str | None = None, period: int | None = None) -> dict:
    """Score a forecast you already have: MAE, RMSE, MAPE, sMAPE, bias (and MASE when the training history is given) from a table holding actual and forecast columns.

    Args:
        actual_column: Column holding the observed values.
        forecast_column: Column holding the forecasts.
        train_column: Column holding the training history (for MASE); optional.
    """
    frame = load_frame(data)
    require_columns(frame, [actual_column, forecast_column] + ([train_column] if train_column else []))
    both = frame[[actual_column, forecast_column]].dropna()
    train = frame[train_column].dropna().to_numpy(float) if train_column else None
    return {"points": len(both), "metrics": forecast_metrics(both[actual_column].to_numpy(float), both[forecast_column].to_numpy(float), train, period)}
