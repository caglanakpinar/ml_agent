"""Forecasting as supervised learning on lag, rolling-window and calendar features — recursive or direct, with ridge, random forest or gradient boosting — plus bulk and feature-building helpers."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.tooling import tool
from supervised.time_series_forecasting.tools._common import (
    FORECASTERS,
    calendar_features,
    default_period,
    forecast_metrics,
    load_series,
    run_forecaster,
)

Learner = Literal["ridge", "random_forest", "hist_gradient_boosting"]


@tool
def forecast_lag_regression(
    data: str, date_column: str, value_column: str, horizon: int = 12, model: Learner = "hist_gradient_boosting",
    lags: list[int] | None = None, windows: list[int] | None = None, calendar: bool = True,
    holdout: int = 0, period: int | None = None, freq: str | None = None, output_path: str | None = None,
) -> dict:
    """Recursive ML forecast: one model learns the next value from lags, rolling means and calendar fields, and its own predictions are fed back to extend the horizon. Captures non-linear and calendar effects; errors compound over long horizons.

    Args:
        model: ridge, random_forest or hist_gradient_boosting.
        lags: Lags to use as features; defaults to 1, 2, 3 and the seasonal lags.
        windows: Trailing-mean window lengths; defaults to the seasonal period.
        calendar: Add day-of-week, month, day-of-month, hour and week-of-year features.
        holdout: Hide the last N points, forecast them and report errors; 0 forecasts the real future.
    """
    return run_forecaster("lag_regression", f"lag_regression({model},recursive)", data, date_column, value_column, horizon, period, holdout, freq, output_path,
                          model=model, lags=lags, windows=windows, calendar=calendar, strategy="recursive")


@tool
def forecast_direct_multi_horizon(
    data: str, date_column: str, value_column: str, horizon: int = 12, model: Learner = "ridge",
    lags: list[int] | None = None, windows: list[int] | None = None, calendar: bool = True,
    holdout: int = 0, period: int | None = None, freq: str | None = None, output_path: str | None = None,
) -> dict:
    """Direct ML forecast: a separate model per step ahead, each trained to predict exactly that step from the latest history. No error compounding, at the cost of training `horizon` models.

    Args:
        model: ridge, random_forest or hist_gradient_boosting.
        lags: Lags to use as features; defaults to 1, 2, 3 and the seasonal lags.
        windows: Trailing-mean window lengths; defaults to the seasonal period.
        calendar: Add calendar features of the target date.
        holdout: Hide the last N points, forecast them and report errors; 0 forecasts the real future.
    """
    return run_forecaster("lag_regression", f"lag_regression({model},direct)", data, date_column, value_column, horizon, period, holdout, freq, output_path,
                          model=model, lags=lags, windows=windows, calendar=calendar, strategy="direct")


@tool
def build_lag_features(
    data: str, date_column: str, value_column: str, lags: list[int] | None = None, windows: list[int] | None = None,
    calendar: bool = True, freq: str | None = None, output_path: str | None = None,
) -> dict:
    """Write a model-ready table of lag, rolling-mean and calendar features for the series (rows with incomplete history dropped) — for fitting your own model or joining exogenous variables.

    Args:
        lags: Lags to create; defaults to 1, 2, 3 and the seasonal lags.
        windows: Trailing-mean window lengths; defaults to the seasonal period.
        calendar: Add calendar fields.
    """
    s, inferred, _ = load_series(data, date_column, value_column, freq)
    m = default_period(inferred)
    lag_list = lags or sorted({1, 2, 3, *([m, 2 * m] if m and m > 1 else [])})
    win_list = windows if windows is not None else ([m] if m and m > 1 else [])
    table = pd.DataFrame({"y": s})
    for l in lag_list:
        table[f"lag_{l}"] = s.shift(l)
    for w in win_list:
        table[f"roll_mean_{w}"] = s.shift(1).rolling(w).mean()
        table[f"roll_std_{w}"] = s.shift(1).rolling(w).std()
    if calendar:
        table = table.join(calendar_features(s.index))
    table = table.dropna().reset_index(names=date_column)
    return {"rows": len(table), "columns": list(table.columns), "written_to": save_frame(table, output_path, "lag_features.csv"), "sample": table.head(3).assign(**{date_column: lambda d: d[date_column].astype(str)})}


@tool
def forecast_many_series(
    data: str, date_column: str, value_column: str, series_column: str, horizon: int = 12,
    method: Literal["naive", "seasonal_naive", "drift", "ses", "holt", "holt_winters", "theta", "harmonic"] = "theta",
    holdout: int = 0, freq: str | None = None, output_path: str | None = None, max_series: int = 500,
) -> dict:
    """Forecast every series in a long table (one row per series per date) with one cheap method, writing a single long output and per-series error when `holdout` is set — the practical route for thousands of SKUs or stores.

    Args:
        series_column: Column identifying which series each row belongs to.
        method: Forecaster applied to every series.
        holdout: Hide the last N points of each series and score them; 0 forecasts the real future.
        max_series: Safety cap on how many series to process.
    """
    frame = load_frame(data)
    require_columns(frame, [date_column, value_column, series_column])
    rows, errors, skipped = [], [], 0
    for name, part in list(frame.groupby(series_column))[:max_series]:
        try:
            s, inferred, _ = load_series(part[[date_column, value_column]], date_column, value_column, freq)
            period = default_period(inferred)
            train = s.iloc[:-holdout] if holdout else s
            steps = holdout or horizon
            if len(train) < 8:
                skipped += 1
                continue
            fc = np.asarray(FORECASTERS[method](train, steps, period), float)
            idx = s.index[-holdout:] if holdout else pd.date_range(s.index[-1] + pd.tseries.frequencies.to_offset(inferred), periods=steps, freq=inferred)
            rows.append(pd.DataFrame({series_column: name, date_column: idx, "forecast": fc}))
            if holdout:
                errors.append({"series": str(name), **forecast_metrics(s.iloc[-holdout:].to_numpy(), fc, train.to_numpy(), period)})
        except Exception:
            skipped += 1
    if not rows:
        raise ValueError("no series could be forecast")
    table = pd.concat(rows)
    out = {"method": method, "series_forecast": len(rows), "series_skipped": skipped, "written_to": save_frame(table, output_path, "forecast_many.csv")}
    if errors:
        e = pd.DataFrame(errors)
        out["median_mase"] = round(float(e["mase"].median()), 4) if e["mase"].notna().any() else None
        out["median_smape"] = round(float(e["smape"].median()), 4)
        out["worst_series"] = e.sort_values("smape", ascending=False).head(5)
    return out
