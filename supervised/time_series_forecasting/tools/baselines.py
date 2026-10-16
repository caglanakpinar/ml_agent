"""Benchmark forecasters: naive, seasonal naive, drift, mean. They are the floor any real model has to beat. With `holdout` > 0 they are scored on the last points; otherwise they forecast the future."""

from __future__ import annotations

from ml_core.tooling import tool
from supervised.time_series_forecasting.tools._common import run_forecaster


@tool
def forecast_naive(data: str, date_column: str, value_column: str, horizon: int = 12, holdout: int = 0, period: int | None = None, freq: str | None = None, output_path: str | None = None) -> dict:
    """Naive forecast: every future value equals the last observation. Surprisingly hard to beat on random-walk-like series (prices, cumulative counts).

    Args:
        holdout: Hide the last N points, forecast them and report errors; 0 forecasts the real future.
        freq: Pandas frequency alias (D, W, MS, h ...); inferred when omitted.
    """
    return run_forecaster("naive", "naive", data, date_column, value_column, horizon, period, holdout, freq, output_path)


@tool
def forecast_seasonal_naive(data: str, date_column: str, value_column: str, horizon: int = 12, holdout: int = 0, period: int | None = None, freq: str | None = None, output_path: str | None = None) -> dict:
    """Seasonal naive forecast: every future value equals the value one season earlier. The benchmark for strongly seasonal series; hard to beat on short horizons.

    Args:
        holdout: Hide the last N points, forecast them and report errors; 0 forecasts the real future.
        freq: Pandas frequency alias; inferred when omitted.
    """
    return run_forecaster("seasonal_naive", "seasonal_naive", data, date_column, value_column, horizon, period, holdout, freq, output_path)


@tool
def forecast_drift(data: str, date_column: str, value_column: str, horizon: int = 12, holdout: int = 0, period: int | None = None, freq: str | None = None, output_path: str | None = None) -> dict:
    """Drift forecast: extend the straight line between the first and last observation — the benchmark for trending series with no seasonality.

    Args:
        holdout: Hide the last N points, forecast them and report errors; 0 forecasts the real future.
        freq: Pandas frequency alias; inferred when omitted.
    """
    return run_forecaster("drift", "drift", data, date_column, value_column, horizon, period, holdout, freq, output_path)


@tool
def forecast_mean(data: str, date_column: str, value_column: str, horizon: int = 12, window: int | None = None, holdout: int = 0, period: int | None = None, freq: str | None = None, output_path: str | None = None) -> dict:
    """Mean forecast: every future value equals the average of the history (or of the last `window` points) — the benchmark for stationary, trendless series; with a window it is a moving average.

    Args:
        window: Average only the last N points; the whole history when omitted.
        holdout: Hide the last N points, forecast them and report errors; 0 forecasts the real future.
        freq: Pandas frequency alias; inferred when omitted.
    """
    return run_forecaster("mean", f"mean(window={window})", data, date_column, value_column, horizon, period, holdout, freq, output_path, window=window)
