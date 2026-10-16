"""Exponential-smoothing and regression-on-time forecasters (no extra packages): SES, Holt, Holt–Winters, Theta, and harmonic regression."""

from __future__ import annotations

from typing import Literal

from ml_core.tooling import tool
from supervised.time_series_forecasting.tools._common import run_forecaster


@tool
def forecast_ses(data: str, date_column: str, value_column: str, horizon: int = 12, holdout: int = 0, period: int | None = None, freq: str | None = None, output_path: str | None = None) -> dict:
    """Simple exponential smoothing: a flat forecast at a weighted average of the past that favours recent points, smoothing parameter fitted automatically. For level-only series without trend or seasonality.

    Args:
        holdout: Hide the last N points, forecast them and report errors; 0 forecasts the real future.
        freq: Pandas frequency alias; inferred when omitted.
    """
    return run_forecaster("ses", "ses", data, date_column, value_column, horizon, period, holdout, freq, output_path)


@tool
def forecast_holt(data: str, date_column: str, value_column: str, horizon: int = 12, damped: bool = False, holdout: int = 0, period: int | None = None, freq: str | None = None, output_path: str | None = None) -> dict:
    """Holt's linear-trend smoothing: level plus a trend that evolves; damping makes the trend flatten out over long horizons, which is usually safer.

    Args:
        damped: Damp the trend so the forecast levels off instead of rising forever.
        holdout: Hide the last N points, forecast them and report errors; 0 forecasts the real future.
        freq: Pandas frequency alias; inferred when omitted.
    """
    return run_forecaster("holt", f"holt(damped={damped})", data, date_column, value_column, horizon, period, holdout, freq, output_path, damped=damped)


@tool
def forecast_holt_winters(
    data: str, date_column: str, value_column: str, horizon: int = 12, trend: bool = True, damped: bool = False,
    seasonal: Literal["add", "mul"] = "add", holdout: int = 0, period: int | None = None, freq: str | None = None, output_path: str | None = None,
) -> dict:
    """Holt–Winters triple exponential smoothing: level, trend and a seasonal pattern, all updated as data arrives. The default for series with a clear trend and seasonality.

    Args:
        trend: Include a trend term.
        damped: Damp the trend.
        seasonal: add when the seasonal swing is constant, mul when it grows with the level.
        holdout: Hide the last N points, forecast them and report errors; 0 forecasts the real future.
        period: Seasonal period in steps; the frequency's default when omitted.
    """
    return run_forecaster("holt_winters", f"holt_winters(trend={trend},{seasonal})", data, date_column, value_column, horizon, period, holdout, freq, output_path, trend=trend, damped=damped, seasonal=seasonal)


@tool
def forecast_theta(data: str, date_column: str, value_column: str, horizon: int = 12, holdout: int = 0, period: int | None = None, freq: str | None = None, output_path: str | None = None) -> dict:
    """Theta method: simple smoothing plus half the linear trend, on a seasonally adjusted series. A strong, cheap benchmark that won the M3 competition; hard to beat on monthly and quarterly data.

    Args:
        holdout: Hide the last N points, forecast them and report errors; 0 forecasts the real future.
        freq: Pandas frequency alias; inferred when omitted.
    """
    return run_forecaster("theta", "theta", data, date_column, value_column, horizon, period, holdout, freq, output_path)


@tool
def forecast_harmonic_regression(
    data: str, date_column: str, value_column: str, horizon: int = 12, periods: list[float] | None = None, harmonics: int = 3,
    trend_degree: int = 1, alpha: float = 1.0, holdout: int = 0, period: int | None = None, freq: str | None = None, output_path: str | None = None,
) -> dict:
    """Regression on a polynomial trend plus Fourier terms — handles several seasonal cycles at once (e.g. daily 24 and weekly 168 on hourly data) and long or non-integer periods that smoothing methods cannot.

    Args:
        periods: Seasonal cycles in steps, e.g. [24, 168]; the frequency's default when omitted.
        harmonics: Fourier pairs per cycle; more captures sharper seasonal shapes.
        trend_degree: Polynomial degree of the trend (0 for none).
        alpha: Ridge penalty.
        holdout: Hide the last N points, forecast them and report errors; 0 forecasts the real future.
    """
    return run_forecaster("harmonic", "harmonic_regression", data, date_column, value_column, horizon, period, holdout, freq, output_path, periods=periods, harmonics=harmonics, trend_degree=trend_degree, alpha=alpha)
