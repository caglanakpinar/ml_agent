"""Classical statistical forecasters built on statsmodels (ARIMA family) and Prophet."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ml_core.data import save_frame
from ml_core.tooling import tool
from supervised.time_series_forecasting.tools._common import (
    default_period,
    forecast_metrics,
    future_index,
    load_series,
    run_forecaster,
)


@tool(requires=("statsmodels",))
def forecast_arima(
    data: str, date_column: str, value_column: str, horizon: int = 12, p: int = 1, d: int = 1, q: int = 1,
    holdout: int = 0, period: int | None = None, freq: str | None = None, output_path: str | None = None,
) -> dict:
    """ARIMA(p,d,q): autoregression on the (differenced) series plus moving-average error terms — the general-purpose statistical model for series with autocorrelation and no strong seasonality.

    Args:
        p: Autoregressive order (from the PACF).
        d: Differencing order (0 if stationary, usually 1 with a trend).
        q: Moving-average order (from the ACF).
        holdout: Hide the last N points, forecast them and report errors; 0 forecasts the real future.
    """
    return run_forecaster("arima", f"arima({p},{d},{q})", data, date_column, value_column, horizon, period, holdout, freq, output_path, order=(p, d, q))


@tool(requires=("statsmodels",))
def forecast_sarima(
    data: str, date_column: str, value_column: str, horizon: int = 12, p: int = 1, d: int = 1, q: int = 1,
    seasonal_p: int = 1, seasonal_d: int = 1, seasonal_q: int = 1, holdout: int = 0, period: int | None = None,
    freq: str | None = None, output_path: str | None = None,
) -> dict:
    """Seasonal ARIMA(p,d,q)(P,D,Q)[period]: ARIMA plus the same structure at the seasonal lag — for trending, strongly seasonal series with autocorrelated errors. Slower than smoothing; needs several full seasons.

    Args:
        p: Non-seasonal AR order.
        d: Non-seasonal differencing.
        q: Non-seasonal MA order.
        seasonal_p: Seasonal AR order.
        seasonal_d: Seasonal differencing.
        seasonal_q: Seasonal MA order.
        holdout: Hide the last N points, forecast them and report errors; 0 forecasts the real future.
    """
    return run_forecaster("arima", f"sarima({p},{d},{q})({seasonal_p},{seasonal_d},{seasonal_q})", data, date_column, value_column, horizon, period, holdout, freq, output_path,
                          order=(p, d, q), seasonal_order=(seasonal_p, seasonal_d, seasonal_q))


@tool(requires=("statsmodels",))
def auto_arima_search(
    data: str, date_column: str, value_column: str, max_p: int = 3, max_d: int = 1, max_q: int = 3, seasonal: bool = False,
    period: int | None = None, criterion: str = "aic", freq: str | None = None, k: int = 5,
) -> dict:
    """Grid-search ARIMA (optionally with a small seasonal grid) orders by AIC/BIC and return the best few — the order to pass to `forecast_arima` / `forecast_sarima`.

    Args:
        max_p: Largest AR order tried.
        max_d: Largest differencing tried.
        max_q: Largest MA order tried.
        seasonal: Also search seasonal orders (0–1) at the series' period.
        criterion: aic or bic.
        k: How many best models to return.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    s, inferred, _ = load_series(data, date_column, value_column, freq)
    m = period or default_period(inferred)
    seasonal_grid = list(itertools.product([0, 1], [0, 1], [0, 1])) if seasonal and m and m > 1 else [(0, 0, 0)]
    results = []
    for (p, d, q), sp in itertools.product(itertools.product(range(max_p + 1), range(max_d + 1), range(max_q + 1)), seasonal_grid):
        try:
            fit = SARIMAX(s.to_numpy(), order=(p, d, q), seasonal_order=(*sp, m if sp != (0, 0, 0) else 0), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            results.append({"order": (p, d, q), "seasonal_order": (*sp, m) if sp != (0, 0, 0) else None, "aic": round(float(fit.aic), 3), "bic": round(float(fit.bic), 3)})
        except Exception:
            continue
    results.sort(key=lambda r: r[criterion])
    return {"tried": len(results), "criterion": criterion, "best": results[:k]}


@tool(requires=("statsmodels",))
def forecast_ets_auto(
    data: str, date_column: str, value_column: str, horizon: int = 12, holdout: int = 0, period: int | None = None,
    freq: str | None = None, output_path: str | None = None,
) -> dict:
    """Automatic ETS: fits every combination of additive/multiplicative error, trend (plain or damped) and seasonality with statsmodels and keeps the lowest AICc — the exponential-smoothing family chosen by the data rather than by hand.

    Args:
        holdout: Hide the last N points, forecast them and report errors; 0 forecasts the real future.
    """
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    series, inferred, repairs = load_series(data, date_column, value_column, freq)
    m = period or default_period(inferred)
    train = series.iloc[:-holdout] if holdout else series
    steps = holdout or horizon
    best, tried = None, []
    positive = bool((train > 0).all())
    for error, trend, damped, seasonal in itertools.product(["add", "mul"] if positive else ["add"], [None, "add"], [False, True], [None, "add", "mul"] if m and m > 1 and len(train) >= 2 * m and positive else [None]):
        if trend is None and damped:
            continue
        try:
            fit = ETSModel(train.to_numpy(), error=error, trend=trend, damped_trend=damped, seasonal=seasonal, seasonal_periods=m if seasonal else None).fit(disp=False)
            tried.append({"error": error, "trend": trend, "damped": damped, "seasonal": seasonal, "aicc": round(float(fit.aicc), 3)})
            if best is None or fit.aicc < best[0]:
                best = (fit.aicc, fit, tried[-1])
        except Exception:
            continue
    if best is None:
        raise ValueError("no ETS configuration converged")
    fc = np.asarray(best[1].forecast(steps))
    index = series.index[-holdout:] if holdout else future_index(series, inferred, steps)
    table = pd.DataFrame({date_column: index, "forecast": fc})
    out = {"model": "ets", "chosen": best[2], "configurations_tried": len(tried), "frequency": inferred, "period": m, "repairs": repairs}
    if holdout:
        out["holdout"] = forecast_metrics(series.iloc[-holdout:].to_numpy(), fc, train.to_numpy(), m)
    out["forecast"] = table.head(12).assign(**{date_column: lambda d: d[date_column].astype(str)})
    out["written_to"] = save_frame(table, output_path, "forecast_ets.csv") if output_path else None
    return out


@tool(requires=("prophet",))
def forecast_prophet(
    data: str, date_column: str, value_column: str, horizon: int = 12, holdout: int = 0, yearly: bool = True, weekly: bool = True,
    daily: bool = False, changepoint_prior_scale: float = 0.05, freq: str | None = None, output_path: str | None = None,
) -> dict:
    """Prophet: piecewise trend with changepoints plus yearly/weekly/daily seasonality and holidays — good for business series with several seasonal cycles, outliers and missing stretches; interpretable components.

    Args:
        holdout: Hide the last N points, forecast them and report errors; 0 forecasts the real future.
        yearly: Fit yearly seasonality.
        weekly: Fit weekly seasonality.
        daily: Fit daily seasonality.
        changepoint_prior_scale: Trend flexibility; larger follows the data more closely.
    """
    from prophet import Prophet

    series, inferred, repairs = load_series(data, date_column, value_column, freq)
    train = series.iloc[:-holdout] if holdout else series
    steps = holdout or horizon
    model = Prophet(yearly_seasonality=yearly, weekly_seasonality=weekly, daily_seasonality=daily, changepoint_prior_scale=changepoint_prior_scale)
    model.fit(pd.DataFrame({"ds": train.index, "y": train.to_numpy()}))
    future = model.make_future_dataframe(periods=steps, freq=inferred)
    pred = model.predict(future).tail(steps)
    out = {"model": "prophet", "frequency": inferred, "repairs": repairs}
    if holdout:
        out["holdout"] = forecast_metrics(series.iloc[-holdout:].to_numpy(), pred["yhat"].to_numpy(), train.to_numpy(), default_period(inferred))
    table = pd.DataFrame({date_column: pred["ds"].to_numpy(), "forecast": pred["yhat"].to_numpy(), "lower": pred["yhat_lower"].to_numpy(), "upper": pred["yhat_upper"].to_numpy()})
    out["forecast"] = table.head(12).assign(**{date_column: lambda d: d[date_column].astype(str)})
    out["written_to"] = save_frame(table, output_path, "forecast_prophet.csv") if output_path else None
    return out
