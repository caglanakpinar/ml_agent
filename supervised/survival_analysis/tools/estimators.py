"""Non-parametric survival: Kaplan–Meier, Nelson–Aalen, log-rank tests, restricted mean survival time, life tables, and competing risks."""

from __future__ import annotations


import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.tooling import tool
from supervised.survival_analysis.tools._common import (
    kaplan_meier,
    load_survival,
    median_time,
    nelson_aalen,
    step_value,
)


def _group_masks(frame: pd.DataFrame, group: str | None, bins: int) -> dict[str, np.ndarray]:
    if not group:
        return {"all": np.ones(len(frame), bool)}
    require_columns(frame, [group])
    g = frame[group]
    if pd.api.types.is_numeric_dtype(g) and g.nunique() > 6:
        g = pd.qcut(g, bins, duplicates="drop").astype(str)
    return {str(k): (g.astype(str) == str(k)).to_numpy() for k in sorted(g.dropna().astype(str).unique())}


@tool
def kaplan_meier_estimate(
    data: str, duration: str, event: str, group_column: str | None = None, times: list[float] | None = None,
    bins: int = 3, output_path: str | None = None,
) -> dict:
    """Kaplan–Meier survival curve, overall or by group: survival probability with 95% intervals at chosen times, the median survival time, and events per group. Numeric groups are cut into quantile bins.

    Args:
        group_column: Compare curves across the levels of this column.
        times: Times at which to report survival; defaults to the quartiles of the event times.
        bins: Quantile bins for a numeric group column.
    """
    frame, t, e, _ = load_survival(data, duration, event, covariates=[])
    raw = load_frame(data)
    raw = raw.loc[frame.index] if len(raw) == len(frame) else frame
    horizon = np.asarray(times if times else np.quantile(t[e == 1], [0.25, 0.5, 0.75]), float)
    groups, tables = {}, []
    for name, mask in _group_masks(frame if group_column in frame else raw, group_column, bins).items():
        table = kaplan_meier(t[mask], e[mask])
        groups[name] = {"n": int(mask.sum()), "events": int(e[mask].sum()), "median_survival": median_time(table),
                        "survival_at": {f"{h:g}": {"survival": round(float(step_value(table, 'survival', np.array([h]))[0]), 4),
                                                   "lower": round(float(step_value(table, 'lower', np.array([h]))[0]), 4),
                                                   "upper": round(float(step_value(table, 'upper', np.array([h]))[0]), 4)} for h in horizon}}
        tables.append(table.assign(group=name))
    saved = save_frame(pd.concat(tables), output_path, "kaplan_meier.csv") if output_path else None
    return {"groups": groups, "written_to": saved}


@tool
def nelson_aalen_estimate(data: str, duration: str, event: str, times: list[float] | None = None) -> dict:
    """Nelson–Aalen cumulative hazard — the natural scale for spotting how the risk of the event changes over time (a rising curve means wear-out, a flattening one means early failures only).

    Args:
        times: Times at which to report the cumulative hazard; defaults to quartiles of the event times.
    """
    _, t, e, _ = load_survival(data, duration, event, covariates=[])
    table = nelson_aalen(t, e)
    horizon = np.asarray(times if times else np.quantile(t[e == 1], [0.25, 0.5, 0.75, 1.0]), float)
    values = step_value(table, "cumulative_hazard", horizon, start=0.0)
    slopes = np.diff(values) / np.diff(horizon) if len(horizon) > 1 else np.array([])
    shape = ("increasing hazard (wear-out)" if len(slopes) > 1 and slopes[-1] > 1.3 * slopes[0] else
             "decreasing hazard (early failures)" if len(slopes) > 1 and slopes[-1] < 0.7 * slopes[0] else "roughly constant hazard")
    return {"cumulative_hazard": {f"{h:g}": round(float(v), 4) for h, v in zip(horizon, values)}, "hazard_shape": shape}


@tool
def logrank_test(data: str, duration: str, event: str, group_column: str, bins: int = 3) -> dict:
    """Log-rank test: do the survival curves of two or more groups differ? Reports the chi-square, degrees of freedom and p-value, with observed vs expected events per group.

    Args:
        group_column: Column defining the groups; numeric columns are cut into quantile bins.
        bins: Quantile bins for a numeric group column.
    """
    from scipy import stats

    frame, t, e, _ = load_survival(data, duration, event, covariates=[])
    masks = _group_masks(frame if group_column in frame else load_frame(data).loc[frame.index], group_column, bins)
    names = list(masks)
    if len(names) < 2:
        raise ValueError("log-rank needs at least two groups")
    times = np.unique(t[e == 1])
    O, E = np.zeros(len(names)), np.zeros(len(names))
    V = np.zeros((len(names), len(names)))
    for u in times:
        at_risk = np.array([(t[m] >= u).sum() for m in masks.values()], float)
        d_g = np.array([((t[m] == u) & (e[m] == 1)).sum() for m in masks.values()], float)
        n, d = at_risk.sum(), d_g.sum()
        O += d_g
        E += d * at_risk / n
        if n > 1:
            V += d * (n - d) / (n - 1) * (np.diag(at_risk / n) - np.outer(at_risk, at_risk) / n**2)
    diff = (O - E)[:-1]
    chi2 = float(diff @ np.linalg.pinv(V[:-1, :-1]) @ diff)
    p = float(stats.chi2.sf(chi2, len(names) - 1))
    return {"groups": {n: {"observed_events": int(o), "expected_events": round(float(x), 2)} for n, o, x in zip(names, O, E)},
            "chi2": round(chi2, 4), "df": len(names) - 1, "p_value": round(p, 6),
            "significant_at_5pct": bool(p < 0.05)}


@tool
def restricted_mean_survival_time(data: str, duration: str, event: str, horizon: float | None = None, group_column: str | None = None, bins: int = 3) -> dict:
    """Restricted mean survival time (RMST): the average event-free time up to a horizon — a summary that stays valid when hazards are not proportional and the median is not reached.

    Args:
        horizon: Upper time limit; defaults to the largest time at which every group still has subjects at risk.
        group_column: Compare RMST across the levels of this column.
        bins: Quantile bins for a numeric group column.
    """
    frame, t, e, _ = load_survival(data, duration, event, covariates=[])
    masks = _group_masks(frame if (group_column is None or group_column in frame) else load_frame(data).loc[frame.index], group_column, bins)
    tau = float(horizon) if horizon else float(min(t[m].max() for m in masks.values()))
    out = {}
    for name, m in masks.items():
        table = kaplan_meier(t[m], e[m])
        grid = np.concatenate([[0.0], table["time"][table["time"] < tau].to_numpy(), [tau]])
        surv = np.concatenate([[1.0], table["survival"][table["time"] < tau].to_numpy()])
        out[name] = {"rmst": round(float(np.sum(surv * np.diff(grid))), 4), "n": int(m.sum())}
    values = [v["rmst"] for v in out.values()]
    return {"horizon": tau, "rmst": out, "difference_max_min": round(max(values) - min(values), 4) if len(values) > 1 else None}


@tool
def life_table(data: str, duration: str, event: str, n_intervals: int = 10) -> dict:
    """Actuarial life table over equal-width intervals: subjects entering, events, censored, hazard and survival per interval — the report format used in clinical and actuarial work.

    Args:
        n_intervals: Number of equal-width time intervals.
    """
    _, t, e, _ = load_survival(data, duration, event, covariates=[])
    edges = np.linspace(0, t.max() + 1e-9, n_intervals + 1)
    rows, surv = [], 1.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        entering = int((t >= lo).sum())
        d = int(((t >= lo) & (t < hi) & (e == 1)).sum())
        c = int(((t >= lo) & (t < hi) & (e == 0)).sum())
        effective = entering - c / 2
        q = d / effective if effective > 0 else 0.0
        surv *= 1 - q
        rows.append({"from": round(float(lo), 3), "to": round(float(hi), 3), "entering": entering, "events": d, "censored": c,
                     "conditional_event_probability": round(q, 4), "survival_at_end": round(surv, 4)})
    return {"table": rows}


@tool
def competing_risks_cumulative_incidence(data: str, duration: str, event_type: str, event_of_interest: str, times: list[float] | None = None) -> dict:
    """Cumulative incidence of one event type when other event types can pre-empt it (Aalen–Johansen). Kaplan–Meier would overstate the risk here because it treats the competing events as censoring.

    Args:
        event_type: Column holding 0 for censored and a label for each kind of event.
        event_of_interest: The event label whose cumulative incidence to estimate.
        times: Times at which to report; defaults to quartiles of the event times.
    """
    frame = load_frame(data)
    require_columns(frame, [duration, event_type])
    t = pd.to_numeric(frame[duration], errors="coerce").to_numpy(float)
    kind = frame[event_type].astype(str).to_numpy()
    ok = ~np.isnan(t)
    t, kind = t[ok], kind[ok]
    any_event = kind != "0"
    grid = np.unique(t[any_event])
    surv, cif, rows = 1.0, 0.0, []
    competing: dict[str, float] = {}
    for u in grid:
        n = (t >= u).sum()
        d_all = ((t == u) & any_event).sum()
        d_int = ((t == u) & (kind == str(event_of_interest))).sum()
        cif += surv * d_int / n
        for other in set(kind[(t == u) & any_event]) - {str(event_of_interest)}:
            competing[other] = competing.get(other, 0.0) + surv * ((t == u) & (kind == other)).sum() / n
        surv *= 1 - d_all / n
        rows.append({"time": float(u), "cif": cif})
    table = pd.DataFrame(rows)
    horizon = np.asarray(times if times else np.quantile(grid, [0.25, 0.5, 0.75, 1.0]), float)
    naive = kaplan_meier(t, ((kind == str(event_of_interest))).astype(int))
    return {"event_of_interest": event_of_interest,
            "cumulative_incidence": {f"{h:g}": round(float(step_value(table, "cif", np.array([h]), 0.0)[0]), 4) for h in horizon},
            "naive_km_1_minus_s": {f"{h:g}": round(float(1 - step_value(naive, "survival", np.array([h]))[0]), 4) for h in horizon},
            "final_competing_incidence": {k: round(v, 4) for k, v in competing.items()},
            "note": "the naive 1−KM figure is always at least as large as the cumulative incidence; the gap is the bias from ignoring competing events"}
