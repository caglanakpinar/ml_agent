"""Read-only checks on a time-to-event problem and its data, before any survival model is fitted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_core.data import load_frame
from ml_core.tooling import tool
from supervised.survival_analysis.tools._common import (
    Design,
    cox_fit,
    kaplan_meier,
    load_survival,
    median_time,
)


@tool
def profile_survival_data(data: str, duration: str, event: str, covariates: list[str] | None = None) -> dict:
    """Profile a time-to-event dataset: rows, events vs censored, follow-up distribution, tied times, zero or negative durations, and the covariates with their types and missing values."""
    raw = load_frame(data)
    frame, t, e, names = load_survival(raw, duration, event, covariates)
    reverse = kaplan_meier(t, 1 - e)  # censoring as the "event": its median is the typical follow-up
    return {
        "rows": len(raw), "rows_used": len(frame), "rows_dropped_invalid": len(raw) - len(frame),
        "events": int(e.sum()), "censored": int((1 - e).sum()), "event_rate": round(float(e.mean()), 4),
        "duration": {"min": float(t.min()), "p25": float(np.quantile(t, .25)), "median": float(np.median(t)), "p75": float(np.quantile(t, .75)), "max": float(t.max())},
        "median_follow_up_reverse_km": median_time(reverse),
        "zero_durations": int((t == 0).sum()),
        "tied_event_times": int(len(t[e == 1]) - len(np.unique(t[e == 1]))),
        "covariates_usable": names,
        "numeric_covariates": [c for c in names if pd.api.types.is_numeric_dtype(frame[c])],
        "categorical_covariates": [c for c in names if not pd.api.types.is_numeric_dtype(frame[c])],
        "covariates_with_missing": {c: round(float(frame[c].isna().mean()), 4) for c in names if frame[c].isna().any()},
        "events_per_covariate": round(float(e.sum() / max(len(names), 1)), 1),
        "warnings": [w for w in [
            "fewer than 10 events per covariate: a Cox/parametric model with this many covariates will overfit — use a penalty (fit_cox_regularised) or fewer covariates" if e.sum() / max(len(names), 1) < 10 else None,
            "more than 90% censored: survival estimates beyond the follow-up are extrapolation" if e.mean() < 0.1 else None,
            "zero-length durations present: they can never be at risk; check the time origin" if (t == 0).any() else None] if w],
    }


@tool
def check_censoring_pattern(data: str, duration: str, event: str, covariates: list[str] | None = None, k: int = 10) -> dict:
    """Test whether censoring looks independent of the covariates, which every standard estimator assumes: compares each covariate between censored and event rows.

    Args:
        k: How many most-different covariates to return.
    """
    from scipy import stats

    frame, t, e, names = load_survival(data, duration, event, covariates)
    rows = []
    for c in names:
        if pd.api.types.is_numeric_dtype(frame[c]):
            a, b = frame.loc[e == 1, c].dropna(), frame.loc[e == 0, c].dropna()
            if len(a) > 2 and len(b) > 2:
                p = stats.mannwhitneyu(a, b).pvalue
                rows.append({"covariate": c, "test": "mann-whitney", "p_value": round(float(p), 5)})
        else:
            table = pd.crosstab(frame[c], e)
            if table.shape[0] > 1 and table.shape[1] > 1:
                rows.append({"covariate": c, "test": "chi-square", "p_value": round(float(stats.chi2_contingency(table)[1]), 5)})
    rows.sort(key=lambda r: r["p_value"])
    late = float(np.mean(t[e == 0] >= np.median(t)))
    return {"censored_share": round(float(1 - e.mean()), 4), "censored_late_share": round(late, 4),
            "most_different": rows[:k], "suspicious": [r["covariate"] for r in rows if r["p_value"] < 0.01 / max(len(rows), 1) * 5],
            "reading": "Covariates that differ sharply between censored and event rows suggest informative censoring (people leave the study *because* of the covariate); administrative censoring at a fixed end date is fine."}


@tool
def check_proportional_hazards(data: str, duration: str, event: str, covariates: list[str] | None = None, l2: float = 0.01) -> dict:
    """Fit a Cox model and test the proportional-hazards assumption per covariate by correlating scaled Schoenfeld residuals with event time — a significant trend means the covariate's effect changes over time.

    Args:
        l2: Small ridge penalty for stability.
    """
    from scipy import stats

    frame, t, e, names = load_survival(data, duration, event, covariates)
    design = Design.fit(frame, names)
    X = design.transform(frame)
    fit = cox_fit(X, t, e, l2)
    order = np.argsort(t, kind="stable")
    Xs, ts, es = X[order], t[order], e[order]
    w = np.exp(Xs @ fit["beta"])
    resid, times = [], []
    for i in np.flatnonzero(es == 1):
        risk = ts >= ts[i]
        mean = (w[risk, None] * Xs[risk]).sum(axis=0) / w[risk].sum()
        resid.append(Xs[i] - mean)
        times.append(ts[i])
    resid, times = np.asarray(resid), np.asarray(times)
    ranks = stats.rankdata(times)
    rows = []
    for j, name in enumerate(design.names):
        rho, p = stats.spearmanr(ranks, resid[:, j])
        rows.append({"covariate": name, "correlation_with_time": round(float(rho), 4), "p_value": round(float(p), 5), "violates": bool(p < 0.05)})
    rows.sort(key=lambda r: r["p_value"])
    bad = [r["covariate"] for r in rows if r["violates"]]
    return {"tests": rows, "violations": bad,
            "advice": "proportional hazards looks fine — a Cox model is appropriate" if not bad else
            "the effect of " + ", ".join(bad) + " changes over time: stratify on it, add time-interaction terms, or use fit_parametric_aft / fit_discrete_time_logistic / a forest instead of a plain Cox model"}


@tool
def check_events_per_variable(data: str, duration: str, event: str, covariates: list[str] | None = None) -> dict:
    """Events per covariate (EPV) — the rule-of-thumb check on whether the data can support the number of covariates; below ~10 a regression model is unreliable."""
    frame, t, e, names = load_survival(data, duration, event, covariates)
    design = Design.fit(frame, names)
    epv = float(e.sum() / max(len(design.names), 1))
    return {"events": int(e.sum()), "model_terms": len(design.names), "events_per_term": round(epv, 2),
            "verdict": "adequate" if epv >= 10 else "marginal" if epv >= 5 else "too few events for this many terms: penalise (fit_cox_regularised), drop covariates, or collapse categories",
            "max_terms_for_10_epv": int(e.sum() // 10)}
