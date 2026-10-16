"""Judge a fitted survival model: concordance, calibration at a horizon, time-dependent discrimination, risk groups, and predicting for new subjects."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.tooling import tool
from supervised.survival_analysis.tools._common import (
    brier_at,
    concordance_index,
    default_horizons,
    kaplan_meier,
    step_value,
)


def _load(model_path: str, data: str, duration: str, event: str) -> tuple[Any, pd.DataFrame, np.ndarray, np.ndarray]:
    model = joblib.load(model_path)
    frame = load_frame(data)
    require_columns(frame, [duration, event] + model.design.columns)
    frame = frame[frame[duration].notna() & frame[event].notna()].reset_index(drop=True)
    return model, frame, frame[duration].to_numpy(float), frame[event].to_numpy(int)


@tool
def concordance_report(model_path: str, data: str, duration: str, event: str, bootstrap: int = 200, seed: int = 42) -> dict:
    """Harrell's concordance of a saved model on a dataset, with a bootstrap confidence interval — 0.5 is a coin flip, above ~0.7 is useful discrimination.

    Args:
        model_path: Path of the saved survival model (joblib).
        bootstrap: Bootstrap resamples for the interval.
    """
    model, frame, t, e = _load(model_path, data, duration, event)
    risk = model.risk(frame)
    c = concordance_index(t, e, risk)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        i = rng.integers(0, len(t), len(t))
        if e[i].sum() > 1:
            draws.append(concordance_index(t[i], e[i], risk[i]))
    return {"concordance": round(c, 5), "ci_95": [round(float(np.quantile(draws, .025)), 4), round(float(np.quantile(draws, .975)), 4)] if draws else None, "n": len(t), "events": int(e.sum())}


@tool
def calibration_at_horizon(model_path: str, data: str, duration: str, event: str, horizon: float | None = None, n_groups: int = 5) -> dict:
    """Calibration at a time horizon: subjects are grouped by predicted survival, and each group's mean prediction is compared with its Kaplan–Meier survival — a calibrated model's two columns match.

    Args:
        model_path: Path of the saved survival model (joblib).
        horizon: Time at which to compare; defaults to the median event time.
        n_groups: Number of risk groups.
    """
    model, frame, t, e = _load(model_path, data, duration, event)
    h = float(horizon) if horizon else float(np.median(t[e == 1]))
    pred = model.survival(frame, np.array([h]))[:, 0]
    groups = pd.qcut(pd.Series(pred).rank(method="first"), n_groups, labels=False)
    rows = []
    for g in sorted(groups.unique()):
        m = (groups == g).to_numpy()
        km = float(step_value(kaplan_meier(t[m], e[m]), "survival", np.array([h]))[0])
        rows.append({"group": int(g) + 1, "n": int(m.sum()), "predicted_survival": round(float(pred[m].mean()), 4), "observed_km_survival": round(km, 4), "gap": round(float(pred[m].mean() - km), 4)})
    worst = max(abs(r["gap"]) for r in rows)
    return {"horizon": h, "groups": rows, "max_abs_gap": round(worst, 4), "verdict": "well calibrated" if worst < 0.05 else "moderately off" if worst < 0.1 else "poorly calibrated: recalibrate or use a different model"}


@tool
def time_dependent_auc(model_path: str, data: str, duration: str, event: str, horizons: list[float] | None = None) -> dict:
    """Cumulative/dynamic AUC at chosen horizons: how well the model's risk separates subjects who had the event by that time from those still event-free. Subjects censored before the horizon are excluded.

    Args:
        model_path: Path of the saved survival model (joblib).
        horizons: Times to evaluate at; defaults to quartiles of the event times.
    """
    from sklearn.metrics import roc_auc_score

    model, frame, t, e = _load(model_path, data, duration, event)
    risk = model.risk(frame)
    out = {}
    for h in (np.asarray(horizons, float) if horizons else default_horizons(t, e, 3)):
        case, control = (t <= h) & (e == 1), t > h
        keep = case | control
        out[f"{h:g}"] = {"auc": round(float(roc_auc_score(case[keep], risk[keep])), 4) if case.sum() and control.sum() else None, "cases": int(case.sum()), "controls": int(control.sum())}
    return {"auc_by_horizon": out}


@tool
def brier_score_report(model_path: str, data: str, duration: str, event: str, horizons: list[float] | None = None) -> dict:
    """Inverse-probability-of-censoring-weighted Brier score at chosen horizons, beside the score of a Kaplan–Meier-only (no covariates) model — the model has to beat that reference to be worth using.

    Args:
        model_path: Path of the saved survival model (joblib).
        horizons: Times to evaluate at; defaults to quartiles of the event times.
    """
    model, frame, t, e = _load(model_path, data, duration, event)
    km = kaplan_meier(t, e)

    class _KM:
        def survival(self, f: pd.DataFrame, times: np.ndarray) -> np.ndarray:
            return np.tile(step_value(km, "survival", times), (len(f), 1))

    out = {}
    for h in (np.asarray(horizons, float) if horizons else default_horizons(t, e, 3)):
        m, ref = brier_at(model, frame, duration, event, h), brier_at(_KM(), frame, duration, event, h)
        out[f"{h:g}"] = {"model": round(m, 5), "km_reference": round(ref, 5), "beats_reference": bool(m < ref)}
    return {"brier_by_horizon": out}


@tool
def risk_stratification(model_path: str, data: str, duration: str, event: str, n_groups: int = 3, horizons: list[float] | None = None) -> dict:
    """Split subjects into low/medium/high risk by the model's score and show each group's Kaplan–Meier survival and event counts — the clearest picture of whether the model separates outcomes.

    Args:
        model_path: Path of the saved survival model (joblib).
        n_groups: Number of risk groups.
        horizons: Times at which to report each group's survival.
    """
    model, frame, t, e = _load(model_path, data, duration, event)
    risk = model.risk(frame)
    grp = pd.qcut(pd.Series(risk).rank(method="first"), n_groups, labels=False).to_numpy()
    hs = np.asarray(horizons, float) if horizons else default_horizons(t, e, 2)
    names = ["low", "medium", "high"] if n_groups == 3 else [f"risk_{i + 1}" for i in range(n_groups)]
    out = {}
    for g in range(n_groups):
        m = grp == g
        table = kaplan_meier(t[m], e[m])
        out[names[g]] = {"n": int(m.sum()), "events": int(e[m].sum()), "mean_risk_score": round(float(risk[m].mean()), 4),
                         "survival_at": {f"{h:g}": round(float(step_value(table, "survival", np.array([h]))[0]), 4) for h in hs}}
    first, last = out[names[0]]["survival_at"], out[names[-1]]["survival_at"]
    key = next(iter(first))
    return {"groups": out, "separation_at_first_horizon": round(first[key] - last[key], 4)}


@tool
def predict_survival_curves(model_path: str, data: str, times: list[float], id_column: str | None = None, output_path: str | None = None) -> dict:
    """Predict each subject's survival probability at the given times, plus a risk score and rank, with a saved model — one output row per subject.

    Args:
        model_path: Path of the saved survival model (joblib).
        times: Times at which to predict S(t).
        id_column: Column to carry through to the output.
    """
    model = joblib.load(model_path)
    frame = load_frame(data)
    require_columns(frame, model.design.columns)
    surv = model.survival(frame, np.asarray(times, float))
    out = pd.DataFrame(surv, columns=[f"survival_at_{t:g}" for t in times])
    out.insert(0, "risk_score", model.risk(frame))
    out["risk_rank"] = out["risk_score"].rank(ascending=False, method="first").astype(int)
    if id_column and id_column in frame.columns:
        out.insert(0, id_column, frame[id_column].to_numpy())
    return {"rows": len(out), "written_to": save_frame(out, output_path, "survival_predictions.csv"), "sample": out.head(5),
            "mean_survival": {c: round(float(out[c].mean()), 4) for c in out.columns if c.startswith("survival_at_")}}
