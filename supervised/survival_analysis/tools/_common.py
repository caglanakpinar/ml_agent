"""Survival-analysis machinery the tool modules share, written against numpy/scipy so nothing here needs
lifelines or scikit-survival: Kaplan–Meier, Nelson–Aalen, Harrell's C, a Cox fit (Breslow ties, Newton–Raphson),
parametric accelerated-failure-time fits, a discrete-time hazard model, and the model classes that carry
a fitted model, its preprocessing and its predictions through joblib."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ml_core.data import as_list, load_frame, require_columns
from ml_core.ml import save_model, sk


# ---- data ---------------------------------------------------------------------------------------
@dataclass
class Design:
    """How raw covariate columns become the numeric matrix a model was fitted on."""

    columns: list[str]
    dummies: dict[str, list[str]] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)
    names: list[str] = field(default_factory=list)
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: list[str], standardise: bool = True) -> "Design":
        d = cls(columns=list(columns))
        for c in columns:
            if pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c]):
                d.medians[c] = float(pd.to_numeric(frame[c], errors="coerce").median())
                d.names.append(c)
            else:
                levels = sorted(frame[c].dropna().astype(str).unique())[1:]  # first level is the reference
                d.dummies[c] = levels
                d.names += [f"{c}={lvl}" for lvl in levels]
        raw = d._raw(frame)
        d.mean = raw.mean(axis=0) if standardise else np.zeros(raw.shape[1])
        d.std = np.where(raw.std(axis=0) > 0, raw.std(axis=0), 1.0) if standardise else np.ones(raw.shape[1])
        return d

    def _raw(self, frame: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.columns:
            if c in self.dummies:
                s = frame[c].astype(str)
                cols += [(s == lvl).astype(float).to_numpy() for lvl in self.dummies[c]]
            else:
                cols.append(pd.to_numeric(frame[c], errors="coerce").fillna(self.medians[c]).astype(float).to_numpy())
        return np.column_stack(cols) if cols else np.zeros((len(frame), 0))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return (self._raw(frame) - self.mean) / self.std


def load_survival(data: Any, duration: str, event: str, covariates: Any = None, drop: Any = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Load rows with a valid (duration > 0, event 0/1) pair; returns frame, durations, events and covariate names."""
    frame = load_frame(data)
    require_columns(frame, [duration, event])
    frame = frame.copy()
    frame[duration] = pd.to_numeric(frame[duration], errors="coerce")
    frame[event] = pd.to_numeric(frame[event], errors="coerce")
    keep = frame[duration].notna() & frame[event].notna() & (frame[duration] >= 0)
    frame = frame[keep].reset_index(drop=True)
    if not set(frame[event].unique()) <= {0, 1, 0.0, 1.0, True, False}:
        raise ValueError(f"`event` column {event!r} must hold 1 (event seen) or 0 (censored); found {sorted(frame[event].unique())[:6]}")
    names = as_list(covariates) or [c for c in frame.columns if c not in (duration, event)]
    skip = set(as_list(drop) or [])
    names = [c for c in names if c not in skip and c not in (duration, event)]
    require_columns(frame, names)
    names = [c for c in names if frame[c].nunique(dropna=True) > 1 and (frame[c].nunique() <= 30 or pd.api.types.is_numeric_dtype(frame[c]))]
    return frame, frame[duration].to_numpy(float), frame[event].to_numpy(int), names


# ---- non-parametric -----------------------------------------------------------------------------
def kaplan_meier(t: np.ndarray, e: np.ndarray) -> pd.DataFrame:
    """Kaplan–Meier table with Greenwood log-log 95% intervals."""
    order = np.argsort(t, kind="stable")
    t, e = t[order], e[order]
    times = np.unique(t[e == 1])
    rows, surv, greenwood = [], 1.0, 0.0
    for u in times:
        n_risk = int((t >= u).sum())
        d = int(((t == u) & (e == 1)).sum())
        surv *= 1 - d / n_risk
        greenwood += d / (n_risk * (n_risk - d)) if n_risk > d else 0.0
        if surv > 0 and greenwood > 0:
            se = np.sqrt(greenwood) / abs(np.log(surv))
            lo, hi = surv ** np.exp(1.96 * se), surv ** np.exp(-1.96 * se)
        else:
            lo = hi = surv
        rows.append({"time": float(u), "at_risk": n_risk, "events": d, "survival": surv, "lower": lo, "upper": hi})
    return pd.DataFrame(rows)


def step_value(table: pd.DataFrame, column: str, at: np.ndarray, start: float = 1.0) -> np.ndarray:
    """Value of a step function (a KM / cumulative-hazard column) at arbitrary times."""
    if table.empty:
        return np.full(len(at), start)
    idx = np.searchsorted(table["time"].to_numpy(), np.asarray(at, float), side="right") - 1
    values = table[column].to_numpy()
    return np.where(idx >= 0, values[np.clip(idx, 0, len(values) - 1)], start)


def median_time(table: pd.DataFrame, level: float = 0.5) -> float | None:
    below = table[table["survival"] <= level]
    return float(below["time"].iloc[0]) if len(below) else None


def nelson_aalen(t: np.ndarray, e: np.ndarray) -> pd.DataFrame:
    order = np.argsort(t, kind="stable")
    t, e = t[order], e[order]
    rows, H = [], 0.0
    for u in np.unique(t[e == 1]):
        n_risk, d = int((t >= u).sum()), int(((t == u) & (e == 1)).sum())
        H += d / n_risk
        rows.append({"time": float(u), "at_risk": n_risk, "events": d, "cumulative_hazard": H})
    return pd.DataFrame(rows)


def concordance_index(t: np.ndarray, e: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C: among comparable pairs (the earlier time is an event), the share where higher risk failed earlier."""
    t, e, risk = np.asarray(t, float), np.asarray(e, int), np.asarray(risk, float)
    concordant = comparable = 0.0
    for i in np.flatnonzero(e == 1):
        later = t > t[i]
        comparable += later.sum()
        concordant += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return float(concordant / comparable) if comparable else float("nan")


# ---- Cox ----------------------------------------------------------------------------------------
def _cox_terms(beta: np.ndarray, X: np.ndarray, t: np.ndarray, e: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    order = np.argsort(t, kind="stable")
    X, t, e = X[order], t[order], e[order]
    eta = X @ beta
    w = np.exp(eta - eta.max())
    first = np.searchsorted(t, t, side="left")  # first row of each tie group = start of its risk set
    S0 = np.cumsum(w[::-1])[::-1][first]
    S1 = np.cumsum((w[:, None] * X)[::-1], axis=0)[::-1][first]
    S2 = np.cumsum((w[:, None, None] * X[:, :, None] * X[:, None, :])[::-1], axis=0)[::-1][first]
    ev = e == 1
    mean = S1[ev] / S0[ev, None]
    loglik = float((eta[ev] - eta.max() - np.log(S0[ev])).sum())
    grad = (X[ev] - mean).sum(axis=0)
    hess = -(S2[ev] / S0[ev, None, None] - mean[:, :, None] * mean[:, None, :]).sum(axis=0)
    return loglik, grad, hess


def cox_fit(X: np.ndarray, t: np.ndarray, e: np.ndarray, l2: float = 0.0, max_iter: int = 50, tol: float = 1e-7) -> dict[str, Any]:
    """Cox partial likelihood by Newton–Raphson with step halving (Breslow ties) and an optional ridge penalty."""
    p = X.shape[1]
    beta = np.zeros(p)
    ll, grad, hess = _cox_terms(beta, X, t, e)
    ll -= 0.5 * l2 * beta @ beta
    for _ in range(max_iter):
        g, H = grad - l2 * beta, hess - l2 * np.eye(p)
        step = np.linalg.solve(-H - 1e-9 * np.eye(p), g)
        scale = 1.0
        while scale > 1e-4:
            cand = beta + scale * step
            new_ll, new_grad, new_hess = _cox_terms(cand, X, t, e)
            new_ll -= 0.5 * l2 * cand @ cand
            if new_ll >= ll - 1e-12:
                break
            scale /= 2
        converged = abs(new_ll - ll) < tol
        beta, ll, grad, hess = cand, new_ll, new_grad, new_hess
        if converged:
            break
    info = -(hess - l2 * np.eye(p))
    cov = np.linalg.pinv(info)
    return {"beta": beta, "cov": cov, "loglik": ll, "se": np.sqrt(np.clip(np.diag(cov), 0, None))}


def breslow_baseline(X: np.ndarray, t: np.ndarray, e: np.ndarray, beta: np.ndarray) -> pd.DataFrame:
    """Baseline cumulative hazard at each event time (Breslow)."""
    risk = np.exp(X @ beta - (X @ beta).max())
    shift = np.exp((X @ beta).max())
    rows, H = [], 0.0
    for u in np.unique(t[e == 1]):
        d = int(((t == u) & (e == 1)).sum())
        H += d / (risk[t >= u].sum() * shift)
        rows.append({"time": float(u), "cumulative_hazard": H})
    return pd.DataFrame(rows)


# ---- models -------------------------------------------------------------------------------------
class SurvivalModel:
    """Base: a design + a fitted model. `risk` is higher for earlier expected events; `survival` is S(t|x)."""

    kind = "base"

    def __init__(self, design: Design):
        self.design = design

    def risk(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def survival(self, frame: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CoxModel(SurvivalModel):
    kind = "cox"

    def __init__(self, design: Design, beta: np.ndarray, baseline: pd.DataFrame):
        super().__init__(design)
        self.beta, self.baseline = beta, baseline

    def risk(self, frame: pd.DataFrame) -> np.ndarray:
        return self.design.transform(frame) @ self.beta

    def survival(self, frame: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        H0 = step_value(self.baseline, "cumulative_hazard", np.asarray(times, float), start=0.0)
        return np.exp(-np.outer(np.exp(self.risk(frame)), H0))


def _dist_terms(dist: str, z: np.ndarray, log_t: np.ndarray, log_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """(log density of T, log survival of T) under an AFT model where z = (log t - mu) / sigma."""
    from scipy import stats

    if dist in ("weibull", "exponential"):
        log_s = -np.exp(z)
        log_f = z - np.exp(z) - log_sigma - log_t
    elif dist == "lognormal":
        log_s = stats.norm.logsf(z)
        log_f = stats.norm.logpdf(z) - log_sigma - log_t
    elif dist == "loglogistic":
        log_s = -np.logaddexp(0, z)
        log_f = z - 2 * np.logaddexp(0, z) - log_sigma - log_t
    else:
        raise ValueError(f"unknown distribution {dist!r}")
    return log_f, log_s


class ParametricModel(SurvivalModel):
    kind = "parametric"

    def __init__(self, design: Design, dist: str, intercept: float, beta: np.ndarray, log_sigma: float):
        super().__init__(design)
        self.dist, self.intercept, self.beta, self.log_sigma = dist, intercept, beta, log_sigma

    def mu(self, frame: pd.DataFrame) -> np.ndarray:
        return self.intercept + self.design.transform(frame) @ self.beta

    def risk(self, frame: pd.DataFrame) -> np.ndarray:
        return -self.mu(frame)  # a shorter expected log-time means higher risk

    def survival(self, frame: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        z = (np.log(np.maximum(np.asarray(times, float), 1e-12))[None, :] - self.mu(frame)[:, None]) / np.exp(self.log_sigma)
        return np.exp(_dist_terms(self.dist, z, np.log(np.maximum(times, 1e-12))[None, :], self.log_sigma)[1])


def parametric_fit(dist: str, X: np.ndarray, t: np.ndarray, e: np.ndarray, l2: float = 0.0) -> dict[str, Any]:
    from scipy.optimize import minimize

    p = X.shape[1]
    log_t = np.log(np.maximum(t, 1e-12))
    fixed_sigma = dist == "exponential"

    def nll(theta: np.ndarray) -> float:
        b0, beta = theta[0], theta[1 : 1 + p]
        log_sigma = 0.0 if fixed_sigma else theta[-1]
        z = (log_t - b0 - X @ beta) / np.exp(log_sigma)
        log_f, log_s = _dist_terms(dist, z, log_t, log_sigma)
        return -float(np.sum(np.where(e == 1, log_f, log_s))) + 0.5 * l2 * float(beta @ beta)

    start = np.concatenate([[log_t.mean()], np.zeros(p)] + ([] if fixed_sigma else [[np.log(max(log_t.std(), 0.1))]]))
    res = minimize(nll, start, method="L-BFGS-B")
    theta = res.x
    # Standard errors from a numerical Hessian of the negative log-likelihood.
    h, k = 1e-4, len(theta)
    H = np.zeros((k, k))
    f0 = nll(theta)
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.eye(k)[i] * h, np.eye(k)[j] * h
            H[i, j] = H[j, i] = (nll(theta + ei + ej) - nll(theta + ei) - nll(theta + ej) + f0) / h**2
    cov = np.linalg.pinv(H + 1e-9 * np.eye(k))
    return {"intercept": float(theta[0]), "beta": theta[1 : 1 + p], "log_sigma": 0.0 if fixed_sigma else float(theta[-1]),
            "loglik": -float(res.fun), "se": np.sqrt(np.clip(np.diag(cov), 0, None)), "n_params": k, "converged": bool(res.success)}


def make_bins(t: np.ndarray, e: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(t[e == 1] if (e == 1).sum() >= n_bins else t, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.concatenate([[0.0], edges, [np.inf]])


def person_period(Z: np.ndarray, t: np.ndarray, e: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expand each subject into one row per interval they were at risk in; label 1 on the interval of their event."""
    rows, labels = [], []
    n_bins = len(edges) - 1
    for i in range(len(t)):
        last = min(np.searchsorted(edges, t[i], side="right") - 1, n_bins - 1)
        for b in range(last + 1):
            onehot = np.zeros(n_bins)
            onehot[b] = 1
            rows.append(np.concatenate([onehot, Z[i]]))
            labels.append(int(e[i] == 1 and b == last))
    return np.asarray(rows), np.asarray(labels)


class DiscreteModel(SurvivalModel):
    kind = "discrete_time"

    def __init__(self, design: Design, clf: Any, edges: np.ndarray):
        super().__init__(design)
        self.clf, self.edges = clf, edges

    def hazards(self, frame: pd.DataFrame) -> np.ndarray:
        Z = self.design.transform(frame)
        n_bins = len(self.edges) - 1
        out = np.zeros((len(Z), n_bins))
        for b in range(n_bins):
            onehot = np.zeros((len(Z), n_bins))
            onehot[:, b] = 1
            out[:, b] = self.clf.predict_proba(np.hstack([onehot, Z]))[:, 1]
        return out

    def risk(self, frame: pd.DataFrame) -> np.ndarray:
        return -np.log(np.clip(np.prod(1 - self.hazards(frame), axis=1), 1e-12, 1))  # cumulative hazard over the whole horizon

    def survival(self, frame: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        cum = np.cumprod(1 - self.hazards(frame), axis=1)
        idx = np.clip(np.searchsorted(self.edges, np.asarray(times, float), side="right") - 1, 0, cum.shape[1] - 1)
        return cum[:, idx]


class SksurvModel(SurvivalModel):
    kind = "sksurv"

    def __init__(self, design: Design, model: Any):
        super().__init__(design)
        self.model = model

    def risk(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(self.design.transform(frame)), float)

    def survival(self, frame: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        fns = self.model.predict_survival_function(self.design.transform(frame))
        return np.array([[fn(min(max(x, fn.domain[0]), fn.domain[1])) for x in times] for fn in fns])


# ---- shared reporting ---------------------------------------------------------------------------
def default_horizons(t: np.ndarray, e: np.ndarray, n: int = 3) -> np.ndarray:
    observed = t[e == 1] if (e == 1).any() else t
    return np.unique(np.quantile(observed, np.linspace(0.25, 0.75, n)))


def holdout_survival(frame: pd.DataFrame, duration: str, event: str, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sk()
    from sklearn.model_selection import train_test_split

    return train_test_split(frame, test_size=test_size, random_state=seed, stratify=frame[event] if frame[event].nunique() > 1 else None)


def evaluate_model(model: SurvivalModel, frame: pd.DataFrame, duration: str, event: str, horizons: np.ndarray | None = None) -> dict[str, Any]:
    t, e = frame[duration].to_numpy(float), frame[event].to_numpy(int)
    out: dict[str, Any] = {"concordance": round(concordance_index(t, e, model.risk(frame)), 5)}
    horizons = default_horizons(t, e) if horizons is None else np.asarray(horizons, float)
    out["horizons"] = [round(float(h), 4) for h in horizons]
    out["brier_at_horizons"] = [round(float(brier_at(model, frame, duration, event, h)), 5) for h in horizons]
    return out


def brier_at(model: SurvivalModel, frame: pd.DataFrame, duration: str, event: str, horizon: float) -> float:
    """IPCW Brier score at `horizon`, weighting by the Kaplan–Meier estimate of the censoring distribution."""
    t, e = frame[duration].to_numpy(float), frame[event].to_numpy(int)
    S = model.survival(frame, np.array([horizon]))[:, 0]
    censor = kaplan_meier(t, 1 - e)
    G = lambda x: step_value(censor, "survival", np.atleast_1d(x))  # noqa: E731
    died = (t <= horizon) & (e == 1)
    alive = t > horizon
    w = np.zeros(len(t))
    w[died] = 1 / np.clip(G(t[died] - 1e-9), 1e-3, None)
    w[alive] = 1 / np.clip(G(horizon), 1e-3, None)
    return float(np.mean(w * np.where(died, S**2, 0) + w * np.where(alive, (1 - S) ** 2, 0)))


def run_cox(data: Any, duration: str, event: str, covariates: Any, drop: Any, l2: float, test_size: float, seed: int, model_path: str | None, label: str) -> dict[str, Any]:
    frame, t, e, names = load_survival(data, duration, event, covariates, drop)
    train, test = holdout_survival(frame, duration, event, test_size, seed)
    design = Design.fit(train, names)
    Xtr = design.transform(train)
    fit = cox_fit(Xtr, train[duration].to_numpy(float), train[event].to_numpy(int), l2)
    model = CoxModel(design, fit["beta"], breslow_baseline(Xtr, train[duration].to_numpy(float), train[event].to_numpy(int), fit["beta"]))
    z = fit["beta"] / np.where(fit["se"] > 0, fit["se"], np.nan)
    from scipy import stats

    sd = design.std
    table = [{"covariate": n, "hazard_ratio_per_sd": round(float(np.exp(b)), 4), "hazard_ratio_per_unit": round(float(np.exp(b / s)), 4),
              "ci_low": round(float(np.exp(b - 1.96 * se)), 4), "ci_high": round(float(np.exp(b + 1.96 * se)), 4),
              "p_value": round(float(2 * stats.norm.sf(abs(zz))), 5) if np.isfinite(zz) else None}
             for n, b, se, s, zz in zip(design.names, fit["beta"], fit["se"], sd, z)]
    table.sort(key=lambda r: (r["p_value"] is None, r["p_value"] or 1))
    return {"model": label, "rows_train": len(train), "rows_test": len(test), "events_train": int(train[event].sum()),
            "coefficients": table, "partial_log_likelihood": round(float(fit["loglik"]), 4),
            "train": evaluate_model(model, train, duration, event), "holdout": evaluate_model(model, test, duration, event),
            "model_path": save_model(model, model_path)}
