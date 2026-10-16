"""Fit a survival model: Cox, regularised Cox, parametric accelerated-failure-time, discrete-time hazard models, and forests."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ml_core.ml import save_model, sk
from ml_core.tooling import tool
from supervised.survival_analysis.tools._common import (
    CoxModel,
    Design,
    DiscreteModel,
    ParametricModel,
    SksurvModel,
    breslow_baseline,
    concordance_index,
    cox_fit,
    evaluate_model,
    holdout_survival,
    load_survival,
    make_bins,
    parametric_fit,
    person_period,
    run_cox,
)


@tool
def fit_cox_proportional_hazards(
    data: str, duration: str, event: str, covariates: list[str] | None = None, drop: list[str] | None = None,
    test_size: float = 0.25, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit a Cox proportional-hazards model (Breslow ties): hazard ratios with confidence intervals and p-values per covariate, plus concordance and Brier scores on train and held-out rows. The standard interpretable survival model.

    Args:
        covariates: Covariate columns; defaults to every column except duration and event.
    """
    return run_cox(data, duration, event, covariates, drop, 0.0, test_size, seed, model_path, "cox_ph")


@tool
def fit_cox_regularised(
    data: str, duration: str, event: str, covariates: list[str] | None = None, drop: list[str] | None = None,
    l2: float = 1.0, test_size: float = 0.25, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit a ridge-penalised Cox model — the version to use when there are many covariates, correlated ones, or fewer than ~10 events per covariate.

    Args:
        covariates: Covariate columns; defaults to every column except duration and event.
        l2: Ridge penalty on the standardised coefficients.
    """
    return run_cox(data, duration, event, covariates, drop, l2, test_size, seed, model_path, f"cox_ridge(l2={l2})")


def _parametric(dist: str, data, duration, event, covariates, drop, l2, test_size, seed, model_path):
    frame, t, e, names = load_survival(data, duration, event, covariates, drop)
    train, test = holdout_survival(frame, duration, event, test_size, seed)
    design = Design.fit(train, names)
    fit = parametric_fit(dist, design.transform(train), train[duration].to_numpy(float), train[event].to_numpy(int), l2)
    model = ParametricModel(design, dist, fit["intercept"], fit["beta"], fit["log_sigma"])
    k = fit["n_params"]
    n_events = int(train[event].sum())
    coefs = [{"covariate": n, "log_time_ratio_per_sd": round(float(b), 4), "time_ratio_per_sd": round(float(np.exp(b)), 4), "std_error": round(float(se), 4)}
             for n, b, se in zip(design.names, fit["beta"], fit["se"][1 : 1 + len(fit["beta"])])]
    shape = None
    if dist == "weibull":
        shape = round(float(1 / np.exp(fit["log_sigma"])), 4)
    return {"model": f"{dist}_aft", "rows_train": len(train), "events_train": n_events, "converged": fit["converged"],
            "log_likelihood": round(fit["loglik"], 3), "aic": round(2 * k - 2 * fit["loglik"], 3),
            "weibull_shape": shape, "scale_sigma": round(float(np.exp(fit["log_sigma"])), 4),
            "coefficients": sorted(coefs, key=lambda r: -abs(r["log_time_ratio_per_sd"])),
            "reading": "time ratio > 1 means the covariate lengthens survival; < 1 shortens it" + ("; shape > 1 means the hazard rises with time, < 1 that it falls" if shape else ""),
            "train": evaluate_model(model, train, duration, event), "holdout": evaluate_model(model, test, duration, event),
            "model_path": save_model(model, model_path)}


@tool
def fit_parametric_aft(
    data: str, duration: str, event: str, distribution: Literal["weibull", "lognormal", "loglogistic", "exponential"] = "weibull",
    covariates: list[str] | None = None, drop: list[str] | None = None, l2: float = 0.0,
    test_size: float = 0.25, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit an accelerated-failure-time model with a chosen distribution (weibull, lognormal, loglogistic, exponential) by censored maximum likelihood — gives full survival curves and time ratios, and works when the proportional-hazards assumption fails.

    Args:
        distribution: weibull (monotone hazard), lognormal / loglogistic (hazard rises then falls), exponential (constant hazard).
        covariates: Covariate columns; defaults to every column except duration and event.
        l2: Ridge penalty on the standardised coefficients.
    """
    return _parametric(distribution, data, duration, event, covariates, drop, l2, test_size, seed, model_path)


@tool
def fit_weibull_aft(
    data: str, duration: str, event: str, covariates: list[str] | None = None, drop: list[str] | None = None,
    l2: float = 0.0, test_size: float = 0.25, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit a Weibull model — the workhorse parametric survival model; its shape parameter says whether the risk rises (wear-out) or falls (early failures) with time.

    Args:
        covariates: Covariate columns; defaults to every column except duration and event.
        l2: Ridge penalty on the standardised coefficients.
    """
    return _parametric("weibull", data, duration, event, covariates, drop, l2, test_size, seed, model_path)


@tool
def fit_exponential_model(
    data: str, duration: str, event: str, covariates: list[str] | None = None, drop: list[str] | None = None,
    test_size: float = 0.25, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit an exponential model — a constant hazard over time; the simplest parametric baseline, right when the event is memoryless (random failures).

    Args:
        covariates: Covariate columns; defaults to every column except duration and event.
    """
    return _parametric("exponential", data, duration, event, covariates, drop, 0.0, test_size, seed, model_path)


@tool
def fit_lognormal_aft(
    data: str, duration: str, event: str, covariates: list[str] | None = None, drop: list[str] | None = None,
    l2: float = 0.0, test_size: float = 0.25, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit a log-normal model — the hazard rises then falls, which suits events with a peak period (time to repair, time to first purchase).

    Args:
        covariates: Covariate columns; defaults to every column except duration and event.
        l2: Ridge penalty on the standardised coefficients.
    """
    return _parametric("lognormal", data, duration, event, covariates, drop, l2, test_size, seed, model_path)


@tool
def fit_discrete_time_logistic(
    data: str, duration: str, event: str, covariates: list[str] | None = None, drop: list[str] | None = None,
    n_bins: int = 8, C: float = 1.0, test_size: float = 0.25, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit a discrete-time hazard model: the follow-up is cut into intervals, each subject becomes one row per interval at risk, and a logistic regression predicts the event in each. Needs no proportional-hazards assumption and handles non-monotone hazards.

    Args:
        covariates: Covariate columns; defaults to every column except duration and event.
        n_bins: Number of time intervals (cut at event-time quantiles).
        C: Inverse regularisation strength.
    """
    sk()
    from sklearn.linear_model import LogisticRegression

    frame, t, e, names = load_survival(data, duration, event, covariates, drop)
    train, test = holdout_survival(frame, duration, event, test_size, seed)
    design = Design.fit(train, names)
    edges = make_bins(train[duration].to_numpy(float), train[event].to_numpy(int), n_bins)
    Xpp, ypp = person_period(design.transform(train), train[duration].to_numpy(float), train[event].to_numpy(int), edges)
    clf = LogisticRegression(C=C, max_iter=2000).fit(Xpp, ypp)
    model = DiscreteModel(design, clf, edges)
    beta = clf.coef_[0][len(edges) - 1 :]
    return {"model": f"discrete_time_logistic(bins={len(edges) - 1})", "person_period_rows": len(ypp), "interval_edges": [round(float(x), 3) for x in edges[:-1]] + ["inf"],
            "coefficients": sorted([{"covariate": n, "log_odds_per_sd": round(float(b), 4)} for n, b in zip(design.names, beta)], key=lambda r: -abs(r["log_odds_per_sd"])),
            "train": evaluate_model(model, train, duration, event), "holdout": evaluate_model(model, test, duration, event), "model_path": save_model(model, model_path)}


@tool
def fit_discrete_time_boosting(
    data: str, duration: str, event: str, covariates: list[str] | None = None, drop: list[str] | None = None,
    n_bins: int = 8, max_iter: int = 200, learning_rate: float = 0.08, test_size: float = 0.25, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit a non-linear survival model without extra packages: the discrete-time person-period expansion with gradient-boosted trees, so interactions and thresholds between covariates and time are learned automatically.

    Args:
        covariates: Covariate columns; defaults to every column except duration and event.
        n_bins: Number of time intervals (cut at event-time quantiles).
        max_iter: Boosting rounds.
        learning_rate: Shrinkage per round.
    """
    sk()
    from sklearn.ensemble import HistGradientBoostingClassifier

    frame, t, e, names = load_survival(data, duration, event, covariates, drop)
    train, test = holdout_survival(frame, duration, event, test_size, seed)
    design = Design.fit(train, names)
    edges = make_bins(train[duration].to_numpy(float), train[event].to_numpy(int), n_bins)
    Xpp, ypp = person_period(design.transform(train), train[duration].to_numpy(float), train[event].to_numpy(int), edges)
    clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=learning_rate, random_state=seed).fit(Xpp, ypp)
    model = DiscreteModel(design, clf, edges)
    return {"model": f"discrete_time_boosting(bins={len(edges) - 1})", "person_period_rows": len(ypp),
            "train": evaluate_model(model, train, duration, event), "holdout": evaluate_model(model, test, duration, event), "model_path": save_model(model, model_path)}


def _sksurv(kind: str, data, duration, event, covariates, drop, test_size, seed, model_path, **params):
    frame, t, e, names = load_survival(data, duration, event, covariates, drop)
    train, test = holdout_survival(frame, duration, event, test_size, seed)
    design = Design.fit(train, names)
    from sksurv.util import Surv

    y = Surv.from_arrays(train[event].astype(bool).to_numpy(), train[duration].to_numpy(float))
    if kind == "forest":
        from sksurv.ensemble import RandomSurvivalForest
        est = RandomSurvivalForest(n_estimators=params.get("n_estimators", 200), min_samples_leaf=params.get("min_samples_leaf", 10), n_jobs=-1, random_state=seed)
    else:
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        est = GradientBoostingSurvivalAnalysis(n_estimators=params.get("n_estimators", 200), learning_rate=params.get("learning_rate", 0.1), random_state=seed)
    est.fit(design.transform(train), y)
    model = SksurvModel(design, est)
    return {"model": f"sksurv_{kind}", "rows_train": len(train), "train": evaluate_model(model, train, duration, event),
            "holdout": evaluate_model(model, test, duration, event), "model_path": save_model(model, model_path)}


@tool(requires=("sksurv",))
def fit_random_survival_forest(
    data: str, duration: str, event: str, covariates: list[str] | None = None, drop: list[str] | None = None,
    n_estimators: int = 200, min_samples_leaf: int = 10, test_size: float = 0.25, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit a random survival forest (scikit-survival) — a non-linear, assumption-free model that captures interactions and non-proportional effects; judge it by concordance and calibration, not coefficients.

    Args:
        covariates: Covariate columns; defaults to every column except duration and event.
        min_samples_leaf: Smallest leaf; larger smooths the survival curves.
    """
    return _sksurv("forest", data, duration, event, covariates, drop, test_size, seed, model_path, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)


@tool(requires=("sksurv",))
def fit_gradient_boosted_survival(
    data: str, duration: str, event: str, covariates: list[str] | None = None, drop: list[str] | None = None,
    n_estimators: int = 200, learning_rate: float = 0.1, test_size: float = 0.25, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Fit gradient-boosted survival trees (scikit-survival, Cox loss) — usually the most accurate ranking of risk on tabular data.

    Args:
        covariates: Covariate columns; defaults to every column except duration and event.
        n_estimators: Boosting rounds.
        learning_rate: Shrinkage per round.
    """
    return _sksurv("boosting", data, duration, event, covariates, drop, test_size, seed, model_path, n_estimators=n_estimators, learning_rate=learning_rate)


@tool
def compare_survival_models(
    data: str, duration: str, event: str, covariates: list[str] | None = None, drop: list[str] | None = None,
    folds: int = 4, seed: int = 42,
) -> dict:
    """Cross-validate Cox, ridge-Cox, Weibull, log-normal and discrete-time models on identical folds and rank them by concordance — the quickest way to see whether a simple model is enough.

    Args:
        covariates: Covariate columns; defaults to every column except duration and event.
        folds: Number of cross-validation folds.
    """
    sk()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    frame, t, e, names = load_survival(data, duration, event, covariates, drop)
    kinds = ["cox", "cox_ridge", "weibull", "lognormal", "discrete_time"]
    scores: dict[str, list[float]] = {k: [] for k in kinds}
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(frame, e):
        train, test = frame.iloc[tr], frame.iloc[te]
        design = Design.fit(train, names)
        Xtr, ttr, etr = design.transform(train), train[duration].to_numpy(float), train[event].to_numpy(int)
        for kind in kinds:
            try:
                if kind in ("cox", "cox_ridge"):
                    fit = cox_fit(Xtr, ttr, etr, 0.0 if kind == "cox" else 1.0)
                    model = CoxModel(design, fit["beta"], breslow_baseline(Xtr, ttr, etr, fit["beta"]))
                elif kind in ("weibull", "lognormal"):
                    fit = parametric_fit(kind, Xtr, ttr, etr)
                    model = ParametricModel(design, kind, fit["intercept"], fit["beta"], fit["log_sigma"])
                else:
                    edges = make_bins(ttr, etr, 6)
                    Xpp, ypp = person_period(Xtr, ttr, etr, edges)
                    model = DiscreteModel(design, LogisticRegression(max_iter=2000).fit(Xpp, ypp), edges)
                scores[kind].append(concordance_index(test[duration].to_numpy(float), test[event].to_numpy(int), model.risk(test)))
            except Exception:
                pass
    board = [{"model": k, "concordance": round(float(np.mean(v)), 5), "std": round(float(np.std(v)), 5), "folds_ok": len(v)} for k, v in scores.items() if v]
    board.sort(key=lambda r: -r["concordance"])
    return {"folds": folds, "metric": "Harrell concordance (0.5 = random, 1 = perfect)", "leaderboard": board, "best": board[0]["model"] if board else None}
