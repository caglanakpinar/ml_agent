"""The self-training loop and its variants. One implementation, parameterised, so every `fit_self_training_*`
tool differs only in how it decides which unlabelled rows to trust."""

from __future__ import annotations

from typing import Any

import numpy as np

from ml_core.semi import Context, Fitted, base_clf, fit_weighted, full_proba, metrics, sk


def make_base(ctx: Context, calibrated: bool = False) -> Any:
    model = base_clf(ctx.base, ctx.seed)
    if calibrated and np.bincount(ctx.yl, minlength=ctx.K).min() >= 3:
        from sklearn.calibration import CalibratedClassifierCV

        return CalibratedClassifierCV(model, cv=3, method="sigmoid")
    return model


def select(proba: np.ndarray, rule: str, it: int, max_iter: int, ctx: Context, *, threshold: float, k_best: int | None, end_threshold: float, per_class_prior: np.ndarray | None) -> np.ndarray:
    """Indices (into the pool) of the rows to pseudo-label this round."""
    conf = proba.max(axis=1)
    if rule == "k_best":
        return np.argsort(-conf)[: k_best or max(1, len(conf) // max_iter)]
    if rule == "curriculum":
        thr = threshold + (end_threshold - threshold) * it / max(max_iter - 1, 1)
        return np.flatnonzero(conf >= thr)
    if rule == "balanced":
        pred = proba.argmax(axis=1)
        budget = max(1, int(len(conf) / max_iter))
        chosen = []
        for k in range(ctx.K):
            rows = np.flatnonzero((pred == k) & (conf >= threshold))
            take = max(1, int(round(budget * per_class_prior[k]))) if len(rows) else 0
            chosen += list(rows[np.argsort(-conf[rows])[:take]])
        return np.asarray(chosen, int)
    return np.flatnonzero(conf >= threshold)


def self_train(ctx: Context, rule: str = "threshold", threshold: float = 0.9, max_iter: int = 10, k_best: int | None = None, end_threshold: float = 0.7,
               calibrated: bool = False, confidence_weights: bool = False, ensemble: bool = False, monitor: bool = True) -> tuple[Fitted, dict[str, Any]]:
    """Fit on the labelled rows, label the unlabelled rows the model is sure about, refit, repeat."""
    sk()
    Zl, yl = ctx.Zl.copy(), ctx.yl.copy()
    weights = np.ones(len(yl))
    pool = np.arange(len(ctx.split.unl))
    Zu = ctx.Zu
    prior = np.bincount(ctx.yl, minlength=ctx.K) / len(ctx.yl)
    added_idx, added_lab, added_conf = [], [], []
    history: list[dict[str, Any]] = []
    Zt, yt = ctx.Z[ctx.split.l_test], ctx.sd.y[ctx.split.l_test]

    def learners() -> list[Any]:
        names = [ctx.base, "random_forest", "logistic_regression"] if ensemble else [ctx.base]
        out = []
        for n in dict.fromkeys(names):
            m = base_clf(n, ctx.seed)
            if calibrated and not ensemble and np.bincount(yl, minlength=ctx.K).min() >= 3:
                from sklearn.calibration import CalibratedClassifierCV

                m = CalibratedClassifierCV(m, cv=3, method="sigmoid")
            out.append(m)
        return out

    models: list[Any] = []
    for it in range(max_iter):
        models = [fit_weighted(m, Zl, yl, weights) for m in learners()]
        if monitor:
            history.append({"iteration": it, "labelled_used": int(len(yl)), "pseudo_added_total": int(len(added_idx)),
                            "holdout_accuracy": metrics(yt, full_proba(models[0], Zt, ctx.K))["accuracy"]})
        if len(pool) == 0:
            break
        probas = np.stack([full_proba(m, Zu[pool], ctx.K) for m in models])
        proba = probas.mean(axis=0)
        if ensemble:
            votes = probas.argmax(axis=2)
            agree = (votes == votes[0]).all(axis=0)
            proba = proba * agree[:, None]  # no agreement -> zero confidence, never selected
        chosen = select(proba, rule, it, max_iter, ctx, threshold=threshold, k_best=k_best, end_threshold=end_threshold, per_class_prior=prior)
        if len(chosen) == 0:
            break
        labels, conf = proba[chosen].argmax(axis=1), proba[chosen].max(axis=1)
        Zl = np.vstack([Zl, Zu[pool[chosen]]])
        yl = np.concatenate([yl, labels])
        weights = np.concatenate([weights, conf if confidence_weights else np.ones(len(chosen))])
        added_idx += list(ctx.split.unl[pool[chosen]])
        added_lab += list(labels)
        added_conf += list(conf)
        pool = np.delete(pool, chosen)
    final = fit_weighted(base_clf(ctx.base, ctx.seed), Zl, yl, weights) if not calibrated else fit_weighted(make_base(ctx, True), Zl, yl, weights)
    if monitor:
        history.append({"iteration": len(history), "labelled_used": int(len(yl)), "pseudo_added_total": int(len(added_idx)), "holdout_accuracy": metrics(yt, full_proba(final, Zt, ctx.K))["accuracy"]})
    extra = {"iterations": max(len(history) - 1, 0), "unlabelled_left_unused": int(len(pool)), "trajectory": history,
             "pseudo": {"index": np.asarray(added_idx, int), "label": np.asarray(added_lab, int), "confidence": np.asarray(added_conf, float)},
             "note": "the trajectory's hold-out accuracy is for monitoring only; it was never used to choose which rows to pseudo-label"}
    return Fitted(final, ctx.K, history), extra
