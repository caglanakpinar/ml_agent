"""Diagnostics every semi-supervised topic shares — written once here, exposed as a tool by each topic's
`tools/diagnostics.py`, so the checks (and their numbers) are identical wherever they are called from."""

from __future__ import annotations

from typing import Any

import numpy as np

from ml_core.ml import build_preprocessor, sk
from ml_core.semi import base_clf, full_proba, load_semi, make_context, metrics


def coverage_profile(data: Any, label_column: str, unlabeled_value: Any, features: Any, drop: Any) -> dict[str, Any]:
    sd = load_semi(data, label_column, features, drop, unlabeled_value)
    counts = np.bincount(sd.y[sd.labeled], minlength=len(sd.classes))
    return {"rows": len(sd.y), "labelled": int(len(sd.labeled)), "unlabelled": int(len(sd.unlabeled)), "labelled_share": round(float(len(sd.labeled) / len(sd.y)), 4),
            "classes": sd.classes, "labelled_per_class": dict(zip(sd.classes, counts.astype(int).tolist())), "class_share_among_labelled": dict(zip(sd.classes, (counts / counts.sum()).round(4).tolist())),
            "labels_per_feature": round(float(len(sd.labeled) / max(sd.X.shape[1], 1)), 2),
            "warnings": [w for w in [
                "a class has under 10 labels: pseudo-labelling will reinforce the classes it knows and starve this one" if counts.min() < 10 else None,
                "labelled share above 50%: unlabelled data adds little; a supervised model is probably enough" if len(sd.labeled) / len(sd.y) > 0.5 else None,
                "fewer than 5 labels per feature: the base learner will be weak, so its early confidence is unreliable" if len(sd.labeled) / max(sd.X.shape[1], 1) < 5 else None] if w]}


def shift_check(data: Any, label_column: str, unlabeled_value: Any, features: Any, drop: Any, k: int, seed: int) -> dict[str, Any]:
    sk()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    sd = load_semi(data, label_column, features, drop, unlabeled_value)
    Z = np.asarray(build_preprocessor(sd.X).fit_transform(sd.X), float)
    flag = (sd.y >= 0).astype(int)
    model = RandomForestClassifier(200, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    auc = float(roc_auc_score(flag, cross_val_predict(model, Z, flag, cv=4, method="predict_proba")[:, 1]))
    model.fit(Z, flag)
    names = sd.X.columns
    step = max(1, len(Z) // 800)
    imp = permutation_importance(model, Z[::step], flag[::step], n_repeats=3, random_state=seed, n_jobs=1) if Z.shape[1] == len(names) else None
    shifted = dict(sorted(zip(names, imp.importances_mean), key=lambda kv: -kv[1])[:k]) if imp is not None else {}
    return {"labelled_vs_unlabelled_auc": round(auc, 4), "most_shifted_features": {c: round(float(v), 4) for c, v in shifted.items()},
            "verdict": "no shift: the unlabelled rows look like the labelled ones" if auc < 0.6 else "moderate shift: pseudo-labels may be unreliable in the shifted region — use a strict threshold" if auc < 0.75
            else "strong shift: the labelled rows are not representative; propagated or pseudo labels will extrapolate — fix the sampling or label a representative batch first"}


def supervised_baseline(data: Any, label_column: str, unlabeled_value: Any, features: Any, drop: Any, base: str, test_size: float, seed: int) -> dict[str, Any]:
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, None, test_size, seed, base)
    model = base_clf(base, seed).fit(ctx.Zl, ctx.yl)
    return {"base": base, "labelled_train": len(ctx.yl), "labelled_test": len(ctx.split.l_test), "holdout": metrics(ctx.sd.y[ctx.split.l_test], full_proba(model, ctx.Z[ctx.split.l_test], ctx.K))}


def scarcity_curve(data: Any, label_column: str, unlabeled_value: Any, features: Any, drop: Any, base: str, test_size: float, seed: int) -> dict[str, Any]:
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, None, test_size, seed, base)
    rng = np.random.default_rng(seed)
    rows = []
    for frac in (0.1, 0.25, 0.5, 1.0):
        n = max(2 * ctx.K, int(len(ctx.yl) * frac))
        idx = np.sort(rng.choice(len(ctx.yl), size=min(n, len(ctx.yl)), replace=False))
        if len(set(ctx.yl[idx])) < 2:
            continue
        m = base_clf(base, seed).fit(ctx.Zl[idx], ctx.yl[idx])
        rows.append({"labels": int(len(idx)), "fraction": frac, **metrics(ctx.sd.y[ctx.split.l_test], full_proba(m, ctx.Z[ctx.split.l_test], ctx.K))})
    slope = rows[-1]["accuracy"] - rows[0]["accuracy"] if len(rows) > 1 else 0
    return {"curve": rows, "verdict": "labels are the bottleneck: unlabelled data has room to help" if slope > 0.05 else "accuracy barely depends on label count: semi-supervised learning will not add much"}


def _oof(ctx: Any, base: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    Z, y = np.vstack([ctx.Zl, ctx.Z[ctx.split.l_test]]), np.concatenate([ctx.yl, ctx.sd.y[ctx.split.l_test]])
    proba = cross_val_predict(base_clf(base, seed), Z, y, cv=StratifiedKFold(min(5, int(np.bincount(y).min())), shuffle=True, random_state=seed), method="predict_proba")
    return Z, y, proba


def confidence_reliability(data: Any, label_column: str, unlabeled_value: Any, features: Any, drop: Any, base: str, seed: int) -> dict[str, Any]:
    sk()
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, None, 0.3, seed, base)
    _, y, proba = _oof(ctx, base, seed)
    conf, correct = proba.max(axis=1), proba.argmax(axis=1) == y
    rows = [{"confidence": f"{lo:.2f}-{min(hi, 1):.2f}", "rows": int(m.sum()), "mean_confidence": round(float(conf[m].mean()), 3), "actual_accuracy": round(float(correct[m].mean()), 3)}
            for lo, hi in [(0.0, 0.6), (0.6, 0.75), (0.75, 0.9), (0.9, 0.97), (0.97, 1.01)] if (m := (conf >= lo) & (conf < hi)).sum() > 0]
    gap = max((r["mean_confidence"] - r["actual_accuracy"] for r in rows if r["rows"] >= 10), default=0)
    return {"bands": rows, "max_overconfidence": round(float(gap), 3),
            "verdict": "confidence is trustworthy" if gap < 0.05 else "over-confident: calibrate the base learner or raise the threshold well above the accuracy you want"}


def threshold_for_precision(data: Any, label_column: str, unlabeled_value: Any, features: Any, drop: Any, base: str, target_precision: float, seed: int) -> dict[str, Any]:
    sk()
    ctx, _ = make_context(data, label_column, features, drop, unlabeled_value, None, 0.3, seed, base)
    Z, y, proba = _oof(ctx, base, seed)
    conf, correct = proba.max(axis=1), proba.argmax(axis=1) == y
    u_conf = full_proba(base_clf(base, seed).fit(Z, y), ctx.Zu, ctx.K).max(axis=1)
    table = []
    for t in np.round(np.arange(0.5, 1.0, 0.05), 2):
        m = conf >= t
        if m.sum() >= 5:
            table.append({"threshold": float(t), "precision": round(float(correct[m].mean()), 4), "labelled_rows_above": int(m.sum()), "unlabelled_rows_above": int((u_conf >= t).sum())})
    ok = [r for r in table if r["precision"] >= target_precision]
    return {"target_precision": target_precision, "recommended_threshold": ok[0]["threshold"] if ok else None, "table": table,
            "note": "no threshold reaches the target precision: the base learner is not accurate enough to pseudo-label safely" if not ok else "the first threshold that meets the target keeps the most rows"}
