"""Fit when the classes are lopsided: reweight, oversample, undersample or synthesise minority rows. Resampling touches the training rows only."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ml_core.ml import build_preprocessor, holdout, save_model, sk, split_xy
from ml_core.tooling import tool
from supervised.classification.tools._common import (
    classification_metrics,
    make_estimator,
)

Base = Literal["logistic_regression", "random_forest", "gradient_boosting", "hist_gradient_boosting", "extra_trees"]


def _smote(Z: np.ndarray, y: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Minimal SMOTE: interpolate between each minority row and one of its k nearest minority neighbours."""
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target_n = counts.max()
    parts_X, parts_y = [Z], [y]
    for cls, n in zip(classes, counts):
        need = target_n - n
        if need <= 0 or n < 2:
            continue
        rows = Z[y == cls]
        nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(rows)
        neighbours = nn.kneighbors(rows, return_distance=False)[:, 1:]
        base = rng.integers(0, n, need)
        pick = neighbours[base, rng.integers(0, neighbours.shape[1], need)]
        gap = rng.random((need, 1))
        parts_X.append(rows[base] + gap * (rows[pick] - rows[base]))
        parts_y.append(np.full(need, cls, dtype=y.dtype))
    return np.vstack(parts_X), np.concatenate(parts_y)


def _random_resample(Z: np.ndarray, y: np.ndarray, mode: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    size = counts.max() if mode == "over" else counts.min()
    idx = []
    for cls in classes:
        rows = np.flatnonzero(y == cls)
        idx.append(rng.choice(rows, size=size, replace=(mode == "over" and len(rows) < size)))
    pick = rng.permutation(np.concatenate(idx))
    return Z[pick], y[pick]


def _fit_resampled(
    sampler: str, base: str, data: Any, target: str, features: Any, drop: Any, k: int, test_size: float,
    seed: int, positive_label: Any, model_path: str | None,
) -> dict[str, Any]:
    sk()
    X, y = split_xy(data, target, features, drop)
    X_tr, X_te, y_tr, y_te = holdout(X, y, test_size, seed, stratify=True)
    prep = build_preprocessor(X_tr, scale=True)
    Z_tr, Z_te = prep.fit_transform(X_tr), prep.transform(X_te)
    labels = y_tr.to_numpy()
    before = {str(c): int(n) for c, n in zip(*np.unique(labels, return_counts=True))}

    if sampler == "smote":
        Z_res, y_res = _smote(np.asarray(Z_tr, float), labels, k, seed)
    else:
        Z_res, y_res = _random_resample(np.asarray(Z_tr, float), labels, sampler, seed)
    after = {str(c): int(n) for c, n in zip(*np.unique(y_res, return_counts=True))}

    model = make_estimator(base, random_state=seed) if base not in ("knn",) else make_estimator(base)
    model.fit(Z_res, y_res)
    classes = list(model.classes_)
    proba = model.predict_proba(Z_te) if hasattr(model, "predict_proba") else None
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("prep", prep), ("model", model)])
    return {
        "model": f"{base}+{sampler}",
        "class_counts_before": before,
        "class_counts_after": after,
        "holdout": classification_metrics(y_te, model.predict(Z_te), proba, classes, positive_label),
        "model_path": save_model(pipe, model_path),
        "note": "Resampling was applied to the training rows only; the held-out rows keep their natural class balance.",
    }


@tool
def fit_with_class_weights(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    base: Base = "logistic_regression", test_size: float = 0.2, seed: int = 42,
    positive_label: str | None = None, model_path: str | None = None,
) -> dict:
    """Fit with class_weight='balanced' so minority-class errors cost proportionally more — no rows are added or removed.

    Args:
        base: Algorithm to fit with balanced class weights.
    """
    from supervised.classification.tools._common import run_classifier

    if base == "gradient_boosting":
        raise ValueError("gradient_boosting has no class_weight; use hist_gradient_boosting or fit_with_smote")
    est = make_estimator(base, random_state=seed, class_weight="balanced")
    return run_classifier(est, f"{base}+class_weight", data, target, features, drop, test_size, seed,
                          base == "logistic_regression", model_path, 0, positive_label)


@tool
def fit_with_random_oversampling(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    base: Base = "random_forest", test_size: float = 0.2, seed: int = 42,
    positive_label: str | None = None, model_path: str | None = None,
) -> dict:
    """Duplicate minority-class training rows until classes are equal, then fit. Simple and effective, but copies can encourage overfitting.

    Args:
        base: Algorithm to fit on the resampled rows.
    """
    return _fit_resampled("over", base, data, target, features, drop, 5, test_size, seed, positive_label, model_path)


@tool
def fit_with_random_undersampling(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    base: Base = "random_forest", test_size: float = 0.2, seed: int = 42,
    positive_label: str | None = None, model_path: str | None = None,
) -> dict:
    """Drop majority-class training rows down to the minority count, then fit. Fast on huge data; throws information away.

    Args:
        base: Algorithm to fit on the resampled rows.
    """
    return _fit_resampled("under", base, data, target, features, drop, 5, test_size, seed, positive_label, model_path)


@tool
def fit_with_smote(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    base: Base = "random_forest", k_neighbors: int = 5, test_size: float = 0.2, seed: int = 42,
    positive_label: str | None = None, model_path: str | None = None,
) -> dict:
    """Synthesise new minority rows by interpolating between neighbours (SMOTE) on the encoded training matrix, then fit.

    Args:
        base: Algorithm to fit on the resampled rows.
        k_neighbors: Minority neighbours to interpolate towards.
    """
    return _fit_resampled("smote", base, data, target, features, drop, k_neighbors, test_size, seed, positive_label, model_path)


@tool
def compare_imbalance_strategies(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    base: Base = "random_forest", test_size: float = 0.2, seed: int = 42,
) -> dict:
    """Fit the same algorithm with no handling, class weights, oversampling, undersampling and SMOTE, and rank them on the same hold-out rows.

    Args:
        base: Algorithm used for every strategy.
    """
    from supervised.classification.tools._common import run_classifier

    rows = []
    plain = run_classifier(make_estimator(base, random_state=seed), base, data, target, features, drop,
                           test_size, seed, base == "logistic_regression", None, 0, None)
    rows.append(("none", plain["holdout"]))
    if base != "gradient_boosting":
        weighted = run_classifier(make_estimator(base, random_state=seed, class_weight="balanced"), base, data,
                                  target, features, drop, test_size, seed, base == "logistic_regression", None, 0, None)
        rows.append(("class_weight", weighted["holdout"]))
    for name in ("over", "under", "smote"):
        res = _fit_resampled(name, base, data, target, features, drop, 5, test_size, seed, None, None)
        rows.append((name, res["holdout"]))
    key = "average_precision" if "average_precision" in plain["holdout"] else "balanced_accuracy"
    ranked = sorted(rows, key=lambda r: r[1].get(key, 0), reverse=True)
    return {
        "ranked_by": key,
        "strategies": [{"strategy": n, **{k: m.get(k) for k in ("balanced_accuracy", "f1_macro", "recall_positive", "precision_positive", "roc_auc", "average_precision") if k in m}} for n, m in ranked],
        "best": ranked[0][0],
    }
