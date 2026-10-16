"""Helpers the ranking tool modules share: ranking metrics, group-aware splitting, and the `Ranker` that
every fit tool returns (so a saved ranker scores new rows the same way regardless of how it was trained)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, as_list
from ml_core.ml import build_preprocessor, save_model, sk


# ---- metrics ------------------------------------------------------------------------------------
def dcg(relevance: np.ndarray, k: int | None = None) -> float:
    rel = np.asarray(relevance, float)[:k]
    return float(((2.0 ** rel - 1) / np.log2(np.arange(2, len(rel) + 2))).sum())


def ndcg(relevance: np.ndarray, scores: np.ndarray, k: int | None = None) -> float | None:
    """NDCG of the order `scores` induces on `relevance`; None when nothing in the list is relevant."""
    relevance, scores = np.asarray(relevance, float), np.asarray(scores, float)
    ideal = dcg(np.sort(relevance)[::-1], k)
    if ideal <= 0:
        return None
    order = np.argsort(-scores, kind="stable")
    return dcg(relevance[order], k) / ideal


def average_precision(relevance: np.ndarray, scores: np.ndarray, k: int | None = None, threshold: float = 1) -> float | None:
    hit = (np.asarray(relevance)[np.argsort(-np.asarray(scores), kind="stable")] >= threshold)[:k]
    total = (np.asarray(relevance) >= threshold).sum()
    if total == 0:
        return None
    precisions = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    return float((precisions * hit).sum() / min(total, k or total))


def reciprocal_rank(relevance: np.ndarray, scores: np.ndarray, threshold: float = 1) -> float | None:
    hit = np.asarray(relevance)[np.argsort(-np.asarray(scores), kind="stable")] >= threshold
    return float(1.0 / (np.argmax(hit) + 1)) if hit.any() else None


def precision_recall(relevance: np.ndarray, scores: np.ndarray, k: int, threshold: float = 1) -> tuple[float, float | None]:
    hit = (np.asarray(relevance)[np.argsort(-np.asarray(scores), kind="stable")] >= threshold)[:k]
    total = int((np.asarray(relevance) >= threshold).sum())
    return float(hit.sum() / k), (float(hit.sum() / total) if total else None)


def per_group_metrics(frame: pd.DataFrame, group: str, relevance: str, score: str, k: int = 10, threshold: float = 1) -> pd.DataFrame:
    """One row per group with NDCG@k, AP@k, reciprocal rank, precision@k, recall@k."""
    rows = []
    for g, part in frame.groupby(group, sort=False):
        rel, sc = part[relevance].to_numpy(float), part[score].to_numpy(float)
        p, r = precision_recall(rel, sc, k, threshold)
        rows.append({group: g, "size": len(part), "ndcg": ndcg(rel, sc, k), "ap": average_precision(rel, sc, k, threshold),
                     "rr": reciprocal_rank(rel, sc, threshold), "precision": p, "recall": r})
    return pd.DataFrame(rows)


def summarise(per_group: pd.DataFrame, k: int) -> dict[str, Any]:
    """Mean of each metric over the groups that have it, plus how many groups had no relevant item."""
    out = {f"ndcg@{k}": per_group["ndcg"].mean(), f"map@{k}": per_group["ap"].mean(), "mrr": per_group["rr"].mean(),
           f"precision@{k}": per_group["precision"].mean(), f"recall@{k}": per_group["recall"].mean()}
    return {"groups": len(per_group), "groups_without_relevant_item": int(per_group["ndcg"].isna().sum()),
            **{key: (round(float(v), 5) if pd.notna(v) else None) for key, v in out.items()}}


# ---- splitting ----------------------------------------------------------------------------------
def split_groups(frame: pd.DataFrame, group: str, test_size: float = 0.25, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by whole groups: a query's rows are never in both train and test."""
    ids = frame[group].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    test_ids = set(rng.choice(ids, size=max(1, int(round(len(ids) * test_size))), replace=False))
    mask = frame[group].isin(test_ids)
    return frame[~mask].copy(), frame[mask].copy()


def load_ranking(data: Any, group: str, relevance: str, features: Any = None, drop: Any = None) -> tuple[pd.DataFrame, list[str]]:
    frame = load_frame(data)
    require_columns(frame, [group, relevance])
    frame = frame[frame[relevance].notna()].reset_index(drop=True)
    names = as_list(features) or [c for c in frame.columns if c not in (group, relevance)]
    names = [c for c in names if c not in set(as_list(drop) or []) and c not in (group, relevance)]
    require_columns(frame, names)
    numeric = [c for c in names if pd.api.types.is_numeric_dtype(frame[c])]
    if not numeric:
        raise ValueError("no numeric feature columns to rank with; pass `features`")
    return frame, numeric


# ---- the ranker ---------------------------------------------------------------------------------
class Ranker:
    """A fitted preprocessor + scoring model. `.scores(frame)` returns one score per row, higher = better."""

    def __init__(self, kind: str, prep: Any, model: Any, columns: list[str]):
        self.kind, self.prep, self.model, self.columns = kind, prep, model, columns

    def scores(self, frame: pd.DataFrame) -> np.ndarray:
        Z = np.asarray(self.prep.transform(frame[self.columns]), float)
        if hasattr(self.model, "scores"):
            return self.model.scores(Z)
        if hasattr(self.model, "predict_proba") and self.kind.startswith("pointwise_class"):
            return self.model.predict_proba(Z)[:, -1]
        return np.asarray(self.model.predict(Z), float)


class ListNetLinear:
    """Linear listwise ranker trained on the top-one softmax cross-entropy (ListNet), full-batch Adam."""

    def __init__(self, epochs: int = 300, lr: float = 0.05, l2: float = 1e-3, seed: int = 0):
        self.epochs, self.lr, self.l2, self.seed = epochs, lr, l2, seed
        self.w: np.ndarray | None = None
        self.loss_history: list[float] = []

    def fit(self, Z: np.ndarray, rel: np.ndarray, groups: np.ndarray) -> "ListNetLinear":
        w = np.zeros(Z.shape[1])
        m, v = np.zeros_like(w), np.zeros_like(w)
        index = [np.flatnonzero(groups == g) for g in np.unique(groups)]
        for step in range(1, self.epochs + 1):
            grad, loss = np.zeros_like(w), 0.0
            for idx in index:
                s = Z[idx] @ w
                ps = np.exp(s - s.max()); ps /= ps.sum()
                gain = 2.0 ** rel[idx] - 1
                py = gain / gain.sum() if gain.sum() > 0 else np.full(len(idx), 1 / len(idx))
                loss -= float((py * np.log(ps + 1e-12)).sum())
                grad += Z[idx].T @ (ps - py)
            grad = grad / len(index) + self.l2 * w
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad ** 2
            w -= self.lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
            if step % 25 == 0:
                self.loss_history.append(round(loss / len(index), 5))
        self.w = w
        return self

    def scores(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.w


def pair_differences(Z: np.ndarray, rel: np.ndarray, groups: np.ndarray, per_group: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample up to `per_group` ordered pairs with different relevance per group; returns (z_i - z_j, 1 if i beats j)."""
    rng = np.random.default_rng(seed)
    diffs, labels = [], []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        pairs = [(i, j) for i in idx for j in idx if rel[i] > rel[j]]
        if not pairs:
            continue
        pick = rng.choice(len(pairs), size=min(per_group, len(pairs)), replace=False)
        for p in pick:
            i, j = pairs[p]
            diffs += [Z[i] - Z[j], Z[j] - Z[i]]
            labels += [1, 0]
    if not diffs:
        raise ValueError("no pairs with different relevance — every group has constant relevance")
    return np.asarray(diffs), np.asarray(labels)


class LinearPairwise:
    """RankNet-style linear scorer: a classifier on feature differences; the score is the learned direction."""

    def __init__(self, estimator: Any):
        self.estimator = estimator
        self.w: np.ndarray | None = None

    def fit(self, Z: np.ndarray, rel: np.ndarray, groups: np.ndarray, per_group: int = 200, seed: int = 0) -> "LinearPairwise":
        D, y = pair_differences(Z, rel, groups, per_group, seed)
        self.estimator.fit(D, y)
        self.w = np.ravel(self.estimator.coef_)
        return self

    def scores(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.w


def fit_ranker(kind: str, train: pd.DataFrame, columns: list[str], group: str, relevance: str, seed: int = 42, **params: Any) -> Ranker:
    """Fit one of the supported ranking approaches on `train`; returns a `Ranker`."""
    sk()
    prep = build_preprocessor(train[columns], scale=True)
    Z = np.asarray(prep.fit_transform(train[columns]), float)
    rel = train[relevance].to_numpy(float)
    groups = train[group].to_numpy()

    if kind == "pointwise_regression":
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=params.get("n_estimators", 200), learning_rate=params.get("learning_rate", 0.08), random_state=seed).fit(Z, rel)
    elif kind == "pointwise_linear":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=params.get("alpha", 1.0)).fit(Z, rel)
    elif kind == "pointwise_random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=params.get("n_estimators", 300), min_samples_leaf=3, n_jobs=-1, random_state=seed).fit(Z, rel)
    elif kind == "pointwise_classification":
        from sklearn.ensemble import HistGradientBoostingClassifier
        y = (rel >= params.get("threshold", 1)).astype(int)
        if y.min() == y.max():
            raise ValueError("the relevance threshold leaves a single class; lower or raise `relevant_from`")
        model = HistGradientBoostingClassifier(max_iter=params.get("n_estimators", 200), learning_rate=0.08, random_state=seed).fit(Z, y)
    elif kind == "pairwise_logistic":
        from sklearn.linear_model import LogisticRegression
        model = LinearPairwise(LogisticRegression(C=params.get("C", 1.0), fit_intercept=False, max_iter=1000)).fit(Z, rel, groups, params.get("pairs_per_group", 200), seed)
    elif kind == "ranksvm":
        from sklearn.svm import LinearSVC
        model = LinearPairwise(LinearSVC(C=params.get("C", 1.0), fit_intercept=False, max_iter=5000, random_state=seed)).fit(Z, rel, groups, params.get("pairs_per_group", 200), seed)
    elif kind == "listnet":
        model = ListNetLinear(params.get("epochs", 300), params.get("learning_rate", 0.05), params.get("l2", 1e-3), seed).fit(Z, rel, groups)
    elif kind in ("lightgbm_lambdarank", "xgboost_rank"):
        order = np.argsort(groups, kind="stable")
        Zs, rs, gs = Z[order], rel[order].astype(int), groups[order]
        sizes = pd.Series(gs).groupby(gs, sort=False).size().to_numpy()
        if kind == "lightgbm_lambdarank":
            from lightgbm import LGBMRanker
            model = LGBMRanker(n_estimators=params.get("n_estimators", 200), learning_rate=params.get("learning_rate", 0.05), num_leaves=params.get("num_leaves", 15), min_child_samples=5, random_state=seed, verbose=-1)
        else:
            from xgboost import XGBRanker
            model = XGBRanker(n_estimators=params.get("n_estimators", 200), learning_rate=params.get("learning_rate", 0.1), random_state=seed)
        model.fit(Zs, rs, group=sizes)
    else:
        raise ValueError(f"unknown ranking approach {kind!r}")
    return Ranker(kind, prep, model, columns)


def run_ranker(
    kind: str, label: str, data: Any, group: str, relevance: str, features: Any, drop: Any, k: int,
    test_size: float, seed: int, model_path: str | None, relevant_from: float = 1, **params: Any,
) -> dict[str, Any]:
    """Split by group, fit `kind`, score the held-out groups and compare with a random order."""
    frame, columns = load_ranking(data, group, relevance, features, drop)
    train, test = split_groups(frame, group, test_size, seed)
    ranker = fit_ranker(kind, train, columns, group, relevance, seed, threshold=relevant_from, **params)
    test = test.assign(_score=ranker.scores(test), _random=np.random.default_rng(seed).random(len(test)))
    model_metrics = summarise(per_group_metrics(test, group, relevance, "_score", k, relevant_from), k)
    random_metrics = summarise(per_group_metrics(test, group, relevance, "_random", k, relevant_from), k)
    result = {"model": label, "groups_train": int(train[group].nunique()), "groups_test": int(test[group].nunique()),
              "features": columns, "holdout": model_metrics, "random_order": {k_: v for k_, v in random_metrics.items() if k_.startswith(("ndcg", "map"))}}
    weights = getattr(ranker.model, "w", None)
    if weights is None:
        weights = getattr(ranker.model, "coef_", None)
    if weights is not None and len(np.ravel(weights)) == len(columns):
        result["feature_weights"] = {c: round(float(w), 4) for c, w in sorted(zip(columns, np.ravel(weights)), key=lambda t: -abs(t[1]))}
    elif hasattr(ranker.model, "feature_importances_") and len(ranker.model.feature_importances_) == len(columns):
        result["feature_importance"] = {c: round(float(w), 4) for c, w in sorted(zip(columns, ranker.model.feature_importances_), key=lambda t: -t[1])}
    result["model_path"] = save_model(ranker, model_path)
    return result
