"""Multi-view semi-supervised machinery: choosing views, the co-training loop, random-subspace co-training,
tri-training and democratic co-learning. A *view* is a group of the original columns; classifiers trained on
different views make different mistakes, which is what lets them teach each other."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_core.data import as_list
from ml_core.semi import Context, base_clf, full_proba, metrics, sk


# ---- views --------------------------------------------------------------------------------------
def choose_views(ctx: Context, prep: Any, views: Any, strategy: str, seed: int) -> list[list[str]]:
    """Two groups of original columns. Given explicitly, or built by `strategy`: random, halves (in column order), or correlation (greedy
    split that puts strongly correlated columns on the same side, so the views carry as little shared information as possible)."""
    if views:
        groups = [as_list(v) for v in views]
        if len(groups) < 2 or any(not g for g in groups):
            raise ValueError("`views` must be at least two non-empty lists of column names")
        return groups
    cols = list(ctx.sd.X.columns)
    if len(cols) < 2:
        raise ValueError("co-training needs at least two feature columns to split into views")
    if strategy == "halves":
        return [cols[: len(cols) // 2], cols[len(cols) // 2 :]]
    if strategy == "random":
        order = np.random.default_rng(seed).permutation(len(cols))
        return [[cols[i] for i in sorted(order[: len(cols) // 2])], [cols[i] for i in sorted(order[len(cols) // 2 :])]]
    num = ctx.sd.X.apply(lambda s: pd.to_numeric(s, errors="coerce") if pd.api.types.is_numeric_dtype(s) else s.astype("category").cat.codes)
    corr = num.corr().abs().fillna(0).to_numpy(copy=True)
    np.fill_diagonal(corr, 0)
    i, j = np.unravel_index(corr.argmin(), corr.shape)  # seed the two views with the two least-related columns
    left, right = {int(i)}, {int(j if j != i else (i + 1) % len(cols))}
    for c in range(len(cols)):
        if c not in left | right:
            (left if corr[c, list(left)].mean() >= corr[c, list(right)].mean() else right).add(c)
    return [[cols[c] for c in sorted(left)], [cols[c] for c in sorted(right)]]


def view_dependence(ctx: Context, groups: list[list[str]]) -> float:
    """Mean absolute correlation between columns of different views, within each class — the conditional-independence assumption of co-training."""
    X = ctx.sd.X.apply(lambda s: pd.to_numeric(s, errors="coerce") if pd.api.types.is_numeric_dtype(s) else s.astype("category").cat.codes)
    a, b = [c for c in groups[0] if c in X], [c for c in groups[1] if c in X]
    vals = []
    for k in range(ctx.K):
        rows = ctx.split.l_train[ctx.yl == k]
        if len(rows) > 5:
            sub = X.iloc[rows]
            corr = sub[a + b].corr().abs().loc[a, b].to_numpy()
            vals.append(np.nanmean(corr))
    return float(np.nanmean(vals)) if vals else float("nan")


# ---- co-training --------------------------------------------------------------------------------
class TwoViewModel:
    """Final predictor of co-training: the two view classifiers' probabilities averaged."""

    def __init__(self, models: list[Any], idx: list[np.ndarray], K: int):
        self.models, self.idx, self.K = models, idx, K
        self.classes_ = np.arange(K)

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        probs = [full_proba(m, Z[:, i], self.K) for m, i in zip(self.models, self.idx)]
        mean = np.mean(probs, axis=0)
        return mean / mean.sum(axis=1, keepdims=True)


def co_train(ctx: Context, idx: list[np.ndarray], rounds: int = 10, per_class: int = 5, threshold: float = 0.8, redraw: bool = False, monitor: bool = True) -> tuple[TwoViewModel, dict[str, Any]]:
    """Blum–Mitchell co-training: each view's classifier labels the pool rows it is surest of, and those rows join the *other* view's training set."""
    Zu = ctx.Zu
    pool = np.arange(len(Zu))
    sets = [(ctx.Zl.copy(), ctx.yl.copy()) for _ in idx]  # per-view labelled sets
    added_idx, added_lab, added_conf, history = [], [], [], []
    Zt, yt = ctx.Z[ctx.split.l_test], ctx.sd.y[ctx.split.l_test]
    rng = np.random.default_rng(ctx.seed)
    all_cols = ctx.Z.shape[1]
    models: list[Any] = []
    for r in range(rounds):
        if redraw and r:
            perm = rng.permutation(all_cols)
            idx = [np.sort(perm[: all_cols // 2]), np.sort(perm[all_cols // 2 :])]
        models = [base_clf(ctx.base, ctx.seed).fit(Zs[:, i], ys) for (Zs, ys), i in zip(sets, idx)]
        if monitor:
            history.append({"round": r, "labelled_per_view": [int(len(s[1])) for s in sets], "pseudo_added_total": int(len(added_idx)),
                            "holdout_accuracy": metrics(yt, TwoViewModel(models, idx, ctx.K).predict_proba(Zt))["accuracy"],
                            "view_accuracy": [metrics(yt, full_proba(m, Zt[:, i], ctx.K))["accuracy"] for m, i in zip(models, idx)]})
        if len(pool) == 0:
            break
        probas = [full_proba(m, Zu[pool][:, i], ctx.K) for m, i in zip(models, idx)]
        taken: dict[int, tuple[int, float]] = {}
        for v, proba in enumerate(probas):
            conf, lab = proba.max(axis=1), proba.argmax(axis=1)
            for k in range(ctx.K):
                rows = np.flatnonzero((lab == k) & (conf >= threshold))
                for row in rows[np.argsort(-conf[rows])[:per_class]]:
                    if row not in taken or conf[row] > taken[row][1]:
                        taken[int(row)] = (k, float(conf[row]), v)
        if not taken:
            break
        rows = np.array(sorted(taken))
        labels = np.array([taken[int(x)][0] for x in rows])
        conf = np.array([taken[int(x)][1] for x in rows])
        teacher = np.array([taken[int(x)][2] for x in rows])
        for v in range(len(idx)):  # a view learns from the *other* view's confident labels
            mine = teacher != v
            Zs, ys = sets[v]
            sets[v] = (np.vstack([Zs, Zu[pool[rows[mine]]]]), np.concatenate([ys, labels[mine]]))
        added_idx += list(ctx.split.unl[pool[rows]])
        added_lab += list(labels)
        added_conf += list(conf)
        pool = np.delete(pool, rows)
    final = TwoViewModel([base_clf(ctx.base, ctx.seed).fit(Zs[:, i], ys) for (Zs, ys), i in zip(sets, idx)], idx, ctx.K)
    return final, {"rounds_run": len(history), "unlabelled_left": int(len(pool)), "trajectory": history, "views": [int(len(i)) for i in idx],
                   "pseudo": {"index": np.asarray(added_idx, int), "label": np.asarray(added_lab, int), "confidence": np.asarray(added_conf, float)}}


# ---- tri-training and democratic co-learning ----------------------------------------------------
class VoteModel:
    def __init__(self, models: list[Any], K: int, weights: np.ndarray | None = None):
        self.models, self.K = models, K
        self.weights = np.ones(len(models)) / len(models) if weights is None else weights / weights.sum()
        self.classes_ = np.arange(K)

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        return sum(w * full_proba(m, Z, self.K) for w, m in zip(self.weights, self.models))


def tri_train(ctx: Context, rounds: int = 10, monitor: bool = True) -> tuple[VoteModel, dict[str, Any]]:
    """Zhou & Li tri-training: three classifiers on bootstrap resamples; a pool row is labelled for one of them when the other two agree on it."""
    rng = np.random.default_rng(ctx.seed)
    n = len(ctx.yl)
    boot = [rng.integers(0, n, n) for _ in range(3)]
    extra_sets: list[tuple[np.ndarray, np.ndarray]] = [(np.empty((0, ctx.Z.shape[1])), np.empty(0, int)) for _ in range(3)]
    Zt, yt = ctx.Z[ctx.split.l_test], ctx.sd.y[ctx.split.l_test]
    history, models = [], []
    final_votes = np.full(len(ctx.Zu), -1)
    for r in range(rounds):
        models = [base_clf(ctx.base, ctx.seed + i).fit(np.vstack([ctx.Zl[boot[i]], extra_sets[i][0]]), np.concatenate([ctx.yl[boot[i]], extra_sets[i][1]])) for i in range(3)]
        preds = np.stack([m.predict(ctx.Zu) if hasattr(m, "predict") else full_proba(m, ctx.Zu, ctx.K).argmax(axis=1) for m in models])
        changed = False
        new_sets = []
        for i in range(3):
            j, k = [x for x in range(3) if x != i]
            agree = preds[j] == preds[k]
            if agree.sum() > 0:
                new_sets.append((ctx.Zu[agree], preds[j][agree]))
                changed |= (len(new_sets[-1][1]) != len(extra_sets[i][1]))
            else:
                new_sets.append((np.empty((0, ctx.Z.shape[1])), np.empty(0, int)))
        extra_sets = new_sets
        agree_all = (preds == preds[0]).all(axis=0)
        final_votes = np.where(agree_all, preds[0], -1)
        if monitor:
            history.append({"round": r, "pseudo_rows_per_classifier": [int(len(s[1])) for s in extra_sets], "holdout_accuracy": metrics(yt, VoteModel(models, ctx.K).predict_proba(Zt))["accuracy"]})
        if not changed:
            break
    got = np.flatnonzero(final_votes >= 0)
    proba = VoteModel(models, ctx.K).predict_proba(ctx.Zu[got]) if len(got) else np.zeros((0, ctx.K))
    return VoteModel(models, ctx.K), {"rounds_run": len(history), "trajectory": history,
                                      "pseudo": {"index": ctx.split.unl[got], "label": final_votes[got], "confidence": proba.max(axis=1) if len(got) else np.array([])}}


def democratic(ctx: Context, rounds: int = 8, threshold: float = 0.7, monitor: bool = True) -> tuple[VoteModel, dict[str, Any]]:
    """Democratic co-learning: learners of different families vote, weighted by their cross-validated accuracy; a row labelled by the weighted
    majority is added only to the training sets of the learners that disagreed with it."""
    sk()
    from sklearn.model_selection import cross_val_score

    names = list(dict.fromkeys([ctx.base, "random_forest", "logistic_regression", "naive_bayes", "knn"]))[:3]
    folds = max(2, min(5, int(np.bincount(ctx.yl).min())))
    weights = np.array([max(float(cross_val_score(base_clf(n, ctx.seed), ctx.Zl, ctx.yl, cv=folds).mean()), 0.05) for n in names])
    sets = [(ctx.Zl.copy(), ctx.yl.copy()) for _ in names]
    pool = np.arange(len(ctx.Zu))
    Zt, yt = ctx.Z[ctx.split.l_test], ctx.sd.y[ctx.split.l_test]
    added_idx, added_lab, added_conf, history = [], [], [], []
    models: list[Any] = []
    for r in range(rounds):
        models = [base_clf(n, ctx.seed).fit(Zs, ys) for n, (Zs, ys) in zip(names, sets)]
        if monitor:
            history.append({"round": r, "labelled_per_learner": [int(len(s[1])) for s in sets], "holdout_accuracy": metrics(yt, VoteModel(models, ctx.K, weights).predict_proba(Zt))["accuracy"]})
        if len(pool) == 0:
            break
        probas = np.stack([full_proba(m, ctx.Zu[pool], ctx.K) for m in models])
        votes = probas.argmax(axis=2)
        score = sum(weights[i] * (np.eye(ctx.K)[votes[i]]) for i in range(len(models))) / weights.sum()
        lab, conf = score.argmax(axis=1), score.max(axis=1)
        chosen = np.flatnonzero(conf >= threshold)
        if len(chosen) == 0:
            break
        for i in range(len(models)):
            wrong = chosen[votes[i][chosen] != lab[chosen]]
            if len(wrong):
                Zs, ys = sets[i]
                sets[i] = (np.vstack([Zs, ctx.Zu[pool[wrong]]]), np.concatenate([ys, lab[wrong]]))
        added_idx += list(ctx.split.unl[pool[chosen]])
        added_lab += list(lab[chosen])
        added_conf += list(conf[chosen])
        pool = np.delete(pool, chosen)
    final = VoteModel([base_clf(n, ctx.seed).fit(Zs, ys) for n, (Zs, ys) in zip(names, sets)], ctx.K, weights)
    return final, {"learners": names, "learner_weights": [round(float(w), 4) for w in weights / weights.sum()], "rounds_run": len(history), "trajectory": history,
                   "pseudo": {"index": np.asarray(added_idx, int), "label": np.asarray(added_lab, int), "confidence": np.asarray(added_conf, float)}}
