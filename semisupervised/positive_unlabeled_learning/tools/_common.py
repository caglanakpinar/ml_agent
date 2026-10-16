"""Positive-unlabelled (PU) learning machinery.

Only some positives carry a label; everything else is *unlabelled* — a mix of hidden positives and true negatives. A
classifier that treats unlabelled as negative is biased (it learns 'labelled vs not labelled', not 'positive vs negative'),
so the methods here correct for it. They rest on the SCAR assumption: the labelled positives are a *random* sample of all
positives, so a labelled positive looks like any other positive.

Everything is expressed as a score model: `.scores(Z)` is the estimated probability of being positive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml_core.data import as_list, load_frame, require_columns
from ml_core.ml import build_preprocessor, save_model, sk


@dataclass
class PUData:
    X: pd.DataFrame
    s: np.ndarray                  # 1 = labelled positive, 0 = unlabelled
    truth: np.ndarray | None       # 1 positive, 0 negative, -1 unknown
    label_column: str


def load_pu(data: Any, label_column: str, positive_label: Any, features: Any, drop: Any, truth_column: str | None, truth_positive: Any = None) -> PUData:
    frame = load_frame(data)
    require_columns(frame, [label_column])
    raw = frame[label_column]
    s = (raw.astype(str) == str(positive_label)).to_numpy(int) if positive_label is not None else (pd.to_numeric(raw, errors="coerce").fillna(0) > 0).to_numpy(int)
    if s.sum() < 5:
        raise ValueError(f"only {int(s.sum())} labelled positive rows in {label_column!r} (positive_label={positive_label!r}); PU learning needs at least 5, ideally many more")
    truth = None
    skip = {label_column} | set(as_list(drop) or [])
    if truth_column:
        require_columns(frame, [truth_column])
        t = frame[truth_column]
        pos = str(truth_positive if truth_positive is not None else positive_label if positive_label is not None else 1)
        truth = (t.astype(str) == pos).to_numpy(int)
        skip.add(truth_column)
    names = [c for c in (as_list(features) or list(frame.columns)) if c not in skip]
    require_columns(frame, names)
    return PUData(frame[names].reset_index(drop=True), s, truth, label_column)


@dataclass
class PUContext:
    Z: np.ndarray
    pu: PUData
    p_train: np.ndarray            # labelled positives the method may use
    p_test: np.ndarray             # labelled positives held out (never trained on)
    unl: np.ndarray                # unlabelled rows
    seed: int
    base: str
    prior: float | None = None     # class prior pi = P(y=1), if known or estimated

    @property
    def Zp(self) -> np.ndarray:
        return self.Z[self.p_train]

    @property
    def Zu(self) -> np.ndarray:
        return self.Z[self.unl]


def make_pu(data: Any, label_column: str, positive_label: Any, features: Any, drop: Any, truth_column: str | None, test_size: float, seed: int, base: str,
            prior: float | None = None, truth_positive: Any = None) -> tuple[PUContext, Any]:
    pu = load_pu(data, label_column, positive_label, features, drop, truth_column, truth_positive)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(pu.s == 1)
    test = rng.choice(pos, size=max(2, int(len(pos) * test_size)), replace=False)
    train = np.setdiff1d(pos, test)
    prep = build_preprocessor(pu.X, scale=True)
    prep.fit(pu.X.iloc[np.concatenate([train, np.flatnonzero(pu.s == 0)])])
    Z = prep.transform(pu.X)
    Z = np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z, float)
    return PUContext(Z, pu, np.sort(train), np.sort(test), np.flatnonzero(pu.s == 0), seed, base, prior), prep


def clf(name: str, seed: int = 0, class_weight: Any = None) -> Any:
    sk()
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB

    return {"logistic_regression": lambda: LogisticRegression(max_iter=1000, class_weight=class_weight),
            "random_forest": lambda: RandomForestClassifier(200, min_samples_leaf=3, n_jobs=-1, random_state=seed, class_weight=class_weight),
            "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=seed),
            "naive_bayes": lambda: GaussianNB()}[name]()


def pos_proba(model: Any, Z: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(Z)
    return raw[:, list(model.classes_).index(1)] if 1 in list(model.classes_) else np.zeros(len(Z))


class ScoreModel:
    """Anything with `.scores(Z)` -> P(positive) in [0, 1]; optionally rescaled by a constant (Elkan–Noto's 1/c)."""

    def __init__(self, model: Any, scale: float = 1.0, kind: str = "classifier"):
        self.model, self.scale, self.kind = model, scale, kind

    def scores(self, Z: np.ndarray) -> np.ndarray:
        raw = self.model.scores(Z) if hasattr(self.model, "scores") else pos_proba(self.model, Z)
        return np.clip(raw * self.scale, 0, 1)


class PUModel:
    """What gets saved: the preprocessor, a score model, and the decision threshold."""

    def __init__(self, kind: str, prep: Any, core: ScoreModel, columns: list[str], threshold: float, prior: float | None):
        self.kind, self.prep, self.core, self.columns, self.threshold, self.prior = kind, prep, core, columns, threshold, prior

    def scores(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.prep.transform(X[self.columns])
        return self.core.scores(np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z, float))


# ---- class prior --------------------------------------------------------------------------------
def labelling_fit(ctx: PUContext, base: str | None = None, folds: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold g(x) = P(labelled | x) from a classifier of labelled positives vs unlabelled; returns g over [p_train, p_test, unl] and the matching s."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    Z = np.vstack([ctx.Zp, ctx.Z[ctx.p_test], ctx.Zu])
    s = np.concatenate([np.ones(len(ctx.p_train) + len(ctx.p_test), int), np.zeros(len(ctx.unl), int)])
    g = cross_val_predict(clf(base or ctx.base, ctx.seed), Z, s, cv=StratifiedKFold(folds, shuffle=True, random_state=ctx.seed), method="predict_proba")[:, 1]
    return g, s


def elkan_noto_c(ctx: PUContext, base: str | None = None) -> dict[str, float]:
    """Estimate c = P(labelled | positive): e1 = mean g on labelled positives, e3 = a high quantile of g (the most positive-looking labelled row)."""
    g, s = labelling_fit(ctx, base)
    gp = g[s == 1]
    c1, c3 = float(gp.mean()), float(np.quantile(gp, 0.95))
    n_pos_labelled = int(s.sum())
    out = {}
    for name, c in (("e1_mean", c1), ("e3_high_quantile", c3)):
        c = min(max(c, 1e-3), 1.0)
        out[name] = c
    out["prior_e1"] = min(1.0, n_pos_labelled / (out["e1_mean"] * len(s)))
    out["prior_e3"] = min(1.0, n_pos_labelled / (out["e3_high_quantile"] * len(s)))
    # e1 under-estimates c (hidden positives sit among the unlabelled rows, dragging the score of labelled positives down), so its prior is too high;
    # e3 over-estimates c, so its prior is too low. They bracket the truth; the midpoint is the working estimate.
    out["prior_mid"] = (out["prior_e1"] + out["prior_e3"]) / 2
    return out


# ---- scoring a PU model ------------------------------------------------------------------------
def pu_metrics(ctx: PUContext, scores_all: np.ndarray, prior: float | None) -> dict[str, Any]:
    """Quality of a score vector over *all* rows. With truth: real AUC / AP / F1. Always: PU proxies that need only labelled positives."""
    from sklearn import metrics as m

    out: dict[str, Any] = {}
    sp, su = scores_all[ctx.p_test], scores_all[ctx.unl]
    y = np.concatenate([np.ones(len(sp)), np.zeros(len(su))])
    sc = np.concatenate([sp, su])
    out["pu_auc_heldout_positives_vs_unlabelled"] = round(float(m.roc_auc_score(y, sc)), 4)
    if prior is not None:
        k = max(1, int(round(prior * len(scores_all))))
        cut = np.sort(scores_all)[::-1][min(k, len(scores_all)) - 1]
        pred = scores_all >= cut
        out["recall_on_heldout_positives_at_prior_cut"] = round(float(pred[ctx.p_test].mean()), 4)
        out["share_of_unlabelled_predicted_positive"] = round(float(pred[ctx.unl].mean()), 4)
        r = float(pred[ctx.p_test].mean())
        out["lee_liu_criterion"] = round(float(r * r / max(pred.mean(), 1e-6)), 4)
    if ctx.pu.truth is not None:
        keep = np.setdiff1d(np.arange(len(scores_all)), ctx.p_train)  # everything the model did not train on as a labelled positive
        t = ctx.pu.truth[keep]
        out["true_roc_auc"] = round(float(m.roc_auc_score(t, scores_all[keep])), 4) if len(set(t)) > 1 else None
        out["true_average_precision"] = round(float(m.average_precision_score(t, scores_all[keep])), 4) if len(set(t)) > 1 else None
        thr = 0.5 if prior is None else np.sort(scores_all[keep])[::-1][max(0, min(len(keep) - 1, int(prior * len(keep)) - 1))]
        pred = scores_all[keep] >= thr
        out["true_f1_at_prior_cut"] = round(float(m.f1_score(t, pred)), 4)
        out["true_precision_at_prior_cut"] = round(float(m.precision_score(t, pred, zero_division=0)), 4)
        out["true_recall_at_prior_cut"] = round(float(m.recall_score(t, pred, zero_division=0)), 4)
    return out


def naive_scores(ctx: PUContext) -> np.ndarray:
    """The biased baseline: treat every unlabelled row as negative and classify positive vs the rest."""
    Z = np.vstack([ctx.Zp, ctx.Zu])
    y = np.concatenate([np.ones(len(ctx.Zp), int), np.zeros(len(ctx.Zu), int)])
    return pos_proba(clf(ctx.base, ctx.seed).fit(Z, y), ctx.Z)


def run_pu(label: str, trainer: Any, data: Any, label_column: str, positive_label: Any, features: Any, drop: Any, truth_column: str | None, base: str, prior: float | None,
           test_size: float, seed: int, model_path: str | None, truth_positive: Any = None) -> dict[str, Any]:
    """Fit `trainer(ctx) -> (ScoreModel, extra)`, score every row, and compare with the naive positive-vs-unlabelled classifier."""
    ctx, prep = make_pu(data, label_column, positive_label, features, drop, truth_column, test_size, seed, base, prior, truth_positive)
    if ctx.prior is None:
        est = elkan_noto_c(ctx)
        ctx.prior = est["prior_mid"]
        prior_source = "estimated (midpoint of the Elkan–Noto e1 and e3 bounds) — pass `prior` if it is known"
    else:
        prior_source = "given"
    core, extra = trainer(ctx)
    scores = core.scores(ctx.Z)
    naive = naive_scores(ctx)
    k = max(1, int(round(ctx.prior * len(scores))))
    threshold = float(np.sort(scores)[::-1][min(k, len(scores)) - 1])
    out = {"model": label, "rows": int(len(scores)), "labelled_positives": int(ctx.pu.s.sum()), "labelled_positives_trained_on": int(len(ctx.p_train)), "labelled_positives_held_out": int(len(ctx.p_test)),
           "unlabelled": int(len(ctx.unl)), "class_prior": round(float(ctx.prior), 4), "class_prior_source": prior_source, "threshold_at_prior": round(threshold, 4),
           "metrics": pu_metrics(ctx, scores, ctx.prior), "naive_positive_vs_unlabelled_baseline": pu_metrics(ctx, naive, ctx.prior),
           "predicted_positive_among_unlabelled": int((scores[ctx.unl] >= threshold).sum()), **extra}
    out["model_path"] = save_model(PUModel(label, prep, core, list(ctx.pu.X.columns), threshold, ctx.prior), model_path)
    return out
