"""Judge PU models: methods head to head, thresholds from the class prior, ranked candidates and reliable negatives for review, and scoring new rows."""

from __future__ import annotations

from typing import Literal

import joblib
import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.tooling import tool
from semisupervised.positive_unlabeled_learning.tools._common import (
    clf,
    elkan_noto_c,
    labelling_fit,
    make_pu,
    naive_scores,
    pos_proba,
    pu_metrics,
)

Base = Literal["logistic_regression", "random_forest", "hist_gradient_boosting", "naive_bayes"]


def _scored(data, label_column, positive_label, features, drop, truth_column, base, prior, test_size, seed):
    pass

    ctx, prep = make_pu(data, label_column, positive_label, features, drop, truth_column, test_size, seed, base, prior)
    if ctx.prior is None:
        ctx.prior = elkan_noto_c(ctx, base)["prior_mid"]
    return ctx, prep


@tool
def compare_pu_methods(
    data: str, label_column: str, positive_label: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "random_forest", prior: float | None = None, test_size: float = 0.3, seed: int = 42,
) -> dict:
    """Run the naive positive-vs-unlabelled baseline, Elkan–Noto, bagging, the spy and Rocchio two-step methods, Lee–Liu weighting, nnPU and EM on the same split and rank them by the PU AUC (held-out positives vs unlabelled) — and by the true AUC / F1 when a truth column exists, which is the only honest comparison.

    Args:
        base: Classifier used inside each method.
        prior: Class prior if known (otherwise estimated, and shared by every method).
    """
    from semisupervised.positive_unlabeled_learning.tools import models as M

    ctx, _ = _scored(data, label_column, positive_label, features, drop, truth_column, base, prior, test_size, seed)
    rows = [{"method": "naive_positive_vs_unlabelled", **pu_metrics(ctx, naive_scores(ctx), ctx.prior)}]
    common = dict(data=data, label_column=label_column, positive_label=positive_label, features=features, drop=drop, truth_column=truth_column, prior=float(ctx.prior), test_size=test_size, seed=seed)
    runs = {"elkan_noto": (M.fit_pu_elkan_noto, {"base": base}), "bagging": (M.fit_pu_bagging, {"base": base, "n_bags": 15}), "two_step_spy": (M.fit_pu_two_step_spy, {"base": "logistic_regression"}),
            "two_step_rocchio": (M.fit_pu_two_step_rocchio, {"base": "logistic_regression"}), "weighted_logistic": (M.fit_pu_weighted_logistic, {}), "nnpu": (M.fit_nnpu, {"epochs": 250}),
            "em": (M.fit_pu_expectation_maximisation, {"base": "logistic_regression", "rounds": 5})}
    for name, (fn, kw) in runs.items():
        r = fn(**common, **kw)
        rows.append({"method": name, **r["metrics"]} if r["status"] == "ok" else {"method": name, "error": r.get("message", "")[:100]})
    key = "true_roc_auc" if ctx.pu.truth is not None else "pu_auc_heldout_positives_vs_unlabelled"
    naive, corrected = rows[0], sorted(rows[1:], key=lambda r: -(r.get(key) or 0))
    return {"ranked_by": key, "class_prior": round(float(ctx.prior), 4), "methods": corrected, "best": corrected[0]["method"], "naive_reference": naive,
            "note": ("no truth column: the PU AUC rewards scoring unlabelled rows low — which is exactly the naive classifier's bias, so the naive baseline can look best on it. Compare the corrected methods with each other, and do not conclude the naive one wins." if ctx.pu.truth is None
                     else "the true metrics use the truth column and are the ones to trust; compare the naive reference with them")}


@tool
def choose_pu_threshold(
    data: str, label_column: str, positive_label: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "random_forest", prior: float | None = None, target_recall: float = 0.9, test_size: float = 0.3, seed: int = 42,
) -> dict:
    """Pick a decision threshold without negative labels: (a) the cut that makes the share of predicted positives equal the class prior, and (b) the highest cut that still recovers `target_recall` of held-out labelled positives — and show how many unlabelled rows each calls positive, with true precision and recall when a truth column exists.

    Args:
        base: Classifier used (Elkan–Noto).
        prior: Class prior if known.
        target_recall: Recall on held-out labelled positives to preserve.
    """

    ctx, _ = _scored(data, label_column, positive_label, features, drop, truth_column, base, prior, test_size, seed)
    c = elkan_noto_c(ctx, base)["e1_mean"]
    model = clf(base, seed).fit(np.vstack([ctx.Zp, ctx.Zu]), np.concatenate([np.ones(len(ctx.Zp), int), np.zeros(len(ctx.Zu), int)]))
    scores = np.clip(pos_proba(model, ctx.Z) / c, 0, 1)
    k = max(1, int(round(ctx.prior * len(scores))))
    cuts = {"prior_matching": float(np.sort(scores)[::-1][min(k, len(scores)) - 1]), "recall_preserving": float(np.quantile(scores[ctx.p_test], 1 - target_recall)), "default_0.5": 0.5}
    out = {}
    for name, cut in cuts.items():
        pred = scores >= cut
        row = {"threshold": round(cut, 4), "recall_on_heldout_positives": round(float(pred[ctx.p_test].mean()), 4), "unlabelled_called_positive": int(pred[ctx.unl].sum()), "share_of_unlabelled": round(float(pred[ctx.unl].mean()), 4)}
        if ctx.pu.truth is not None:
            keep = np.setdiff1d(np.arange(len(scores)), ctx.p_train)
            t, p = ctx.pu.truth[keep], pred[keep]
            row.update({"true_precision": round(float((t[p] == 1).mean()), 4) if p.any() else None, "true_recall": round(float(p[t == 1].mean()), 4)})
        out[name] = row
    return {"class_prior": round(float(ctx.prior), 4), "thresholds": out, "advice": "use the prior-matching cut when the prior is trustworthy; use the recall-preserving cut when missing positives is the costly error"}


@tool
def rank_unlabelled_by_positive_probability(
    data: str, label_column: str, positive_label: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, id_column: str | None = None,
    base: Base = "random_forest", prior: float | None = None, top: int = 50, seed: int = 42, output_path: str | None = None,
) -> dict:
    """The deliverable of most PU projects: a ranked list of the unlabelled rows most likely to be hidden positives, from a model trained on every labelled positive, with the score and a prior-based cut — the list to send for review or follow-up.

    Args:
        base: Classifier used (Elkan–Noto).
        prior: Class prior if known.
        top: How many rows to show in the reply (the file has them all).
    """
    ctx, _ = _scored(data, label_column, positive_label, features, drop, None, base, prior, 0.0001 if False else 0.2, seed)
    all_pos = np.flatnonzero(ctx.pu.s == 1)
    c = elkan_noto_c(ctx, base)["e1_mean"]
    model = clf(base, seed).fit(np.vstack([ctx.Z[all_pos], ctx.Zu]), np.concatenate([np.ones(len(all_pos), int), np.zeros(len(ctx.Zu), int)]))
    scores = np.clip(pos_proba(model, ctx.Zu) / c, 0, 1)
    frame = load_frame(data)
    out = frame.iloc[ctx.unl].copy()
    out.insert(0, "row", ctx.unl)
    out["positive_score"] = scores
    out["rank"] = out["positive_score"].rank(ascending=False, method="first").astype(int)
    out = out.sort_values("rank")
    expected = max(ctx.prior * len(frame) - len(all_pos), 0)
    cut = int(round(expected))
    if id_column and id_column in out.columns:
        out = out[["row", id_column] + [c2 for c2 in out.columns if c2 not in ("row", id_column)]]
    return {"unlabelled_ranked": int(len(out)), "class_prior": round(float(ctx.prior), 4), "expected_hidden_positives": int(cut), "score_at_expected_cut": round(float(out["positive_score"].iloc[min(max(cut - 1, 0), len(out) - 1)]), 4),
            "top": out.head(top), "written_to": save_frame(out, output_path, "pu_ranked_unlabelled.csv")}


@tool
def extract_reliable_negatives(
    data: str, label_column: str, positive_label: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, id_column: str | None = None,
    method: Literal["spy", "rocchio", "low_score"] = "low_score", share: float = 0.3, base: Base = "logistic_regression", seed: int = 42, output_path: str | None = None,
) -> dict:
    """Pick out the unlabelled rows that are very probably negative (by the spy technique, Rocchio distance, or the lowest labelled-vs-unlabelled score) and write them — turns a PU problem into an ordinary classification dataset (positives + reliable negatives) that any supervised tool can use.

    Args:
        method: spy, rocchio or low_score.
        share: Share of the unlabelled rows to take as reliable negatives.
        base: Classifier for the spy / low_score scores.
    """
    ctx, _ = _scored(data, label_column, positive_label, features, drop, None, base, None, 0.2, seed)
    if method == "rocchio":
        cp, cu = ctx.Zp.mean(axis=0), ctx.Zu.mean(axis=0)
        badness = np.linalg.norm(ctx.Zu - cp, axis=1) - np.linalg.norm(ctx.Zu - cu, axis=1)
    else:
        g, s = labelling_fit(ctx, base)
        badness = -g[s == 0]
    pick = np.argsort(-badness)[: max(5, int(len(ctx.Zu) * share))]
    frame = load_frame(data)
    pos = frame.iloc[np.flatnonzero(ctx.pu.s == 1)].assign(pu_class="positive")
    neg = frame.iloc[ctx.unl[pick]].assign(pu_class="reliable_negative")
    table = pd.concat([pos, neg])
    return {"labelled_positives": int(len(pos)), "reliable_negatives": int(len(neg)), "unlabelled_left_out": int(len(ctx.unl) - len(neg)), "written_to": save_frame(table, output_path, "pu_classification_dataset.csv"),
            "note": "the reliable negatives are the easy ones; a classifier trained on them will be over-optimistic about the hard region near the boundary"}


@tool
def predict_with_pu_model(model_path: str, data: str, id_column: str | None = None, threshold: float | None = None, output_path: str | None = None) -> dict:
    """Score new rows with a saved PU model: the estimated probability of being positive and the decision at the saved (prior-matching) threshold or at one you give.

    Args:
        model_path: Path of the saved PU model (joblib).
        id_column: Column to carry through to the output.
        threshold: Override the saved decision threshold.
    """
    model = joblib.load(model_path)
    frame = load_frame(data)
    require_columns(frame, model.columns)
    s = model.scores(frame)
    cut = model.threshold if threshold is None else threshold
    out = pd.DataFrame({"positive_score": s, "predicted_positive": (s >= cut).astype(int)}, index=frame.index)
    if id_column and id_column in frame.columns:
        out.insert(0, id_column, frame[id_column])
    return {"rows": len(out), "threshold": round(float(cut), 4), "predicted_positive": int(out["predicted_positive"].sum()), "share_positive": round(float(out["predicted_positive"].mean()), 4),
            "class_prior_when_trained": round(float(model.prior), 4) if model.prior else None, "written_to": save_frame(out, output_path, "pu_predictions.csv"), "sample": out.head(5)}
