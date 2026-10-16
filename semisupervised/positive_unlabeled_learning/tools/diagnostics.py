"""Read-only checks on a positive-unlabelled problem: how many positives are labelled, what share of the data is probably positive, whether the SCAR assumption is plausible, and whether positives can be told apart at all."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ml_core.tooling import tool
from semisupervised.positive_unlabeled_learning.tools._common import (
    clf,
    elkan_noto_c,
    labelling_fit,
    load_pu,
    make_pu,
    naive_scores,
)

Base = Literal["logistic_regression", "random_forest", "hist_gradient_boosting", "naive_bayes"]


@tool
def profile_pu_data(data: str, label_column: str, positive_label: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None) -> dict:
    """Profile a positive-unlabelled dataset: labelled positives vs unlabelled rows, the labelled share, feature columns, and whether a truth column exists for auditing — and what the numbers imply about the task.

    Args:
        positive_label: The value in the label column that marks a labelled positive; everything else counts as unlabelled. Numeric 0/1 columns need no value.
    """
    pu = load_pu(data, label_column, positive_label, features, drop, truth_column)
    n, p = len(pu.s), int(pu.s.sum())
    out = {"rows": n, "labelled_positives": p, "unlabelled": n - p, "labelled_share": round(p / n, 4), "features": list(pu.X.columns), "truth_available": pu.truth is not None,
           "warnings": [w for w in ["fewer than 30 labelled positives: scores and prior estimates will be unstable" if p < 30 else None,
                                    "labelled positives are over 30% of all rows: the unlabelled set is mostly negatives, so the naive classifier is less biased" if p / n > 0.3 else None,
                                    "labelled positives under 1% of rows: the labelled set is a tiny sample of the positives — recall will be limited by how typical it is" if p / n < 0.01 else None] if w]}
    if pu.truth is not None:
        out["true_positives_total"] = int(pu.truth.sum())
        out["true_label_frequency_c"] = round(float(pu.s.sum() / max(pu.truth.sum(), 1)), 4)
        out["hidden_positives_in_unlabelled"] = int(((pu.truth == 1) & (pu.s == 0)).sum())
    return out


@tool
def estimate_class_prior(data: str, label_column: str, positive_label: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, base: Base = "random_forest", seed: int = 42) -> dict:
    """Estimate the class prior — the share of ALL rows that are positive — with the Elkan–Noto estimators: train labelled-vs-unlabelled, take c = P(labelled | positive) as the mean (e1) or a high quantile (e3) of the out-of-fold score on labelled positives, and prior = labelled share / c. The two give an upper and a lower bound. Every PU method needs this number; a wrong prior shifts every threshold.

    Args:
        base: Classifier used to model 'labelled vs unlabelled'.
    """
    ctx, _ = make_pu(data, label_column, positive_label, features, drop, truth_column, 0.3, seed, base)
    est = elkan_noto_c(ctx, base)
    labelled_share = float(ctx.pu.s.mean())
    out = {"labelled_share": round(labelled_share, 4), "c_label_frequency_e1": round(est["e1_mean"], 4), "c_label_frequency_e3": round(est["e3_high_quantile"], 4),
           "prior_e1_upper_bound": round(est["prior_e1"], 4), "prior_e3_lower_bound": round(est["prior_e3"], 4), "recommended_prior": round(est["prior_mid"], 4),
           "reading": "e1 under-estimates c, so its prior is an upper bound; e3 over-estimates c, so its prior is a lower bound. When SCAR holds they bracket the truth, and the midpoint is the working estimate — the wider the gap, the less certain the prior; say so, and prefer a prior known from outside the data."}
    if ctx.pu.truth is not None:
        out["true_prior"] = round(float(ctx.pu.truth.mean()), 4)
    return out


@tool
def check_scar_assumption(data: str, label_column: str, positive_label: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: Base = "random_forest", top_share: float = 0.1, seed: int = 42) -> dict:
    """Test 'selected completely at random': if every positive had the same chance of being labelled, labelled positives should be indistinguishable from the most positive-looking unlabelled rows. A classifier separating the two (AUC well above 0.5) means labelling was biased towards certain kinds of positive, and scores will under-rate the others.

    Args:
        base: Classifier used for the check.
        top_share: Share of the unlabelled rows (highest-scoring) treated as likely positives.
    """
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score

    ctx, _ = make_pu(data, label_column, positive_label, features, drop, None, 0.3, seed, base)
    g, s = labelling_fit(ctx, base)
    unl_g = g[s == 0]
    top = np.argsort(-unl_g)[: max(10, int(len(unl_g) * top_share))]
    Zp = np.vstack([ctx.Zp, ctx.Z[ctx.p_test]])
    Zt = ctx.Zu[top]
    Z = np.vstack([Zp, Zt])
    y = np.concatenate([np.ones(len(Zp), int), np.zeros(len(Zt), int)])
    auc = float(roc_auc_score(y, cross_val_predict(clf(base, seed), Z, y, cv=StratifiedKFold(4, shuffle=True, random_state=seed), method="predict_proba")[:, 1]))
    return {"labelled_positives": int(len(Zp)), "top_unlabelled_compared": int(len(Zt)), "auc_labelled_vs_top_unlabelled": round(auc, 4),
            "verdict": "SCAR looks plausible: labelled positives resemble the most positive-looking unlabelled rows" if auc < 0.65 else
            "SCAR is doubtful: the labelled positives are a biased subset — expect low recall on positives unlike them, and treat the prior estimate and thresholds with caution"}


@tool
def check_pu_separability(data: str, label_column: str, positive_label: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: Base = "random_forest", seed: int = 42, k: int = 8) -> dict:
    """Can labelled positives be told apart from unlabelled rows at all? Cross-validated AUC of labelled-vs-unlabelled, the shape of the score distribution on unlabelled rows, and the features that drive it. An AUC near 0.5 means the positives are not distinguishable from the rest and no PU method will work.

    Args:
        base: Classifier used for the check.
        k: How many top features to list (random forest only).
    """
    from sklearn.metrics import roc_auc_score

    ctx, _ = make_pu(data, label_column, positive_label, features, drop, None, 0.3, seed, base)
    g, s = labelling_fit(ctx, base)
    auc = float(roc_auc_score(s, g))
    gu = g[s == 0]
    out = {"labelled_vs_unlabelled_auc": round(auc, 4), "unlabelled_score_quantiles": {q: round(float(np.quantile(gu, q)), 4) for q in (0.5, 0.9, 0.99)},
           "unlabelled_rows_scoring_like_labelled_positives": int((gu >= np.median(g[s == 1])).sum()),
           "verdict": "positives are well separated from the rest: PU learning can work" if auc > 0.8 else "positives overlap the unlabelled rows: expect modest results" if auc > 0.6 else "labelled positives look like the unlabelled rows: no PU method can find structure here"}
    if base == "random_forest":
        Z = np.vstack([ctx.Zp, ctx.Z[ctx.p_test], ctx.Zu])
        yy = np.concatenate([np.ones(len(ctx.Zp) + len(ctx.p_test), int), np.zeros(len(ctx.Zu), int)])
        rf = clf("random_forest", seed).fit(Z, yy)
        if Z.shape[1] == len(ctx.pu.X.columns):
            out["top_features"] = {c: round(float(v), 4) for c, v in sorted(zip(ctx.pu.X.columns, rf.feature_importances_), key=lambda kv: -kv[1])[:k]}
    return out


@tool
def baseline_positive_vs_unlabelled(data: str, label_column: str, positive_label: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, base: Base = "logistic_regression", prior: float | None = None, test_size: float = 0.3, seed: int = 42) -> dict:
    """The naive baseline PU methods must beat: label every unlabelled row negative and train positive-vs-rest. Reports the PU proxies (held-out positives vs unlabelled) and, with a truth column, the real AUC/F1 — it ranks fine but its probabilities and counts are biased low.

    Args:
        base: Classifier used.
        prior: Class prior if known; estimated otherwise.
    """
    from semisupervised.positive_unlabeled_learning.tools._common import pu_metrics

    ctx, _ = make_pu(data, label_column, positive_label, features, drop, truth_column, test_size, seed, base, prior)
    if ctx.prior is None:
        ctx.prior = elkan_noto_c(ctx)["prior_mid"]
    sc = naive_scores(ctx)
    return {"class_prior_used": round(float(ctx.prior), 4), "metrics": pu_metrics(ctx, sc, ctx.prior), "mean_score_on_unlabelled": round(float(sc[ctx.unl].mean()), 4),
            "note": "the mean score on unlabelled rows is far below the prior: that is the bias — the classifier learned 'was labelled', not 'is positive'"}
