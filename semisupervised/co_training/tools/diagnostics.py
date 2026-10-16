"""Read-only checks before co-training: how much is labelled, whether the unlabelled rows look like the labelled ones, how far the labels alone go, and whether the base learner's confidence can be trusted."""

from __future__ import annotations

from ml_core.semi_diag import (
    confidence_reliability,
    coverage_profile,
    scarcity_curve,
    shift_check,
    supervised_baseline,
    threshold_for_precision,
)
from ml_core.tooling import tool


@tool
def profile_label_coverage(data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None) -> dict:
    """Profile a partially labelled dataset: labelled vs unlabelled rows, the class balance among the labelled ones, labels per class and per feature, and whether there are enough labels for a base learner to start from at all."""
    return coverage_profile(data, label_column, unlabeled_value, features, drop)


@tool
def check_labeled_unlabeled_shift(data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, k: int = 8, seed: int = 42) -> dict:
    """Adversarial validation: train a classifier to tell labelled rows from unlabelled ones. An AUC near 0.5 means they come from the same distribution (semi-supervised learning is safe); a high AUC means the labels were collected on a biased subset and pseudo or propagated labels will extrapolate badly.

    Args:
        k: How many most-shifted features to list.
    """
    return shift_check(data, label_column, unlabeled_value, features, drop, k, seed)


@tool
def baseline_supervised_only(data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "logistic_regression", test_size: float = 0.3, seed: int = 42) -> dict:
    """Fit the base learner on the labelled rows alone and score it on held-out labelled rows — the number every semi-supervised method has to beat.

    Args:
        base: logistic_regression, random_forest, hist_gradient_boosting, svm, knn or naive_bayes.
    """
    return supervised_baseline(data, label_column, unlabeled_value, features, drop, base, test_size, seed)


@tool
def label_scarcity_curve(data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "logistic_regression", test_size: float = 0.3, seed: int = 42) -> dict:
    """Supervised-only accuracy as the number of labels grows (10%, 25%, 50%, 100% of what is available): a steep curve means more labels — or pseudo-labels — would help a lot; a flat one means the labels are not the bottleneck.

    Args:
        base: Base learner.
    """
    return scarcity_curve(data, label_column, unlabeled_value, features, drop, base, test_size, seed)


@tool
def check_base_learner_confidence(data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "logistic_regression", seed: int = 42) -> dict:
    """Reliability of the base learner's confidence, from cross-validation on the labelled rows: in each confidence band, how often is the top prediction actually right? Methods that trust confidence poison themselves if it is over-confident.

    Args:
        base: Base learner.
    """
    return confidence_reliability(data, label_column, unlabeled_value, features, drop, base, seed)


@tool
def suggest_confidence_threshold(data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "logistic_regression", target_precision: float = 0.95, seed: int = 42) -> dict:
    """Pick the confidence threshold at which the base learner's pseudo-labels would be right at least `target_precision` of the time, measured by cross-validation on the labelled rows, and how many unlabelled rows would clear it.

    Args:
        base: Base learner.
        target_precision: Wanted share of correct pseudo-labels, e.g. 0.95.
    """
    return threshold_for_precision(data, label_column, unlabeled_value, features, drop, base, target_precision, seed)


@tool
def suggest_feature_views(data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, strategy: str = "correlation", seed: int = 42) -> dict:
    """Propose two views (groups of columns) for co-training and judge them: sufficiency (each view alone predicts the label), and dependence (how correlated the views are within each class). Good views are each sufficient and as independent as possible.

    Args:
        strategy: correlation (split by correlation structure), halves (column order) or random.
    """
    from ml_core.semi import base_clf, full_proba, make_context, metrics, view_indices
    from semisupervised.co_training.tools._common import choose_views, view_dependence

    ctx, prep = make_context(data, label_column, features, drop, unlabeled_value, None, 0.3, seed, "logistic_regression")
    groups = choose_views(ctx, prep, None, strategy, seed)
    yt = ctx.sd.y[ctx.split.l_test]
    sufficiency = []
    for g in groups:
        i = view_indices(prep, g)
        m = base_clf("logistic_regression", seed).fit(ctx.Zl[:, i], ctx.yl)
        sufficiency.append(metrics(yt, full_proba(m, ctx.Z[ctx.split.l_test][:, i], ctx.K))["accuracy"])
    full = metrics(yt, full_proba(base_clf("logistic_regression", seed).fit(ctx.Zl, ctx.yl), ctx.Z[ctx.split.l_test], ctx.K))["accuracy"]
    dep = view_dependence(ctx, groups)
    return {"strategy": strategy, "views": groups, "view_accuracy": sufficiency, "all_features_accuracy": full, "within_class_dependence": round(dep, 4) if dep == dep else None,
            "verdict": "usable views: each is sufficient and they are fairly independent" if min(sufficiency) >= 0.8 * full and (dep != dep or dep < 0.3) else
            "views are too weak or too dependent: co-training will behave like self-training — prefer fit_tri_training or fit_democratic_co_learning"}


@tool
def check_view_agreement(data: str, label_column: str, views: list[list[str]] | None = None, view_strategy: str = "correlation", unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, seed: int = 42) -> dict:
    """Train one classifier per view on the labelled rows and measure how often they agree on the unlabelled rows, and how accurate they are where they do. Co-training needs views that disagree on some rows (so there is something to teach) yet are right when they agree.

    Args:
        views: Two or more lists of column names; built automatically when omitted.
        view_strategy: How to build views when none are given: correlation, halves or random.
    """
    from ml_core.semi import base_clf, full_proba, make_context, view_indices
    from semisupervised.co_training.tools._common import choose_views

    ctx, prep = make_context(data, label_column, features, drop, unlabeled_value, None, 0.3, seed, "logistic_regression")
    groups = choose_views(ctx, prep, views, view_strategy, seed)
    idx = [view_indices(prep, g) for g in groups]
    probas = [full_proba(base_clf("logistic_regression", seed).fit(ctx.Zl[:, i], ctx.yl), ctx.Zu[:, i], ctx.K) for i in idx]
    pa, pb = probas[0].argmax(axis=1), probas[1].argmax(axis=1)
    agree = pa == pb
    conf_agree = (probas[0].max(axis=1) > 0.8) & (probas[1].max(axis=1) > 0.8) & agree
    return {"views": groups, "unlabelled_rows": int(len(pa)), "agreement_rate": round(float(agree.mean()), 4), "confident_agreement_rate": round(float(conf_agree.mean()), 4),
            "view_a_confident_view_b_unsure": int(((probas[0].max(axis=1) > 0.9) & (probas[1].max(axis=1) < 0.6)).sum()),
            "view_b_confident_view_a_unsure": int(((probas[1].max(axis=1) > 0.9) & (probas[0].max(axis=1) < 0.6)).sum()),
            "reading": "the last two counts are the rows co-training can teach with: one view is sure where the other is not; if both are near zero the views carry the same information"}
