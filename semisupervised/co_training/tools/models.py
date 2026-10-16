"""Fit a co-training model: classifiers trained on different views (or different learners) that label rows for each other."""

from __future__ import annotations

from typing import Literal

from ml_core.semi import run_semi, view_indices
from ml_core.tooling import tool
from semisupervised.co_training.tools._common import choose_views, co_train, democratic, tri_train

Base = Literal["logistic_regression", "random_forest", "hist_gradient_boosting", "svm", "knn", "naive_bayes"]
Strategy = Literal["correlation", "halves", "random"]


def _co(label, redraw, data, label_column, views, view_strategy, unlabeled_value, features, drop, truth_column, base, rounds, per_class, threshold, test_size, seed, model_path):
    def trainer(ctx):
        import numpy as np

        # the preprocessor is needed to map columns to encoded positions; rebuild it the way make_context did
        from ml_core.ml import build_preprocessor

        prep = build_preprocessor(ctx.sd.X, scale=True).fit(ctx.sd.X.iloc[np.concatenate([ctx.split.l_train, ctx.split.unl])])
        groups = choose_views(ctx, prep, views, view_strategy, seed)
        idx = [view_indices(prep, g) for g in groups]
        if redraw:
            idx = [np.sort(np.random.default_rng(seed).permutation(ctx.Z.shape[1])[: ctx.Z.shape[1] // 2]), np.sort(np.random.default_rng(seed + 1).permutation(ctx.Z.shape[1])[: ctx.Z.shape[1] // 2])]
        model, extra = co_train(ctx, idx, rounds, per_class, threshold, redraw)
        extra["views"] = groups if not redraw else "random column subsets, redrawn every round"
        return model, extra

    return run_semi(label, trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, base)


@tool
def fit_co_training(
    data: str, label_column: str, views: list[list[str]] | None = None, view_strategy: Strategy = "correlation", unlabeled_value: str | None = None, features: list[str] | None = None,
    drop: list[str] | None = None, truth_column: str | None = None, base: Base = "logistic_regression", rounds: int = 10, per_class: int = 5, threshold: float = 0.8,
    test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Blum–Mitchell co-training: two classifiers, each on its own view of the columns; every round each labels the unlabelled rows it is most sure of, and those rows are added to the *other* classifier's training set. The final model averages both views.

    Args:
        views: Two lists of column names; built automatically (see view_strategy) when omitted.
        view_strategy: How to build views when none are given: correlation, halves or random.
        rounds: Rounds of mutual teaching.
        per_class: Most rows each view adds per class per round.
        threshold: Confidence a view needs before it teaches.
    """
    return _co(f"co_training({base})", False, data, label_column, views, view_strategy, unlabeled_value, features, drop, truth_column, base, rounds, per_class, threshold, test_size, seed, model_path)


@tool
def fit_co_training_random_subspaces(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", rounds: int = 10, per_class: int = 5, threshold: float = 0.8, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Co-training for data with no natural views: each round the encoded columns are randomly split into two fresh subsets, so the two classifiers stay diverse even though no real view split exists.

    Args:
        rounds: Rounds of mutual teaching.
        per_class: Most rows each view adds per class per round.
        threshold: Confidence a view needs before it teaches.
    """
    return _co(f"co_training_random_subspaces({base})", True, data, label_column, None, "random", unlabeled_value, features, drop, truth_column, base, rounds, per_class, threshold, test_size, seed, model_path)


@tool
def fit_tri_training(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "random_forest", rounds: int = 10, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Tri-training: three classifiers trained on bootstrap resamples of the labelled data; an unlabelled row is labelled for one classifier whenever the other two agree on it. Needs neither views nor confidence scores, so it works with any learner.

    Args:
        rounds: Most rounds.
    """
    return run_semi(f"tri_training({base})", lambda ctx: tri_train(ctx, rounds), data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, base)


@tool
def fit_democratic_co_learning(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", rounds: int = 8, threshold: float = 0.7, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Democratic co-learning: three learners of different families vote, weighted by their cross-validated accuracy; a row the weighted majority is sure of is added only to the learners that disagreed with it — they learn precisely where they were wrong.

    Args:
        rounds: Most rounds.
        threshold: Weighted vote share a row needs to be labelled.
    """
    return run_semi("democratic_co_learning", lambda ctx: democratic(ctx, rounds, threshold), data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, base)


@tool
def fit_multiview_agreement_filter(
    data: str, label_column: str, views: list[list[str]] | None = None, view_strategy: Strategy = "correlation", unlabeled_value: str | None = None, features: list[str] | None = None,
    drop: list[str] | None = None, truth_column: str | None = None, base: Base = "logistic_regression", threshold: float = 0.8, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """One-shot multi-view labelling: pseudo-label only the rows on which every view's classifier agrees and is confident, then train a single model on all columns with them — the conservative option, cleaner labels than any one view gives.

    Args:
        views: Two lists of column names; built automatically when omitted.
        view_strategy: How to build views when none are given.
        threshold: Confidence each view needs.
    """
    def trainer(ctx):
        import numpy as np

        from ml_core.ml import build_preprocessor
        from ml_core.semi import Fitted, base_clf, full_proba

        prep = build_preprocessor(ctx.sd.X, scale=True).fit(ctx.sd.X.iloc[np.concatenate([ctx.split.l_train, ctx.split.unl])])
        groups = choose_views(ctx, prep, views, view_strategy, seed)
        idx = [view_indices(prep, g) for g in groups]
        probas = [full_proba(base_clf(base, seed).fit(ctx.Zl[:, i], ctx.yl), ctx.Zu[:, i], ctx.K) for i in idx]
        labels = [p.argmax(axis=1) for p in probas]
        keep = np.flatnonzero(np.all([l == labels[0] for l in labels], axis=0) & np.all([p.max(axis=1) >= threshold for p in probas], axis=0))
        conf = np.mean([p.max(axis=1) for p in probas], axis=0)[keep]
        model = base_clf(base, seed).fit(np.vstack([ctx.Zl, ctx.Zu[keep]]), np.concatenate([ctx.yl, labels[0][keep]]))
        return Fitted(model, ctx.K), {"views": groups, "pseudo": {"index": ctx.split.unl[keep], "label": labels[0][keep], "confidence": conf}}

    return run_semi(f"multiview_agreement({base})", trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, base)
