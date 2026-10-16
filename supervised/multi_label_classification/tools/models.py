"""Fit one multi-label approach and score it on held-out rows: problem transformations (binary relevance, chains, label powerset, RAkEL) and native multi-output models (forests, k-NN, ML-kNN, neural net)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ml_core.ml import sk
from ml_core.tooling import tool
from supervised.multi_label_classification.tools._common import (
    decide,
    fit_multilabel,
    load_multilabel,
    multilabel_metrics,
    run_multilabel,
)

Base = Literal["logistic_regression", "random_forest", "hist_gradient_boosting", "linear_svm", "naive_bayes"]


@tool
def fit_binary_relevance(
    data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", features: list[str] | None = None, drop: list[str] | None = None,
    text_column: str | None = None, base: Base = "logistic_regression", threshold: float = 0.5, test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Binary relevance: one independent yes/no classifier per label. The simplest and most scalable approach, and the baseline to beat — it ignores dependencies between labels.

    Args:
        labels: Columns holding one 0/1 indicator per label.
        label_column: Alternatively, one column listing each row's labels joined by `separator`.
        separator: Delimiter inside `label_column` cells.
        text_column: Column of free text to use as features (TF-IDF).
        base: Binary classifier fitted for every label.
        threshold: Probability above which a label is predicted.
    """
    return run_multilabel("binary_relevance", f"binary_relevance({base})", data, labels, label_column, separator, features, drop, text_column, test_size, seed, model_path, base, threshold)


@tool
def fit_classifier_chain(
    data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", features: list[str] | None = None, drop: list[str] | None = None,
    text_column: str | None = None, base: Base = "logistic_regression", n_chains: int = 5, threshold: float = 0.5, test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Classifier chains: each label's classifier also sees the previous labels, so label dependencies are modelled; an ensemble of chains with random orders removes the dependence on any one ordering.

    Args:
        labels: Columns holding one 0/1 indicator per label.
        label_column: Alternatively, one column listing each row's labels joined by `separator`.
        separator: Delimiter inside `label_column` cells.
        text_column: Column of free text to use as features (TF-IDF).
        base: Binary classifier used along the chain.
        n_chains: Chains in the ensemble (random label orders).
        threshold: Probability above which a label is predicted.
    """
    return run_multilabel("classifier_chain", f"classifier_chain({base}x{n_chains})", data, labels, label_column, separator, features, drop, text_column, test_size, seed, model_path, base, threshold, n_chains=n_chains)


@tool
def fit_label_powerset(
    data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", features: list[str] | None = None, drop: list[str] | None = None,
    text_column: str | None = None, base: Base = "random_forest", min_count: int = 2, threshold: float = 0.5, test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Label powerset: every distinct combination of labels is one class of a multiclass problem. Captures label dependence completely, but needs each combination to occur several times and cannot predict combinations unseen in training.

    Args:
        labels: Columns holding one 0/1 indicator per label.
        label_column: Alternatively, one column listing each row's labels joined by `separator`.
        separator: Delimiter inside `label_column` cells.
        text_column: Column of free text to use as features (TF-IDF).
        base: Multiclass classifier over label combinations.
        min_count: Combinations rarer than this are folded into the closest common one.
        threshold: Probability above which a label is predicted.
    """
    return run_multilabel("label_powerset", f"label_powerset({base})", data, labels, label_column, separator, features, drop, text_column, test_size, seed, model_path, base, threshold, min_count=min_count)


@tool
def fit_rakel(
    data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", features: list[str] | None = None, drop: list[str] | None = None,
    text_column: str | None = None, base: Base = "random_forest", k: int = 3, n_models: int = 12, threshold: float = 0.5, test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """RAkEL (random k-labelsets): an ensemble of small label-powerset models, each over a random subset of k labels, voting on every label — captures label dependence without the combinatorial blow-up of the full powerset.

    Args:
        labels: Columns holding one 0/1 indicator per label.
        label_column: Alternatively, one column listing each row's labels joined by `separator`.
        separator: Delimiter inside `label_column` cells.
        text_column: Column of free text to use as features (TF-IDF).
        base: Multiclass classifier inside each subset model.
        k: Labels per subset.
        n_models: Subset models in the ensemble.
        threshold: Probability above which a label is predicted.
    """
    return run_multilabel("rakel", f"rakel(k={k},m={n_models},{base})", data, labels, label_column, separator, features, drop, text_column, test_size, seed, model_path, base, threshold, k=k, n_models=n_models)


@tool
def fit_multioutput_random_forest(
    data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", features: list[str] | None = None, drop: list[str] | None = None,
    text_column: str | None = None, n_estimators: int = 300, min_samples_leaf: int = 2, threshold: float = 0.5, test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """One random forest whose trees split to predict all labels together — shares structure across labels, fast, and a strong default for tabular multi-label data.

    Args:
        labels: Columns holding one 0/1 indicator per label.
        label_column: Alternatively, one column listing each row's labels joined by `separator`.
        separator: Delimiter inside `label_column` cells.
        text_column: Column of free text to use as features (TF-IDF).
        min_samples_leaf: Larger values smooth the probabilities.
        threshold: Probability above which a label is predicted.
    """
    return run_multilabel("random_forest", "multioutput_random_forest", data, labels, label_column, separator, features, drop, text_column, test_size, seed, model_path, "random_forest", threshold, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)


@tool
def fit_multioutput_extra_trees(
    data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", features: list[str] | None = None, drop: list[str] | None = None,
    text_column: str | None = None, n_estimators: int = 300, min_samples_leaf: int = 2, threshold: float = 0.5, test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Extremely randomised trees predicting all labels jointly — like the multi-output forest but with random split points: faster and smoother.

    Args:
        labels: Columns holding one 0/1 indicator per label.
        label_column: Alternatively, one column listing each row's labels joined by `separator`.
        separator: Delimiter inside `label_column` cells.
        text_column: Column of free text to use as features (TF-IDF).
        min_samples_leaf: Larger values smooth the probabilities.
        threshold: Probability above which a label is predicted.
    """
    return run_multilabel("extra_trees", "multioutput_extra_trees", data, labels, label_column, separator, features, drop, text_column, test_size, seed, model_path, "random_forest", threshold, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)


@tool
def fit_knn_multilabel(
    data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", features: list[str] | None = None, drop: list[str] | None = None,
    text_column: str | None = None, n_neighbors: int = 15, threshold: float = 0.5, test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Plain k-nearest-neighbours: a label's probability is the share of the neighbours that carry it — a non-parametric check on whether similar rows share labels.

    Args:
        labels: Columns holding one 0/1 indicator per label.
        label_column: Alternatively, one column listing each row's labels joined by `separator`.
        separator: Delimiter inside `label_column` cells.
        text_column: Column of free text to use as features (TF-IDF).
        threshold: Probability above which a label is predicted.
    """
    return run_multilabel("knn", f"knn(k={n_neighbors})", data, labels, label_column, separator, features, drop, text_column, test_size, seed, model_path, "random_forest", threshold, n_neighbors=n_neighbors)


@tool
def fit_mlknn(
    data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", features: list[str] | None = None, drop: list[str] | None = None,
    text_column: str | None = None, n_neighbors: int = 10, threshold: float = 0.5, test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """ML-kNN: k-nearest neighbours with a Bayesian correction — each label's posterior combines its prior frequency with how many neighbours carry it. Better calibrated than plain k-NN, especially for rare labels.

    Args:
        labels: Columns holding one 0/1 indicator per label.
        label_column: Alternatively, one column listing each row's labels joined by `separator`.
        separator: Delimiter inside `label_column` cells.
        text_column: Column of free text to use as features (TF-IDF).
        threshold: Probability above which a label is predicted.
    """
    return run_multilabel("mlknn", f"mlknn(k={n_neighbors})", data, labels, label_column, separator, features, drop, text_column, test_size, seed, model_path, "random_forest", threshold, n_neighbors=n_neighbors)


@tool
def fit_mlp_multilabel(
    data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", features: list[str] | None = None, drop: list[str] | None = None,
    text_column: str | None = None, hidden_layers: str = "64", threshold: float = 0.5, test_size: float = 0.2, seed: int = 42, model_path: str | None = None,
) -> dict:
    """A feed-forward neural network with one sigmoid output per label — learns shared representations across labels; worth trying with many rows or TF-IDF text.

    Args:
        labels: Columns holding one 0/1 indicator per label.
        label_column: Alternatively, one column listing each row's labels joined by `separator`.
        separator: Delimiter inside `label_column` cells.
        text_column: Column of free text to use as features (TF-IDF).
        hidden_layers: Comma-separated layer widths, e.g. "128,64".
        threshold: Probability above which a label is predicted.
    """
    hidden = tuple(int(p) for p in hidden_layers.replace(" ", "").split(",") if p) or (64,)
    return run_multilabel("mlp", f"mlp({hidden_layers})", data, labels, label_column, separator, features, drop, text_column, test_size, seed, model_path, "random_forest", threshold, hidden=hidden)


@tool
def compare_multilabel_models(
    data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", features: list[str] | None = None, drop: list[str] | None = None,
    text_column: str | None = None, cv: int = 4, seed: int = 42, metric: Literal["f1_micro", "f1_macro", "f1_samples", "hamming_loss", "subset_accuracy", "average_precision_macro"] = "f1_micro",
) -> dict:
    """Cross-validate the main approaches on identical folds — binary relevance, classifier chains, label powerset, RAkEL, multi-output forest, ML-kNN — and rank them; the gap between binary relevance and the dependence-aware methods says how much label dependence matters.

    Args:
        labels: Columns holding one 0/1 indicator per label.
        label_column: Alternatively, one column listing each row's labels joined by `separator`.
        separator: Delimiter inside `label_column` cells.
        text_column: Column of free text to use as features (TF-IDF).
        metric: Metric to rank by.
    """
    sk()
    from sklearn.model_selection import KFold

    X, Y, names = load_multilabel(data, labels, label_column, separator, features, drop, text_column)
    specs = [("binary_relevance", "binary_relevance", {"base": "logistic_regression"}), ("classifier_chain", "classifier_chain", {"base": "logistic_regression", "n_chains": 3}),
             ("label_powerset", "label_powerset", {"base": "random_forest"}), ("rakel", "rakel", {"base": "random_forest", "k": 3, "n_models": 8}),
             ("multioutput_forest", "random_forest", {}), ("mlknn", "mlknn", {})]
    scores: dict[str, list[dict]] = {n: [] for n, _, _ in specs}
    for tr, te in KFold(cv, shuffle=True, random_state=seed).split(X):
        for name, kind, params in specs:
            try:
                model = fit_multilabel(kind, X.iloc[tr], Y[tr], names, text_column, seed=seed, **params)
                proba = model.proba(X.iloc[te])
                scores[name].append(multilabel_metrics(Y[te], decide(proba), proba))
            except Exception:
                pass
    keys = ("f1_micro", "f1_macro", "f1_samples", "hamming_loss", "subset_accuracy", "average_precision_macro")
    board = [{"model": n, **{k: round(float(np.mean([s[k] for s in v if k in s])), 5) for k in keys if any(k in s for s in v)}} for n, v in scores.items() if v]
    board.sort(key=lambda r: r.get(metric, 0), reverse=metric != "hamming_loss")
    return {"labels": len(names), "folds": cv, "ranked_by": metric, "leaderboard": board, "best": board[0]["model"] if board else None}
