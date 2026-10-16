"""Fit one learning-to-rank approach — pointwise, pairwise or listwise — and score it on held-out groups."""

from __future__ import annotations


import numpy as np

from ml_core.tooling import tool
from supervised.ranking.tools._common import (
    fit_ranker,
    load_ranking,
    per_group_metrics,
    run_ranker,
)


@tool
def fit_pointwise_regression_ranker(
    data: str, group_column: str, relevance: str, features: list[str] | None = None, drop: list[str] | None = None,
    n_estimators: int = 200, learning_rate: float = 0.08, k: int = 10, relevant_from: float = 1,
    test_size: float = 0.25, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Pointwise ranking: boosted trees predict each item's graded relevance, items are sorted by the prediction. The simplest strong baseline for graded labels.

    Args:
        group_column: Column identifying the query/group each row belongs to.
        n_estimators: Boosting rounds.
        learning_rate: Shrinkage per round.
        k: Cut-off for NDCG/MAP/precision/recall.
        relevant_from: Relevance at or above which an item counts as relevant for MAP/MRR/precision/recall.
    """
    return run_ranker("pointwise_regression", "pointwise_regression(hist_gbm)", data, group_column, relevance, features, drop, k,
                      test_size, seed, model_path, relevant_from, n_estimators=n_estimators, learning_rate=learning_rate)


@tool
def fit_pointwise_linear_ranker(
    data: str, group_column: str, relevance: str, features: list[str] | None = None, drop: list[str] | None = None,
    alpha: float = 1.0, k: int = 10, relevant_from: float = 1, test_size: float = 0.25, seed: int = 42,
    model_path: str | None = None,
) -> dict:
    """Pointwise ranking with ridge regression on relevance — an interpretable linear scorer whose weights show which features move items up.

    Args:
        group_column: Column identifying the query/group each row belongs to.
        alpha: Ridge penalty.
        k: Cut-off for NDCG/MAP/precision/recall.
        relevant_from: Relevance at or above which an item counts as relevant.
    """
    return run_ranker("pointwise_linear", f"pointwise_ridge(alpha={alpha})", data, group_column, relevance, features, drop, k,
                      test_size, seed, model_path, relevant_from, alpha=alpha)


@tool
def fit_pointwise_random_forest_ranker(
    data: str, group_column: str, relevance: str, features: list[str] | None = None, drop: list[str] | None = None,
    n_estimators: int = 300, k: int = 10, relevant_from: float = 1, test_size: float = 0.25, seed: int = 42,
    model_path: str | None = None,
) -> dict:
    """Pointwise ranking with a random forest on relevance — robust, little tuning, handles non-linear feature effects.

    Args:
        group_column: Column identifying the query/group each row belongs to.
        n_estimators: Number of trees.
        k: Cut-off for NDCG/MAP/precision/recall.
        relevant_from: Relevance at or above which an item counts as relevant.
    """
    return run_ranker("pointwise_random_forest", "pointwise_random_forest", data, group_column, relevance, features, drop, k,
                      test_size, seed, model_path, relevant_from, n_estimators=n_estimators)


@tool
def fit_pointwise_classifier_ranker(
    data: str, group_column: str, relevance: str, features: list[str] | None = None, drop: list[str] | None = None,
    relevant_from: float = 1, n_estimators: int = 200, k: int = 10, test_size: float = 0.25, seed: int = 42,
    model_path: str | None = None,
) -> dict:
    """Pointwise ranking as classification: predict P(relevant) and sort by it. Right when relevance is binary (clicked / not) or only 'relevant or not' matters.

    Args:
        group_column: Column identifying the query/group each row belongs to.
        relevant_from: Relevance at or above which an item is the positive class.
        n_estimators: Boosting rounds.
        k: Cut-off for NDCG/MAP/precision/recall.
    """
    return run_ranker("pointwise_classification", "pointwise_classification(hist_gbm)", data, group_column, relevance, features, drop, k,
                      test_size, seed, model_path, relevant_from, n_estimators=n_estimators)


@tool
def fit_pairwise_logistic_ranker(
    data: str, group_column: str, relevance: str, features: list[str] | None = None, drop: list[str] | None = None,
    C: float = 1.0, pairs_per_group: int = 200, k: int = 10, relevant_from: float = 1, test_size: float = 0.25,
    seed: int = 42, model_path: str | None = None,
) -> dict:
    """Pairwise ranking (RankNet-style, linear): learn from pairs of items in the same group which one should rank higher, using logistic regression on feature differences.

    Args:
        group_column: Column identifying the query/group each row belongs to.
        C: Inverse regularisation strength.
        pairs_per_group: Pairs sampled per group.
        k: Cut-off for NDCG/MAP/precision/recall.
        relevant_from: Relevance at or above which an item counts as relevant.
    """
    return run_ranker("pairwise_logistic", "pairwise_logistic(RankNet-linear)", data, group_column, relevance, features, drop, k,
                      test_size, seed, model_path, relevant_from, C=C, pairs_per_group=pairs_per_group)


@tool
def fit_ranksvm(
    data: str, group_column: str, relevance: str, features: list[str] | None = None, drop: list[str] | None = None,
    C: float = 1.0, pairs_per_group: int = 200, k: int = 10, relevant_from: float = 1, test_size: float = 0.25,
    seed: int = 42, model_path: str | None = None,
) -> dict:
    """Pairwise ranking with a linear SVM on pair differences (RankSVM) — a max-margin ordering of items within each group.

    Args:
        group_column: Column identifying the query/group each row belongs to.
        C: Margin softness.
        pairs_per_group: Pairs sampled per group.
        k: Cut-off for NDCG/MAP/precision/recall.
        relevant_from: Relevance at or above which an item counts as relevant.
    """
    return run_ranker("ranksvm", "ranksvm", data, group_column, relevance, features, drop, k, test_size, seed, model_path,
                      relevant_from, C=C, pairs_per_group=pairs_per_group)


@tool
def fit_listnet_ranker(
    data: str, group_column: str, relevance: str, features: list[str] | None = None, drop: list[str] | None = None,
    epochs: int = 300, learning_rate: float = 0.05, l2: float = 0.001, k: int = 10, relevant_from: float = 1,
    test_size: float = 0.25, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Listwise ranking (ListNet, linear): optimise the probability distribution over the whole list directly with a top-one softmax cross-entropy, which aligns training with a list metric like NDCG.

    Args:
        group_column: Column identifying the query/group each row belongs to.
        epochs: Gradient steps over all groups.
        learning_rate: Adam step size.
        l2: Weight decay.
        k: Cut-off for NDCG/MAP/precision/recall.
        relevant_from: Relevance at or above which an item counts as relevant.
    """
    return run_ranker("listnet", "listnet(linear)", data, group_column, relevance, features, drop, k, test_size, seed, model_path,
                      relevant_from, epochs=epochs, learning_rate=learning_rate, l2=l2)


@tool(requires=("lightgbm",))
def fit_lightgbm_lambdarank(
    data: str, group_column: str, relevance: str, features: list[str] | None = None, drop: list[str] | None = None,
    n_estimators: int = 200, learning_rate: float = 0.05, num_leaves: int = 15, k: int = 10, relevant_from: float = 1,
    test_size: float = 0.25, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Listwise boosted trees with the LambdaRank objective (LightGBM) — the standard production choice for graded relevance and NDCG; needs integer relevance labels.

    Args:
        group_column: Column identifying the query/group each row belongs to.
        n_estimators: Boosting rounds.
        learning_rate: Shrinkage per round.
        num_leaves: Leaves per tree.
        k: Cut-off for NDCG/MAP/precision/recall.
        relevant_from: Relevance at or above which an item counts as relevant.
    """
    return run_ranker("lightgbm_lambdarank", "lightgbm_lambdarank", data, group_column, relevance, features, drop, k, test_size, seed,
                      model_path, relevant_from, n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves)


@tool(requires=("xgboost",))
def fit_xgboost_ranker(
    data: str, group_column: str, relevance: str, features: list[str] | None = None, drop: list[str] | None = None,
    n_estimators: int = 200, learning_rate: float = 0.1, k: int = 10, relevant_from: float = 1,
    test_size: float = 0.25, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Pairwise-objective boosted trees (XGBoost rank:pairwise) — an alternative to LightGBM LambdaRank; needs integer relevance labels.

    Args:
        group_column: Column identifying the query/group each row belongs to.
        n_estimators: Boosting rounds.
        learning_rate: Shrinkage per round.
        k: Cut-off for NDCG/MAP/precision/recall.
        relevant_from: Relevance at or above which an item counts as relevant.
    """
    return run_ranker("xgboost_rank", "xgboost_rank", data, group_column, relevance, features, drop, k, test_size, seed, model_path,
                      relevant_from, n_estimators=n_estimators, learning_rate=learning_rate)


@tool
def compare_rankers(
    data: str, group_column: str, relevance: str, features: list[str] | None = None, drop: list[str] | None = None,
    approaches: list[str] | None = None, k: int = 10, relevant_from: float = 1, folds: int = 4, seed: int = 42,
) -> dict:
    """Cross-validate several ranking approaches over the same group folds and rank them by NDCG@k — group-level folds, so no query leaks across the split.

    Args:
        group_column: Column identifying the query/group each row belongs to.
        approaches: Any of pointwise_regression, pointwise_linear, pointwise_random_forest, pointwise_classification, pairwise_logistic, ranksvm, listnet, lightgbm_lambdarank, xgboost_rank. Defaults to six that need no extra packages.
        k: Cut-off for NDCG/MAP.
        relevant_from: Relevance at or above which an item counts as relevant.
        folds: Number of group-level folds.
    """
    frame, columns = load_ranking(data, group_column, relevance, features, drop)
    kinds = approaches or ["pointwise_linear", "pointwise_regression", "pointwise_random_forest", "pointwise_classification", "pairwise_logistic", "listnet"]
    groups = frame[group_column].drop_duplicates().to_numpy(copy=True)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    parts = np.array_split(groups, max(2, min(folds, len(groups))))
    board = []
    for kind in kinds:
        scores = []
        try:
            for held in parts:
                test_mask = frame[group_column].isin(set(held))
                ranker = fit_ranker(kind, frame[~test_mask], columns, group_column, relevance, seed, threshold=relevant_from)
                test = frame[test_mask].assign(_s=lambda d: ranker.scores(d))
                scores.append(per_group_metrics(test, group_column, relevance, "_s", k, relevant_from)["ndcg"].mean())
            board.append({"approach": kind, f"ndcg@{k}": round(float(np.nanmean(scores)), 5), "std": round(float(np.nanstd(scores)), 5)})
        except Exception as error:
            board.append({"approach": kind, "error": str(error)[:120]})
    board.sort(key=lambda r: -r.get(f"ndcg@{k}", -1))
    return {"folds": len(parts), "metric": f"ndcg@{k}", "leaderboard": board, "best": board[0]["approach"] if board and "error" not in board[0] else None}
