"""Fit one classification algorithm and score it on held-out rows — one function per algorithm family."""

from __future__ import annotations

from typing import Literal


from ml_core.ml import logistic_kwargs, make_pipeline, sk, split_xy
from ml_core.tooling import tool
from supervised.classification.tools._common import (
    MODEL_NAMES,
    make_estimator,
    parse_layers,
    run_classifier,
)

CW = Literal["none", "balanced"]


def _cw(value: str) -> str | None:
    return None if value == "none" else value


@tool
def fit_logistic_regression(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    penalty: Literal["l2", "l1", "elasticnet"] = "l2", C: float = 1.0, class_weight: CW = "none",
    test_size: float = 0.2, seed: int = 42, cv: int = 0, positive_label: str | None = None,
    model_path: str | None = None,
) -> dict:
    """Fit a (regularised) logistic regression — the interpretable linear baseline; coefficients show direction and strength.

    Args:
        penalty: l2 shrinks, l1 selects features, elasticnet mixes the two.
        C: Inverse regularisation strength; smaller is stronger.
        class_weight: 'balanced' reweights classes by inverse frequency.
        positive_label: Class treated as positive for binary precision/recall/AUC.
    """
    from sklearn.linear_model import LogisticRegression

    est = LogisticRegression(**logistic_kwargs(penalty, C, class_weight=_cw(class_weight), max_iter=2000))
    return run_classifier(est, f"logistic_regression({penalty}, C={C})", data, target, features, drop,
                          test_size, seed, True, model_path, cv, positive_label)


@tool
def fit_random_forest(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    n_estimators: int = 300, max_depth: int | None = None, min_samples_leaf: int = 1,
    class_weight: Literal["none", "balanced", "balanced_subsample"] = "none",
    test_size: float = 0.2, seed: int = 42, cv: int = 0, positive_label: str | None = None,
    model_path: str | None = None,
) -> dict:
    """Fit a random forest — a strong default for tabular data: handles non-linearity and interactions, little tuning needed.

    Args:
        n_estimators: Number of trees.
        max_depth: Depth limit; unset grows trees fully.
        min_samples_leaf: Larger values smooth the model and curb overfitting.
        class_weight: none, balanced, or balanced_subsample.
    """
    est = make_estimator("random_forest", n_estimators=n_estimators, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf, class_weight=_cw(class_weight), random_state=seed)
    return run_classifier(est, "random_forest", data, target, features, drop, test_size, seed, False,
                          model_path, cv, positive_label)


@tool
def fit_extra_trees(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    n_estimators: int = 300, max_depth: int | None = None, class_weight: CW = "none",
    test_size: float = 0.2, seed: int = 42, cv: int = 0, positive_label: str | None = None,
    model_path: str | None = None,
) -> dict:
    """Fit extremely randomised trees — like a random forest with random split points: faster, lower variance, often a touch smoother.

    Args:
        n_estimators: Number of trees.
        max_depth: Depth limit; unset grows trees fully.
        class_weight: none or balanced.
    """
    est = make_estimator("extra_trees", n_estimators=n_estimators, max_depth=max_depth,
                         class_weight=_cw(class_weight), random_state=seed)
    return run_classifier(est, "extra_trees", data, target, features, drop, test_size, seed, False,
                          model_path, cv, positive_label)


@tool
def fit_gradient_boosting(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    n_estimators: int = 200, learning_rate: float = 0.1, max_depth: int = 3, subsample: float = 1.0,
    test_size: float = 0.2, seed: int = 42, cv: int = 0, positive_label: str | None = None,
    model_path: str | None = None,
) -> dict:
    """Fit gradient-boosted trees — usually the most accurate on tabular data, at the price of more tuning and slower fits.

    Args:
        n_estimators: Boosting rounds.
        learning_rate: Shrinkage per round; lower needs more rounds.
        max_depth: Depth of each tree; 3 is a good start.
        subsample: Fraction of rows per round; below 1 adds stochastic regularisation.
    """
    est = make_estimator("gradient_boosting", n_estimators=n_estimators, learning_rate=learning_rate,
                         max_depth=max_depth, subsample=subsample, random_state=seed)
    return run_classifier(est, "gradient_boosting", data, target, features, drop, test_size, seed, False,
                          model_path, cv, positive_label)


@tool
def fit_hist_gradient_boosting(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    max_iter: int = 300, learning_rate: float = 0.1, max_leaf_nodes: int = 31,
    class_weight: CW = "none", l2_regularization: float = 0.0,
    test_size: float = 0.2, seed: int = 42, cv: int = 0, positive_label: str | None = None,
    model_path: str | None = None,
) -> dict:
    """Fit histogram gradient boosting — boosted trees that stay fast on 100k+ rows, with early stopping built in.

    Args:
        max_iter: Maximum boosting rounds.
        learning_rate: Shrinkage per round.
        max_leaf_nodes: Leaves per tree; the main complexity control.
        class_weight: none or balanced.
        l2_regularization: L2 penalty on leaf values.
    """
    est = make_estimator("hist_gradient_boosting", max_iter=max_iter, learning_rate=learning_rate,
                         max_leaf_nodes=max_leaf_nodes, class_weight=_cw(class_weight),
                         l2_regularization=l2_regularization, random_state=seed)
    return run_classifier(est, "hist_gradient_boosting", data, target, features, drop, test_size, seed,
                          False, model_path, cv, positive_label)


@tool(requires=("lightgbm",))
def fit_lightgbm_classifier(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    n_estimators: int = 400, learning_rate: float = 0.05, num_leaves: int = 31,
    class_weight: CW = "none", test_size: float = 0.2, seed: int = 42, cv: int = 0,
    positive_label: str | None = None, model_path: str | None = None,
) -> dict:
    """Fit LightGBM — the usual choice for large tabular data and many categorical levels.

    Args:
        n_estimators: Boosting rounds.
        learning_rate: Shrinkage per round.
        num_leaves: Leaves per tree.
        class_weight: none or balanced.
    """
    from lightgbm import LGBMClassifier

    est = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                         class_weight=_cw(class_weight), random_state=seed, verbose=-1)
    return run_classifier(est, "lightgbm", data, target, features, drop, test_size, seed, False,
                          model_path, cv, positive_label)


@tool
def fit_decision_tree(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    max_depth: int = 5, min_samples_leaf: int = 10, class_weight: CW = "none",
    test_size: float = 0.2, seed: int = 42, positive_label: str | None = None,
    model_path: str | None = None,
) -> dict:
    """Fit a single shallow decision tree — fully explainable as if/then rules, at some cost in accuracy.

    Args:
        max_depth: Tree depth; keep it small for readable rules.
        min_samples_leaf: Smallest leaf allowed.
        class_weight: none or balanced.
    """
    est = make_estimator("decision_tree", max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                         class_weight=_cw(class_weight), random_state=seed)
    return run_classifier(est, "decision_tree", data, target, features, drop, test_size, seed, False,
                          model_path, 0, positive_label)


@tool
def fit_svm(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    kernel: Literal["rbf", "linear", "poly", "sigmoid"] = "rbf", C: float = 1.0,
    class_weight: CW = "none", test_size: float = 0.2, seed: int = 42, positive_label: str | None = None,
    model_path: str | None = None,
) -> dict:
    """Fit a support-vector machine — good on small/medium, high-dimensional data; slow beyond ~50k rows.

    Args:
        kernel: rbf for non-linear boundaries, linear for text-like sparse data.
        C: Margin softness; larger fits the training data harder.
        class_weight: none or balanced.
    """
    est = make_estimator("svm", kernel=kernel, C=C, class_weight=_cw(class_weight), random_state=seed)
    return run_classifier(est, f"svm({kernel})", data, target, features, drop, test_size, seed, True,
                          model_path, 0, positive_label)


@tool
def fit_knn(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    n_neighbors: int = 15, weights: Literal["uniform", "distance"] = "uniform",
    test_size: float = 0.2, seed: int = 42, positive_label: str | None = None,
    model_path: str | None = None,
) -> dict:
    """Fit k-nearest neighbours — a non-parametric look at how well 'similar rows share a label' holds.

    Args:
        n_neighbors: Neighbours that vote.
        weights: distance weights closer neighbours more.
    """
    est = make_estimator("knn", n_neighbors=n_neighbors, weights=weights)
    return run_classifier(est, f"knn(k={n_neighbors})", data, target, features, drop, test_size, seed,
                          True, model_path, 0, positive_label)


@tool
def fit_naive_bayes(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    variant: Literal["gaussian", "bernoulli"] = "gaussian", test_size: float = 0.2, seed: int = 42,
    positive_label: str | None = None, model_path: str | None = None,
) -> dict:
    """Fit naive Bayes — instant to train, a good sanity check, and surprisingly strong with little data.

    Args:
        variant: gaussian for continuous features, bernoulli for binary indicators.
    """
    est = make_estimator("gaussian_nb" if variant == "gaussian" else "bernoulli_nb")
    return run_classifier(est, f"naive_bayes({variant})", data, target, features, drop, test_size, seed,
                          True, model_path, 0, positive_label)


@tool
def fit_mlp(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    hidden_layers: str = "64,32", alpha: float = 0.0001, max_iter: int = 500,
    test_size: float = 0.2, seed: int = 42, positive_label: str | None = None, model_path: str | None = None,
) -> dict:
    """Fit a small feed-forward neural network — worth trying when interactions are rich and rows are plentiful.

    Args:
        hidden_layers: Comma-separated layer widths, e.g. "64,32".
        alpha: L2 penalty.
        max_iter: Training epochs.
    """
    est = make_estimator("mlp", hidden_layer_sizes=parse_layers(hidden_layers), alpha=alpha,
                         max_iter=max_iter, random_state=seed)
    return run_classifier(est, f"mlp({hidden_layers})", data, target, features, drop, test_size, seed,
                          True, model_path, 0, positive_label)


@tool
def fit_adaboost(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    n_estimators: int = 200, learning_rate: float = 0.5, test_size: float = 0.2, seed: int = 42,
    positive_label: str | None = None, model_path: str | None = None,
) -> dict:
    """Fit AdaBoost — boosted decision stumps; a light model that focuses on the rows earlier rounds got wrong.

    Args:
        n_estimators: Boosting rounds.
        learning_rate: Weight of each round.
    """
    est = make_estimator("adaboost", n_estimators=n_estimators, learning_rate=learning_rate, random_state=seed)
    return run_classifier(est, "adaboost", data, target, features, drop, test_size, seed, False,
                          model_path, 0, positive_label)


@tool
def fit_lda(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    shrinkage: float | None = None, test_size: float = 0.2, seed: int = 42,
    positive_label: str | None = None, model_path: str | None = None,
) -> dict:
    """Fit linear discriminant analysis — a closed-form linear classifier that also projects to class-separating axes.

    Args:
        shrinkage: Covariance shrinkage between 0 and 1; unset means none. Helps when features outnumber rows.
    """
    kwargs = {"solver": "lsqr", "shrinkage": shrinkage} if shrinkage is not None else {}
    est = make_estimator("lda", **kwargs)
    return run_classifier(est, "lda", data, target, features, drop, test_size, seed, True, model_path,
                          0, positive_label)


@tool
def fit_sgd_classifier(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    loss: Literal["log_loss", "modified_huber", "hinge"] = "log_loss", alpha: float = 0.0001,
    class_weight: CW = "none", test_size: float = 0.2, seed: int = 42,
    positive_label: str | None = None, model_path: str | None = None,
) -> dict:
    """Fit a linear model by stochastic gradient descent — the choice when the data is too big for the solvers above.

    Args:
        loss: log_loss gives probabilities; hinge is a linear SVM.
        alpha: Regularisation strength.
        class_weight: none or balanced.
    """
    est = make_estimator("sgd", loss=loss, alpha=alpha, class_weight=_cw(class_weight))
    return run_classifier(est, f"sgd({loss})", data, target, features, drop, test_size, seed, True,
                          model_path, 0, positive_label)


@tool
def compare_classifiers(
    data: str, target: str, features: list[str] | None = None, drop: list[str] | None = None,
    models: list[str] | None = None, cv: int = 5, scoring: Literal["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "roc_auc", "average_precision"] = "f1_macro", seed: int = 42,
) -> dict:
    """Cross-validate several classifiers on the same folds and rank them — the fastest way to see which family suits the data.

    Args:
        models: Which to compare (logistic_regression, random_forest, extra_trees, gradient_boosting, hist_gradient_boosting, decision_tree, svm, knn, gaussian_nb, mlp, adaboost, lda, sgd). Defaults to a representative six.
        scoring: Metric to rank by.
    """
    sk()
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X, y = split_xy(data, target, features, drop)
    chosen = models or ["logistic_regression", "random_forest", "gradient_boosting",
                        "hist_gradient_boosting", "knn", "gaussian_nb"]
    unknown = [m for m in chosen if m not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"unknown model(s) {unknown}; choose from {list(MODEL_NAMES)}")
    folds = StratifiedKFold(n_splits=max(2, min(cv, int(y.value_counts().min()))), shuffle=True, random_state=seed)
    metric = scoring if (y.nunique() == 2 or scoring not in ("roc_auc", "average_precision")) else "f1_macro"
    board = []
    for name in chosen:
        try:
            pipe = make_pipeline(make_estimator(name, random_state=seed) if name not in ("knn", "gaussian_nb", "lda", "bernoulli_nb") else make_estimator(name), X, scale=True)
            scores = cross_val_score(pipe, X, y, cv=folds, scoring=metric, n_jobs=1)
            board.append({"model": name, "mean": round(float(scores.mean()), 5), "std": round(float(scores.std()), 5)})
        except Exception as error:
            board.append({"model": name, "error": str(error)[:120]})
    board.sort(key=lambda r: r.get("mean", -1), reverse=True)
    return {"scoring": metric, "cv_folds": folds.get_n_splits(), "leaderboard": board,
            "best": board[0]["model"] if board and "mean" in board[0] else None}
