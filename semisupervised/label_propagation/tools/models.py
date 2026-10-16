"""Fit a graph-based semi-supervised model: label propagation and label spreading on k-NN or RBF graphs, with dimensionality reduction, feature weighting, and an inductive classifier on top."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ml_core.semi import Fitted, base_clf, run_semi, sk
from ml_core.tooling import tool
from semisupervised.label_propagation.tools._common import propagate, pseudo_block

Base = Literal["logistic_regression", "random_forest", "hist_gradient_boosting", "svm", "knn", "naive_bayes"]


class Scale:
    """Picklable column scaling (a lambda would not survive joblib)."""

    def __init__(self, w: np.ndarray):
        self.w = w

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.w


class Propagated:
    """Wraps a fitted propagator so unseen rows are scored through the graph kernel."""

    def __init__(self, model, K, transform=None):
        self.model, self.K, self.transform = model, K, transform
        self.classes_ = np.arange(K)

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        Z = self.transform(Z) if self.transform else Z
        raw = self.model.predict_proba(Z)
        out = np.zeros((len(Z), self.K))
        out[:, [int(c) for c in self.model.classes_]] = raw
        return out


def _prop(label, kind, kernel, n_neighbors, gamma, alpha, max_iter, data, label_column, unlabeled_value, features, drop, truth_column, test_size, seed, model_path):
    def trainer(ctx):
        model, extra, dist = propagate(ctx, kind, kernel, n_neighbors, gamma, alpha, max_iter)
        extra["pseudo"] = pseudo_block(ctx, dist)
        extra["unlabelled_confidence"] = {"mean": round(float(dist.max(axis=1).mean()), 4), "share_below_0.6": round(float((dist.max(axis=1) < 0.6).mean()), 4)}
        return Propagated(model, ctx.K), extra

    return run_semi(label, trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, "knn")


@tool
def fit_label_spreading_knn(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    n_neighbors: int = 10, alpha: float = 0.2, max_iter: int = 1000, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Label spreading on a k-nearest-neighbour graph: labels diffuse along edges with a normalised graph Laplacian, and `alpha` controls how much a labelled row may be overridden by its neighbours (soft clamping) — robust to a few wrong labels. The default graph method; scales to large data.

    Args:
        n_neighbors: Neighbours per row in the graph.
        alpha: 0 keeps the given labels fixed; values near 1 let neighbours override them.
        max_iter: Most diffusion iterations.
    """
    return _prop(f"label_spreading(knn,k={n_neighbors},alpha={alpha})", "spreading", "knn", n_neighbors, 1.0, alpha, max_iter, data, label_column, unlabeled_value, features, drop, truth_column, test_size, seed, model_path)


@tool
def fit_label_spreading_rbf(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    gamma: float = 0.5, alpha: float = 0.2, max_iter: int = 1000, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Label spreading on a fully connected RBF (Gaussian) similarity graph — every pair of rows is linked with a weight that decays with distance, so influence is smooth rather than limited to k neighbours. Dense: practical up to a few thousand rows.

    Args:
        gamma: RBF width; larger means labels travel a shorter distance.
        alpha: 0 keeps the given labels fixed; values near 1 let neighbours override them.
        max_iter: Most diffusion iterations.
    """
    return _prop(f"label_spreading(rbf,gamma={gamma},alpha={alpha})", "spreading", "rbf", 10, gamma, alpha, max_iter, data, label_column, unlabeled_value, features, drop, truth_column, test_size, seed, model_path)


@tool
def fit_label_propagation_knn(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    n_neighbors: int = 10, max_iter: int = 1000, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Classic label propagation on a k-NN graph: the given labels are clamped (hard) and everything else is the fixed point of the diffusion. Use when the labels are trusted completely.

    Args:
        n_neighbors: Neighbours per row in the graph.
        max_iter: Most diffusion iterations.
    """
    return _prop(f"label_propagation(knn,k={n_neighbors})", "propagation", "knn", n_neighbors, 1.0, 0.0, max_iter, data, label_column, unlabeled_value, features, drop, truth_column, test_size, seed, model_path)


@tool
def fit_label_propagation_rbf(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    gamma: float = 0.5, max_iter: int = 1000, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Classic label propagation on a dense RBF graph with hard-clamped labels.

    Args:
        gamma: RBF width; larger means labels travel a shorter distance.
        max_iter: Most diffusion iterations.
    """
    return _prop(f"label_propagation(rbf,gamma={gamma})", "propagation", "rbf", 10, gamma, 0.0, max_iter, data, label_column, unlabeled_value, features, drop, truth_column, test_size, seed, model_path)


@tool
def fit_label_spreading_after_pca(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    n_components: int = 5, n_neighbors: int = 10, alpha: float = 0.2, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Reduce the features with PCA (fitted on labelled + unlabelled rows) before building the graph — distances in many noisy dimensions are nearly equal, which destroys the neighbourhood structure propagation relies on.

    Args:
        n_components: PCA dimensions kept.
        n_neighbors: Neighbours per row in the graph.
        alpha: Soft-clamping factor.
    """
    sk()
    from sklearn.decomposition import PCA

    def trainer(ctx):
        pca = PCA(n_components=min(n_components, ctx.Z.shape[1]), random_state=seed).fit(np.vstack([ctx.Zl, ctx.Zu]))
        sub = type(ctx)(pca.transform(ctx.Z), ctx.sd, ctx.split, ctx.base, ctx.seed)
        model, extra, dist = propagate(sub, "spreading", "knn", n_neighbors, 1.0, alpha, 1000)
        extra["variance_explained"] = round(float(pca.explained_variance_ratio_.sum()), 4)
        extra["pseudo"] = pseudo_block(ctx, dist)
        return Propagated(model, ctx.K, transform=pca.transform), extra

    return run_semi(f"label_spreading(pca={n_components},knn,k={n_neighbors})", trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, "knn")


@tool
def fit_label_spreading_weighted_features(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    n_neighbors: int = 10, alpha: float = 0.2, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Learn a metric from the labels before building the graph: a random forest fitted on the labelled rows scores each feature's usefulness, and the features are scaled by it, so neighbours are judged on the dimensions that actually carry the label rather than on every column equally.

    Args:
        n_neighbors: Neighbours per row in the graph.
        alpha: Soft-clamping factor.
    """
    sk()
    from sklearn.ensemble import RandomForestClassifier

    def trainer(ctx):
        rf = RandomForestClassifier(200, min_samples_leaf=2, n_jobs=-1, random_state=seed).fit(ctx.Zl, ctx.yl)
        w = rf.feature_importances_ / max(rf.feature_importances_.max(), 1e-9)
        w = np.sqrt(w)
        sub = type(ctx)(ctx.Z * w, ctx.sd, ctx.split, ctx.base, ctx.seed)
        model, extra, dist = propagate(sub, "spreading", "knn", n_neighbors, 1.0, alpha, 1000)
        extra["feature_weights"] = {f"col{i}": round(float(x), 3) for i, x in enumerate(w) if x > 0.3}
        extra["pseudo"] = pseudo_block(ctx, dist)
        return Propagated(model, ctx.K, transform=Scale(w)), extra

    return run_semi(f"label_spreading(weighted_features,k={n_neighbors})", trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, "knn")


@tool
def fit_propagation_then_classifier(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "random_forest", n_neighbors: int = 10, alpha: float = 0.2, threshold: float = 0.8, test_size: float = 0.3, seed: int = 42, model_path: str | None = None, output_path: str | None = None,
) -> dict:
    """Inductive finish: propagate labels over the graph, keep the unlabelled rows whose propagated label is confident, and train an ordinary classifier on labelled + propagated rows. The result scores new rows instantly, with no graph — propagation itself cannot do that cheaply.

    Args:
        base: Classifier trained on the combined rows.
        n_neighbors: Neighbours per row in the graph.
        alpha: Soft-clamping factor.
        threshold: Confidence a propagated label needs to be used for training.
        output_path: Where to write every unlabelled row with its propagated label and confidence.
    """
    def trainer(ctx):
        from ml_core.data import save_frame

        model, extra, dist = propagate(ctx, "spreading", "knn", n_neighbors, 1.0, alpha, 1000)
        keep = np.flatnonzero(dist.max(axis=1) >= threshold)
        clf = base_clf(base, seed).fit(np.vstack([ctx.Zl, ctx.Zu[keep]]), np.concatenate([ctx.yl, dist[keep].argmax(axis=1)]))
        extra["propagated_rows_used"] = int(len(keep))
        extra["pseudo"] = pseudo_block(ctx, dist, threshold)
        if output_path:
            table = ctx.sd.X.iloc[ctx.split.unl].reset_index(drop=True).assign(propagated_label=[ctx.sd.classes[c] for c in dist.argmax(axis=1)], confidence=dist.max(axis=1))
            extra["written_to"] = save_frame(table, output_path, "propagated_labels.csv")
        return Fitted(clf, ctx.K), extra

    return run_semi(f"propagation_then_{base}", trainer, data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, model_path, base)
