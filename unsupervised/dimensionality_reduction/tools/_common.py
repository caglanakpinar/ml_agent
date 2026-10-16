"""Dimensionality-reduction machinery the tool modules share: building each reducer, measuring how faithful an embedding is
(trustworthiness, continuity, neighbourhood preservation, distance correlation, reconstruction), and the picklable model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_core.data import save_frame
from ml_core.ml import save_model, sk
from ml_core.unsup import Prepared, prepare

NO_OUT_OF_SAMPLE = ("tsne", "mds", "spectral_embedding")


def make_reducer(kind: str, n: int, params: dict[str, Any], seed: int) -> Any:
    sk()
    from sklearn import decomposition as dc, manifold as mf, random_projection as rp
    from sklearn.cluster import FeatureAgglomeration

    if kind == "pca":
        return dc.PCA(n_components=n, whiten=params.get("whiten", False), svd_solver=params.get("svd_solver", "auto"), random_state=seed)
    if kind == "incremental_pca":
        return dc.IncrementalPCA(n_components=n, batch_size=params.get("batch_size"))
    if kind == "truncated_svd":
        return dc.TruncatedSVD(n_components=n, random_state=seed)
    if kind == "sparse_pca":
        return dc.SparsePCA(n_components=n, alpha=params.get("alpha", 1.0), random_state=seed)
    if kind == "nmf":
        return dc.NMF(n_components=n, init="nndsvda", max_iter=500, random_state=seed)
    if kind == "ica":
        return dc.FastICA(n_components=n, whiten="unit-variance", max_iter=500, random_state=seed)
    if kind == "factor_analysis":
        return dc.FactorAnalysis(n_components=n, rotation=params.get("rotation"), random_state=seed)
    if kind == "kernel_pca":
        return dc.KernelPCA(n_components=n, kernel=params.get("kernel", "rbf"), gamma=params.get("gamma"), fit_inverse_transform=True, random_state=seed)
    if kind == "tsne":
        return mf.TSNE(n_components=n, perplexity=params.get("perplexity", 30), learning_rate="auto", init="pca", random_state=seed)
    if kind == "isomap":
        return mf.Isomap(n_components=n, n_neighbors=params.get("n_neighbors", 10))
    if kind == "lle":
        return mf.LocallyLinearEmbedding(n_components=n, n_neighbors=params.get("n_neighbors", 12), method=params.get("method", "standard"), random_state=seed)
    if kind == "mds":
        try:  # scikit-learn 1.8+ takes `init`, and changes its default in 1.10
            return mf.MDS(n_components=n, n_init=1, max_iter=200, init="random", random_state=seed)
        except TypeError:
            return mf.MDS(n_components=n, n_init=1, max_iter=200, random_state=seed)
    if kind == "spectral_embedding":
        return mf.SpectralEmbedding(n_components=n, n_neighbors=params.get("n_neighbors", 10), random_state=seed)
    if kind == "random_projection":
        return (rp.GaussianRandomProjection if params.get("sparse") is False else rp.SparseRandomProjection)(n_components=n, random_state=seed)
    if kind == "feature_agglomeration":
        return FeatureAgglomeration(n_clusters=n)
    if kind == "umap":
        import umap

        return umap.UMAP(n_components=n, n_neighbors=params.get("n_neighbors", 15), min_dist=params.get("min_dist", 0.1), random_state=seed)
    raise ValueError(f"unknown reducer {kind!r}")


def nonnegative(Z: np.ndarray) -> np.ndarray:
    return Z - Z.min(axis=0)


def neighbour_sets(Z: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    return NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]


def quality(Z: np.ndarray, E: np.ndarray, k: int = 10, seed: int = 0, max_rows: int = 2500) -> dict[str, float]:
    """How faithfully the embedding E keeps the neighbourhoods and distances of Z."""
    from scipy.stats import spearmanr
    from sklearn.manifold import trustworthiness

    if len(Z) > max_rows:
        idx = np.random.default_rng(seed).choice(len(Z), max_rows, replace=False)
        Z, E = Z[idx], E[idx]
    k = min(k, max(2, len(Z) // 5))
    a, b = neighbour_sets(Z, k), neighbour_sets(E, k)
    overlap = float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(Z), 4000), rng.integers(0, len(Z), 4000)
    dz, de = np.linalg.norm(Z[i] - Z[j], axis=1), np.linalg.norm(E[i] - E[j], axis=1)
    return {"trustworthiness": round(float(trustworthiness(Z, E, n_neighbors=k)), 4), "continuity": round(float(trustworthiness(E, Z, n_neighbors=k)), 4),
            "neighbour_overlap": round(overlap, 4), "distance_rank_correlation": round(float(spearmanr(dz, de)[0]), 4)}


def probe_accuracy(E: np.ndarray, truth: np.ndarray, seed: int = 0) -> float:
    """5-fold k-NN accuracy of predicting known labels from the embedding."""
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    return round(float(cross_val_score(KNeighborsClassifier(10), E, truth, cv=5).mean()), 4)


class ReducerModel:
    def __init__(self, kind: str, prep: Any, columns: list[str], model: Any, nonneg_shift: np.ndarray | None):
        self.kind, self.prep, self.columns, self.model, self.shift = kind, prep, columns, model, nonneg_shift

    def transform(self, Z: np.ndarray) -> np.ndarray:
        if self.kind in NO_OUT_OF_SAMPLE:
            raise ValueError(f"{self.kind} has no transform for new rows — it only embeds the rows it was fitted on; use PCA, kernel PCA, Isomap, UMAP or an autoencoder to embed new data")
        return np.asarray(self.model.transform(Z - self.shift if self.shift is not None else Z))


def fit_reducer(P: Prepared, kind: str, n: int, params: dict[str, Any], seed: int) -> tuple[Any, np.ndarray, np.ndarray | None]:
    Z = P.Z
    shift = None
    if kind == "nmf":
        shift = P.Z.min(axis=0)
        Z = Z - shift
    model = make_reducer(kind, n, params, seed)
    E = np.asarray(model.fit_transform(Z))
    return model, E, shift


def run_reduction(kind: str, label: str, data: Any, n: int, columns: Any, drop: Any, scale: bool, truth_column: str | None, params: dict[str, Any], seed: int, output_path: str | None,
                  model_path: str | None, max_rows: int | None = None, keep_columns: Any = None, P: Prepared | None = None) -> dict[str, Any]:
    P = P or prepare(data, columns, drop, scale, truth_column, max_rows, seed)
    if kind != "random_projection":
        n = min(n, P.Z.shape[1])
    model, E, shift = fit_reducer(P, kind, n, params, seed)
    out: dict[str, Any] = {"method": label, "rows": int(len(P.Z)), "input_dimensions": int(P.Z.shape[1]), "output_dimensions": int(E.shape[1]), "quality": quality(P.Z, E, seed=seed)}
    if hasattr(model, "explained_variance_ratio_"):
        ev = np.asarray(model.explained_variance_ratio_)
        out["explained_variance_ratio"] = ev.round(4).tolist()
        out["cumulative_explained_variance"] = round(float(ev.sum()), 4)
    if hasattr(model, "inverse_transform") and kind not in ("feature_agglomeration",):
        try:
            R = np.asarray(model.inverse_transform(E))
            out["reconstruction_rmse"] = round(float(np.sqrt(np.mean((P.Z - (R + (shift if shift is not None else 0))) ** 2))), 4)
            out["reconstruction_r2"] = round(float(1 - np.sum((P.Z - R - (shift if shift is not None else 0)) ** 2) / np.sum((P.Z - P.Z.mean(axis=0)) ** 2)), 4)
        except Exception:
            pass
    if hasattr(model, "components_") and kind not in ("kernel_pca",):
        comp = model.components_.toarray() if hasattr(model.components_, "toarray") else np.asarray(model.components_)
        if comp.ndim == 2 and comp.shape[1] == len(P.names):
            out["top_loadings"] = {f"dim_{i + 1}": {P.names[j]: round(float(comp[i, j]), 3) for j in np.argsort(-np.abs(comp[i]))[:4]} for i in range(min(comp.shape[0], 6))}
    if hasattr(model, "kl_divergence_"):
        out["kl_divergence"] = round(float(model.kl_divergence_), 4)
    if hasattr(model, "reconstruction_error_"):
        out["reconstruction_error"] = round(float(model.reconstruction_error_), 4)
    if hasattr(model, "stress_"):
        out["stress"] = round(float(model.stress_), 4)
    if P.truth is not None:
        out["downstream_knn_accuracy_original"] = probe_accuracy(P.Z, P.truth, seed)
        out["downstream_knn_accuracy_embedding"] = probe_accuracy(E, P.truth, seed)
    table = pd.DataFrame(E, columns=[f"dim_{i + 1}" for i in range(E.shape[1])])
    keep = [c for c in (keep_columns or []) if c in P.frame.columns]
    if truth_column and truth_column in P.frame.columns:
        keep.append(truth_column)
    for c in dict.fromkeys(keep):
        table.insert(0, c, P.frame[c].to_numpy())
    out["written_to"] = save_frame(table, output_path, f"embedding_{kind}.csv") if output_path else None
    out["model_path"] = save_model(ReducerModel(kind, P.prep, P.columns, model, shift), model_path) if kind not in NO_OUT_OF_SAMPLE else None
    if kind in NO_OUT_OF_SAMPLE:
        out["note"] = f"{kind} embeds only the fitted rows and has no model for new rows"
    return out
