"""Clustering machinery the tool modules share: building each algorithm, scoring a partition (internal metrics, and external
ones when hold-out labels exist), describing what distinguishes the clusters, and the picklable model that assigns new rows."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_core.data import save_frame
from ml_core.ml import save_model, sk
from ml_core.unsup import Prepared, prepare

MAX_SILHOUETTE_ROWS = 4000


def internal_metrics(Z: np.ndarray, labels: np.ndarray, seed: int = 0) -> dict[str, float | None]:
    """Silhouette, Calinski–Harabasz, Davies–Bouldin over the clustered (non-noise) rows."""
    from sklearn import metrics as m

    mask = labels >= 0
    z, y = Z[mask], labels[mask]
    if len(set(y)) < 2 or len(y) < 5:
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}
    sil = m.silhouette_score(z, y, sample_size=min(MAX_SILHOUETTE_ROWS, len(y)), random_state=seed)
    return {"silhouette": round(float(sil), 4), "calinski_harabasz": round(float(m.calinski_harabasz_score(z, y)), 2), "davies_bouldin": round(float(m.davies_bouldin_score(z, y)), 4)}


def external_metrics(truth: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Agreement with hold-out labels that were never used for fitting."""
    from sklearn import metrics as m

    h, c, v = m.homogeneity_completeness_v_measure(truth, labels)
    table = pd.crosstab(labels, truth)
    return {"adjusted_rand_index": round(float(m.adjusted_rand_score(truth, labels)), 4), "normalized_mutual_info": round(float(m.normalized_mutual_info_score(truth, labels)), 4),
            "homogeneity": round(float(h), 4), "completeness": round(float(c), 4), "v_measure": round(float(v), 4), "fowlkes_mallows": round(float(m.fowlkes_mallows_score(truth, labels)), 4),
            "purity": round(float(table.max(axis=1).sum() / table.to_numpy().sum()), 4)}


def describe_clusters(P: Prepared, labels: np.ndarray, k: int = 6) -> list[dict[str, Any]]:
    """For each cluster: size, share, and the numeric features that differ most from the whole table (effect size in standard deviations)."""
    X = P.X
    num = X.select_dtypes("number")
    overall_mean, overall_std = num.mean(), num.std().replace(0, 1)
    rows = []
    for c in sorted(set(labels)):
        m = labels == c
        diff = ((num[m].mean() - overall_mean) / overall_std).sort_values(key=abs, ascending=False)
        entry: dict[str, Any] = {"cluster": int(c) if c >= 0 else "noise", "size": int(m.sum()), "share": round(float(m.mean()), 4),
                                 "distinguishing_features": {f: round(float(v), 3) for f, v in diff.head(k).items()}}
        cats = {}
        for col in X.columns:
            if col not in num.columns and X[col].nunique() <= 20:
                top = X.loc[m, col].astype(str).value_counts(normalize=True).head(1)
                base = X[col].astype(str).value_counts(normalize=True).get(top.index[0], 0)
                if top.iloc[0] - base > 0.15:
                    cats[col] = f"{top.index[0]} ({top.iloc[0]:.0%} vs {base:.0%} overall)"
        if cats:
            entry["over_represented_categories"] = cats
        rows.append(entry)
    return rows


def partition_summary(P: Prepared, labels: np.ndarray, seed: int = 0) -> dict[str, Any]:
    sizes = pd.Series(labels).value_counts().sort_index()
    out: dict[str, Any] = {"clusters": int((sizes.index >= 0).sum()), "noise_rows": int((labels < 0).sum()), "noise_share": round(float((labels < 0).mean()), 4),
                           "cluster_sizes": {(int(k) if k >= 0 else "noise"): int(v) for k, v in sizes.items()}, "internal": internal_metrics(P.Z, labels, seed)}
    if P.truth is not None:
        out["external_vs_truth"] = external_metrics(P.truth, labels)
    return out


class ClusterModel:
    """Preprocessor + a way to assign new rows: the algorithm's own `predict` when it has one, otherwise the nearest cluster centroid."""

    def __init__(self, kind: str, prep: Any, columns: list[str], model: Any, centroids: dict[int, np.ndarray], labels: np.ndarray):
        self.kind, self.prep, self.columns, self.model, self.centroids, self.train_labels = kind, prep, columns, model, centroids, labels

    def assign(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(cluster, distance to that cluster's centroid) for each encoded row."""
        ids = np.array(sorted(self.centroids))
        C = np.stack([self.centroids[i] for i in ids])
        d = np.linalg.norm(Z[:, None, :] - C[None, :, :], axis=2)
        nearest = d.argmin(axis=1)
        labels = ids[nearest]
        if self.model is not None and hasattr(self.model, "predict") and self.kind not in ("dbscan", "optics", "agglomerative", "spectral", "hdbscan"):
            try:
                labels = np.asarray(self.model.predict(Z))
            except Exception:
                pass
        return labels, d[np.arange(len(Z)), nearest]


def make_model(kind: str, k: int | None, params: dict[str, Any], seed: int) -> Any:
    sk()
    from sklearn import cluster as c, mixture as mx

    if kind == "kmeans":
        return c.KMeans(n_clusters=k, n_init=params.get("n_init", 10), random_state=seed)
    if kind == "minibatch_kmeans":
        return c.MiniBatchKMeans(n_clusters=k, batch_size=params.get("batch_size", 1024), n_init=3, random_state=seed)
    if kind == "gmm":
        return mx.GaussianMixture(n_components=k, covariance_type=params.get("covariance_type", "full"), n_init=3, random_state=seed)
    if kind == "bayesian_gmm":
        return mx.BayesianGaussianMixture(n_components=k, weight_concentration_prior_type="dirichlet_process", weight_concentration_prior=params.get("concentration", 0.01), max_iter=500, random_state=seed)
    if kind == "agglomerative":
        return c.AgglomerativeClustering(n_clusters=k, linkage=params.get("linkage", "ward"))
    if kind == "dbscan":
        return c.DBSCAN(eps=params["eps"], min_samples=params.get("min_samples", 5))
    if kind == "hdbscan":
        try:
            return c.HDBSCAN(min_cluster_size=params.get("min_cluster_size", 10), min_samples=params.get("min_samples"), copy=True)
        except TypeError:  # scikit-learn versions before the `copy` argument
            return c.HDBSCAN(min_cluster_size=params.get("min_cluster_size", 10), min_samples=params.get("min_samples"))
    if kind == "optics":
        return c.OPTICS(min_samples=params.get("min_samples", 10), xi=params.get("xi", 0.05), min_cluster_size=params.get("min_cluster_size", 0.05))
    if kind == "birch":
        return c.Birch(n_clusters=k, threshold=params.get("threshold", 0.5))
    if kind == "spectral":
        return c.SpectralClustering(n_clusters=k, affinity=params.get("affinity", "nearest_neighbors"), n_neighbors=params.get("n_neighbors", 10), assign_labels="kmeans", random_state=seed)
    if kind == "affinity_propagation":
        return c.AffinityPropagation(damping=params.get("damping", 0.7), random_state=seed)
    raise ValueError(f"unknown clustering algorithm {kind!r}")


def centroids_of(Z: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    return {int(c): Z[labels == c].mean(axis=0) for c in sorted(set(labels)) if c >= 0}


def run_clustering(kind: str, label: str, data: Any, columns: Any, drop: Any, scale: bool, truth_column: str | None, k: int | None, params: dict[str, Any], seed: int,
                   output_path: str | None, model_path: str | None, max_rows: int | None = None, labels_override: np.ndarray | None = None, P: Prepared | None = None) -> dict[str, Any]:
    """Fit one algorithm and report the partition: sizes, noise, internal + external metrics, what distinguishes each cluster, and where the labelled file went."""
    P = P or prepare(data, columns, drop, scale, truth_column, max_rows, seed)
    model = make_model(kind, k, params, seed) if labels_override is None else None
    if labels_override is not None:
        labels = labels_override
    elif hasattr(model, "fit_predict"):
        labels = np.asarray(model.fit_predict(P.Z))
    else:
        labels = np.asarray(model.fit(P.Z).labels_)
    out: dict[str, Any] = {"algorithm": label, "rows": int(len(P.Z)), "encoded_features": int(P.Z.shape[1]), **partition_summary(P, labels, seed), "profile": describe_clusters(P, labels)}
    if hasattr(model, "inertia_"):
        out["inertia"] = round(float(model.inertia_), 3)
    if hasattr(model, "bic"):
        out["bic"] = round(float(model.bic(P.Z)), 2)
    if hasattr(model, "n_iter_"):
        out["iterations"] = int(model.n_iter_)
    table = P.frame.assign(cluster=labels)
    out["written_to"] = save_frame(table, output_path, f"clusters_{kind}.csv") if output_path else None
    cm = ClusterModel(kind, P.prep, P.columns, model, centroids_of(P.Z, labels), labels)
    out["model_path"] = save_model(cm, model_path)
    return out
