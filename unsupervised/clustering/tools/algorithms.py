"""Fit one clustering algorithm and describe the partition it finds: centroid-, model-, hierarchical-, density-, graph- and exemplar-based methods, plus k-medoids, k-modes and Gower-distance clustering for mixed data."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ml_core.tooling import tool
from ml_core.unsup import prepare
from unsupervised.clustering.tools._common import run_clustering


def _go(kind, label, data, columns, drop, scale, truth_column, k, params, seed, output_path, model_path, max_rows=None):
    return run_clustering(kind, label, data, columns, drop, scale, truth_column, k, params, seed, output_path, model_path, max_rows)


@tool
def cluster_kmeans(data: str, n_clusters: int = 3, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, n_init: int = 10, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """K-means: partition the rows into k compact, roughly spherical, similarly sized groups around centroids. The default first method — fast, interpretable, and a baseline for everything else; poor on elongated, nested or very unequal clusters.

    Args:
        truth_column: Column of known groups kept aside to *score* the clustering (never used for fitting).
        n_init: Random restarts; the best is kept.
        output_path: Where to write the table with a `cluster` column.
    """
    return _go("kmeans", f"kmeans(k={n_clusters})", data, columns, drop, scale, truth_column, n_clusters, {"n_init": n_init}, seed, output_path, model_path)


@tool
def cluster_minibatch_kmeans(data: str, n_clusters: int = 3, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, batch_size: int = 1024, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Mini-batch k-means: k-means fitted on small random batches — an order of magnitude faster on millions of rows, for a slightly worse objective.

    Args:
        truth_column: Column of known groups kept aside to score the clustering.
        batch_size: Rows per mini-batch.
    """
    return _go("minibatch_kmeans", f"minibatch_kmeans(k={n_clusters})", data, columns, drop, scale, truth_column, n_clusters, {"batch_size": batch_size}, seed, output_path, model_path)


@tool
def cluster_gaussian_mixture(data: str, n_clusters: int = 3, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, covariance_type: Literal["full", "tied", "diag", "spherical"] = "full", seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Gaussian mixture model: each cluster is an ellipsoid with its own centre, shape and size, and rows get *soft* membership probabilities (written as `max_probability`-style certainty in the summary). Handles overlapping, elongated and unequal clusters that k-means cannot.

    Args:
        truth_column: Column of known groups kept aside to score the clustering.
        covariance_type: full (any ellipsoid), tied (same shape), diag (axis-aligned) or spherical.
    """
    return _go("gmm", f"gaussian_mixture(k={n_clusters},{covariance_type})", data, columns, drop, scale, truth_column, n_clusters, {"covariance_type": covariance_type}, seed, output_path, model_path)


@tool
def cluster_bayesian_mixture(data: str, max_clusters: int = 10, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, concentration: float = 0.01, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Dirichlet-process Gaussian mixture: give an upper bound on the number of clusters and the model switches off the components it does not need, so k is *learned* — a principled alternative to scanning k when the number of groups is unknown.

    Args:
        max_clusters: Upper bound on the components.
        truth_column: Column of known groups kept aside to score the clustering.
        concentration: Lower values favour fewer clusters.
    """
    return _go("bayesian_gmm", f"bayesian_mixture(max={max_clusters})", data, columns, drop, scale, truth_column, max_clusters, {"concentration": concentration}, seed, output_path, model_path)


@tool
def cluster_agglomerative(data: str, n_clusters: int = 3, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, linkage: Literal["ward", "average", "complete", "single"] = "ward", max_rows: int = 6000, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Agglomerative (hierarchical) clustering: start with every row alone and repeatedly merge the closest groups. Ward gives compact clusters like k-means; single linkage follows chains and elongated shapes; average and complete sit between. Quadratic in rows, so large tables are subsampled.

    Args:
        truth_column: Column of known groups kept aside to score the clustering.
        linkage: ward, average, complete or single.
    """
    return _go("agglomerative", f"agglomerative(k={n_clusters},{linkage})", data, columns, drop, scale, truth_column, n_clusters, {"linkage": linkage}, seed, output_path, model_path, max_rows)


@tool
def cluster_dbscan(data: str, eps: float, min_samples: int = 5, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """DBSCAN: clusters are dense regions separated by sparse ones; rows in no dense region are labelled noise (cluster −1). Finds arbitrary shapes and needs no k, but one density for all clusters — pick eps with `estimate_dbscan_eps`.

    Args:
        eps: Neighbourhood radius (in standardised units).
        min_samples: Rows needed within eps for a core point.
        truth_column: Column of known groups kept aside to score the clustering.
    """
    return _go("dbscan", f"dbscan(eps={eps},min_samples={min_samples})", data, columns, drop, scale, truth_column, None, {"eps": eps, "min_samples": min_samples}, seed, output_path, model_path)


@tool
def cluster_hdbscan(data: str, min_cluster_size: int = 10, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, min_samples: int | None = None, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """HDBSCAN: DBSCAN over *all* radii, keeping the clusters that persist — finds clusters of different densities with no eps to tune, and marks the rest as noise. The best general-purpose density method.

    Args:
        min_cluster_size: Smallest group worth calling a cluster.
        min_samples: How conservative the noise decision is; defaults to min_cluster_size.
        truth_column: Column of known groups kept aside to score the clustering.
    """
    return _go("hdbscan", f"hdbscan(min_cluster_size={min_cluster_size})", data, columns, drop, scale, truth_column, None, {"min_cluster_size": min_cluster_size, "min_samples": min_samples}, seed, output_path, model_path)


@tool
def cluster_optics(data: str, min_samples: int = 10, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, xi: float = 0.05, max_rows: int = 5000, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """OPTICS: orders rows by reachability distance and extracts clusters of varying density from the resulting profile — DBSCAN's flexibility without a single radius, with the reachability plot as a diagnostic.

    Args:
        min_samples: Neighbours defining a dense point.
        xi: Minimum steepness of the reachability drop that starts a cluster.
        truth_column: Column of known groups kept aside to score the clustering.
    """
    return _go("optics", f"optics(min_samples={min_samples},xi={xi})", data, columns, drop, scale, truth_column, None, {"min_samples": min_samples, "xi": xi}, seed, output_path, model_path, max_rows)


@tool
def cluster_birch(data: str, n_clusters: int = 3, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, threshold: float = 0.5, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """BIRCH: builds a compact tree summary of the data in one pass and clusters the summaries — memory-light and incremental, for very large tables.

    Args:
        truth_column: Column of known groups kept aside to score the clustering.
        threshold: Radius of a leaf subcluster; smaller keeps more detail.
    """
    return _go("birch", f"birch(k={n_clusters})", data, columns, drop, scale, truth_column, n_clusters, {"threshold": threshold}, seed, output_path, model_path)


@tool
def cluster_spectral(data: str, n_clusters: int = 3, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, n_neighbors: int = 10, max_rows: int = 4000, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Spectral clustering: cluster the eigenvectors of a nearest-neighbour similarity graph — separates non-convex shapes (rings, moons, intertwined spirals) that centroid methods split wrongly. Heavy on large data, so it is subsampled.

    Args:
        truth_column: Column of known groups kept aside to score the clustering.
        n_neighbors: Neighbours in the similarity graph.
    """
    return _go("spectral", f"spectral(k={n_clusters})", data, columns, drop, scale, truth_column, n_clusters, {"n_neighbors": n_neighbors}, seed, output_path, model_path, max_rows)


@tool
def cluster_affinity_propagation(data: str, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, damping: float = 0.7, max_rows: int = 2500, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Affinity propagation: rows exchange 'responsibility' messages and elect exemplar rows; the number of clusters emerges from the data. Gives real rows as cluster representatives, but is quadratic in memory, so keep it to a few thousand rows.

    Args:
        truth_column: Column of known groups kept aside to score the clustering.
        damping: Message damping between 0.5 and 1; higher is more stable.
    """
    return _go("affinity_propagation", "affinity_propagation", data, columns, drop, scale, truth_column, None, {"damping": damping}, seed, output_path, model_path, max_rows)


@tool
def cluster_kmedoids(data: str, n_clusters: int = 3, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, metric: Literal["euclidean", "manhattan", "cosine"] = "manhattan", max_rows: int = 3000, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """K-medoids (alternating PAM): like k-means but each centre is an actual row, and any distance works — more robust to outliers, and the medoids are real examples to show people. Quadratic in rows, so subsampled.

    Args:
        truth_column: Column of known groups kept aside to score the clustering.
        metric: Distance used (manhattan is robust to outliers).
    """
    from sklearn.metrics import pairwise_distances

    P = prepare(data, columns, drop, scale, truth_column, max_rows, seed)
    D = pairwise_distances(P.Z, metric=metric)
    rng = np.random.default_rng(seed)
    medoids = rng.choice(len(P.Z), n_clusters, replace=False)
    for _ in range(50):
        labels = D[:, medoids].argmin(axis=1)
        new = np.array([np.flatnonzero(labels == k)[D[np.ix_(labels == k, labels == k)].sum(axis=0).argmin()] if (labels == k).any() else medoids[k] for k in range(n_clusters)])
        if set(new) == set(medoids):
            break
        medoids = new
    labels = D[:, medoids].argmin(axis=1)
    out = run_clustering("kmedoids", f"kmedoids(k={n_clusters},{metric})", data, columns, drop, scale, truth_column, n_clusters, {}, seed, output_path, model_path, max_rows, labels_override=labels, P=P)
    out["medoid_rows"] = [int(i) for i in medoids]
    out["total_cost"] = round(float(D[np.arange(len(P.Z)), medoids[labels]].sum()), 3)
    return out


@tool
def cluster_kmodes_categorical(data: str, n_clusters: int = 3, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, n_init: int = 5, max_iter: int = 30, seed: int = 42, output_path: str | None = None) -> dict:
    """K-modes for categorical data: centres are the most common value of each column and distance is the number of mismatching columns (numeric columns are cut into quantile bins first). The right k-means analogue when the data is codes, labels and flags rather than measurements.

    Args:
        truth_column: Column of known groups kept aside to score the clustering.
        n_init: Random restarts; the lowest-cost one is kept.
        max_iter: Reassignment iterations per restart.
    """
    from ml_core.data import as_list, load_frame, save_frame
    from unsupervised.clustering.tools._common import external_metrics

    frame = load_frame(data)
    skip = set(as_list(drop) or []) | ({truth_column} if truth_column else set())
    cols = [c for c in (as_list(columns) or frame.columns) if c not in skip]
    X = frame[cols].copy()
    for c in cols:
        if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 10:
            X[c] = pd.qcut(X[c], 5, duplicates="drop").astype(str)
    factored = [pd.factorize(X[c].astype(str)) for c in cols]
    codes, uniques = np.column_stack([f[0] for f in factored]), [f[1] for f in factored]
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        modes = codes[rng.choice(len(codes), n_clusters, replace=False)].copy()
        for _ in range(max_iter):
            cost = (codes[:, None, :] != modes[None, :, :]).sum(axis=2)
            labels = cost.argmin(axis=1)
            new = np.array([[np.bincount(codes[labels == k, j]).argmax() if (labels == k).any() else modes[k, j] for j in range(codes.shape[1])] for k in range(n_clusters)])
            if (new == modes).all():
                break
            modes = new
        total = int((codes != modes[labels]).sum())
        if best is None or total < best[0]:
            best = (total, labels, modes)
    total, labels, modes = best
    sizes = pd.Series(labels).value_counts().sort_index()
    out = {"algorithm": f"kmodes(k={n_clusters})", "rows": len(frame), "total_mismatches": total, "cluster_sizes": {int(k): int(v) for k, v in sizes.items()},
           "cluster_modes": [{c: str(uniques[j][modes[k, j]]) for j, c in enumerate(cols)} for k in range(n_clusters)]}
    if truth_column:
        out["external_vs_truth"] = external_metrics(pd.factorize(frame[truth_column].astype(str))[0], labels)
    out["written_to"] = save_frame(frame.assign(cluster=labels), output_path, "clusters_kmodes.csv") if output_path else None
    return out


@tool
def cluster_gower_hierarchical(data: str, n_clusters: int = 3, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, linkage: Literal["average", "complete", "single"] = "average", max_rows: int = 3000, seed: int = 42, output_path: str | None = None) -> dict:
    """Hierarchical clustering on Gower distance — numeric columns contribute range-scaled differences, categorical columns contribute 0/1 mismatches, missing values are skipped — the standard way to cluster genuinely mixed data without forcing one-hot encodings into Euclidean space.

    Args:
        truth_column: Column of known groups kept aside to score the clustering.
        linkage: average, complete or single.
        max_rows: Gower is quadratic; larger tables are subsampled.
    """
    from sklearn.cluster import AgglomerativeClustering

    from ml_core.data import as_list, load_frame, save_frame
    from unsupervised.clustering.tools._common import external_metrics

    frame = load_frame(data)
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=seed).reset_index(drop=True)
    skip = set(as_list(drop) or []) | ({truth_column} if truth_column else set())
    cols = [c for c in (as_list(columns) or frame.columns) if c not in skip]
    n = len(frame)
    D, w = np.zeros((n, n)), np.zeros((n, n))
    for c in cols:
        s = frame[c]
        if pd.api.types.is_numeric_dtype(s):
            v = s.to_numpy(float)
            rng_ = np.nanmax(v) - np.nanmin(v)
            d = np.abs(v[:, None] - v[None, :]) / (rng_ if rng_ > 0 else 1)
        else:
            v = s.astype(str).where(s.notna(), None).to_numpy(object)
            d = (v[:, None] != v[None, :]).astype(float)
        ok = (~pd.isna(s).to_numpy())[:, None] & (~pd.isna(s).to_numpy())[None, :]
        D += np.where(ok, np.nan_to_num(d), 0)
        w += ok
    D = D / np.maximum(w, 1)
    labels = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage=linkage).fit_predict(D)
    sizes = pd.Series(labels).value_counts().sort_index()
    out = {"algorithm": f"gower_hierarchical(k={n_clusters},{linkage})", "rows": n, "cluster_sizes": {int(k): int(v) for k, v in sizes.items()}}
    from sklearn.metrics import silhouette_score

    out["silhouette_on_gower_distance"] = round(float(silhouette_score(D, labels, metric="precomputed")), 4) if len(set(labels)) > 1 else None
    if truth_column:
        out["external_vs_truth"] = external_metrics(pd.factorize(frame[truth_column].astype(str))[0], labels)
    out["written_to"] = save_frame(frame.assign(cluster=labels), output_path, "clusters_gower.csv") if output_path else None
    return out
