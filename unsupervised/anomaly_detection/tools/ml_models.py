"""Model-based detectors: isolation forest, local outlier factor, one-class SVM, elliptic envelope, k-NN distance, PCA reconstruction error, an autoencoder, k-means distance, and DBSCAN noise."""

from __future__ import annotations

import numpy as np

from ml_core.tooling import tool
from ml_core.unsup import prepare
from unsupervised.anomaly_detection.tools._common import run_detector


def _d(kind, label, data, columns, drop, truth_column, contamination, params, seed, output_path, model_path, id_column, top, max_rows=None):
    return run_detector(kind, label, data, columns, drop, True, truth_column, contamination, params, seed, output_path, model_path, id_column, top, max_rows)


@tool
def detect_isolation_forest(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, contamination: float = 0.05, n_estimators: int = 300, id_column: str | None = None, top: int = 15, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Isolation forest: anomalies are few and different, so random splits isolate them in fewer steps than normal rows — the score is the (inverted) average path length. No distance, no distribution assumed; scales to many rows and columns. The default first detector.

    Args:
        truth_column: Column of known anomalies, used only to score the detector.
        n_estimators: Trees.
        id_column: Column to show beside the flagged rows.
        top: Rows listed in the reply.
    """
    return _d("isolation_forest", "isolation_forest", data, columns, drop, truth_column, contamination, {"n_estimators": n_estimators}, seed, output_path, model_path, id_column, top)


@tool
def detect_local_outlier_factor(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, contamination: float = 0.05, n_neighbors: int = 20, id_column: str | None = None, top: int = 15, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Local outlier factor: compares each row's local density with its neighbours' — a row is anomalous if it is much sparser than the rows around it. Catches anomalies near a *dense* cluster that a global rule would call normal; sensitive to the neighbourhood size.

    Args:
        truth_column: Column of known anomalies, used only to score the detector.
        n_neighbors: Neighbourhood size.
        id_column: Column to show beside the flagged rows.
        top: Rows listed in the reply.
    """
    return _d("lof", f"lof(k={n_neighbors})", data, columns, drop, truth_column, contamination, {"n_neighbors": n_neighbors}, seed, output_path, model_path, id_column, top, 10000)


@tool
def detect_one_class_svm(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, contamination: float = 0.05, nu: float = 0.05, id_column: str | None = None, top: int = 15, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """One-class SVM: learns a boundary around the bulk of the data in a kernel feature space; rows outside it are anomalies. Flexible shapes, but slow beyond ~10,000 rows and sensitive to the kernel width and to scaling.

    Args:
        truth_column: Column of known anomalies, used only to score the detector.
        nu: Upper bound on the share of training rows allowed outside the boundary.
        id_column: Column to show beside the flagged rows.
        top: Rows listed in the reply.
    """
    return _d("one_class_svm", f"one_class_svm(nu={nu})", data, columns, drop, truth_column, contamination, {"nu": nu}, seed, output_path, model_path, id_column, top, 6000)


@tool
def detect_elliptic_envelope(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, contamination: float = 0.05, id_column: str | None = None, top: int = 15, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Elliptic envelope: fits a robust Gaussian to the central bulk of the data and flags what falls outside the ellipse — a parametric detector for roughly unimodal, normal data.

    Args:
        truth_column: Column of known anomalies, used only to score the detector.
        id_column: Column to show beside the flagged rows.
        top: Rows listed in the reply.
    """
    return _d("elliptic_envelope", "elliptic_envelope", data, columns, drop, truth_column, contamination, {"contamination": contamination}, seed, output_path, model_path, id_column, top, 8000)


@tool
def detect_knn_distance(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, contamination: float = 0.05, n_neighbors: int = 10, aggregate: str = "mean", id_column: str | None = None, top: int = 15, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """k-nearest-neighbour distance: score = the mean (or the largest) distance to the k nearest rows. A simple, hard-to-fool global detector; the baseline that fancier distance methods must beat.

    Args:
        truth_column: Column of known anomalies, used only to score the detector.
        n_neighbors: k.
        aggregate: mean or kth (distance to the k-th neighbour).
        id_column: Column to show beside the flagged rows.
        top: Rows listed in the reply.
    """
    return _d("knn_distance", f"knn_distance(k={n_neighbors})", data, columns, drop, truth_column, contamination, {"n_neighbors": n_neighbors, "aggregate": aggregate}, seed, output_path, model_path, id_column, top, 20000)


@tool
def detect_pca_reconstruction(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, contamination: float = 0.05, variance: float = 0.9, id_column: str | None = None, top: int = 15, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """PCA reconstruction error: keep the components that explain most of the variance and score rows by how badly the compressed version reproduces them. Normal rows follow the data's main correlations and reconstruct well; anomalies do not. Strong for correlated, high-dimensional data.

    Args:
        truth_column: Column of known anomalies, used only to score the detector.
        variance: Share of variance the kept components must explain.
        id_column: Column to show beside the flagged rows.
        top: Rows listed in the reply.
    """
    return _d("pca_reconstruction", f"pca_reconstruction({variance})", data, columns, drop, truth_column, contamination, {"variance": variance}, seed, output_path, model_path, id_column, top)


@tool
def detect_autoencoder(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, contamination: float = 0.05, hidden_layers: str | None = None, max_iter: int = 600, id_column: str | None = None, top: int = 15, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Autoencoder reconstruction error: a small neural network is trained to reproduce its own input through a narrow middle layer; rows it cannot reproduce are anomalies. The non-linear counterpart of PCA reconstruction, for curved normal structure.

    Args:
        truth_column: Column of known anomalies, used only to score the detector.
        hidden_layers: Comma-separated layer widths with a bottleneck, e.g. "8,3,8"; chosen from the data when omitted.
        max_iter: Training epochs.
        id_column: Column to show beside the flagged rows.
        top: Rows listed in the reply.
    """
    hidden = tuple(int(p) for p in hidden_layers.replace(" ", "").split(",") if p) if hidden_layers else None
    return _d("autoencoder", "autoencoder", data, columns, drop, truth_column, contamination, {"hidden": hidden, "max_iter": max_iter}, seed, output_path, model_path, id_column, top, 20000)


@tool
def detect_kmeans_distance(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, contamination: float = 0.05, n_clusters: int = 5, id_column: str | None = None, top: int = 15, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Distance to the nearest k-means centre, divided by that cluster's own typical distance — anomalous means unusually far from *its* cluster, so tight and loose clusters are judged fairly. Good when the normal data has several distinct groups.

    Args:
        truth_column: Column of known anomalies, used only to score the detector.
        n_clusters: Number of normal groups.
        id_column: Column to show beside the flagged rows.
        top: Rows listed in the reply.
    """
    return _d("kmeans_distance", f"kmeans_distance(k={n_clusters})", data, columns, drop, truth_column, contamination, {"n_clusters": n_clusters}, seed, output_path, model_path, id_column, top)


@tool
def detect_dbscan_noise(data: str, eps: float | None = None, min_samples: int = 5, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, id_column: str | None = None, top: int = 15, seed: int = 42, output_path: str | None = None) -> dict:
    """DBSCAN noise points: rows that belong to no dense region are the anomalies. The number of anomalies is decided by the radius, not by a contamination setting — eps comes from the k-distance knee when omitted.

    Args:
        eps: Neighbourhood radius in standardised units.
        min_samples: Rows needed within eps for a dense core.
        truth_column: Column of known anomalies, used only to score the detector.
        id_column: Column to show beside the flagged rows.
        top: Rows listed in the reply.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    from ml_core.data import save_frame
    from unsupervised.anomaly_detection.tools._common import evaluate_scores, explain_rows

    P = prepare(data, columns, drop, True, truth_column, 20000, seed)
    d = np.sort(NearestNeighbors(n_neighbors=min_samples + 1).fit(P.Z).kneighbors(P.Z)[0][:, -1])
    x = np.arange(len(d), dtype=float)
    knee = float(d[int(np.argmax(np.abs((d[-1] - d[0]) * x - (len(d) - 1) * (d - d[0]))))])
    radius = eps or knee
    labels = DBSCAN(eps=radius, min_samples=min_samples).fit_predict(P.Z)
    noise = labels == -1
    score = NearestNeighbors(n_neighbors=min_samples + 1).fit(P.Z).kneighbors(P.Z)[0][:, -1] / radius
    order = np.argsort(-score)[:top]
    worst = P.frame.iloc[order].copy()
    worst.insert(0, "row", order)
    worst["isolation_ratio"] = score[order]
    worst["why"] = explain_rows(P, order)
    out = {"detector": f"dbscan_noise(eps={radius:.3g},min_samples={min_samples})", "eps_from_knee": eps is None, "clusters_found": int(len(set(labels) - {-1})), "flagged": int(noise.sum()), "share": round(float(noise.mean()), 4), "top_anomalies": worst}
    if P.truth is not None and set(np.unique(P.truth)) == {0, 1}:
        out["evaluation_against_labels"] = evaluate_scores(P.truth, score, float(noise.mean()) or 0.01)
    out["written_to"] = save_frame(P.frame.assign(is_flagged=noise.astype(int), isolation_ratio=score), output_path, "anomalies_dbscan.csv") if output_path else None
    return out
