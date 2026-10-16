"""Judge a clustering: algorithms compared on the same data, stability, what separates the clusters, agreement with known groups, and assigning new rows."""

from __future__ import annotations

from typing import Literal

import joblib
import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.ml import sk, top_items
from ml_core.tooling import tool
from ml_core.unsup import prepare
from unsupervised.clustering.tools._common import (
    describe_clusters,
    external_metrics,
    make_model,
    partition_summary,
)


@tool
def compare_clusterings(data: str, n_clusters: int = 3, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, algorithms: list[str] | None = None, max_rows: int = 3000, seed: int = 42) -> dict:
    """Run several algorithms on the same prepared data and compare silhouette, Davies–Bouldin, cluster sizes, noise share and (with a truth column) ARI / NMI — the quickest way to see whether the structure is algorithm-independent.

    Args:
        n_clusters: k for the algorithms that need one.
        truth_column: Column of known groups kept aside to score the clusterings.
        algorithms: Any of kmeans, minibatch_kmeans, gmm, agglomerative, birch, spectral, dbscan, hdbscan, optics; defaults to six.
    """
    P = prepare(data, columns, drop, True, truth_column, max_rows, seed)
    from sklearn.neighbors import NearestNeighbors

    eps = float(np.sort(NearestNeighbors(n_neighbors=6).fit(P.Z).kneighbors(P.Z)[0][:, -1])[int(0.9 * len(P.Z))])
    rows = []
    for kind in algorithms or ["kmeans", "gmm", "agglomerative", "birch", "dbscan", "hdbscan"]:
        try:
            model = make_model(kind, n_clusters, {"eps": eps, "min_samples": 5, "min_cluster_size": max(10, len(P.Z) // 50)}, seed)
            labels = np.asarray(model.fit_predict(P.Z) if hasattr(model, "fit_predict") else model.fit(P.Z).labels_)
            s = partition_summary(P, labels, seed)
            row = {"algorithm": kind, "clusters": s["clusters"], "noise_share": s["noise_share"], **s["internal"]}
            if "external_vs_truth" in s:
                row.update({k: s["external_vs_truth"][k] for k in ("adjusted_rand_index", "normalized_mutual_info")})
            rows.append(row)
        except Exception as error:
            rows.append({"algorithm": kind, "error": str(error)[:100]})
    key = "adjusted_rand_index" if truth_column else "silhouette"
    rows.sort(key=lambda r: -(r.get(key) if r.get(key) is not None else -2))
    return {"ranked_by": key, "dbscan_eps_used": round(eps, 3), "algorithms": rows, "best": rows[0]["algorithm"],
            "note": "internal scores favour compact round clusters, so they flatter k-means-like methods; trust external agreement when truth exists" if not truth_column else "ARI/NMI use the truth column, which was never used to fit"}


@tool
def cluster_stability(data: str, n_clusters: int = 3, algorithm: Literal["kmeans", "gmm", "agglomerative", "birch"] = "kmeans", columns: list[str] | None = None, drop: list[str] | None = None, n_resamples: int = 15, sample_share: float = 0.8, max_rows: int = 4000, seed: int = 42) -> dict:
    """Bootstrap stability: re-cluster random subsamples and compare each with the full-data partition (adjusted Rand index) — and report which clusters are stable and which dissolve. An unstable partition is an artefact of the particular rows, whatever its silhouette.

    Args:
        n_clusters: k.
        algorithm: Algorithm to test.
        n_resamples: Number of subsamples.
        sample_share: Share of rows in each subsample.
    """
    from sklearn.metrics import adjusted_rand_score

    P = prepare(data, columns, drop, True, None, max_rows, seed)
    full = np.asarray(make_model(algorithm, n_clusters, {}, seed).fit_predict(P.Z))
    rng = np.random.default_rng(seed)
    aris, per_cluster = [], {int(c): [] for c in set(full)}
    for _ in range(n_resamples):
        idx = rng.choice(len(P.Z), int(len(P.Z) * sample_share), replace=False)
        sub = np.asarray(make_model(algorithm, n_clusters, {}, int(rng.integers(1 << 30))).fit_predict(P.Z[idx]))
        aris.append(adjusted_rand_score(full[idx], sub))
        for c in per_cluster:  # best Jaccard match of the full-data cluster in the subsample
            a = full[idx] == c
            per_cluster[c].append(max((a & (sub == s)).sum() / max((a | (sub == s)).sum(), 1) for s in set(sub)))
    return {"algorithm": algorithm, "n_clusters": n_clusters, "mean_ari": round(float(np.mean(aris)), 4), "std_ari": round(float(np.std(aris)), 4),
            "cluster_jaccard_stability": {c: round(float(np.mean(v)), 3) for c, v in per_cluster.items()},
            "verdict": "stable" if np.mean(aris) > 0.8 else "moderately stable" if np.mean(aris) > 0.5 else "unstable: the partition changes with the sample — fewer clusters or a different method"}


@tool
def profile_clusters(data: str, cluster_column: str, columns: list[str] | None = None, drop: list[str] | None = None, k: int = 6, output_path: str | None = None) -> dict:
    """Describe an existing clustering: for every cluster its size, the numeric columns that differ most from the overall table (in standard deviations) and the over-represented categories — the step that turns cluster numbers into names a person can use.

    Args:
        cluster_column: Column holding each row's cluster id (e.g. the `cluster` column written by a cluster_* tool).
        k: How many distinguishing features to list per cluster.
    """
    frame = load_frame(data)
    require_columns(frame, [cluster_column])
    labels = pd.factorize(frame[cluster_column])[0] if not pd.api.types.is_numeric_dtype(frame[cluster_column]) else frame[cluster_column].to_numpy(int)
    P = prepare(frame, columns, list(drop or []) + [cluster_column], True, None)
    prof = describe_clusters(P, labels, k)
    table = pd.DataFrame([{"cluster": p["cluster"], "size": p["size"], **p["distinguishing_features"]} for p in prof])
    return {"clusters": len(prof), "profile": prof, "written_to": save_frame(table, output_path, "cluster_profiles.csv") if output_path else None}


@tool
def explain_clusters_with_tree(data: str, cluster_column: str, columns: list[str] | None = None, drop: list[str] | None = None, max_depth: int = 3, seed: int = 42) -> dict:
    """Fit a shallow decision tree that predicts the cluster from the original columns and print its rules — a readable if/then description of each cluster ('income > 50k and age < 35'), plus the accuracy of those rules and a permutation importance of the columns.

    Args:
        cluster_column: Column holding each row's cluster id.
        max_depth: Depth of the explaining tree; small keeps the rules readable.
    """
    sk()
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier, export_text

    frame = load_frame(data)
    require_columns(frame, [cluster_column])
    y = frame[cluster_column].astype(str)
    P = prepare(frame, columns, list(drop or []) + [cluster_column], False, None)
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed).fit(P.Z, y)
    imp = dict(zip(P.names, tree.feature_importances_))
    return {"rule_accuracy_cv": round(float(cross_val_score(DecisionTreeClassifier(max_depth=max_depth, random_state=seed), P.Z, y, cv=5).mean()), 4), "rules": export_text(tree, feature_names=P.names).splitlines()[:60],
            "feature_importance": top_items(imp, 10), "reading": "high rule accuracy means the clusters are describable with a few simple conditions; low accuracy means they are defined by combinations of many columns"}


@tool
def validate_against_known_groups(data: str, cluster_column: str, truth_column: str) -> dict:
    """Compare a clustering you already have with known groups (a label, a segment, a ground-truth class): adjusted Rand index, normalised mutual information, homogeneity, completeness, V-measure, purity, and the contingency table — how much of the known structure the clusters recover.

    Args:
        cluster_column: Column holding each row's cluster id.
        truth_column: Column holding the known group.
    """
    frame = load_frame(data)
    require_columns(frame, [cluster_column, truth_column])
    labels, truth = pd.factorize(frame[cluster_column].astype(str))[0], pd.factorize(frame[truth_column].astype(str))[0]
    table = pd.crosstab(frame[cluster_column], frame[truth_column])
    return {"metrics": external_metrics(truth, labels), "contingency": table.reset_index().to_dict("records"),
            "reading": "ARI near 0 is no better than random; completeness low means a known group is split across clusters, homogeneity low means a cluster mixes groups"}


@tool
def find_cluster_outliers(data: str, cluster_column: str, columns: list[str] | None = None, drop: list[str] | None = None, z_threshold: float = 3.0, k: int = 20, output_path: str | None = None) -> dict:
    """Rows that sit far from their own cluster's centre (distance in standard deviations of that cluster's distance distribution) — borderline or mis-assigned rows, and a cheap anomaly list that respects cluster structure.

    Args:
        cluster_column: Column holding each row's cluster id.
        z_threshold: Distance z-score above which a row is flagged.
        k: How many rows to list.
    """
    frame = load_frame(data)
    require_columns(frame, [cluster_column])
    P = prepare(frame, columns, list(drop or []) + [cluster_column], True, None)
    labels = frame[cluster_column].to_numpy()
    dist = np.zeros(len(frame))
    for c in pd.unique(labels):
        m = labels == c
        d = np.linalg.norm(P.Z[m] - P.Z[m].mean(axis=0), axis=1)
        dist[m] = (d - d.mean()) / (d.std() or 1)
    out = frame.assign(distance_z=dist)
    flagged = out[out["distance_z"] >= z_threshold].sort_values("distance_z", ascending=False)
    return {"flagged": int(len(flagged)), "share": round(float(len(flagged) / len(frame)), 4), "worst": flagged.head(k), "written_to": save_frame(flagged, output_path, "cluster_outliers.csv") if output_path else None}


@tool
def assign_new_points(model_path: str, data: str, id_column: str | None = None, output_path: str | None = None) -> dict:
    """Assign new rows to the clusters of a saved clustering — through the algorithm's own predict when it has one (k-means, mixtures, BIRCH), otherwise to the nearest cluster centroid — with the distance to that centroid so far-away rows can be flagged as not belonging.

    Args:
        model_path: Path of the saved clustering (joblib).
        id_column: Column to carry through to the output.
    """
    model = joblib.load(model_path)
    frame = load_frame(data)
    require_columns(frame, model.columns)
    Z = model.prep.transform(frame[model.columns])
    Z = np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z, float)
    labels, dist = model.assign(Z)
    out = pd.DataFrame({"cluster": labels, "distance_to_centroid": dist}, index=frame.index)
    if id_column and id_column in frame.columns:
        out.insert(0, id_column, frame[id_column])
    return {"rows": len(out), "algorithm": model.kind, "assignment_counts": out["cluster"].value_counts().sort_index().to_dict(), "mean_distance": round(float(dist.mean()), 4),
            "rows_far_from_any_centroid": int((dist > 3 * np.median(dist)).sum()), "written_to": save_frame(out, output_path, "cluster_assignments.csv"), "sample": out.head(5)}
