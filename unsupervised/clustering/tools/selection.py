"""Choosing the number of clusters (or a density radius): elbow, silhouette, Calinski–Harabasz / Davies–Bouldin, gap statistic, mixture BIC, stability, DBSCAN eps — and a consensus across them."""

from __future__ import annotations

import numpy as np

from ml_core.tooling import tool
from ml_core.unsup import prepare
from unsupervised.clustering.tools._common import internal_metrics


def _range(k_min: int, k_max: int, n: int) -> range:
    return range(max(2, k_min), min(k_max, max(3, n // 5)) + 1)


def _kmeans(Z: np.ndarray, k: int, seed: int):
    from sklearn.cluster import KMeans

    return KMeans(k, n_init=5, random_state=seed).fit(Z)


@tool
def find_k_elbow(data: str, columns: list[str] | None = None, drop: list[str] | None = None, k_min: int = 2, k_max: int = 10, max_rows: int = 5000, seed: int = 42) -> dict:
    """Elbow method: k-means inertia (within-cluster sum of squares) for each k, and the knee — the k furthest from the straight line joining the first and last points, where adding clusters stops paying. A heuristic; look at the curve as well as the pick.

    Args:
        k_min: Smallest k.
        k_max: Largest k.
    """
    P = prepare(data, columns, drop, True, None, max_rows, seed)
    ks = list(_range(k_min, k_max, len(P.Z)))
    inertia = np.array([_kmeans(P.Z, k, seed).inertia_ for k in ks])
    x, y = np.array(ks, float), (inertia - inertia.min()) / max(np.ptp(inertia), 1e-9)
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    d = np.abs(dy * (x - x[0]) - dx * (y - y[0])) / np.hypot(dx, dy)  # distance of each point from the chord joining the ends
    return {"k": ks, "inertia": inertia.round(2).tolist(), "relative_drop_per_step": (-np.diff(inertia) / inertia[:-1]).round(4).tolist(), "suggested_k": int(ks[int(np.argmax(d))])}


@tool
def find_k_silhouette(data: str, columns: list[str] | None = None, drop: list[str] | None = None, k_min: int = 2, k_max: int = 10, max_rows: int = 4000, seed: int = 42) -> dict:
    """Silhouette sweep: for each k the mean silhouette (how much closer a row is to its own cluster than to the next one, from −1 to 1). The best k maximises it; a best value under ~0.25 means no k separates the data well.

    Args:
        k_min: Smallest k.
        k_max: Largest k.
    """
    P = prepare(data, columns, drop, True, None, max_rows, seed)
    ks = list(_range(k_min, k_max, len(P.Z)))
    sil = [internal_metrics(P.Z, _kmeans(P.Z, k, seed).labels_, seed)["silhouette"] for k in ks]
    best = ks[int(np.argmax(sil))]
    return {"k": ks, "silhouette": sil, "suggested_k": int(best), "best_silhouette": max(sil), "reading": "above 0.5 strong, 0.25–0.5 reasonable, below 0.25 weak or artificial structure"}


@tool
def find_k_calinski_davies(data: str, columns: list[str] | None = None, drop: list[str] | None = None, k_min: int = 2, k_max: int = 10, max_rows: int = 5000, seed: int = 42) -> dict:
    """Calinski–Harabasz (higher = better separated, denser clusters) and Davies–Bouldin (lower = better) for each k — two cheap variance-ratio indices that often agree with the silhouette and break ties when it does not.

    Args:
        k_min: Smallest k.
        k_max: Largest k.
    """
    P = prepare(data, columns, drop, True, None, max_rows, seed)
    ks = list(_range(k_min, k_max, len(P.Z)))
    rows = [internal_metrics(P.Z, _kmeans(P.Z, k, seed).labels_, seed) for k in ks]
    ch, db = [r["calinski_harabasz"] for r in rows], [r["davies_bouldin"] for r in rows]
    return {"k": ks, "calinski_harabasz": ch, "davies_bouldin": db, "best_k_by_calinski_harabasz": int(ks[int(np.argmax(ch))]), "best_k_by_davies_bouldin": int(ks[int(np.argmin(db))])}


@tool
def find_k_gap_statistic(data: str, columns: list[str] | None = None, drop: list[str] | None = None, k_min: int = 1, k_max: int = 8, n_references: int = 10, max_rows: int = 3000, seed: int = 42) -> dict:
    """Gap statistic (Tibshirani et al.): compares the log within-cluster dispersion of the data with that of uniform reference data in the same bounding box; chooses the smallest k whose gap is within one standard error of the next. Unlike the elbow it can answer k = 1 (no clusters).

    Args:
        k_min: Smallest k (1 allowed).
        k_max: Largest k.
        n_references: Uniform reference datasets per k.
    """
    P = prepare(data, columns, drop, True, None, max_rows, seed)
    rng = np.random.default_rng(seed)
    lo, hi = P.Z.min(axis=0), P.Z.max(axis=0)
    ks = list(range(max(1, k_min), min(k_max, max(3, len(P.Z) // 5)) + 1))
    gaps, sds = [], []
    for k in ks:
        w = np.log(_kmeans(P.Z, k, seed).inertia_)
        ref = np.array([np.log(_kmeans(rng.uniform(lo, hi, size=P.Z.shape), k, seed).inertia_) for _ in range(n_references)])
        gaps.append(ref.mean() - w)
        sds.append(ref.std() * np.sqrt(1 + 1 / n_references))
    choice = next((ks[i] for i in range(len(ks) - 1) if gaps[i] >= gaps[i + 1] - sds[i + 1]), ks[-1])
    return {"k": ks, "gap": np.round(gaps, 4).tolist(), "gap_standard_error": np.round(sds, 4).tolist(), "suggested_k": int(choice),
            "reading": "suggested_k = 1 means the data is not distinguishable from uniform noise in the same box"}


@tool
def find_k_gmm_bic(data: str, columns: list[str] | None = None, drop: list[str] | None = None, k_min: int = 1, k_max: int = 10, covariance_type: str = "full", max_rows: int = 5000, seed: int = 42) -> dict:
    """Fit Gaussian mixtures for each number of components and compare BIC and AIC — a likelihood-based choice of k that also accounts for ellipsoidal, differently sized clusters, and penalises complexity.

    Args:
        k_min: Smallest number of components.
        k_max: Largest number of components.
        covariance_type: full, tied, diag or spherical.
    """
    from sklearn.mixture import GaussianMixture

    P = prepare(data, columns, drop, True, None, max_rows, seed)
    ks = list(range(max(1, k_min), min(k_max, max(3, len(P.Z) // 10)) + 1))
    fits = [GaussianMixture(k, covariance_type=covariance_type, n_init=3, random_state=seed).fit(P.Z) for k in ks]
    bic, aic = [float(f.bic(P.Z)) for f in fits], [float(f.aic(P.Z)) for f in fits]
    return {"k": ks, "bic": np.round(bic, 1).tolist(), "aic": np.round(aic, 1).tolist(), "best_k_by_bic": int(ks[int(np.argmin(bic))]), "best_k_by_aic": int(ks[int(np.argmin(aic))]), "covariance_type": covariance_type}


@tool
def find_k_stability(data: str, columns: list[str] | None = None, drop: list[str] | None = None, k_min: int = 2, k_max: int = 8, n_resamples: int = 10, max_rows: int = 3000, seed: int = 42) -> dict:
    """Stability-based choice of k: cluster pairs of random half-samples with k-means and measure how well the two partitions agree on the rows they share (adjusted Rand index). A k that is reproducible under resampling is more believable than one that merely scores well once.

    Args:
        k_min: Smallest k.
        k_max: Largest k.
        n_resamples: Half-sample pairs per k.
    """
    from sklearn.metrics import adjusted_rand_score

    P = prepare(data, columns, drop, True, None, max_rows, seed)
    rng = np.random.default_rng(seed)
    ks = list(_range(k_min, k_max, len(P.Z)))
    out = []
    for k in ks:
        scores = []
        for _ in range(n_resamples):
            a, b = rng.choice(len(P.Z), len(P.Z) // 2, replace=False), rng.choice(len(P.Z), len(P.Z) // 2, replace=False)
            ma, mb = _kmeans(P.Z[a], k, int(rng.integers(1 << 30))), _kmeans(P.Z[b], k, int(rng.integers(1 << 30)))
            common = np.arange(len(P.Z))
            scores.append(adjusted_rand_score(ma.predict(P.Z[common]), mb.predict(P.Z[common])))
        out.append({"k": k, "mean_ari": round(float(np.mean(scores)), 4), "std": round(float(np.std(scores)), 4)})
    best = max(out, key=lambda r: r["mean_ari"])
    return {"stability": out, "most_stable_k": best["k"], "reading": "ARI above 0.8 is reproducible; below 0.5 the partition changes with the sample"}


@tool
def estimate_dbscan_eps(data: str, columns: list[str] | None = None, drop: list[str] | None = None, min_samples: int = 5, max_rows: int = 5000, seed: int = 42) -> dict:
    """Choose DBSCAN's radius from the sorted k-distance curve (distance of each row to its `min_samples`-th neighbour): eps is read at the knee where distances start to climb steeply — points beyond it are isolated. Also reports eps at several percentiles.

    Args:
        min_samples: DBSCAN's min_samples (a common rule is 2 × the number of features).
    """
    from sklearn.neighbors import NearestNeighbors

    P = prepare(data, columns, drop, True, None, max_rows, seed)
    d = np.sort(NearestNeighbors(n_neighbors=min_samples + 1).fit(P.Z).kneighbors(P.Z)[0][:, -1])
    x = np.arange(len(d), dtype=float)
    chord = np.abs((d[-1] - d[0]) * x - (len(d) - 1) * (d - d[0]))
    knee = float(d[int(np.argmax(chord))])
    return {"min_samples": min_samples, "suggested_eps": round(knee, 4), "eps_at_percentile": {p: round(float(np.quantile(d, p / 100)), 4) for p in (50, 75, 90, 95, 99)},
            "reading": "too small an eps labels most rows noise; too large merges everything — check the noise share with cluster_dbscan"}


@tool
def suggest_number_of_clusters(data: str, columns: list[str] | None = None, drop: list[str] | None = None, k_min: int = 2, k_max: int = 8, max_rows: int = 3000, seed: int = 42) -> dict:
    """Run the elbow, silhouette, Calinski–Harabasz, Davies–Bouldin, gap-statistic and mixture-BIC criteria together and report the consensus — each criterion has failure modes, and k is believable when several of them agree.

    Args:
        k_min: Smallest k.
        k_max: Largest k.
    """
    from collections import Counter

    kw = dict(data=data, columns=columns, drop=drop, max_rows=max_rows, seed=seed)
    votes = {"elbow": find_k_elbow(**kw, k_min=k_min, k_max=k_max)["suggested_k"], "silhouette": find_k_silhouette(**kw, k_min=k_min, k_max=k_max)["suggested_k"]}
    ch = find_k_calinski_davies(**kw, k_min=k_min, k_max=k_max)
    votes["calinski_harabasz"], votes["davies_bouldin"] = ch["best_k_by_calinski_harabasz"], ch["best_k_by_davies_bouldin"]
    votes["gap_statistic"] = find_k_gap_statistic(**kw, k_min=1, k_max=k_max, n_references=6)["suggested_k"]
    votes["gmm_bic"] = find_k_gmm_bic(**kw, k_min=1, k_max=k_max)["best_k_by_bic"]
    count = Counter(votes.values())
    top, n = count.most_common(1)[0]
    return {"votes": votes, "consensus_k": int(top), "criteria_agreeing": int(n), "of": len(votes), "confidence": "high" if n >= 4 else "moderate" if n >= 3 else "low: the criteria disagree — the data has no single obvious k; choose by what the clusters must be used for"}
