"""Read-only checks before clustering: what the table holds, whether it has cluster structure at all, and whether scale or redundancy will distort distances."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_core.data import load_frame
from ml_core.ml import numeric_and_categorical
from ml_core.tooling import tool
from ml_core.unsup import prepare


@tool
def profile_clustering_data(data: str, columns: list[str] | None = None, drop: list[str] | None = None) -> dict:
    """Profile a table for clustering: rows, numeric vs categorical columns, missing values, constant and id-like columns, duplicate rows, and the spread of column scales — the facts that decide how to prepare features before any distance is computed."""
    frame = load_frame(data)
    names = [c for c in (columns or frame.columns) if c not in set(drop or [])]
    X = frame[names]
    num, cat = numeric_and_categorical(X)
    desc = X[num].describe().T if num else pd.DataFrame()
    spread = (desc["std"].max() / max(desc["std"].min(), 1e-12)) if len(desc) > 1 and (desc["std"] > 0).any() else None
    skew = X[num].skew().abs().sort_values(ascending=False) if num else pd.Series(dtype=float)
    return {"rows": len(frame), "columns_considered": len(names), "numeric": num, "categorical": cat,
            "high_cardinality_columns": [c for c in names if c not in num and c not in cat], "constant_columns": [c for c in names if X[c].nunique(dropna=False) <= 1],
            "id_like_columns": [c for c in names if len(frame) > 20 and X[c].nunique() >= 0.98 * len(frame) and c not in num],
            "columns_with_missing": {c: round(float(X[c].isna().mean()), 4) for c in names if X[c].isna().any()}, "duplicate_rows": int(frame.duplicated().sum()),
            "scale_ratio_largest_to_smallest_std": round(float(spread), 1) if spread else None, "heavily_skewed_columns": [c for c, v in skew.items() if v > 2][:8],
            "advice": [a for a in ["standardise before clustering: column scales differ by orders of magnitude" if spread and spread > 10 else None,
                                   "heavy skew: log-transform those columns first, or distances will be dominated by a few extreme rows" if (skew > 2).any() else None,
                                   "categorical columns present: one-hot encoding is applied, but k-means on mixed data is crude — consider cluster_gower_hierarchical or cluster_kmodes" if cat and num else None,
                                   "drop id-like columns: they carry no structure" if any(X[c].nunique() >= 0.98 * len(frame) for c in names if c not in num) else None] if a]}


@tool
def check_clusterability(data: str, columns: list[str] | None = None, drop: list[str] | None = None, max_rows: int = 3000, n_trials: int = 10, seed: int = 42) -> dict:
    """Is there cluster structure at all? The Hopkins statistic compares nearest-neighbour distances of real rows with those of uniformly random points (0.5 = uniform noise, towards 1 = clustered), and the silhouette of the best 2–8 cluster k-means is compared with the same fit on a column-shuffled copy of the data. Clustering structureless data returns confident-looking nonsense.

    Args:
        n_trials: Hopkins repetitions averaged.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors

    P = prepare(data, columns, drop, True, None, max_rows, seed)
    Z, rng = P.Z, np.random.default_rng(seed)
    m = max(10, min(len(Z) // 10, 300))
    nn = NearestNeighbors(n_neighbors=2).fit(Z)
    lo, hi = Z.min(axis=0), Z.max(axis=0)
    h = []
    for _ in range(n_trials):
        idx = rng.choice(len(Z), m, replace=False)
        u = nn.kneighbors(rng.uniform(lo, hi, size=(m, Z.shape[1])), n_neighbors=1)[0].sum()
        w = nn.kneighbors(Z[idx], n_neighbors=2)[0][:, 1].sum()
        h.append(u / (u + w))
    shuffled = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])

    def best_sil(M: np.ndarray) -> float:
        return max(float(silhouette_score(M, KMeans(k, n_init=3, random_state=seed).fit_predict(M), sample_size=min(2000, len(M)), random_state=seed)) for k in range(2, 9))

    real, null = best_sil(Z), best_sil(shuffled)
    hop = float(np.mean(h))
    return {"hopkins": round(hop, 4), "best_silhouette_real": round(real, 4), "best_silhouette_shuffled_columns": round(null, 4),
            "verdict": "clear cluster structure" if hop > 0.75 and real > null + 0.1 else "weak structure: expect overlapping, unstable clusters — treat results as a segmentation convenience, not discovered groups" if hop > 0.6 else "no meaningful cluster structure: the data looks like one cloud"}


@tool
def check_feature_redundancy(data: str, columns: list[str] | None = None, drop: list[str] | None = None, threshold: float = 0.9) -> dict:
    """Find strongly correlated numeric columns — a group of near-duplicates counts several times in a distance, so it silently dominates the clustering — and the share of variance carried by the first principal components.

    Args:
        threshold: Absolute correlation above which two columns count as redundant.
    """
    from sklearn.decomposition import PCA

    P = prepare(data, columns, drop, True, None, 20000)
    num = P.X.select_dtypes("number")
    corr = num.corr().abs()
    pairs = [{"a": a, "b": b, "correlation": round(float(corr.loc[a, b]), 3)} for i, a in enumerate(num.columns) for b in num.columns[i + 1 :] if corr.loc[a, b] >= threshold]
    ev = PCA().fit(P.Z).explained_variance_ratio_
    return {"redundant_pairs": pairs, "components_for_90pct_variance": int(np.searchsorted(np.cumsum(ev), 0.9) + 1), "encoded_features": int(P.Z.shape[1]),
            "advice": "drop one of each redundant pair or cluster on principal components" if pairs else "no redundancy problem"}


@tool
def check_feature_influence_on_distance(data: str, columns: list[str] | None = None, drop: list[str] | None = None, max_rows: int = 2000, seed: int = 42) -> dict:
    """Which columns dominate pairwise distances after preprocessing: the average share of the squared distance each original column contributes. A column that carries most of it is effectively the only thing being clustered.

    Args:
        max_rows: Rows sampled to estimate the shares.
    """
    from ml_core.semi import column_blocks

    P = prepare(data, columns, drop, True, None, max_rows, seed)
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, len(P.Z), 4000), rng.integers(0, len(P.Z), 4000)
    d2 = (P.Z[i] - P.Z[j]) ** 2
    total = d2.sum(axis=1).mean()
    blocks = column_blocks(P.prep)
    share = {c: float(d2[:, idx].sum(axis=1).mean() / total) for c, idx in blocks.items()}
    ranked = dict(sorted(((c, round(v, 4)) for c, v in share.items()), key=lambda kv: -kv[1]))
    top = next(iter(ranked.values()))
    return {"distance_share_by_column": ranked, "dominant_column": next(iter(ranked)), "verdict": "balanced" if top < 0.35 else "one column dominates the distance: rescale, transform or drop it"}
