"""Read-only checks before reducing dimensions: shape of the table, the intrinsic dimensionality of the data, and whether a linear method is enough."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_core.data import load_frame
from ml_core.ml import numeric_and_categorical
from ml_core.tooling import tool
from ml_core.unsup import prepare
from unsupervised.dimensionality_reduction.tools._common import fit_reducer, quality


@tool
def profile_features_for_reduction(data: str, columns: list[str] | None = None, drop: list[str] | None = None) -> dict:
    """Profile the table for dimensionality reduction: rows vs columns (and the ratio), numeric vs categorical columns, missing values, constant columns, how many column pairs are strongly correlated, and whether all values are non-negative (a precondition of NMF)."""
    frame = load_frame(data)
    names = [c for c in (columns or frame.columns) if c not in set(drop or [])]
    X = frame[names]
    num, cat = numeric_and_categorical(X)
    corr = X[num].corr().abs() if len(num) > 1 else pd.DataFrame()
    strong = int(((corr.to_numpy() > 0.8).sum() - len(num)) / 2) if len(corr) else 0
    return {"rows": len(frame), "columns": len(names), "rows_per_column": round(len(frame) / max(len(names), 1), 1), "numeric": len(num), "categorical": cat,
            "constant_columns": [c for c in names if X[c].nunique(dropna=False) <= 1], "columns_with_missing": {c: round(float(X[c].isna().mean()), 4) for c in names if X[c].isna().any()},
            "strongly_correlated_pairs": strong, "all_non_negative": bool(len(num) and (X[num].min() >= 0).all()),
            "advice": [a for a in ["fewer than 10 rows per column: reduce before modelling, but expect unstable components" if len(frame) / max(len(names), 1) < 10 else None,
                                   "many correlated pairs: linear methods (PCA) will compress a lot with little loss" if strong > len(num) / 2 else None,
                                   "categorical columns are one-hot encoded before reduction; for mostly categorical data consider MCA-like treatment or an autoencoder" if cat else None] if a]}


@tool
def estimate_intrinsic_dimension(data: str, columns: list[str] | None = None, drop: list[str] | None = None, n_neighbors: int = 10, max_rows: int = 4000, seed: int = 42) -> dict:
    """How many dimensions does the data really occupy? Three estimates: the PCA components needed for 95% of the variance (linear), the participation ratio of the eigenvalue spectrum, and the Levina–Bickel maximum-likelihood estimate from nearest-neighbour distances (local, handles curved manifolds). A gap between the linear and the local estimate means non-linear methods can compress further.

    Args:
        n_neighbors: Neighbours used by the local estimator.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    P = prepare(data, columns, drop, True, None, max_rows, seed)
    ev = PCA().fit(P.Z).explained_variance_
    r = ev / ev.sum()
    d = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(P.Z).kneighbors(P.Z)[0][:, 1:]
    d = np.maximum(d, 1e-12)
    local = (n_neighbors - 2) / np.sum(np.log(d[:, -1:] / d[:, :-1]), axis=1)
    mle = float(1 / np.mean(1 / local[np.isfinite(local) & (local > 0)]))
    linear95 = int(np.searchsorted(np.cumsum(r), 0.95) + 1)
    return {"ambient_dimension": int(P.Z.shape[1]), "pca_components_for_90pct": int(np.searchsorted(np.cumsum(r), 0.90) + 1), "pca_components_for_95pct": linear95, "participation_ratio": round(float(ev.sum() ** 2 / (ev**2).sum()), 2),
            "local_mle_dimension": round(mle, 2), "reading": "a local estimate well below the linear one means the data lies on a curved surface: Isomap, kernel PCA or UMAP can reach a smaller dimension than PCA" if mle < 0.7 * linear95 else "the linear and local estimates agree: PCA is enough"}


@tool
def check_linearity_of_structure(data: str, columns: list[str] | None = None, drop: list[str] | None = None, n_components: int = 2, n_neighbors: int = 10, max_rows: int = 2000, seed: int = 42) -> dict:
    """Does the data need a non-linear method? Embeds into the same number of dimensions with PCA, kernel PCA, Isomap and LLE and compares how well each preserves neighbourhoods (trustworthiness) — if the non-linear methods are clearly more faithful, the structure is curved.

    Args:
        n_components: Output dimensions to test at.
        n_neighbors: Neighbourhood size for Isomap, LLE and the quality scores.
    """
    P = prepare(data, columns, drop, True, None, max_rows, seed)
    rows = []
    for kind, params in (("pca", {}), ("kernel_pca", {"kernel": "rbf"}), ("isomap", {"n_neighbors": n_neighbors}), ("lle", {"n_neighbors": max(n_neighbors, n_components + 2)})):
        try:
            _, E, _ = fit_reducer(P, kind, n_components, params, seed)
            rows.append({"method": kind, **quality(P.Z, E, n_neighbors, seed)})
        except Exception as error:
            rows.append({"method": kind, "error": str(error)[:100]})
    pca = next(r for r in rows if r["method"] == "pca")
    best = max((r for r in rows if "trustworthiness" in r), key=lambda r: r["trustworthiness"])
    gain = best["trustworthiness"] - pca.get("trustworthiness", 0)
    return {"n_components": n_components, "methods": rows, "best": best["method"], "gain_over_pca": round(gain, 4),
            "verdict": "linear structure: PCA preserves neighbourhoods about as well as anything" if gain < 0.02 else "curved structure: a non-linear method keeps neighbourhoods noticeably better than PCA"}
