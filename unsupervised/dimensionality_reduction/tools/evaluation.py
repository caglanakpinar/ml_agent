"""Judge a reduction: methods side by side, trustworthiness of an embedding you already have, what a downstream model loses, what the components mean, and embedding new rows."""

from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.tooling import tool
from ml_core.unsup import prepare
from unsupervised.dimensionality_reduction.tools._common import (
    NO_OUT_OF_SAMPLE,
    fit_reducer,
    probe_accuracy,
    quality,
)


@tool
def compare_reductions(data: str, n_components: int = 2, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, methods: list[str] | None = None, max_rows: int = 2000, seed: int = 42) -> dict:
    """Run several reducers to the same number of dimensions on the same rows and compare trustworthiness, continuity, neighbour overlap, distance correlation, reconstruction error, a downstream k-NN accuracy (with a truth column) and run time — shows which method keeps what the data is for.

    Args:
        n_components: Output dimensions for every method.
        truth_column: Column of known groups, kept aside to test what survives the reduction.
        methods: Any of pca, truncated_svd, ica, factor_analysis, kernel_pca, isomap, lle, spectral_embedding, tsne, mds, random_projection, umap; defaults to seven.
    """
    P = prepare(data, columns, drop, True, truth_column, max_rows, seed)
    rows = []
    for kind in methods or ["pca", "factor_analysis", "kernel_pca", "isomap", "lle", "tsne", "random_projection"]:
        t0 = time.time()
        try:
            model, E, shift = fit_reducer(P, kind, n_components, {}, seed)
            row = {"method": kind, **quality(P.Z, E, 10, seed), "seconds": round(time.time() - t0, 2), "can_embed_new_rows": kind not in NO_OUT_OF_SAMPLE}
            if hasattr(model, "inverse_transform"):
                try:
                    R = model.inverse_transform(E)
                    row["reconstruction_r2"] = round(float(1 - np.sum((P.Z - R) ** 2) / np.sum((P.Z - P.Z.mean(axis=0)) ** 2)), 4)
                except Exception:
                    pass
            if P.truth is not None:
                row["knn_accuracy"] = probe_accuracy(E, P.truth, seed)
            rows.append(row)
        except Exception as error:
            rows.append({"method": kind, "error": str(error)[:100]})
    ok = [r for r in rows if "trustworthiness" in r]
    best = max(ok, key=lambda r: r["trustworthiness"] + r["continuity"]) if ok else None
    out = {"n_components": n_components, "methods": rows, "most_faithful": best["method"] if best else None}
    if P.truth is not None:
        out["knn_accuracy_on_original_features"] = probe_accuracy(P.Z, P.truth, seed)
    return out


@tool
def evaluate_embedding(original: str, embedding: str, columns: list[str] | None = None, drop: list[str] | None = None, embedding_columns: list[str] | None = None, k: int = 10, max_rows: int = 2500, seed: int = 42) -> dict:
    """Score an embedding you already have against the original data: trustworthiness (do embedded neighbours come from true neighbours?), continuity (do true neighbours stay together?), neighbour overlap and distance-rank correlation. The rows of the two files must correspond one to one.

    Args:
        original: Path to the original table.
        embedding: Path to the embedding table (e.g. the file written by a reduce_* tool).
        embedding_columns: Embedding columns; defaults to every `dim_*` column.
        k: Neighbourhood size for the scores.
    """
    P = prepare(original, columns, drop, True, None, None, seed)
    E = load_frame(embedding)
    cols = embedding_columns or [c for c in E.columns if c.startswith("dim_")]
    require_columns(E, cols)
    if len(E) != len(P.Z):
        raise ValueError(f"the embedding has {len(E)} rows but the original has {len(P.Z)}; they must correspond row for row")
    return {"rows": len(E), "embedding_dimensions": len(cols), **quality(P.Z, E[cols].to_numpy(float), k, seed, max_rows),
            "reading": "trustworthiness below ~0.9 means the embedding shows neighbours that are not real; continuity below ~0.9 means real neighbours were torn apart"}


@tool
def downstream_probe(data: str, truth_column: str, columns: list[str] | None = None, drop: list[str] | None = None, dimensions: list[int] | None = None, method: str = "pca", seed: int = 42) -> dict:
    """What does reduction cost a downstream model? Cross-validated accuracy of a k-NN and a logistic regression predicting the known groups from 1, 2, 3, 5, 10 … reduced dimensions, against the full feature set — shows how few dimensions keep the information that matters.

    Args:
        truth_column: Column of known groups to predict.
        dimensions: Output sizes to try.
        method: Reducer used (any linear reducer, or kernel_pca / isomap).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    P = prepare(data, columns, drop, True, truth_column, 6000, seed)
    if P.truth is None:
        raise ValueError("`truth_column` is required")
    full = {"knn": float(cross_val_score(KNeighborsClassifier(10), P.Z, P.truth, cv=5).mean()), "logreg": float(cross_val_score(LogisticRegression(max_iter=1000), P.Z, P.truth, cv=5).mean())}
    rows = []
    for d in dimensions or [1, 2, 3, 5, 10]:
        if d >= P.Z.shape[1]:
            continue
        _, E, _ = fit_reducer(P, method, d, {}, seed)
        rows.append({"dimensions": d, "knn": round(float(cross_val_score(KNeighborsClassifier(10), E, P.truth, cv=5).mean()), 4), "logreg": round(float(cross_val_score(LogisticRegression(max_iter=1000), E, P.truth, cv=5).mean()), 4)})
    return {"method": method, "full_features": {k: round(v, 4) for k, v in full.items()}, "by_dimension": rows,
            "smallest_dimension_within_2pct": next((r["dimensions"] for r in rows if r["knn"] >= full["knn"] - 0.02), None),
            "note": "the reduction here is unsupervised: a supervised method (LDA, PLS) would keep more label information in the same dimensions"}


@tool
def explain_components(data: str, n_components: int = 3, method: str = "pca", columns: list[str] | None = None, drop: list[str] | None = None, top: int = 6, seed: int = 42) -> dict:
    """Name the components: for each, the original columns with the largest loadings (and their sign), the variance it carries, and a plain-language reading — the step that turns 'PC1' into 'overall size' or 'engagement'.

    Args:
        n_components: Components to explain.
        method: pca, factor_analysis, ica, nmf, sparse_pca or truncated_svd.
        top: Columns listed per component.
    """
    P = prepare(data, columns, drop, True, None, 8000, seed)
    model, E, _ = fit_reducer(P, method, n_components, {}, seed)
    comp = np.asarray(model.components_)
    ev = getattr(model, "explained_variance_ratio_", None)
    out = []
    for i in range(comp.shape[0]):
        order = np.argsort(-np.abs(comp[i]))[:top]
        out.append({"component": i + 1, "variance_share": round(float(ev[i]), 4) if ev is not None else None, "loadings": {P.names[j]: round(float(comp[i, j]), 3) for j in order},
                    "reading": ("a contrast between " + ", ".join(P.names[j] for j in order if comp[i, j] > 0)[:60] + " and " + ", ".join(P.names[j] for j in order if comp[i, j] < 0)[:60]) if (comp[i, order] > 0).any() and (comp[i, order] < 0).any()
                    else "a general level / size factor: its columns move together"})
    return {"method": method, "components": out}


@tool
def flag_poorly_reconstructed_rows(data: str, n_components: int = 3, method: str = "pca", columns: list[str] | None = None, drop: list[str] | None = None, z_threshold: float = 3.0, k: int = 20, seed: int = 42, output_path: str | None = None) -> dict:
    """Rows the reduced representation cannot reconstruct (reconstruction error in standard deviations): they do not follow the structure the components capture — outliers, data errors, or a different regime. A by-product of reduction that doubles as an anomaly list.

    Args:
        n_components: Components kept.
        method: pca, truncated_svd, ica, factor_analysis or kernel_pca.
        z_threshold: Error z-score above which a row is flagged.
        k: Rows listed.
    """
    P = prepare(data, columns, drop, True, None, None, seed)
    model, E, _ = fit_reducer(P, method, n_components, {}, seed)
    err = np.linalg.norm(P.Z - model.inverse_transform(E), axis=1)
    z = (err - err.mean()) / (err.std() or 1)
    table = P.frame.assign(reconstruction_error=err, error_z=z)
    flagged = table[table["error_z"] >= z_threshold].sort_values("error_z", ascending=False)
    return {"method": method, "n_components": n_components, "mean_error": round(float(err.mean()), 4), "flagged": int(len(flagged)), "share": round(float(len(flagged) / len(table)), 4), "worst": flagged.head(k),
            "written_to": save_frame(flagged, output_path, "poorly_reconstructed_rows.csv") if output_path else None}


@tool
def transform_new_rows(model_path: str, data: str, id_column: str | None = None, output_path: str | None = None) -> dict:
    """Embed new rows with a saved reducer, using the preprocessing and components it was fitted with (t-SNE, MDS and spectral embedding cannot do this — they have no model).

    Args:
        model_path: Path of the saved reducer (joblib).
        id_column: Column to carry through to the output.
    """
    model = joblib.load(model_path)
    frame = load_frame(data)
    require_columns(frame, model.columns)
    Z = model.prep.transform(frame[model.columns])
    E = model.transform(np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z, float))
    out = pd.DataFrame(E, columns=[f"dim_{i + 1}" for i in range(E.shape[1])])
    if id_column and id_column in frame.columns:
        out.insert(0, id_column, frame[id_column].to_numpy())
    return {"rows": len(out), "method": model.kind, "dimensions": int(E.shape[1]), "written_to": save_frame(out, output_path, "embedding_new_rows.csv"), "sample": out.head(5)}
