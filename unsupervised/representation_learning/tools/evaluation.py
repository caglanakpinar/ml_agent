"""Judge a representation by what it is *for*: probes against labels, neighbour retrieval, geometry (rank, isotropy), embeddings compared with each other and with the raw features, and embedding new rows."""

from __future__ import annotations

from typing import Literal

import joblib
import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.tooling import tool
from ml_core.unsup import prepare
from unsupervised.representation_learning.tools._common import effective_rank, isotropy, neighbour_purity, probe


def _load(embedding: str, label_column: str | None, label_file: str | None, embedding_columns: list[str] | None):
    E = load_frame(embedding)
    cols = embedding_columns or [c for c in E.columns if c.startswith("emb_")]
    require_columns(E, cols)
    y = None
    if label_column:
        src = load_frame(label_file) if label_file else E
        require_columns(src, [label_column])
        if len(src) != len(E):
            raise ValueError(f"the label file has {len(src)} rows but the embedding has {len(E)}; rows must correspond")
        y = src[label_column]
    return E, cols, y


@tool
def evaluate_embedding_quality(embedding: str, label_column: str | None = None, label_file: str | None = None, embedding_columns: list[str] | None = None, task: Literal["classification", "regression"] = "classification", k: int = 10, seed: int = 42) -> dict:
    """Judge an embedding file: geometry (effective rank — does it use its dimensions or has it collapsed?; isotropy — does it fill the space or crowd into a cone?; norm spread) and, with labels, linear and k-NN probe scores and the label purity of each row's nearest neighbours. An embedding that scores no better than the raw features on its probe has not learned anything useful.

    Args:
        embedding: Path to the embedding table (`emb_*` columns, as written by the embed_* tools).
        label_column: Column of known labels or targets, used only to probe.
        label_file: Table holding the label column when it is not in the embedding file (rows must match).
        embedding_columns: Embedding columns; every `emb_*` column when omitted.
        task: classification or regression (the probe used).
        k: Neighbours for the purity score.
    """
    E, cols, y = _load(embedding, label_column, label_file, embedding_columns)
    X = E[cols].to_numpy(float)
    norms = np.linalg.norm(X, axis=1)
    out = {"rows": len(X), "dimensions": len(cols), "effective_rank": round(effective_rank(X), 2), "rank_share": round(effective_rank(X) / len(cols), 3), "isotropy_mean_abs_cosine": round(isotropy(X, seed=seed), 4),
           "norm_mean": round(float(norms.mean()), 4), "norm_cv": round(float(norms.std() / max(norms.mean(), 1e-9)), 4), "constant_dimensions": int((X.std(axis=0) < 1e-8).sum())}
    if y is not None:
        if task == "classification":
            codes = pd.factorize(y.astype(str))[0]
            out["probe"], out["neighbour_label_purity"] = probe(X, codes, "classification", seed), round(neighbour_purity(X, codes, k), 4)
            out["majority_class_rate"] = round(float(np.bincount(codes).max() / len(codes)), 4)
        else:
            out["probe"] = probe(X, y.to_numpy(float), "regression", seed)
    out["verdict"] = "dimensional collapse: the embedding uses few of its dimensions" if out["rank_share"] < 0.2 else "usable geometry"
    return out


@tool
def compare_embeddings(embeddings: list[str], label_column: str, label_file: str, raw_data: str | None = None, raw_drop: list[str] | None = None, task: Literal["classification", "regression"] = "classification", seed: int = 42) -> dict:
    """Put several embeddings on one scoreboard — the same probe, on the same rows, with the same labels — and, if the raw feature table is given, against the raw features as the baseline. The comparison that decides which representation to keep (and whether any of them beats not learning one).

    Args:
        embeddings: Paths of embedding tables, one per method (rows aligned with the label file).
        label_column: Column of labels in `label_file`.
        label_file: Table with one label per row, aligned with every embedding.
        raw_data: Optional raw feature table (rows aligned) as the no-learning baseline.
        raw_drop: Columns of the raw table to exclude (ids, the label).
        task: classification or regression.
    """
    labels = load_frame(label_file)[label_column]
    y = pd.factorize(labels.astype(str))[0] if task == "classification" else labels.to_numpy(float)
    rows = []
    for path in embeddings:
        E = load_frame(path)
        cols = [c for c in E.columns if c.startswith("emb_")]
        if len(E) != len(y):
            rows.append({"embedding": path, "error": f"{len(E)} rows vs {len(y)} labels"})
            continue
        X = E[cols].to_numpy(float)
        rows.append({"embedding": path.split("/")[-1], "dimensions": len(cols), **probe(X, y, task, seed), "effective_rank": round(effective_rank(X), 2), **({"neighbour_label_purity": round(neighbour_purity(X, y), 4)} if task == "classification" else {})})
    if raw_data:
        P = prepare(raw_data, None, list(raw_drop or []) + [label_column], True, None)
        if len(P.Z) == len(y):
            rows.append({"embedding": "RAW FEATURES (baseline)", "dimensions": int(P.Z.shape[1]), **probe(P.Z, y, task, seed), **({"neighbour_label_purity": round(neighbour_purity(P.Z, y), 4)} if task == "classification" else {})})
    key = "linear_accuracy" if task == "classification" else "linear_r2"
    ranked = sorted([r for r in rows if key in r], key=lambda r: -r[key])
    return {"ranked_by": key, "scoreboard": ranked, "failed": [r for r in rows if "error" in r], "best": ranked[0]["embedding"] if ranked else None,
            "note": "a learned embedding that does not beat the raw features on a linear probe has compressed information away rather than organised it"}


@tool
def probe_embedding(embedding: str, label_column: str, label_file: str | None = None, embedding_columns: list[str] | None = None, task: Literal["classification", "regression"] = "classification", dimensions: list[int] | None = None, seed: int = 42) -> dict:
    """How much of a label is *linearly* readable from the first d dimensions of an embedding, for several d — shows how many dimensions carry the information and whether the representation is front-loaded (as PCA/SVD output is) or spread evenly.

    Args:
        embedding: Path to the embedding table.
        label_column: Column of labels.
        label_file: Table holding the label column when it is not in the embedding file.
        embedding_columns: Embedding columns; every `emb_*` column when omitted.
        task: classification or regression.
        dimensions: Prefix sizes to try; defaults to 1, 2, 4, 8, 16, … up to all.
    """
    E, cols, y = _load(embedding, label_column, label_file, embedding_columns)
    X = E[cols].to_numpy(float)
    t = pd.factorize(y.astype(str))[0] if task == "classification" else y.to_numpy(float)
    ds = [d for d in (dimensions or [1, 2, 4, 8, 16, 32, 64, 128]) if d < len(cols)] + [len(cols)]
    key = "linear_accuracy" if task == "classification" else "linear_r2"
    rows = [{"dimensions": d, **probe(X[:, :d], t, task, seed)} for d in ds]
    full = rows[-1][key]
    return {"by_dimension": rows, "dimensions_for_95pct_of_full_score": next((r["dimensions"] for r in rows if r[key] >= 0.95 * full), None), "full_score": full}


@tool
def retrieve_nearest_neighbours(embedding: str, rows: list[int], k: int = 10, id_column: str | None = None, label_column: str | None = None, label_file: str | None = None, embedding_columns: list[str] | None = None, metric: Literal["cosine", "euclidean"] = "cosine",
                                output_path: str | None = None) -> dict:
    """Nearest neighbours of chosen rows in an embedding — the qualitative check (do the neighbours make sense to a person?) and the basis of similarity search and 'more like this'. With labels it also reports how many of each row's neighbours share its label.

    Args:
        embedding: Path to the embedding table.
        rows: Row numbers (0-based) to look up.
        k: Neighbours per row.
        id_column: Column of the embedding file that names a row.
        label_column: Column of labels, to report neighbour agreement.
        label_file: Table holding the label column when it is not in the embedding file.
        embedding_columns: Embedding columns; every `emb_*` column when omitted.
        metric: cosine or euclidean.
        output_path: Where to write the neighbour table.
    """
    from sklearn.neighbors import NearestNeighbors

    E, cols, y = _load(embedding, label_column, label_file, embedding_columns)
    X = E[cols].to_numpy(float)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    dist, idx = nn.kneighbors(X[rows])
    out = []
    for r, d, ix in zip(rows, dist, idx):
        for rank, (dd, j) in enumerate(zip(d[1:], ix[1:]), start=1):
            row = {"query_row": r, "rank": rank, "neighbour_row": int(j), "distance": round(float(dd), 4)}
            if id_column and id_column in E.columns:
                row["query_id"], row["neighbour_id"] = E.loc[r, id_column], E.loc[j, id_column]
            if y is not None:
                row["same_label"] = bool(y.iloc[r] == y.iloc[j])
            out.append(row)
    table = pd.DataFrame(out)
    return {"queries": len(rows), "k": k, "neighbours": table.head(60), **({"share_of_neighbours_with_same_label": round(float(table["same_label"].mean()), 4)} if "same_label" in table else {}), "written_to": save_frame(table, output_path, "nearest_neighbours.csv") if output_path else None}


@tool
def embedding_dimension_sweep(data: str, method: Literal["autoencoder", "kmeans_features", "random_fourier", "tree_ensemble"] = "autoencoder", label_column: str | None = None, dimensions: list[int] | None = None, columns: list[str] | None = None, drop: list[str] | None = None,
                              max_rows: int = 5000, seed: int = 42) -> dict:
    """How big should the embedding be? Train the chosen tabular method at several sizes and probe each against the labels — accuracy that stops rising marks the smallest adequate dimension, and a method that gets worse with size is over-fitting.

    Args:
        method: autoencoder, kmeans_features, random_fourier or tree_ensemble.
        label_column: Column of known labels, used only to probe.
        dimensions: Sizes to try; defaults to 2, 4, 8, 16, 32.
    """
    from unsupervised.representation_learning.tools import tabular

    fn = {"autoencoder": tabular.embed_autoencoder, "kmeans_features": tabular.embed_kmeans_features, "random_fourier": tabular.embed_random_fourier_features, "tree_ensemble": tabular.embed_tree_ensemble}[method]
    rows = []
    for d in dimensions or [2, 4, 8, 16, 32]:
        r = fn(data=data, n_components=d, columns=columns, drop=drop, truth_column=label_column, max_rows=max_rows, seed=seed)
        if r["status"] != "ok":
            rows.append({"dimensions": d, "error": r.get("message", "")[:80]})
            continue
        q = r["quality"]
        rows.append({"dimensions": d, "effective_rank": q["effective_rank"], **q.get("probe_on_embedding", {})})
    key = "linear_accuracy"
    ok = [r for r in rows if key in r]
    best = max((r[key] for r in ok), default=None)
    return {"method": method, "sweep": rows, "smallest_within_1pct_of_best": next((r["dimensions"] for r in ok if r[key] >= best - 0.01), None) if ok else None}


@tool
def embed_new_rows(model_path: str, data: str, text_column: str | None = None, id_column: str | None = None, output_path: str | None = None) -> dict:
    """Embed new rows (or new texts, with `text_column`) with a saved embedder — the same preprocessing and the same learned map as at training time, so the vectors live in the same space as the original embedding.

    Args:
        model_path: Path of the saved embedder (joblib) written by an embed_* tool.
        text_column: Column of texts, for a text embedder.
        id_column: Column to carry through.
        output_path: Where to write the new embeddings.
    """
    model = joblib.load(model_path)
    frame = load_frame(data)
    if text_column:
        require_columns(frame, [text_column])
        E = np.asarray(model.transform(frame[text_column].astype(str).tolist()), float)
    else:
        if hasattr(model, "columns"):
            require_columns(frame, model.columns)
        E = np.asarray(model.transform(frame), float)
    out = pd.DataFrame(E, columns=[f"emb_{i + 1}" for i in range(E.shape[1])])
    if id_column and id_column in frame.columns:
        out.insert(0, id_column, frame[id_column].to_numpy())
    return {"rows": len(out), "dimensions": int(E.shape[1]), "mean_norm": round(float(np.linalg.norm(E, axis=1).mean()), 4), "rows_with_zero_vector": int((np.linalg.norm(E, axis=1) == 0).sum()), "written_to": save_frame(out, output_path, "embedding_new_rows.csv"), "sample": out.head(3)}
