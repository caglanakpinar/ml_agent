"""Representation-learning machinery the tool modules share: the embedding quality measures (probe accuracy, neighbour purity, isotropy,
effective rank — whether a representation is any good is an empirical question), PPMI + SVD word/entity vectors, and the picklable embedder."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_core.data import save_frame
from ml_core.ml import save_model
from ml_core.unsup import Prepared


# ---- quality ------------------------------------------------------------------------------------
def effective_rank(E: np.ndarray) -> float:
    """exp(entropy of the normalised singular values): near the number of dimensions = information spread across them; near 1 = dimensional collapse."""
    s = np.linalg.svd(E - E.mean(axis=0), compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def isotropy(E: np.ndarray, n_pairs: int = 4000, seed: int = 0) -> float:
    """Mean cosine similarity of random pairs (centred): near 0 = embeddings use the whole space; a large positive value = they crowd into a narrow cone."""
    rng = np.random.default_rng(seed)
    X = E - E.mean(axis=0)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    i, j = rng.integers(0, len(X), n_pairs), rng.integers(0, len(X), n_pairs)
    return float(np.abs((X[i] * X[j]).sum(axis=1)).mean())


def probe(E: np.ndarray, y: np.ndarray, task: str = "classification", seed: int = 0) -> dict[str, float]:
    """Cross-validated linear and k-NN probes: how much label information a *simple* model can read straight off the embedding."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if task == "regression":
        return {"linear_r2": round(float(cross_val_score(make_pipeline(StandardScaler(), Ridge(1.0)), E, y, cv=5, scoring="r2").mean()), 4),
                "knn_r2": round(float(cross_val_score(KNeighborsRegressor(10), E, y, cv=5, scoring="r2").mean()), 4)}
    return {"linear_accuracy": round(float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), E, y, cv=5).mean()), 4),
            "knn_accuracy": round(float(cross_val_score(KNeighborsClassifier(10), E, y, cv=5).mean()), 4)}


def neighbour_purity(E: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    """Share of each row's k nearest neighbours (in the embedding) that carry its label — retrieval quality."""
    from sklearn.neighbors import NearestNeighbors

    idx = NearestNeighbors(n_neighbors=k + 1).fit(E).kneighbors(E, return_distance=False)[:, 1:]
    return float((y[idx] == y[:, None]).mean())


def quality_report(P: Prepared, E: np.ndarray, truth: np.ndarray | None, task: str = "classification", seed: int = 0) -> dict[str, Any]:
    from sklearn.manifold import trustworthiness

    idx = np.random.default_rng(seed).choice(len(E), min(len(E), 2000), replace=False)
    out: dict[str, Any] = {"dimensions": int(E.shape[1]), "effective_rank": round(effective_rank(E), 2), "isotropy_mean_abs_cosine": round(isotropy(E, seed=seed), 4),
                           "trustworthiness_vs_input": round(float(trustworthiness(P.Z[idx], E[idx], n_neighbors=10)), 4)}
    if truth is not None:
        out["probe_on_embedding"] = probe(E, truth, task, seed)
        out["probe_on_raw_features"] = probe(P.Z, truth, task, seed)
        if task == "classification":
            out["neighbour_label_purity"] = round(neighbour_purity(E, truth), 4)
    return out


# ---- PPMI vectors -------------------------------------------------------------------------------
def ppmi_vectors(cooc: np.ndarray, dim: int, shift: float = 0.0, alpha: float = 0.75) -> np.ndarray:
    """Shifted positive pointwise mutual information of a co-occurrence matrix, factorised by SVD — the matrix-factorisation form of skip-gram with negative sampling (Levy & Goldberg), with context smoothing alpha."""
    from scipy.sparse.linalg import svds

    total = cooc.sum()
    row = cooc.sum(axis=1, keepdims=True)
    ctx = cooc.sum(axis=0, keepdims=True) ** alpha
    ctx = ctx / ctx.sum() * total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log(np.where(cooc > 0, cooc * total / (row * ctx + 1e-12), 1.0)) - shift
    pmi = np.where(cooc > 0, np.maximum(pmi, 0), 0)
    k = min(dim, min(pmi.shape) - 1)
    U, s, _ = svds(pmi, k=k)
    order = np.argsort(-s)
    return U[:, order] * np.sqrt(s[order])


def cooccurrence(sequences: list[list[str]], window: int, vocab: list[str] | None = None, distance_weighted: bool = True) -> tuple[np.ndarray, list[str]]:
    vocab = vocab or sorted({t for s in sequences for t in s})
    ix = {t: i for i, t in enumerate(vocab)}
    C = np.zeros((len(vocab), len(vocab)))
    for s in sequences:
        ids = [ix[t] for t in s if t in ix]
        for a, i in enumerate(ids):
            for b in range(max(0, a - window), min(len(ids), a + window + 1)):
                if b != a:
                    C[i, ids[b]] += 1.0 / abs(a - b) if distance_weighted else 1.0
    return C, vocab


class Embedder:
    """Preprocessor + a transform function for new rows (any object with `transform(Z)`)."""

    def __init__(self, kind: str, prep: Any, columns: list[str], model: Any):
        self.kind, self.prep, self.columns, self.model = kind, prep, columns, model

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.prep.transform(X[self.columns])
        return np.asarray(self.model.transform(np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z, float)))


def finish(kind: str, label: str, P: Prepared, E: np.ndarray, model: Any, truth_column: str | None, keep_columns: Any, task: str, seed: int, output_path: str | None, model_path: str | None, extra: dict | None = None) -> dict[str, Any]:
    out = {"method": label, "rows": int(len(E)), "input_dimensions": int(P.Z.shape[1]), "quality": quality_report(P, E, P.truth, task, seed), **(extra or {})}
    table = pd.DataFrame(E, columns=[f"emb_{i + 1}" for i in range(E.shape[1])])
    for c in [c for c in dict.fromkeys((list(keep_columns or []) + ([truth_column] if truth_column else []))) if c in P.frame.columns][::-1]:
        table.insert(0, c, P.frame[c].to_numpy())
    out["written_to"] = save_frame(table, output_path, f"embedding_{kind}.csv") if output_path else None
    out["model_path"] = save_model(Embedder(kind, P.prep, P.columns, model), model_path) if model is not None else None
    return out
