"""Choosing how many dimensions to keep, and the neighbourhood / perplexity parameters of the non-linear methods."""

from __future__ import annotations

import numpy as np

from ml_core.tooling import tool
from ml_core.unsup import prepare
from unsupervised.dimensionality_reduction.tools._common import fit_reducer, quality


@tool
def choose_n_components_pca(data: str, columns: list[str] | None = None, drop: list[str] | None = None, variance_target: float = 0.95, n_permutations: int = 20, max_rows: int = 5000, seed: int = 42) -> dict:
    """How many principal components to keep, by five rules side by side: the variance target, the Kaiser rule (eigenvalue > 1 on standardised data), the scree knee, Horn's parallel analysis (components whose eigenvalue beats those of column-shuffled data) and Minka's maximum-likelihood choice.

    Args:
        variance_target: Cumulative explained variance to reach for the first rule.
        n_permutations: Shuffled copies for parallel analysis.
    """
    from sklearn.decomposition import PCA

    P = prepare(data, columns, drop, True, None, max_rows, seed)
    pca = PCA().fit(P.Z)
    ev, ratio = pca.explained_variance_, pca.explained_variance_ratio_
    cum = np.cumsum(ratio)
    rng = np.random.default_rng(seed)
    null = np.array([PCA().fit(np.column_stack([rng.permutation(P.Z[:, j]) for j in range(P.Z.shape[1])])).explained_variance_ for _ in range(n_permutations)])
    thr = np.quantile(null, 0.95, axis=0)
    x, y = np.arange(len(ev), dtype=float), (ev - ev.min()) / max(np.ptp(ev), 1e-9)
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    knee = int(np.argmax(np.abs(dy * (x - x[0]) - dx * (y - y[0])) / np.hypot(dx, dy))) + 1
    try:
        mle = int(PCA(n_components="mle", svd_solver="full").fit(P.Z).n_components_) if len(P.Z) > P.Z.shape[1] else None
    except Exception:
        mle = None
    parallel = int((ev > thr).sum())
    return {"explained_variance_ratio": ratio[:15].round(4).tolist(), "cumulative": cum[:15].round(4).tolist(),
            "rules": {f"variance_{int(variance_target * 100)}pct": int(np.searchsorted(cum, variance_target) + 1), "kaiser_eigenvalue_over_1": int((ev > 1).sum()), "scree_knee": knee, "parallel_analysis": parallel, "minka_mle": mle},
            "suggested": parallel if parallel > 0 else knee, "reading": "parallel analysis is the most defensible rule: it keeps only components that carry more variance than structureless data would"}


@tool
def reconstruction_error_curve(data: str, columns: list[str] | None = None, drop: list[str] | None = None, method: str = "pca", max_components: int = 15, max_rows: int = 4000, seed: int = 42) -> dict:
    """Held-out reconstruction error against the number of components: fit on 80% of the rows, reconstruct the other 20% — the error stops falling once the extra components are fitting noise. A cross-validated alternative to variance targets.

    Args:
        method: pca, truncated_svd, ica, factor_analysis or nmf.
        max_components: Largest number of components tried.
    """
    from unsupervised.dimensionality_reduction.tools._common import make_reducer

    P = prepare(data, columns, drop, True, None, max_rows, seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(P.Z))
    cut = int(len(idx) * 0.8)
    tr, te = P.Z[idx[:cut]], P.Z[idx[cut:]]
    shift = tr.min(axis=0) if method == "nmf" else 0
    rows = []
    for n in range(1, min(max_components, P.Z.shape[1] - 1) + 1):
        m = make_reducer(method, n, {}, seed).fit(tr - shift)
        rec = m.inverse_transform(m.transform(te - shift)) + shift
        rows.append({"components": n, "heldout_rmse": round(float(np.sqrt(np.mean((te - rec) ** 2))), 4)})
    errs = [r["heldout_rmse"] for r in rows]
    best = int(np.argmin(errs)) + 1
    within = next((r["components"] for r in rows if r["heldout_rmse"] <= min(errs) * 1.02), best)
    return {"method": method, "curve": rows, "components_at_minimum": best, "fewest_components_within_2pct_of_minimum": within}


@tool
def tune_neighbourhood_size(data: str, method: str = "isomap", columns: list[str] | None = None, drop: list[str] | None = None, n_components: int = 2, values: list[int] | None = None, max_rows: int = 2000, seed: int = 42) -> dict:
    """Choose the neighbourhood size (Isomap, LLE, spectral embedding, UMAP) or the perplexity (t-SNE) by embedding quality: trustworthiness and neighbour overlap for each value. Too small fragments the manifold; too large short-circuits it.

    Args:
        method: isomap, lle, spectral_embedding, umap or tsne.
        n_components: Output dimensions.
        values: n_neighbors (or perplexity for tsne) to try; defaults to 5, 10, 20, 40.
    """
    P = prepare(data, columns, drop, True, None, max_rows, seed)
    rows = []
    for v in values or [5, 10, 20, 40]:
        key = {"perplexity": v} if method == "tsne" else {"n_neighbors": max(v, n_components + 2) if method == "lle" else v}
        try:
            _, E, _ = fit_reducer(P, method, n_components, key, seed)
            rows.append({("perplexity" if method == "tsne" else "n_neighbors"): v, **quality(P.Z, E, 10, seed)})
        except Exception as error:
            rows.append({"value": v, "error": str(error)[:90]})
    ok = [r for r in rows if "trustworthiness" in r]
    best = max(ok, key=lambda r: r["trustworthiness"] + r["continuity"]) if ok else None
    return {"method": method, "sweep": rows, "best": best}
