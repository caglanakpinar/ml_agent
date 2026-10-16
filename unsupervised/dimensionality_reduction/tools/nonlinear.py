"""Non-linear reducers for curved structure and for visualisation: kernel PCA, t-SNE, UMAP, Isomap, LLE, MDS and spectral embedding."""

from __future__ import annotations

from typing import Literal

from ml_core.tooling import tool
from ml_core.unsup import prepare
from unsupervised.dimensionality_reduction.tools._common import run_reduction


def _go(kind, label, data, n, columns, drop, scale, truth_column, params, seed, output_path, model_path, keep_columns=None, max_rows=None):
    return run_reduction(kind, label, data, n, columns, drop, scale, truth_column, params, seed, output_path, model_path, max_rows, keep_columns)


@tool
def reduce_kernel_pca(data: str, n_components: int = 2, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, kernel: Literal["rbf", "poly", "cosine", "sigmoid"] = "rbf", gamma: float | None = None, keep_columns: list[str] | None = None, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Kernel PCA: PCA in an implicit non-linear feature space, so curved structure (rings, spirals) that no straight axis captures becomes separable. Can embed new rows. Quadratic in rows — large tables are subsampled.

    Args:
        truth_column: Column of known groups, kept aside for evaluation.
        kernel: rbf, poly, cosine or sigmoid.
        gamma: Kernel width for rbf / poly / sigmoid; the library default (1 / features) when omitted.
        keep_columns: Original columns to copy next to the embedding.
    """
    return _go("kernel_pca", f"kernel_pca({kernel},{n_components})", data, n_components, columns, drop, scale, truth_column, {"kernel": kernel, "gamma": gamma}, seed, output_path, model_path, keep_columns, 4000)


@tool
def reduce_tsne(data: str, n_components: int = 2, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, perplexity: float = 30.0, keep_columns: list[str] | None = None, seed: int = 42, output_path: str | None = None) -> dict:
    """t-SNE: places rows in 2–3 dimensions so that neighbours stay neighbours — the standard picture for *seeing* clusters. Distances between far-apart groups and cluster sizes are NOT meaningful, it cannot embed new rows, and the result depends on perplexity; use for visualisation only.

    Args:
        truth_column: Column of known groups, kept aside for evaluation.
        perplexity: Effective neighbourhood size; 5–50, smaller than the number of rows.
        keep_columns: Original columns to copy next to the embedding.
    """
    return _go("tsne", f"tsne(perplexity={perplexity})", data, n_components, columns, drop, scale, truth_column, {"perplexity": perplexity}, seed, output_path, None, keep_columns, 5000)


@tool(requires=("umap",))
def reduce_umap(data: str, n_components: int = 2, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, n_neighbors: int = 15, min_dist: float = 0.1, keep_columns: list[str] | None = None, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """UMAP: like t-SNE but faster, better at keeping global structure, and able to embed new rows — the default non-linear method for both visualisation and as a feature extractor before clustering.

    Args:
        truth_column: Column of known groups, kept aside for evaluation.
        n_neighbors: Local neighbourhood size; larger keeps more global structure.
        min_dist: How tightly points may pack; smaller gives tighter clusters.
        keep_columns: Original columns to copy next to the embedding.
    """
    return _go("umap", f"umap(n_neighbors={n_neighbors})", data, n_components, columns, drop, scale, truth_column, {"n_neighbors": n_neighbors, "min_dist": min_dist}, seed, output_path, model_path, keep_columns)


@tool
def reduce_isomap(data: str, n_components: int = 2, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, n_neighbors: int = 10, keep_columns: list[str] | None = None, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Isomap: preserves *geodesic* distances — distances along the data manifold, measured through a neighbourhood graph — so it unrolls curved surfaces (a Swiss roll) that PCA flattens. Can embed new rows; sensitive to the neighbourhood size and to noise that short-circuits the graph.

    Args:
        truth_column: Column of known groups, kept aside for evaluation.
        n_neighbors: Neighbours defining the graph.
        keep_columns: Original columns to copy next to the embedding.
    """
    return _go("isomap", f"isomap(n_neighbors={n_neighbors})", data, n_components, columns, drop, scale, truth_column, {"n_neighbors": n_neighbors}, seed, output_path, model_path, keep_columns, 4000)


@tool
def reduce_lle(data: str, n_components: int = 2, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, n_neighbors: int = 12, method: Literal["standard", "modified", "hessian", "ltsa"] = "standard", keep_columns: list[str] | None = None, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Locally linear embedding: describes each row as a weighted mix of its neighbours and finds low-dimensional coordinates that keep those weights — preserves local geometry of smooth manifolds. Fragile on noisy data; the modified variant is steadier.

    Args:
        truth_column: Column of known groups, kept aside for evaluation.
        n_neighbors: Neighbours per row.
        method: standard, modified, hessian or ltsa.
        keep_columns: Original columns to copy next to the embedding.
    """
    return _go("lle", f"lle({method},n_neighbors={n_neighbors})", data, n_components, columns, drop, scale, truth_column, {"n_neighbors": n_neighbors, "method": method}, seed, output_path, model_path, keep_columns, 4000)


@tool
def reduce_mds(data: str, n_components: int = 2, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, keep_columns: list[str] | None = None, seed: int = 42, output_path: str | None = None) -> dict:
    """Metric multidimensional scaling: finds coordinates whose pairwise distances match the original ones as closely as possible (minimising stress) — the choice when *distances* between rows must be faithful, not just neighbourhoods. Quadratic in rows; no model for new rows.

    Args:
        truth_column: Column of known groups, kept aside for evaluation.
        keep_columns: Original columns to copy next to the embedding.
    """
    return _go("mds", f"mds({n_components})", data, n_components, columns, drop, scale, truth_column, {}, seed, output_path, None, keep_columns, 1500)


@tool
def reduce_spectral_embedding(data: str, n_components: int = 2, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, n_neighbors: int = 10, keep_columns: list[str] | None = None, seed: int = 42, output_path: str | None = None) -> dict:
    """Spectral embedding (Laplacian eigenmaps): embeds the rows using the leading eigenvectors of a neighbourhood-graph Laplacian — keeps connected groups together and is the first half of spectral clustering. No model for new rows.

    Args:
        truth_column: Column of known groups, kept aside for evaluation.
        n_neighbors: Neighbours in the graph.
        keep_columns: Original columns to copy next to the embedding.
    """
    return _go("spectral_embedding", f"spectral_embedding(n_neighbors={n_neighbors})", data, n_components, columns, drop, scale, truth_column, {"n_neighbors": n_neighbors}, seed, output_path, None, keep_columns, 4000)


@tool
def reduce_pca_then_tsne(data: str, n_components: int = 2, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, pca_components: int = 30, perplexity: float = 30.0, keep_columns: list[str] | None = None, seed: int = 42, output_path: str | None = None) -> dict:
    """The standard recipe for wide data: PCA down to a few dozen dimensions first (removes noise and cuts the cost of every distance), then t-SNE to 2–3. Quality is measured against the original, full-dimensional data.

    Args:
        truth_column: Column of known groups, kept aside for evaluation.
        pca_components: Dimensions kept by the first stage.
        perplexity: t-SNE perplexity.
        keep_columns: Original columns to copy next to the embedding.
    """
    import pandas as pd

    from ml_core.data import save_frame
    from ml_core.unsup import Prepared
    from unsupervised.dimensionality_reduction.tools._common import fit_reducer, probe_accuracy, quality

    P = prepare(data, columns, drop, scale, truth_column, 5000, seed)
    _, mid, _ = fit_reducer(P, "pca", min(pca_components, P.Z.shape[1]), {}, seed)
    mid_P = Prepared(P.frame, P.X, P.prep, mid, [f"pc{i}" for i in range(mid.shape[1])], P.columns, P.truth)
    _, E, _ = fit_reducer(mid_P, "tsne", n_components, {"perplexity": perplexity}, seed)
    out = {"method": f"pca({mid.shape[1]})->tsne", "rows": int(len(P.Z)), "input_dimensions": int(P.Z.shape[1]), "intermediate_dimensions": int(mid.shape[1]), "output_dimensions": n_components, "quality_vs_original": quality(P.Z, E, seed=seed)}
    if P.truth is not None:
        out["downstream_knn_accuracy_original"], out["downstream_knn_accuracy_embedding"] = probe_accuracy(P.Z, P.truth, seed), probe_accuracy(E, P.truth, seed)
    table = pd.DataFrame(E, columns=[f"dim_{i + 1}" for i in range(E.shape[1])])
    for c in [c for c in dict.fromkeys((keep_columns or []) + ([truth_column] if truth_column else [])) if c in P.frame.columns]:
        table.insert(0, c, P.frame[c].to_numpy())
    out["written_to"] = save_frame(table, output_path, "embedding_pca_tsne.csv") if output_path else None
    return out
