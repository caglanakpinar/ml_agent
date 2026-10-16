"""Linear reducers and feature-reduction methods: PCA and its variants, SVD, sparse PCA, NMF, ICA, factor analysis, random projection, feature agglomeration, and unsupervised column dropping."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ml_core.tooling import tool
from unsupervised.dimensionality_reduction.tools._common import run_reduction


def _go(kind, label, data, n, columns, drop, scale, truth_column, params, seed, output_path, model_path, keep_columns=None, max_rows=None):
    return run_reduction(kind, label, data, n, columns, drop, scale, truth_column, params, seed, output_path, model_path, max_rows, keep_columns)


@tool
def reduce_pca(data: str, n_components: int = 2, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, whiten: bool = False, keep_columns: list[str] | None = None, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """PCA: rotate to the orthogonal directions of greatest variance and keep the first few — the default reducer; fast, deterministic, invertible, and its loadings say what each component means.

    Args:
        truth_column: Column of known groups, kept aside to test whether the embedding preserves them (never used for fitting).
        whiten: Rescale components to unit variance.
        keep_columns: Original columns to copy next to the embedding (ids, labels).
        output_path: Where to write the embedding.
    """
    return _go("pca", f"pca({n_components})", data, n_components, columns, drop, scale, truth_column, {"whiten": whiten}, seed, output_path, model_path, keep_columns)


@tool
def reduce_incremental_pca(data: str, n_components: int = 2, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, batch_size: int | None = None, keep_columns: list[str] | None = None, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Incremental PCA: the same decomposition computed batch by batch, so tables too large for memory can be reduced — the right choice for millions of rows.

    Args:
        truth_column: Column of known groups, kept aside for evaluation.
        batch_size: Rows per batch.
        keep_columns: Original columns to copy next to the embedding.
    """
    return _go("incremental_pca", f"incremental_pca({n_components})", data, n_components, columns, drop, scale, truth_column, {"batch_size": batch_size}, seed, output_path, model_path, keep_columns)


@tool
def reduce_truncated_svd(data: str, n_components: int = 2, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = False, truth_column: str | None = None, keep_columns: list[str] | None = None, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Truncated SVD (latent semantic analysis): like PCA but without centring, so it works on sparse or count data (TF-IDF, one-hot, term counts) without destroying the sparsity.

    Args:
        truth_column: Column of known groups, kept aside for evaluation.
        keep_columns: Original columns to copy next to the embedding.
    """
    return _go("truncated_svd", f"truncated_svd({n_components})", data, n_components, columns, drop, scale, truth_column, {}, seed, output_path, model_path, keep_columns)


@tool
def reduce_sparse_pca(data: str, n_components: int = 3, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, alpha: float = 1.0, keep_columns: list[str] | None = None, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Sparse PCA: components built from only a few columns each (an L1 penalty zeroes the rest), trading a little variance for loadings a person can read — 'component 1 is just these three columns'.

    Args:
        truth_column: Column of known groups, kept aside for evaluation.
        alpha: Sparsity penalty; larger gives sparser components.
        keep_columns: Original columns to copy next to the embedding.
    """
    return _go("sparse_pca", f"sparse_pca({n_components})", data, n_components, columns, drop, scale, truth_column, {"alpha": alpha}, seed, output_path, model_path, keep_columns, 8000)


@tool
def reduce_nmf(data: str, n_components: int = 3, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, keep_columns: list[str] | None = None, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Non-negative matrix factorisation: explain each row as a non-negative mix of a few non-negative 'parts' — parts-based and usually easier to interpret than PCA's signed components (topics, spectra, usage profiles). Columns are shifted to be non-negative first.

    Args:
        truth_column: Column of known groups, kept aside for evaluation.
        keep_columns: Original columns to copy next to the embedding.
    """
    return _go("nmf", f"nmf({n_components})", data, n_components, columns, drop, scale, truth_column, {}, seed, output_path, model_path, keep_columns)


@tool
def reduce_ica(data: str, n_components: int = 3, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, keep_columns: list[str] | None = None, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Independent component analysis: finds statistically independent (not merely uncorrelated) source signals — unmixes overlapping sources such as sensors recording the same few physical processes.

    Args:
        truth_column: Column of known groups, kept aside for evaluation.
        keep_columns: Original columns to copy next to the embedding.
    """
    return _go("ica", f"ica({n_components})", data, n_components, columns, drop, scale, truth_column, {}, seed, output_path, model_path, keep_columns)


@tool
def reduce_factor_analysis(data: str, n_components: int = 3, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, rotation: Literal["none", "varimax", "quartimax"] = "varimax", keep_columns: list[str] | None = None, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Factor analysis: models each column as a mix of a few latent factors *plus its own noise*, so unlike PCA it separates shared variance from column-specific noise; a varimax rotation makes the factors easier to name (survey and psychometric style latent traits).

    Args:
        truth_column: Column of known groups, kept aside for evaluation.
        rotation: none, varimax or quartimax.
        keep_columns: Original columns to copy next to the embedding.
    """
    return _go("factor_analysis", f"factor_analysis({n_components},{rotation})", data, n_components, columns, drop, scale, truth_column, {"rotation": None if rotation == "none" else rotation}, seed, output_path, model_path, keep_columns)


@tool
def reduce_random_projection(data: str, n_components: int = 10, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, sparse: bool = True, keep_columns: list[str] | None = None, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Random projection (Johnson–Lindenstrauss): multiply by a random matrix to a lower dimension — pairwise distances are preserved with high probability, with no fitting at all, so it scales to enormous, very wide data. Needs more dimensions than PCA for the same fidelity.

    Args:
        truth_column: Column of known groups, kept aside for evaluation.
        sparse: Sparse random matrix (faster) rather than Gaussian.
        keep_columns: Original columns to copy next to the embedding.
    """
    return _go("random_projection", f"random_projection({n_components})", data, n_components, columns, drop, scale, truth_column, {"sparse": sparse}, seed, output_path, model_path, keep_columns)


@tool
def reduce_feature_agglomeration(data: str, n_components: int = 5, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, keep_columns: list[str] | None = None, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Feature agglomeration: cluster the *columns* by similarity and replace each cluster of columns with its mean — every new feature is an average of related original columns, so it stays directly interpretable.

    Args:
        n_components: Number of column groups (new features).
        truth_column: Column of known groups, kept aside for evaluation.
        keep_columns: Original columns to copy next to the embedding.
    """
    return _go("feature_agglomeration", f"feature_agglomeration({n_components})", data, n_components, columns, drop, scale, truth_column, {}, seed, output_path, model_path, keep_columns)


@tool
def reduce_by_variance_and_correlation(data: str, columns: list[str] | None = None, drop: list[str] | None = None, min_variance: float = 0.01, max_correlation: float = 0.95, output_path: str | None = None) -> dict:
    """Unsupervised column selection: drop columns with almost no variance, and from every group of near-duplicate columns (correlation above the limit) keep only the one with the most variance. No components — the surviving columns are original and interpretable.

    Args:
        min_variance: Columns whose standardised-scale variance share is below this are dropped (after scaling to unit range).
        max_correlation: Absolute correlation above which one of two columns is dropped.
        output_path: Where to write the table with only the kept columns.
    """
    from ml_core.data import load_frame, save_frame

    frame = load_frame(data)
    names = [c for c in (columns or frame.columns) if c not in set(drop or [])]
    num = frame[names].select_dtypes("number")
    rng_ = (num - num.min()) / (num.max() - num.min()).replace(0, np.nan)
    var = rng_.var()
    low = [c for c in num.columns if not (var[c] >= min_variance)]
    keep = [c for c in num.columns if c not in low]
    corr = num[keep].corr().abs()
    dropped = {}
    for c in sorted(keep, key=lambda c: -var[c]):
        if c in dropped:
            continue
        for o in keep:
            if o != c and o not in dropped and corr.loc[c, o] > max_correlation:
                dropped[o] = {"redundant_with": c, "correlation": round(float(corr.loc[c, o]), 3)}
    kept = [c for c in names if c not in low and c not in dropped]
    return {"input_columns": len(names), "dropped_low_variance": low, "dropped_redundant": dropped, "kept": kept, "kept_count": len(kept), "written_to": save_frame(frame[kept], output_path, "reduced_columns.csv") if output_path else None}
