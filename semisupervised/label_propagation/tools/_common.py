"""Graph-based semi-supervised learning: labels flow from the labelled rows to their neighbours along a similarity graph.
Built on scikit-learn's LabelPropagation / LabelSpreading with the graph diagnostics that decide whether the idea can work at all —
propagation assumes *smoothness*: nearby rows tend to share a label."""

from __future__ import annotations

from typing import Any

import numpy as np

from ml_core.semi import Context, sk

MAX_DENSE_ROWS = 6000


def make_propagator(kind: str, kernel: str, n_neighbors: int, gamma: float, alpha: float, max_iter: int) -> Any:
    sk()
    from sklearn.semi_supervised import LabelPropagation, LabelSpreading

    common: dict[str, Any] = {"kernel": kernel, "max_iter": max_iter}
    common["n_neighbors" if kernel == "knn" else "gamma"] = n_neighbors if kernel == "knn" else gamma
    return LabelSpreading(alpha=alpha, **common) if kind == "spreading" else LabelPropagation(**common)


def propagate(ctx: Context, kind: str = "spreading", kernel: str = "knn", n_neighbors: int = 10, gamma: float = 1.0, alpha: float = 0.2, max_iter: int = 1000, Z: np.ndarray | None = None) -> tuple[Any, dict[str, Any], np.ndarray]:
    """Fit on labelled + unlabelled rows; returns the model, extras, and the propagated distribution for the unlabelled rows."""
    rows = np.vstack([ctx.Zl, ctx.Zu]) if Z is None else Z
    if kernel == "rbf" and len(rows) > MAX_DENSE_ROWS:
        raise ValueError(f"the rbf kernel builds a dense {len(rows)}x{len(rows)} graph; use kernel='knn' above {MAX_DENSE_ROWS} rows")
    y = np.concatenate([ctx.yl, -np.ones(len(ctx.Zu), int)])
    model = make_propagator(kind, kernel, n_neighbors, gamma, alpha, max_iter).fit(rows, y)
    dist = model.label_distributions_[len(ctx.yl):]
    dist = dist / np.maximum(dist.sum(axis=1, keepdims=True), 1e-12)
    full = np.zeros((len(dist), ctx.K))
    full[:, [int(c) for c in model.classes_]] = dist
    return model, {"iterations": int(getattr(model, "n_iter_", 0)), "kernel": kernel}, full


def pseudo_block(ctx: Context, dist: np.ndarray, threshold: float = 0.0) -> dict[str, np.ndarray]:
    conf = dist.max(axis=1)
    keep = np.flatnonzero(conf >= threshold)
    return {"index": ctx.split.unl[keep], "label": dist[keep].argmax(axis=1), "confidence": conf[keep]}


def graph_stats(Z: np.ndarray, y: np.ndarray, k: int) -> dict[str, Any]:
    """Connectivity and label smoothness of the k-nearest-neighbour graph over labelled (y>=0) and unlabelled (y=-1) rows."""
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Z))).fit(Z)
    dist, idx = nn.kneighbors(Z)
    from scipy.sparse import csr_matrix

    n = len(Z)
    rows = np.repeat(np.arange(n), idx.shape[1] - 1)
    graph = csr_matrix((np.ones(len(rows)), (rows, idx[:, 1:].ravel())), shape=(n, n))
    n_comp, comp = connected_components(graph, directed=False)
    sizes = np.bincount(comp)
    labelled_in = np.bincount(comp[y >= 0], minlength=n_comp)
    unreachable = int(sum(sizes[c] for c in range(n_comp) if labelled_in[c] == 0))
    both = (y[rows] >= 0) & (y[idx[:, 1:].ravel()] >= 0)
    homophily = float((y[rows][both] == y[idx[:, 1:].ravel()][both]).mean()) if both.any() else float("nan")
    return {"components": int(n_comp), "largest_component_share": round(float(sizes.max() / n), 4), "rows_in_components_without_labels": unreachable,
            "labelled_labelled_edge_homophily": round(homophily, 4) if homophily == homophily else None, "mean_neighbour_distance": round(float(dist[:, 1:].mean()), 4),
            "isolated_rows": int((graph.sum(axis=0).A1 + graph.sum(axis=1).A1 == 0).sum())}
