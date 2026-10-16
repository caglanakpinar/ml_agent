"""Read-only checks before learning a representation: what the inputs are, the raw-feature baseline an embedding must beat, whether co-occurrence data is dense enough, and the shape of a graph."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns
from ml_core.ml import numeric_and_categorical
from ml_core.tooling import tool
from ml_core.unsup import prepare
from unsupervised.representation_learning.tools._common import probe


@tool
def profile_representation_inputs(data: str, columns: list[str] | None = None, drop: list[str] | None = None, text_column: str | None = None) -> dict:
    """Profile the table an embedding will be learned from: rows, numeric and categorical columns, missing values, high-cardinality columns, scale spread — and for a text column the number of documents, words per document and vocabulary size. Decides which family of method applies (tabular, text, co-occurrence) and what preprocessing it needs.

    Args:
        text_column: Column of free text, if the entities are documents.
    """
    frame = load_frame(data)
    names = [c for c in (columns or frame.columns) if c not in set(drop or []) and c != text_column]
    X = frame[names]
    num, cat = numeric_and_categorical(X)
    out = {"rows": len(frame), "numeric": num, "categorical": cat, "high_cardinality": [c for c in names if c not in num and c not in cat], "columns_with_missing": {c: round(float(X[c].isna().mean()), 4) for c in names if X[c].isna().any()},
           "scale_ratio_largest_to_smallest_std": round(float(X[num].std().max() / max(X[num].std().min(), 1e-12)), 1) if len(num) > 1 else None, "constant_columns": [c for c in names if X[c].nunique(dropna=False) <= 1]}
    if text_column:
        require_columns(frame, [text_column])
        words = frame[text_column].dropna().astype(str).str.split().map(len)
        vocab = pd.Series(" ".join(frame[text_column].dropna().astype(str).str.lower().head(20000)).split()).nunique()
        out["text"] = {"documents": int(words.shape[0]), "words_per_document_median": float(words.median()), "short_documents_under_10_words": round(float((words < 10).mean()), 4), "vocabulary_in_first_20000_documents": int(vocab)}
        out["advice"] = "short texts: character n-gram or word-vector embeddings hold up better than bag-of-words LSA" if (words < 10).mean() > 0.5 else "documents are long enough for LSA / TF-IDF based embeddings"
    return out


@tool
def check_raw_feature_baseline(data: str, label_column: str, columns: list[str] | None = None, drop: list[str] | None = None, task: str = "classification", max_rows: int = 10000, seed: int = 42) -> dict:
    """The score to beat: cross-validated linear and k-NN probes of the *raw* (encoded, standardised) features against a label, with the majority-class rate beside them. A learned embedding that cannot beat this has lost information; one that comes close with a fraction of the dimensions has compressed well.

    Args:
        label_column: Column of known labels or targets, used only to probe.
        task: classification or regression.
        max_rows: Rows sampled.
    """
    P = prepare(data, columns, list(drop or []) + [label_column], True, label_column, max_rows, seed)
    y = P.truth if task == "classification" else pd.to_numeric(P.frame[label_column], errors="coerce").fillna(0).to_numpy(float)
    out = {"rows": len(P.Z), "dimensions": int(P.Z.shape[1]), "probe_on_raw_features": probe(P.Z, y, task, seed)}
    if task == "classification":
        out["majority_class_rate"] = round(float(np.bincount(y).max() / len(y)), 4)
    out["reading"] = "raw-feature probe scores are the ceiling a compressed embedding is judged against, and the floor a *useful* learned one must clear"
    return out


@tool
def check_cooccurrence_data(data: str, group_column: str, item_column: str, min_count: int = 5) -> dict:
    """Is there enough co-occurrence to learn item / token vectors from? Number of groups (baskets, sessions), distinct items, items seen at least `min_count` times, average group size, and the density of the co-occurrence matrix — sparse, tiny data gives vectors dominated by noise.

    Args:
        group_column: Column identifying the basket / session / sequence.
        item_column: Column holding the item or event.
        min_count: Items seen fewer times are dropped by the embedding tools.
    """
    frame = load_frame(data)
    require_columns(frame, [group_column, item_column])
    sizes = frame.groupby(group_column)[item_column].nunique()
    counts = frame.groupby(item_column)[group_column].nunique()
    keep = counts[counts >= min_count]
    pairs = frame[frame[item_column].isin(keep.index)].groupby(group_column)[item_column].apply(lambda s: len(set(s)) * (len(set(s)) - 1) / 2).sum()
    return {"groups": int(len(sizes)), "distinct_items": int(len(counts)), "items_with_min_count": int(len(keep)), "mean_items_per_group": round(float(sizes.mean()), 2), "groups_with_one_item": int((sizes <= 1).sum()),
            "co_occurring_pairs": int(pairs), "pairs_per_kept_item": round(float(pairs / max(len(keep), 1)), 1),
            "verdict": "enough co-occurrence to learn from" if pairs / max(len(keep), 1) > 20 else "very sparse: vectors will be mostly noise — lower min_count, merge rare items into categories, or collect more data"}


@tool
def check_graph_structure(data: str, source_column: str, target_column: str) -> dict:
    """The shape of a graph before embedding it: nodes, edges, density, connected components (and the share of nodes in the largest), degree distribution, isolated nodes and the clustering of the graph — random-walk and spectral embeddings only describe nodes that are connected to something.

    Args:
        source_column: Column holding the edge's source node.
        target_column: Column holding the edge's target node.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    frame = load_frame(data)
    require_columns(frame, [source_column, target_column])
    e = frame[[source_column, target_column]].dropna().astype(str)
    nodes = sorted(set(e[source_column]) | set(e[target_column]))
    ix = {n: i for i, n in enumerate(nodes)}
    a, b = e[source_column].map(ix).to_numpy(), e[target_column].map(ix).to_numpy()
    A = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(nodes),) * 2)
    n_comp, comp = connected_components(A, directed=False)
    deg = np.bincount(np.concatenate([a, b]), minlength=len(nodes))
    sizes = np.bincount(comp)
    return {"nodes": len(nodes), "edges": int(len(e)), "density": round(float(2 * len(e) / max(len(nodes) * (len(nodes) - 1), 1)), 5), "components": int(n_comp), "largest_component_share": round(float(sizes.max() / len(nodes)), 4),
            "degree": {"mean": round(float(deg.mean()), 2), "median": float(np.median(deg)), "max": int(deg.max())}, "isolated_nodes": int((deg == 0).sum()), "self_loops": int((a == b).sum()), "duplicate_edges": int(e.duplicated().sum()),
            "advice": "several components: nodes in different components share no walks, so their vectors are not comparable — embed the largest component or each separately" if n_comp > 1 and sizes.max() / len(nodes) < 0.9 else "one dominant component: fine for random-walk and spectral embeddings"}
