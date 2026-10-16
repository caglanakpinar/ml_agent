"""Representations of entities that have no features of their own — items, users, tokens, nodes — learned from how they co-occur: baskets, event sequences and graphs (PPMI + SVD, the matrix-factorisation form of word2vec / DeepWalk)."""

from __future__ import annotations


import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.ml import save_model
from ml_core.tooling import tool
from unsupervised.representation_learning.tools._common import cooccurrence, effective_rank, isotropy, ppmi_vectors


class EntityVectors:
    """Name → vector lookup with cosine nearest-neighbour search."""

    def __init__(self, names: list[str], vectors: np.ndarray):
        self.names, self.vectors = names, vectors
        self.index = {n: i for i, n in enumerate(names)}

    def neighbours(self, name: str, k: int = 10) -> list[tuple[str, float]]:
        v = self.vectors / np.maximum(np.linalg.norm(self.vectors, axis=1, keepdims=True), 1e-9)
        sims = v @ v[self.index[name]]
        return [(self.names[i], float(sims[i])) for i in np.argsort(-sims)[1 : k + 1]]

    def transform(self, names: list[str]) -> np.ndarray:
        return np.stack([self.vectors[self.index[n]] if n in self.index else np.zeros(self.vectors.shape[1]) for n in names])


def _finish(label, names, V, model_path, output_path, id_name, extra, probe_labels=None):
    out = {"method": label, "entities": len(names), "dimensions": int(V.shape[1]), "effective_rank": round(effective_rank(V), 2), "isotropy_mean_abs_cosine": round(isotropy(V), 4), **extra}
    if probe_labels is not None:
        from unsupervised.representation_learning.tools._common import neighbour_purity, probe

        y = np.asarray([probe_labels.get(n, "?") for n in names])
        keep = y != "?"
        if keep.sum() > 20 and len(set(y[keep])) > 1:
            codes = pd.factorize(y[keep])[0]
            out["probe"], out["neighbour_label_purity"] = probe(V[keep], codes), round(neighbour_purity(V[keep], codes), 4)
    table = pd.DataFrame(V, columns=[f"emb_{i + 1}" for i in range(V.shape[1])])
    table.insert(0, id_name, names)
    out["written_to"] = save_frame(table, output_path, f"embedding_{label.split('(')[0]}.csv") if output_path else None
    out["model_path"] = save_model(EntityVectors(names, V), model_path)
    return out


@tool
def embed_items_from_baskets(data: str, transaction_column: str, item_column: str, n_components: int = 32, min_count: int = 5, category_column: str | None = None, shift: float = 0.0, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Item embeddings from baskets (item2vec as matrix factorisation): items bought in the same basket co-occur; shifted positive PMI of the co-occurrence matrix, factorised by SVD, gives each item a vector where *items used in the same way are close*. With a category column the vectors are probed: do an item's nearest neighbours share its category?

    Args:
        transaction_column: Column identifying the basket / session.
        item_column: Column holding the item.
        n_components: Embedding size.
        min_count: Items in fewer baskets than this are dropped.
        category_column: Column giving each item's category, used only to probe the embedding.
        shift: PMI shift (log of the negative-sampling count); larger gives sparser, more selective vectors.
        output_path: Where to write the item vectors.
        model_path: Where to save the vectors for nearest-neighbour search.
    """
    frame = load_frame(data)
    require_columns(frame, [transaction_column, item_column] + ([category_column] if category_column else []))
    frame = frame.dropna(subset=[item_column]).assign(**{item_column: frame[item_column].astype(str)})
    counts = frame.groupby(item_column)[transaction_column].nunique()
    keep = set(counts[counts >= min_count].index)
    baskets = [[i for i in g[item_column] if i in keep] for _, g in frame.groupby(transaction_column)]
    baskets = [sorted(set(b)) for b in baskets if len(b) > 1]
    C, vocab = cooccurrence(baskets, window=10_000, distance_weighted=False)  # a basket is a set: every pair counts equally
    V = ppmi_vectors(C, min(n_components, len(vocab) - 2), shift)
    cats = dict(zip(frame[item_column], frame[category_column].astype(str))) if category_column else None
    return _finish(f"item_ppmi_svd({V.shape[1]})", vocab, V, model_path, output_path, "item", {"baskets_used": len(baskets), "items_dropped_below_min_count": int((counts < min_count).sum())}, cats)


@tool
def embed_tokens_from_sequences(data: str, sequence_column: str, token_column: str, order_column: str | None = None, n_components: int = 32, window: int = 5, min_count: int = 5, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Token embeddings from event sequences (page views, log events, product journeys): tokens that appear near each other, weighted by closeness within a window, get close vectors — PPMI + SVD, the factorisation form of skip-gram. Use the vectors as features for sessions, or to find events that play the same role.

    Args:
        sequence_column: Column identifying the sequence (session, user).
        token_column: Column holding the event / token.
        order_column: Column that orders events within a sequence; the file order when omitted.
        n_components: Embedding size.
        window: Events either side counted as context.
        min_count: Tokens rarer than this are dropped.
        output_path: Where to write the vectors.
        model_path: Where to save the vectors.
    """
    frame = load_frame(data)
    require_columns(frame, [sequence_column, token_column])
    frame = frame.assign(**{token_column: frame[token_column].astype(str)})
    seqs = [list(g.sort_values(order_column)[token_column]) if order_column else list(g[token_column]) for _, g in frame.groupby(sequence_column, sort=False)]
    from collections import Counter

    c = Counter(t for s in seqs for t in s)
    keep = {t for t, n in c.items() if n >= min_count}
    seqs = [[t for t in s if t in keep] for s in seqs]
    C, vocab = cooccurrence([s for s in seqs if len(s) > 1], window)
    V = ppmi_vectors(C, min(n_components, len(vocab) - 2))
    return _finish(f"sequence_ppmi_svd({V.shape[1]},window={window})", vocab, V, model_path, output_path, "token", {"sequences": len(seqs), "mean_length": round(float(np.mean([len(s) for s in seqs])), 2)})


@tool
def embed_graph_deepwalk(data: str, source_column: str, target_column: str, n_components: int = 32, walks_per_node: int = 10, walk_length: int = 20, window: int = 4, directed: bool = False, label_file: str | None = None, label_node_column: str | None = None,
                         label_column: str | None = None, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """DeepWalk as matrix factorisation: random walks on the graph are treated as sentences; nodes that appear near each other on walks get close vectors (PPMI + SVD of the walk co-occurrence). Nodes with similar neighbourhoods — not only directly linked ones — end up near each other. With a file of node labels the embedding is probed (does a node's neighbourhood predict its label?).

    Args:
        source_column: Column holding the edge's source node.
        target_column: Column holding the edge's target node.
        n_components: Embedding size.
        walks_per_node: Random walks started at every node.
        walk_length: Steps per walk.
        window: Context window along each walk.
        directed: Follow edges only from source to target.
        label_file: Optional CSV with a node label, only to probe the embedding.
        label_node_column: Node column of the label file.
        label_column: Label column of the label file.
        output_path: Where to write the node vectors.
        model_path: Where to save the vectors.
    """
    frame = load_frame(data)
    require_columns(frame, [source_column, target_column])
    edges = frame[[source_column, target_column]].dropna().astype(str).to_numpy()
    nodes = sorted(set(edges.ravel()))
    ix = {n: i for i, n in enumerate(nodes)}
    adj: list[list[int]] = [[] for _ in nodes]
    for a, b in edges:
        adj[ix[a]].append(ix[b])
        if not directed:
            adj[ix[b]].append(ix[a])
    rng = np.random.default_rng(seed)
    walks = []
    for _ in range(walks_per_node):
        for start in rng.permutation(len(nodes)):
            walk, cur = [str(start)], start
            for _ in range(walk_length - 1):
                if not adj[cur]:
                    break
                cur = adj[cur][rng.integers(len(adj[cur]))]
                walk.append(str(cur))
            walks.append(walk)
    C, vocab = cooccurrence(walks, window, [str(i) for i in range(len(nodes))])
    V = ppmi_vectors(C, min(n_components, len(nodes) - 2))
    labels = None
    if label_file and label_node_column and label_column:
        lab = load_frame(label_file)
        labels = dict(zip(lab[label_node_column].astype(str), lab[label_column].astype(str)))
    return _finish(f"deepwalk_ppmi_svd({V.shape[1]})", nodes, V, model_path, output_path, "node", {"nodes": len(nodes), "edges": len(edges), "walks": len(walks), "isolated_nodes": int(sum(1 for a in adj if not a))}, labels)


@tool
def embed_graph_spectral(data: str, source_column: str, target_column: str, n_components: int = 16, weight_column: str | None = None, output_path: str | None = None, model_path: str | None = None, seed: int = 42) -> dict:
    """Spectral graph embedding: the leading eigenvectors of the normalised graph Laplacian place connected, tightly knit groups of nodes together — the classical, deterministic alternative to random-walk embeddings, and the basis of spectral clustering. Reports the eigengap, which suggests how many communities the graph has.

    Args:
        source_column: Column holding the edge's source node.
        target_column: Column holding the edge's target node.
        n_components: Embedding size.
        weight_column: Optional edge weight column.
        output_path: Where to write the node vectors.
        model_path: Where to save the vectors.
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import eigsh

    frame = load_frame(data)
    require_columns(frame, [source_column, target_column])
    nodes = sorted(set(frame[source_column].astype(str)) | set(frame[target_column].astype(str)))
    ix = {n: i for i, n in enumerate(nodes)}
    w = frame[weight_column].astype(float).to_numpy() if weight_column else np.ones(len(frame))
    a = np.array([ix[x] for x in frame[source_column].astype(str)])
    b = np.array([ix[x] for x in frame[target_column].astype(str)])
    A = coo_matrix((np.concatenate([w, w]), (np.concatenate([a, b]), np.concatenate([b, a]))), shape=(len(nodes),) * 2).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    Dinv = diags(1 / np.sqrt(np.maximum(deg, 1e-9)))
    L = Dinv @ A @ Dinv
    k = min(n_components + 1, len(nodes) - 2)
    vals, vecs = eigsh(L, k=k, which="LA")
    order = np.argsort(-vals)
    vals, vecs = vals[order], vecs[:, order]
    V = (Dinv @ vecs)[:, 1 : n_components + 1] if k > 1 else vecs
    gaps = -np.diff(vals)
    return _finish(f"spectral({V.shape[1]})", nodes, V, model_path, output_path, "node", {"nodes": len(nodes), "edges": int(len(frame)), "eigenvalues": np.round(vals[:12], 4).tolist(), "largest_eigengap_after": int(np.argmax(gaps[1:]) + 2) if len(gaps) > 2 else None})


@tool
def nearest_entities(model_path: str, entity: str, k: int = 10) -> dict:
    """The entities closest to a given one (item, token or node) in a saved vector model, by cosine similarity — the check that learned vectors reflect real relationships (neighbours of a product should be substitutes or complements), and the basis for 'similar items' recommendations.

    Args:
        model_path: Path of the saved vector model (joblib) from an `embed_items_*`, `embed_tokens_*` or `embed_graph_*` tool.
        entity: The item / token / node name.
        k: Neighbours returned.
    """
    import joblib

    m = joblib.load(model_path)
    if not hasattr(m, "neighbours") or not hasattr(m, "names"):
        raise ValueError("this model is not an entity-vector model")
    if entity not in m.index:
        raise KeyError(f"{entity!r} is not among the {len(m.names)} entities")
    return {"entity": entity, "neighbours": [{"entity": n, "cosine": round(s, 4)} for n, s in m.neighbours(entity, k)]}
