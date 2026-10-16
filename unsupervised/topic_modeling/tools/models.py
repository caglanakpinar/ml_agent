"""Fit a topic model: probabilistic (LDA), matrix-factorisation (NMF, LSA), clustering-based with class TF-IDF, hierarchical, seeded by keywords, or BERTopic."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import diags

from ml_core.data import save_frame
from ml_core.ml import save_model
from ml_core.tooling import tool
from unsupervised.topic_modeling.tools._common import (
    TopicModel,
    ctfidf,
    load_corpus,
    normalise_rows,
    summarise_model,
    vectorise,
)


def _finish(label, tm, X, vec, frame, texts, dt, label_column, id_column, output_path, model_path, extra=None):
    from sklearn.feature_extraction.text import CountVectorizer

    binary = CountVectorizer(vocabulary=vec.vocabulary_, binary=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b").fit_transform(texts)
    labels = pd.factorize(frame[label_column].astype(str))[0] if label_column else None
    out = {"model": label, "documents": len(texts), "vocabulary": len(vec.vocabulary_), **summarise_model(tm, dt, binary, vec.vocabulary_, labels), **(extra or {})}
    table = pd.DataFrame(dt, columns=[f"topic_{i}" for i in range(dt.shape[1])])
    table.insert(0, "dominant_topic", dt.argmax(axis=1))
    table.insert(0, "dominant_share", dt.max(axis=1))
    if id_column and id_column in frame.columns:
        table.insert(0, id_column, frame[id_column].to_numpy())
    out["written_to"] = save_frame(table, output_path, f"doc_topics_{tm.kind}.csv") if output_path else None
    out["model_path"] = save_model(tm, model_path)
    return out


def _prep(data, text_column, id_column, label_column, kind, min_df, max_df, ngram_max, stop_words, max_rows, seed):
    frame, texts = load_corpus(data, text_column, id_column, label_column, None, max_rows, seed)
    vec, X = vectorise(texts, kind, min_df, max_df, ngram_max, 20000, "english" if stop_words == "english" else None)
    return frame, texts, vec, X


@tool
def fit_lda(data: str, text_column: str, n_topics: int = 8, id_column: str | None = None, label_column: str | None = None, doc_topic_prior: float | None = None, topic_word_prior: float | None = None, min_df: int = 3, max_df: float = 0.5,
            ngram_max: int = 1, stop_words: Literal["english", "none"] = "english", max_iter: int = 30, max_rows: int = 20000, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Latent Dirichlet allocation: a probabilistic model in which each document is a mixture of topics and each topic a distribution over words, fitted by variational inference on word *counts*. The classic; gives soft document-topic shares and a perplexity, works best on longer documents (50+ words) and needs the stop-words and rare words trimmed.

    Args:
        text_column: Column holding each document's text.
        n_topics: Number of topics.
        id_column: Column carried to the document-topic file.
        label_column: Known categories, used only to report agreement with the topics.
        doc_topic_prior: Dirichlet prior on a document's topic mixture (alpha); smaller gives documents fewer topics. 1/n_topics when omitted.
        topic_word_prior: Dirichlet prior on a topic's words (eta); smaller gives sparser topics. 1/n_topics when omitted.
        min_df: Terms in fewer documents are dropped.
        max_df: Terms in more than this share of documents are dropped.
        ngram_max: Longest word n-gram.
        stop_words: english or none.
        max_iter: Variational iterations.
        max_rows: Documents sampled.
        output_path: Where to write document-topic shares.
        model_path: Where to save the model.
    """
    from sklearn.decomposition import LatentDirichletAllocation

    frame, texts, vec, X = _prep(data, text_column, id_column, label_column, "counts", min_df, max_df, ngram_max, stop_words, max_rows, seed)
    lda = LatentDirichletAllocation(n_components=n_topics, doc_topic_prior=doc_topic_prior, topic_word_prior=topic_word_prior, learning_method="batch", max_iter=max_iter, random_state=seed, n_jobs=-1).fit(X)
    tm = TopicModel("lda", vec, lda, lda.components_, vec.get_feature_names_out())
    return _finish(f"lda(k={n_topics})", tm, X, vec, frame, texts, normalise_rows(lda.transform(X)), label_column, id_column, output_path, model_path, {"perplexity": round(float(lda.perplexity(X)), 2), "log_likelihood_per_word": round(float(lda.score(X) / X.sum()), 4)})


@tool
def fit_nmf_topics(data: str, text_column: str, n_topics: int = 8, id_column: str | None = None, label_column: str | None = None, sparsity: float = 0.0, min_df: int = 3, max_df: float = 0.5, ngram_max: int = 1, stop_words: Literal["english", "none"] = "english",
                   max_rows: int = 20000, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Non-negative matrix factorisation of the TF-IDF matrix: documents ≈ non-negative mixtures of non-negative topic vectors. Faster and deterministic compared with LDA, usually gives crisper, more coherent topics on short or medium texts, and an L1 `sparsity` makes topics use fewer words.

    Args:
        text_column: Column holding each document's text.
        n_topics: Number of topics.
        id_column: Column carried to the document-topic file.
        label_column: Known categories, used only to report agreement with the topics.
        sparsity: L1 regularisation share (0–1) on the factors; larger gives sparser topics.
        min_df: Terms in fewer documents are dropped.
        max_df: Terms in more than this share of documents are dropped.
        ngram_max: Longest word n-gram.
        stop_words: english or none.
        max_rows: Documents sampled.
        output_path: Where to write document-topic shares.
        model_path: Where to save the model.
    """
    from sklearn.decomposition import NMF

    frame, texts, vec, X = _prep(data, text_column, id_column, label_column, "tfidf", min_df, max_df, ngram_max, stop_words, max_rows, seed)
    nmf = NMF(n_components=n_topics, init="nndsvda", l1_ratio=0.5 if sparsity else 0.0, alpha_W=sparsity, alpha_H=sparsity, max_iter=500, random_state=seed).fit(X)
    tm = TopicModel("nmf", vec, nmf, nmf.components_, vec.get_feature_names_out())
    return _finish(f"nmf(k={n_topics})", tm, X, vec, frame, texts, normalise_rows(nmf.transform(X)), label_column, id_column, output_path, model_path, {"reconstruction_error": round(float(nmf.reconstruction_err_), 4)})


@tool
def fit_lsa_topics(data: str, text_column: str, n_topics: int = 8, id_column: str | None = None, label_column: str | None = None, min_df: int = 3, max_df: float = 0.5, ngram_max: int = 1, stop_words: Literal["english", "none"] = "english", max_rows: int = 20000,
                   seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Latent semantic analysis topics: truncated SVD of the TF-IDF matrix. The fastest model, with an exact variance accounting — but its components are signed (words pull in both directions), so topics are less crisp than NMF's; use it as a quick structural look and as the base of clustering-based topics.

    Args:
        text_column: Column holding each document's text.
        n_topics: Number of components.
        id_column: Column carried to the document-topic file.
        label_column: Known categories, used only to report agreement with the topics.
        min_df: Terms in fewer documents are dropped.
        max_df: Terms in more than this share of documents are dropped.
        ngram_max: Longest word n-gram.
        stop_words: english or none.
        max_rows: Documents sampled.
        output_path: Where to write document-topic shares.
        model_path: Where to save the model.
    """
    from sklearn.decomposition import TruncatedSVD

    frame, texts, vec, X = _prep(data, text_column, id_column, label_column, "tfidf", min_df, max_df, ngram_max, stop_words, max_rows, seed)
    svd = TruncatedSVD(n_topics, random_state=seed).fit(X)
    tm = TopicModel("lsa", vec, svd, np.abs(svd.components_), vec.get_feature_names_out())
    return _finish(f"lsa(k={n_topics})", tm, X, vec, frame, texts, normalise_rows(np.abs(svd.transform(X))), label_column, id_column, output_path, model_path, {"variance_explained": round(float(svd.explained_variance_ratio_.sum()), 4)})


def _cluster_topics(kind, label, data, text_column, n_topics, id_column, label_column, min_df, max_df, ngram_max, stop_words, n_components, algorithm, max_rows, seed, output_path, model_path):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer

    frame, texts, vec, X = _prep(data, text_column, id_column, label_column, "tfidf", min_df, max_df, ngram_max, stop_words, max_rows, seed)
    svd = TruncatedSVD(min(n_components, X.shape[1] - 1), random_state=seed).fit(X)
    Z = svd.transform(X)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    if algorithm == "hdbscan":
        from sklearn.cluster import HDBSCAN

        size = max(10, len(Z) // (n_topics * 3))
        try:
            raw = HDBSCAN(min_cluster_size=size, copy=True).fit_predict(Z)
        except TypeError:  # versions before the `copy` argument
            raw = HDBSCAN(min_cluster_size=size).fit_predict(Z)
        raw = np.where(raw < 0, raw.max() + 1 if raw.max() >= 0 else 0, raw)  # noise gets its own cluster so every document has a topic
        km_labels = pd.factorize(raw)[0]
    else:
        km_labels = KMeans(n_topics, n_init=5, random_state=seed).fit_predict(Z)
    k = int(km_labels.max() + 1)
    counts = CountVectorizer(vocabulary=vec.vocabulary_, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b").fit_transform(texts)
    by_topic = np.vstack([np.asarray(counts[km_labels == c].sum(axis=0)).ravel() for c in range(k)])
    tw = ctfidf(by_topic)
    centroids = np.vstack([Z[km_labels == c].mean(axis=0) for c in range(k)])
    centroids = centroids / np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9)
    tm = TopicModel("kmeans", vec, None, tw, vec.get_feature_names_out(), centroids, svd)
    dt = tm.transform(texts)
    dt[np.arange(len(dt)), km_labels] = np.maximum(dt[np.arange(len(dt)), km_labels], dt.max(axis=1))  # a document's own cluster is its dominant topic
    return _finish(f"{label}(k={k})", tm, X, vec, frame, texts, normalise_rows(dt), label_column, id_column, output_path, model_path, {"embedding_dimensions": int(Z.shape[1]), "clusters": k})


@tool
def fit_kmeans_topics(data: str, text_column: str, n_topics: int = 8, id_column: str | None = None, label_column: str | None = None, min_df: int = 3, max_df: float = 0.5, ngram_max: int = 1, stop_words: Literal["english", "none"] = "english",
                      n_components: int = 50, max_rows: int = 20000, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Clustering-based topics: embed the documents (TF-IDF → SVD), cluster them with k-means, then describe each cluster by its most *distinctive* words using class-based TF-IDF (the BERTopic recipe). Every document gets exactly one topic — a hard partition rather than a mixture — which suits short texts (tweets, titles, tickets) where LDA's mixtures are unstable.

    Args:
        text_column: Column holding each document's text.
        n_topics: Number of clusters / topics.
        id_column: Column carried to the document-topic file.
        label_column: Known categories, used only to report agreement with the topics.
        min_df: Terms in fewer documents are dropped.
        max_df: Terms in more than this share of documents are dropped.
        ngram_max: Longest word n-gram.
        stop_words: english or none.
        n_components: SVD dimensions used to embed the documents.
        max_rows: Documents sampled.
        output_path: Where to write document-topic shares.
        model_path: Where to save the model.
    """
    return _cluster_topics("kmeans", "kmeans_ctfidf", data, text_column, n_topics, id_column, label_column, min_df, max_df, ngram_max, stop_words, n_components, "kmeans", max_rows, seed, output_path, model_path)


@tool
def fit_density_cluster_topics(data: str, text_column: str, expected_topics: int = 8, id_column: str | None = None, label_column: str | None = None, min_df: int = 3, max_df: float = 0.5, ngram_max: int = 1, stop_words: Literal["english", "none"] = "english",
                               n_components: int = 30, max_rows: int = 15000, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Density-based topics: documents are embedded (TF-IDF → SVD) and grouped with HDBSCAN, so the *number of topics is discovered* rather than chosen and documents that belong to no cluster are set aside as their own group, then each cluster is described by class-based TF-IDF. The right model when the number of themes is unknown and the corpus contains a lot of off-topic text.

    Args:
        text_column: Column holding each document's text.
        expected_topics: Rough number of topics wanted; only sets the smallest cluster size.
        id_column: Column carried to the document-topic file.
        label_column: Known categories, used only to report agreement with the topics.
        min_df: Terms in fewer documents are dropped.
        max_df: Terms in more than this share of documents are dropped.
        ngram_max: Longest word n-gram.
        stop_words: english or none.
        n_components: SVD dimensions used to embed the documents.
        max_rows: Documents sampled.
        output_path: Where to write document-topic shares.
        model_path: Where to save the model.
    """
    return _cluster_topics("embedding_clusters", "hdbscan_ctfidf", data, text_column, expected_topics, id_column, label_column, min_df, max_df, ngram_max, stop_words, n_components, "hdbscan", max_rows, seed, output_path, model_path)


@tool
def fit_seeded_topics(data: str, text_column: str, seed_topics: dict, id_column: str | None = None, label_column: str | None = None, extra_topics: int = 2, weight: float = 3.0, min_df: int = 3, max_df: float = 0.5, stop_words: Literal["english", "none"] = "english",
                      max_rows: int = 20000, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Guided (seeded) topics: you name topics and give a few seed words for each ({"sport": ["goal","match"], "finance": ["bank","loan"]}); the seed terms are up-weighted and the NMF topics are initialised on them, so they stay anchored to those themes and are filled in with the corpus' own vocabulary, plus a few free topics for everything else. Use when the themes are known in advance.

    Args:
        text_column: Column holding each document's text.
        seed_topics: Topic name → list of seed words.
        id_column: Column carried to the document-topic file.
        label_column: Known categories, used only to report agreement with the topics.
        extra_topics: Free topics added beside the seeded ones.
        weight: Factor by which seed words are up-weighted in the TF-IDF matrix.
        min_df: Terms in fewer documents are dropped.
        max_df: Terms in more than this share of documents are dropped.
        stop_words: english or none.
        max_rows: Documents sampled.
        output_path: Where to write document-topic shares.
        model_path: Where to save the model.
    """
    from sklearn.decomposition import NMF

    frame, texts, vec, X = _prep(data, text_column, id_column, label_column, "tfidf", min_df, max_df, 1, stop_words, max_rows, seed)
    vocab = vec.vocabulary_
    names = list(seed_topics)
    boost = np.ones(X.shape[1])
    H0 = np.full((len(names) + extra_topics, X.shape[1]), 1e-3)
    found = {}
    for t, (name, words) in enumerate(seed_topics.items()):
        hit = [w.lower() for w in words if w.lower() in vocab]
        found[name] = hit
        for w in hit:
            boost[vocab[w]] = weight
            H0[t, vocab[w]] = 1.0
    Xb = X @ diags(boost)
    W0 = np.full((X.shape[0], H0.shape[0]), 0.1)
    nmf = NMF(n_components=H0.shape[0], init="custom", max_iter=500, random_state=seed)
    W = nmf.fit_transform(Xb, W=W0, H=H0)
    tm = TopicModel("nmf", vec, _SeededProjector(nmf, boost), nmf.components_, vec.get_feature_names_out())
    out = _finish(f"seeded_nmf({len(names)}+{extra_topics})", tm, X, vec, frame, texts, normalise_rows(W), label_column, id_column, output_path, model_path, {"seed_words_found": found, "seed_words_missing": {n: [w for w in ws if w.lower() not in vocab] for n, ws in seed_topics.items()}})
    for t, name in enumerate(names):
        out["topic_summaries"] = [dict(s, name=name) if s["topic"] == t else s for s in out["topic_summaries"]]
    return out


class _SeededProjector:
    def __init__(self, nmf, boost):
        self.nmf, self.boost = nmf, boost

    def transform(self, X):
        return self.nmf.transform(X @ diags(self.boost))


@tool
def fit_hierarchical_topics(data: str, text_column: str, n_topics: int = 12, id_column: str | None = None, label_column: str | None = None, linkage: Literal["ward", "average"] = "ward", min_df: int = 3, max_df: float = 0.5, stop_words: Literal["english", "none"] = "english",
                            n_components: int = 50, max_rows: int = 5000, seed: int = 42, output_path: str | None = None) -> dict:
    """A topic *tree*: documents are embedded (TF-IDF → SVD) and merged bottom-up; cutting at several levels gives broad themes that split into narrower ones, each described by class-based TF-IDF — the view for a large corpus where 'how many topics?' has no single answer.

    Args:
        text_column: Column holding each document's text.
        n_topics: Number of leaf topics.
        id_column: Column carried to the output.
        label_column: Known categories, used only to report agreement.
        linkage: ward or average.
        n_components: SVD dimensions used to embed the documents.
        max_rows: Documents sampled (the merge tree is quadratic).
        output_path: Where to write the document-to-leaf assignments.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import CountVectorizer

    frame, texts, vec, X = _prep(data, text_column, id_column, label_column, "tfidf", min_df, max_df, 1, stop_words, max_rows, seed)
    Z = TruncatedSVD(min(n_components, X.shape[1] - 1), random_state=seed).fit_transform(X)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    levels = sorted({max(2, n_topics // 4), max(3, n_topics // 2), n_topics})
    counts = CountVectorizer(vocabulary=vec.vocabulary_, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b").fit_transform(texts)
    terms = vec.get_feature_names_out()
    tree, labels_by_level = {}, {}
    for k in levels:
        lab = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(Z)
        labels_by_level[k] = lab
        tw = ctfidf(np.vstack([np.asarray(counts[lab == c].sum(axis=0)).ravel() for c in range(k)]))
        tree[k] = [{"topic": c, "documents": int((lab == c).sum()), "top_words": [str(terms[i]) for i in np.argsort(-tw[c])[:8]]} for c in range(k)]
    leaf = labels_by_level[levels[-1]]
    parent = {c: int(pd.Series(labels_by_level[levels[-2]][leaf == c]).mode().iloc[0]) for c in range(levels[-1])} if len(levels) > 1 else {}
    out = {"model": f"agglomerative_ctfidf(levels={levels})", "documents": len(texts), "levels": {str(k): v for k, v in tree.items()}, "leaf_to_parent": parent}
    if label_column:
        from sklearn.metrics import normalized_mutual_info_score

        out["leaf_agreement_nmi"] = round(float(normalized_mutual_info_score(pd.factorize(frame[label_column].astype(str))[0], leaf)), 4)
    table = pd.DataFrame({f"level_{k}": v for k, v in labels_by_level.items()})
    if id_column and id_column in frame.columns:
        table.insert(0, id_column, frame[id_column].to_numpy())
    out["written_to"] = save_frame(table, output_path, "topic_tree_assignments.csv") if output_path else None
    return out


@tool(requires=("bertopic",))
def fit_bertopic(data: str, text_column: str, id_column: str | None = None, min_topic_size: int = 10, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", nr_topics: int | None = None, max_rows: int = 10000, seed: int = 42,
                 output_path: str | None = None) -> dict:
    """BERTopic: sentence-transformer embeddings → UMAP → HDBSCAN → class-based TF-IDF. State of the art for short or varied texts; the number of topics is discovered and outlier documents get topic −1. Needs the bertopic package and a model download.

    Args:
        text_column: Column holding each document's text.
        id_column: Column carried to the output.
        min_topic_size: Smallest topic.
        embedding_model: Hugging Face id of the sentence encoder.
        nr_topics: Reduce to this many topics after fitting; keep the discovered number when omitted.
        max_rows: Documents sampled.
        output_path: Where to write the document-topic assignments.
    """
    from bertopic import BERTopic

    frame, texts = load_corpus(data, text_column, id_column, None, None, max_rows, seed)
    bt = BERTopic(embedding_model=embedding_model, min_topic_size=min_topic_size, nr_topics=nr_topics, verbose=False, calculate_probabilities=False)
    topics, _ = bt.fit_transform(texts)
    info = bt.get_topic_info()
    table = pd.DataFrame({"topic": topics})
    if id_column and id_column in frame.columns:
        table.insert(0, id_column, frame[id_column].to_numpy())
    return {"model": "bertopic", "documents": len(texts), "topics_found": int((info["Topic"] >= 0).sum()), "outlier_documents": int((np.array(topics) == -1).sum()),
            "topic_summaries": [{"topic": int(r.Topic), "size": int(r.Count), "top_words": [w for w, _ in bt.get_topic(int(r.Topic))[:8]]} for r in info.head(25).itertuples()], "written_to": save_frame(table, output_path, "bertopic_assignments.csv") if output_path else None}
