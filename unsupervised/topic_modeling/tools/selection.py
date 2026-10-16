"""Choosing and comparing topic models: the number of topics (coherence, diversity, perplexity), stability across seeds, LDA priors, and the model families against each other."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ml_core.tooling import tool
from unsupervised.topic_modeling.tools._common import (
    coherence,
    diversity,
    load_corpus,
    normalise_rows,
    stability,
    vectorise,
)


def _fit(kind: str, X, k: int, seed: int, **kw):
    from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

    if kind == "lda":
        m = LatentDirichletAllocation(n_components=k, max_iter=kw.get("max_iter", 15), random_state=seed, doc_topic_prior=kw.get("alpha"), topic_word_prior=kw.get("eta"), n_jobs=-1).fit(X)
        return m.components_, normalise_rows(m.transform(X)), m
    if kind == "nmf":
        m = NMF(k, init="nndsvda", max_iter=300, random_state=seed).fit(X)
        return m.components_, normalise_rows(m.transform(X)), m
    m = TruncatedSVD(k, random_state=seed).fit(X)
    return np.abs(m.components_), normalise_rows(np.abs(m.transform(X))), m


def _binary(vec, texts):
    from sklearn.feature_extraction.text import CountVectorizer

    return CountVectorizer(vocabulary=vec.vocabulary_, binary=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b").fit_transform(texts)


def _score(tw, terms, binary, vocab, n=10):
    tops = [[str(terms[i]) for i in np.argsort(-r)[:n]] for r in tw]
    return float(np.mean(coherence(tops, binary, vocab, "npmi", n))), float(np.mean(coherence(tops, binary, vocab, "umass", n))), diversity(tops, n)


@tool
def choose_n_topics(data: str, text_column: str, model: Literal["nmf", "lda", "lsa"] = "nmf", k_values: list[int] | None = None, min_df: int = 3, max_df: float = 0.5, max_rows: int = 8000, seed: int = 42) -> dict:
    """The decision tool for 'how many topics': fit the model for several values of k and report NPMI and UMass coherence, topic diversity and (LDA) perplexity for each. Coherence usually peaks and then falls as topics split into fragments; diversity falls when topics start repeating each other. Pick the k at the coherence peak that still has high diversity, and check it with a person.

    Args:
        text_column: Column holding each document's text.
        model: nmf, lda or lsa.
        k_values: Numbers of topics to try; defaults to 3, 5, 8, 12, 16, 24.
        min_df: Terms in fewer documents are dropped.
        max_df: Terms in more than this share of documents are dropped.
        max_rows: Documents sampled.
    """
    frame, texts = load_corpus(data, text_column, None, None, None, max_rows, seed)
    vec, X = vectorise(texts, "counts" if model == "lda" else "tfidf", min_df, max_df)
    binary, terms = _binary(vec, texts), vec.get_feature_names_out()
    rows = []
    for k in k_values or [3, 5, 8, 12, 16, 24]:
        if k >= X.shape[1] - 1:
            continue
        tw, dt, m = _fit(model, X, k, seed)
        npmi, umass, div = _score(tw, terms, binary, vec.vocabulary_)
        rows.append({"k": k, "npmi_coherence": round(npmi, 4), "umass_coherence": round(umass, 3), "diversity": round(div, 3), **({"perplexity": round(float(m.perplexity(X)), 1)} if model == "lda" else {})})
    best = max(rows, key=lambda r: r["npmi_coherence"] + 0.3 * r["diversity"])
    return {"model": model, "documents": len(texts), "sweep": rows, "suggested_k": best["k"], "reading": "perplexity falls monotonically with k and is a poor guide; trust coherence and diversity — and read the topics before settling"}


@tool
def topic_stability(data: str, text_column: str, n_topics: int = 8, model: Literal["nmf", "lda", "lsa"] = "nmf", n_seeds: int = 4, min_df: int = 3, max_df: float = 0.5, max_rows: int = 8000, seed: int = 42) -> dict:
    """Fit the same model with several random seeds and measure how well the topics match across runs (best-match cosine similarity of the topic-word vectors, Hungarian assignment) — topics that change with the seed are artefacts of the optimiser, not features of the corpus. LSA and (with this initialisation) NMF are near-deterministic; LDA is not.

    Args:
        n_topics: Number of topics.
        model: nmf, lda or lsa.
        n_seeds: Seeds compared.
    """
    frame, texts = load_corpus(data, text_column, None, None, None, max_rows, seed)
    vec, X = vectorise(texts, "counts" if model == "lda" else "tfidf", min_df, max_df)
    seeds = [seed + i * 97 for i in range(n_seeds)]
    s = stability(lambda sd: _fit(model, X, n_topics, sd)[0], seeds)
    return {"model": model, "n_topics": n_topics, "seeds": seeds, "mean_topic_match_similarity": round(s, 4),
            "verdict": "stable topics" if s > 0.9 else "partly stable: some topics change between runs — treat only the recurrent ones as real" if s > 0.7 else "unstable: the topics are largely artefacts of the random start; use fewer topics, more data, or a deterministic model (LSA / NMF with NNDSVD)"}


@tool
def tune_lda_priors(data: str, text_column: str, n_topics: int = 8, alphas: list[float] | None = None, etas: list[float] | None = None, min_df: int = 3, max_df: float = 0.5, max_rows: int = 6000, seed: int = 42) -> dict:
    """Grid-search LDA's two priors by topic coherence: alpha (smaller = each document has fewer topics) and eta (smaller = each topic uses fewer words). The defaults (1/k) are rarely best — documents in most corpora have one or two dominant topics, which wants a small alpha, and a small eta gives sharper topics.

    Args:
        n_topics: Number of topics.
        alphas: doc_topic_prior values; defaults to 0.01, 0.05, 0.2, 1/k.
        etas: topic_word_prior values; defaults to 0.01, 0.05, 0.2.
    """
    frame, texts = load_corpus(data, text_column, None, None, None, max_rows, seed)
    vec, X = vectorise(texts, "counts", min_df, max_df)
    binary, terms = _binary(vec, texts), vec.get_feature_names_out()
    rows = []
    for a in alphas or [0.01, 0.05, 0.2, 1 / n_topics]:
        for e in etas or [0.01, 0.05, 0.2]:
            tw, dt, m = _fit("lda", X, n_topics, seed, alpha=a, eta=e)
            npmi, umass, div = _score(tw, terms, binary, vec.vocabulary_)
            rows.append({"alpha": round(a, 4), "eta": e, "npmi_coherence": round(npmi, 4), "diversity": round(div, 3), "documents_with_clear_topic": round(float((dt.max(axis=1) >= 0.5).mean()), 3)})
    best = max(rows, key=lambda r: r["npmi_coherence"] + 0.3 * r["diversity"])
    return {"n_topics": n_topics, "grid": rows, "best": best}


@tool
def compare_topic_models(data: str, text_column: str, n_topics: int = 8, label_column: str | None = None, min_df: int = 3, max_df: float = 0.5, max_rows: int = 8000, seed: int = 42) -> dict:
    """Fit LDA, NMF, LSA and k-means-with-class-TF-IDF at the same number of topics and compare them on NPMI / UMass coherence, topic diversity, the share of documents with a clear dominant topic, run time, and agreement with known categories (when given) — to choose the family on evidence rather than habit.

    Args:
        text_column: Column holding each document's text.
        n_topics: Number of topics for every model.
        label_column: Known categories, used only to score agreement.
        min_df: Terms in fewer documents are dropped.
        max_df: Terms in more than this share of documents are dropped.
        max_rows: Documents sampled.
    """
    import time

    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import normalized_mutual_info_score

    from unsupervised.topic_modeling.tools._common import ctfidf

    frame, texts = load_corpus(data, text_column, None, label_column, None, max_rows, seed)
    labels = pd.factorize(frame[label_column].astype(str))[0] if label_column else None
    vc, Xc = vectorise(texts, "counts", min_df, max_df)
    vt, Xt = vectorise(texts, "tfidf", min_df, max_df)
    binary, terms = _binary(vt, texts), vt.get_feature_names_out()
    rows = []

    def add(name, tw, dt, t0):
        npmi, umass, div = _score(tw, terms, binary, vt.vocabulary_)
        row = {"model": name, "npmi_coherence": round(npmi, 4), "umass_coherence": round(umass, 3), "diversity": round(div, 3), "documents_with_clear_topic": round(float((dt.max(axis=1) >= 0.5).mean()), 3), "seconds": round(time.time() - t0, 2)}
        if labels is not None:
            row["nmi_with_categories"] = round(float(normalized_mutual_info_score(labels, dt.argmax(axis=1))), 4)
        rows.append(row)

    for name, kind, X in (("lda", "lda", Xc), ("nmf", "nmf", Xt), ("lsa", "lsa", Xt)):
        t0 = time.time()
        if name == "lda":
            from sklearn.decomposition import LatentDirichletAllocation

            m = LatentDirichletAllocation(n_components=n_topics, max_iter=15, random_state=seed, n_jobs=-1).fit(Xc)
            tw, dt = m.components_, normalise_rows(m.transform(Xc))
            vocab_terms = vc.get_feature_names_out()
            tops = [[str(vocab_terms[i]) for i in np.argsort(-r)[:10]] for r in tw]
            npmi = float(np.mean(coherence(tops, binary, vc.vocabulary_, "npmi")))
            rows.append({"model": "lda", "npmi_coherence": round(npmi, 4), "umass_coherence": round(float(np.mean(coherence(tops, binary, vc.vocabulary_, "umass"))), 3), "diversity": round(diversity(tops), 3),
                         "documents_with_clear_topic": round(float((dt.max(axis=1) >= 0.5).mean()), 3), "seconds": round(time.time() - t0, 2), **({"nmi_with_categories": round(float(normalized_mutual_info_score(labels, dt.argmax(axis=1))), 4)} if labels is not None else {})})
        else:
            tw, dt, _ = _fit(kind, X, n_topics, seed)
            add(name, tw, dt, t0)
    t0 = time.time()
    Z = TruncatedSVD(min(50, Xt.shape[1] - 1), random_state=seed).fit_transform(Xt)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    km = KMeans(n_topics, n_init=5, random_state=seed).fit_predict(Z)
    from sklearn.feature_extraction.text import CountVectorizer

    counts = CountVectorizer(vocabulary=vt.vocabulary_, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b").fit_transform(texts)
    tw = ctfidf(np.vstack([np.asarray(counts[km == c].sum(axis=0)).ravel() for c in range(n_topics)]))
    add("kmeans_ctfidf", tw, np.eye(n_topics)[km], t0)
    rows.sort(key=lambda r: -(r["npmi_coherence"] + 0.3 * r["diversity"]))
    return {"n_topics": n_topics, "documents": len(texts), "models": rows, "best": rows[0]["model"], "note": "k-means topics are a hard partition, so 'clear topic' is always 100% for it; compare coherence and diversity first"}
