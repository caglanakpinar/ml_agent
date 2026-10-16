"""Read-only checks on a text corpus before modelling topics: size and length, vocabulary, boilerplate, duplicates, and which kind of model suits it."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from ml_core.tooling import tool
from unsupervised.topic_modeling.tools._common import load_corpus, vectorise

_TOKEN = re.compile(r"[a-z][a-z']+")


@tool
def profile_corpus(data: str, text_column: str, label_column: str | None = None, time_column: str | None = None, top: int = 15, max_rows: int = 50000, seed: int = 42) -> dict:
    """Profile a text corpus: documents, empty and duplicate texts, document length in words (median, quartiles, share under 20 words), vocabulary size, share of words seen once, the most frequent terms, and — if given — the known categories and time span.

    Args:
        text_column: Column holding each document's text.
        label_column: Column of known categories (used only to describe).
        time_column: Column of timestamps (used only to describe the span).
        top: Most frequent terms listed.
        max_rows: Documents sampled.
    """
    from ml_core.data import load_frame

    raw = load_frame(data)
    n_raw = len(raw)
    frame, texts = load_corpus(data, text_column, None, label_column, time_column, max_rows, seed)
    toks = [_TOKEN.findall(t) for t in texts]
    lengths = np.array([len(t) for t in toks])
    counts = Counter(w for t in toks for w in t)
    out = {"rows": n_raw, "documents_used": len(frame), "empty_or_missing": int(n_raw - len(frame)) if n_raw <= max_rows else None, "duplicate_texts": int(frame[text_column].duplicated().sum()),
           "words_per_document": {"median": float(np.median(lengths)), "p25": float(np.quantile(lengths, .25)), "p75": float(np.quantile(lengths, .75)), "max": int(lengths.max())},
           "short_documents_under_20_words": round(float((lengths < 20).mean()), 4), "vocabulary": len(counts), "words_seen_once": round(float(sum(1 for c in counts.values() if c == 1) / max(len(counts), 1)), 4),
           "most_frequent_terms": dict(counts.most_common(top))}
    if label_column:
        out["known_categories"] = frame[label_column].astype(str).value_counts().head(15).to_dict()
    if time_column:
        t = pd.to_datetime(frame[time_column], errors="coerce")
        out["time_span"] = [str(t.min()), str(t.max())]
    out["advice"] = [a for a in ["short documents dominate: LDA is unreliable below ~20 words per document — prefer fit_kmeans_topics, fit_nmf_topics or BERTopic" if out["short_documents_under_20_words"] > 0.5 else None,
                                 "many duplicate texts will form their own topics: drop duplicates first" if out["duplicate_texts"] > 0.05 * len(frame) else None,
                                 "very long documents: consider splitting into passages so a topic is not an average of unrelated sections" if np.median(lengths) > 1500 else None] if a]
    return out


@tool
def suggest_vocabulary_settings(data: str, text_column: str, max_rows: int = 20000, seed: int = 42) -> dict:
    """Show what each combination of `min_df` (rare-word cut) and `max_df` (common-word cut) does to the vocabulary and to the documents: vocabulary size, share of documents left with fewer than 3 terms, and how many tokens survive — rare words add noise and common ones swamp topics, so these two settings matter more than the algorithm.

    Args:
        max_rows: Documents sampled.
    """
    frame, texts = load_corpus(data, text_column, None, None, None, max_rows, seed)
    rows = []
    for min_df in (1, 2, 3, 5, 10):
        for max_df in (0.3, 0.5, 0.8):
            try:
                vec, X = vectorise(texts, "counts", min_df, max_df, 1, 100000)
            except ValueError:
                rows.append({"min_df": min_df, "max_df": max_df, "vocabulary": 0})
                continue
            per_doc = np.asarray((X > 0).sum(axis=1)).ravel()
            rows.append({"min_df": min_df, "max_df": max_df, "vocabulary": int(X.shape[1]), "tokens_kept": int(X.sum()), "documents_under_3_terms": round(float((per_doc < 3).mean()), 4)})
    ok = [r for r in rows if r["vocabulary"] and r.get("documents_under_3_terms", 1) < 0.05 and 500 <= r["vocabulary"] <= 20000]
    pick = min(ok, key=lambda r: abs(r["vocabulary"] - 5000)) if ok else None
    return {"grid": rows, "suggested": pick, "reading": "choose a setting that leaves a vocabulary of a few thousand terms and almost no documents with fewer than 3 terms"}


@tool
def find_boilerplate_terms(data: str, text_column: str, threshold: float = 0.3, top: int = 30, max_rows: int = 20000, seed: int = 42) -> dict:
    """Terms that appear in a large share of documents (signatures, headers, 'unsubscribe', the company name, a template phrase) and so form meaningless 'topics' of their own — listed with their document frequency, ready to add to a stop-word list or to remove with `max_df`.

    Args:
        threshold: Document-frequency share above which a term counts as boilerplate.
        top: Terms listed.
        max_rows: Documents sampled.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    frame, texts = load_corpus(data, text_column, None, None, None, max_rows, seed)
    vec = CountVectorizer(binary=True, stop_words="english", min_df=3, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b")
    X = vec.fit_transform(texts)
    df = np.asarray(X.mean(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    order = np.argsort(-df)
    hits = [{"term": str(terms[i]), "document_frequency": round(float(df[i]), 4)} for i in order[:top] if df[i] >= threshold]
    bigrams = CountVectorizer(binary=True, ngram_range=(2, 3), stop_words="english", min_df=3, max_features=5000).fit(texts)
    Xb = bigrams.transform(texts)
    dfb = np.asarray(Xb.mean(axis=0)).ravel()
    ob = np.argsort(-dfb)[:10]
    return {"threshold": threshold, "boilerplate_terms": hits, "repeated_phrases": [{"phrase": str(bigrams.get_feature_names_out()[i]), "document_frequency": round(float(dfb[i]), 4)} for i in ob if dfb[i] >= threshold / 2],
            "most_common_non_stopwords": [{"term": str(terms[i]), "document_frequency": round(float(df[i]), 4)} for i in order[:10]],
            "advice": "set max_df below the highest boilerplate frequency, or pass these terms as stop words" if hits else "no term dominates the corpus"}


@tool
def check_duplicate_documents(data: str, text_column: str, similarity: float = 0.9, max_rows: int = 5000, seed: int = 42, k: int = 10) -> dict:
    """Exact and near-duplicate documents (cosine similarity of TF-IDF vectors at or above a threshold) — copies and templated messages concentrate into spurious tight topics and inflate cluster sizes; reports how many documents are redundant and a few example pairs.

    Args:
        similarity: Cosine similarity at or above which two documents are near-duplicates.
        max_rows: Documents sampled (comparison is quadratic).
        k: Example pairs listed.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    frame, texts = load_corpus(data, text_column, None, None, None, max_rows, seed)
    X = TfidfVectorizer(stop_words="english", min_df=1).fit_transform(texts)
    S = (X @ X.T).tocoo()
    mask = (S.row < S.col) & (S.data >= similarity)
    pairs = sorted(zip(S.data[mask], S.row[mask], S.col[mask]), reverse=True)
    redundant = len({int(c) for _, _, c in pairs})
    return {"documents_checked": len(texts), "near_duplicate_pairs": len(pairs), "redundant_documents": redundant, "share_redundant": round(redundant / len(texts), 4), "examples": [{"similarity": round(float(s), 3), "a": texts[int(a)][:90], "b": texts[int(b)][:90]} for s, a, b in pairs[:k]],
            "advice": "deduplicate before fitting topics" if redundant > 0.05 * len(texts) else "duplication is not a problem"}


@tool
def estimate_topic_structure(data: str, text_column: str, max_topics: int = 20, min_df: int = 3, max_df: float = 0.5, max_rows: int = 10000, seed: int = 42) -> dict:
    """A quick, model-free look at how much thematic structure there is: the singular-value spectrum of the TF-IDF matrix and its knee (a spectrum that falls steeply then flattens marks the number of real themes; a smooth slope means a continuum with no distinct topics), and the share of documents whose nearest neighbours are very similar.

    Args:
        max_topics: Largest number of components examined.
        min_df: Terms in fewer documents are dropped.
        max_df: Terms in more than this share of documents are dropped.
        max_rows: Documents sampled.
    """
    from sklearn.decomposition import TruncatedSVD

    frame, texts = load_corpus(data, text_column, None, None, None, max_rows, seed)
    vec, X = vectorise(texts, "tfidf", min_df, max_df, 1)
    k = min(max_topics, X.shape[1] - 1)
    ev = TruncatedSVD(k, random_state=seed).fit(X).explained_variance_
    x, y = np.arange(k, dtype=float), (ev - ev.min()) / max(np.ptp(ev), 1e-9)
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    knee = int(np.argmax(np.abs(dy * (x - x[0]) - dx * (y - y[0])) / np.hypot(dx, dy))) + 1
    ratio = ev[0] / max(ev[1], 1e-12)
    return {"documents": len(texts), "vocabulary": X.shape[1], "singular_value_variance": np.round(ev, 5).tolist(), "knee_at": knee, "first_to_second_ratio": round(float(ratio), 2),
            "reading": "the first component is usually 'document length / general language'; a knee well before the end indicates distinct themes, no knee a continuum — then fewer, broader topics, or no topic model at all"}
