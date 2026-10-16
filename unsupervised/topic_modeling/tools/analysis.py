"""Use a fitted topic model: describe and name topics, assign documents, find redundant topics and off-topic documents, follow topics over time and across groups, extract keywords, and score new text."""

from __future__ import annotations


import joblib
import numpy as np
import pandas as pd

from ml_core.data import save_frame
from ml_core.tooling import tool
from unsupervised.topic_modeling.tools._common import coherence, load_corpus, topic_similarity


def _model(model_path: str):
    m = joblib.load(model_path)
    if not hasattr(m, "topic_word"):
        raise ValueError("not a topic model: pass the file written by a fit_* tool in this topic")
    return m


def _dt(m, frame, text_column):
    return m.transform(frame[text_column].astype(str).tolist())


@tool
def describe_topics(model_path: str, data: str, text_column: str, n_words: int = 10, n_examples: int = 3, label_column: str | None = None, id_column: str | None = None, max_rows: int = 20000, seed: int = 42, output_path: str | None = None) -> dict:
    """The topic report: for every topic its share of the corpus, top words (also the most *exclusive* ones — high weight here, low elsewhere), a suggested name built from the strongest words, the most representative documents, and (with categories) how the known labels distribute across it. The deliverable that lets a person read and name the topics.

    Args:
        model_path: Path of the saved topic model (joblib).
        text_column: Column holding each document's text.
        n_words: Words listed per topic.
        n_examples: Representative documents per topic.
        label_column: Known categories, to show their distribution within each topic.
        id_column: Column shown beside each example.
        max_rows: Documents sampled.
        output_path: Where to write the topic table.
    """
    m = _model(model_path)
    frame, _ = load_corpus(data, text_column, id_column, label_column, None, max_rows, seed)
    dt = _dt(m, frame, text_column)
    dom = dt.argmax(axis=1)
    sizes = np.bincount(dom, minlength=m.n_topics)
    tw = m.topic_word / np.maximum(m.topic_word.sum(axis=1, keepdims=True), 1e-12)
    excl = tw / np.maximum(tw.mean(axis=0, keepdims=True), 1e-12)
    rows = []
    for t in np.argsort(-sizes):
        words = [str(m.terms[i]) for i in np.argsort(-tw[t])[:n_words]]
        distinct = [str(m.terms[i]) for i in np.argsort(-(tw[t] * excl[t]))[:n_words]]
        idx = np.argsort(-dt[:, t])[:n_examples]
        row = {"topic": int(t), "documents": int(sizes[t]), "share": round(float(sizes[t] / len(frame)), 4), "suggested_name": " / ".join(distinct[:3]), "top_words": words, "most_exclusive_words": distinct,
               "examples": [{**({id_column: frame.loc[i, id_column]} if id_column and id_column in frame.columns else {"row": int(i)}), "share": round(float(dt[i, t]), 3), "text": str(frame.loc[i, text_column])[:160]} for i in idx]}
        if label_column:
            row["known_categories"] = frame.loc[dom == t, label_column].astype(str).value_counts(normalize=True).head(3).round(3).to_dict()
        rows.append(row)
    table = pd.DataFrame([{k: (", ".join(v) if isinstance(v, list) and v and isinstance(v[0], str) else v) for k, v in r.items() if k not in ("examples", "known_categories")} for r in rows])
    return {"model": m.kind, "topics": m.n_topics, "documents": len(frame), "report": rows, "written_to": save_frame(table, output_path, "topic_report.csv") if output_path else None}


@tool
def assign_topics_to_documents(model_path: str, data: str, text_column: str, id_column: str | None = None, min_share: float = 0.0, max_rows: int | None = None, seed: int = 42, output_path: str | None = None) -> dict:
    """Score every document with a saved model: its dominant topic, that topic's share, the runner-up, and the full topic mixture — written as a table to join back to the source data. Documents whose top share is below `min_share` are marked as having no clear topic.

    Args:
        model_path: Path of the saved topic model (joblib).
        text_column: Column holding each document's text.
        id_column: Column carried through to the output.
        min_share: Dominant share below which a document is marked 'no clear topic'.
        max_rows: Documents sampled; all when omitted.
        output_path: Where to write the assignments.
    """
    m = _model(model_path)
    frame, _ = load_corpus(data, text_column, id_column, None, None, max_rows, seed)
    dt = _dt(m, frame, text_column)
    top2 = np.sort(dt, axis=1)[:, -2:]
    out = pd.DataFrame({"dominant_topic": dt.argmax(axis=1), "dominant_share": dt.max(axis=1), "runner_up_share": top2[:, 0] if dt.shape[1] > 1 else 0.0})
    out.loc[out["dominant_share"] < min_share, "dominant_topic"] = -1
    out = pd.concat([out, pd.DataFrame(dt, columns=[f"topic_{i}" for i in range(dt.shape[1])])], axis=1)
    if id_column and id_column in frame.columns:
        out.insert(0, id_column, frame[id_column].to_numpy())
    return {"documents": len(out), "topic_counts": out["dominant_topic"].value_counts().sort_index().to_dict(), "no_clear_topic": int((out["dominant_topic"] == -1).sum()), "mean_dominant_share": round(float(out["dominant_share"].mean()), 4),
            "written_to": save_frame(out, output_path, "document_topics.csv"), "sample": out.head(5)}


@tool
def find_redundant_topics(model_path: str, threshold: float = 0.6, n_words: int = 10) -> dict:
    """Topics that say nearly the same thing: pairs whose word distributions have cosine similarity above the threshold, with their shared top words and a merge suggestion — the usual symptom of too many topics, and the evidence for reducing k.

    Args:
        model_path: Path of the saved topic model (joblib).
        threshold: Cosine similarity at or above which two topics are redundant.
        n_words: Top words compared and listed.
    """
    m = _model(model_path)
    S = topic_similarity(m.topic_word)
    tops = m.top_words(n_words)
    pairs = [{"topic_a": a, "topic_b": b, "similarity": round(float(S[a, b]), 3), "shared_top_words": sorted(set(tops[a]) & set(tops[b]))} for a in range(m.n_topics) for b in range(a + 1, m.n_topics) if S[a, b] >= threshold]
    pairs.sort(key=lambda p: -p["similarity"])
    return {"topics": m.n_topics, "redundant_pairs": pairs, "max_similarity": round(float(np.max(S - np.eye(m.n_topics))), 3),
            "advice": f"merge or drop topics: try k = {m.n_topics - len({p['topic_b'] for p in pairs})}" if pairs else "no redundant topics at this threshold"}


@tool
def find_offtopic_documents(model_path: str, data: str, text_column: str, id_column: str | None = None, max_share: float = 0.3, k: int = 20, max_rows: int = 20000, seed: int = 42, output_path: str | None = None) -> dict:
    """Documents the model cannot place: those whose strongest topic share is low (spread thinly over many topics) or that contain almost none of the model's vocabulary — new themes, spam, mixed or very short texts, and candidates for a missing topic.

    Args:
        model_path: Path of the saved topic model (joblib).
        text_column: Column holding each document's text.
        max_share: Dominant share at or below which a document is off-topic.
        k: Documents listed.
        max_rows: Documents sampled.
    """
    m = _model(model_path)
    frame, texts = load_corpus(data, text_column, id_column, None, None, max_rows, seed)
    dt = _dt(m, frame, text_column)
    known = np.asarray(m.vec.transform([t for t in texts]).sum(axis=1)).ravel()
    low = (dt.max(axis=1) <= max_share) | (known == 0)
    rows = frame[low].copy()
    rows["dominant_share"], rows["known_terms"] = dt.max(axis=1)[low], known[low]
    rows = rows.sort_values("dominant_share")
    return {"documents": len(frame), "off_topic": int(low.sum()), "share": round(float(low.mean()), 4), "without_any_known_term": int((known == 0).sum()), "examples": rows.head(k)[[c for c in rows.columns if c in (id_column, text_column, "dominant_share", "known_terms")]],
            "written_to": save_frame(rows, output_path, "offtopic_documents.csv") if output_path else None, "advice": "a large off-topic share means the model is missing themes: raise the number of topics, or fit a second model on these documents" if low.mean() > 0.2 else "the model covers the corpus"}


@tool
def topics_over_time(model_path: str, data: str, text_column: str, time_column: str, freq: str = "MS", id_column: str | None = None, max_rows: int = 50000, seed: int = 42, output_path: str | None = None) -> dict:
    """The share of each topic per time period (monthly by default), using one fixed model for the whole corpus so topics are comparable over time — with each topic's trend (slope per period, relative change between the first and last third) and the periods where it peaked. Shows emerging and fading themes.

    Args:
        model_path: Path of the saved topic model (joblib).
        text_column: Column holding each document's text.
        time_column: Column of timestamps.
        freq: Pandas period alias: D, W, MS (month), QS (quarter), YS (year).
        max_rows: Documents sampled.
        output_path: Where to write the period × topic table.
    """
    m = _model(model_path)
    frame, _ = load_corpus(data, text_column, id_column, None, time_column, max_rows, seed)
    frame = frame.assign(**{time_column: pd.to_datetime(frame[time_column], errors="coerce")}).dropna(subset=[time_column])
    dt = _dt(m, frame, text_column)
    D = pd.DataFrame(dt, columns=[f"topic_{i}" for i in range(dt.shape[1])], index=pd.DatetimeIndex(frame[time_column]))
    shares = D.resample(freq).mean().dropna(how="all")
    counts = D.resample(freq).size()
    shares = shares[counts.reindex(shares.index) >= 5]
    x = np.arange(len(shares), dtype=float)
    trends = []
    for c in shares.columns:
        y = shares[c].to_numpy()
        slope = float(np.polyfit(x, y, 1)[0]) if len(x) > 2 else 0.0
        third = max(1, len(y) // 3)
        trends.append({"topic": c, "mean_share": round(float(y.mean()), 4), "slope_per_period": round(slope, 5), "relative_change_last_vs_first_third": round(float(y[-third:].mean() / max(y[:third].mean(), 1e-9) - 1), 3), "peak_period": str(shares.index[int(y.argmax())].date()),
                       "words": ", ".join(m.top_words(5)[int(c.split("_")[1])])})
    trends.sort(key=lambda r: -r["relative_change_last_vs_first_third"])
    return {"periods": int(len(shares)), "documents": int(counts.sum()), "emerging": trends[:3], "fading": trends[-3:][::-1], "trends": trends, "written_to": save_frame(shares.reset_index(names="period"), output_path, "topics_over_time.csv") if output_path else None}


@tool
def topics_by_group(model_path: str, data: str, text_column: str, group_column: str, top_groups: int = 12, max_rows: int = 50000, seed: int = 42, output_path: str | None = None) -> dict:
    """Which groups (authors, channels, segments, sources, regions) talk about which topics: the mean topic share per group, and for each group the topics it over- and under-represents relative to the whole corpus (lift) — the topic model as a segmentation lens.

    Args:
        model_path: Path of the saved topic model (joblib).
        text_column: Column holding each document's text.
        group_column: Column that defines the groups.
        top_groups: Largest groups reported.
        max_rows: Documents sampled.
        output_path: Where to write the group × topic table.
    """
    m = _model(model_path)
    frame, _ = load_corpus(data, text_column, None, group_column, None, max_rows, seed)
    dt = _dt(m, frame, text_column)
    D = pd.DataFrame(dt, columns=[f"topic_{i}" for i in range(dt.shape[1])])
    D["group"] = frame[group_column].astype(str).to_numpy()
    overall = D.drop(columns="group").mean()
    means = D.groupby("group").mean()
    sizes = D["group"].value_counts()
    keep = sizes.head(top_groups).index
    rows = []
    for g in keep:
        lift = (means.loc[g] / overall.replace(0, np.nan)).dropna()
        rows.append({"group": g, "documents": int(sizes[g]), "over_represented": {t: round(float(v), 2) for t, v in lift.sort_values(ascending=False).head(2).items()}, "under_represented": {t: round(float(v), 2) for t, v in lift.sort_values().head(2).items()}})
    return {"groups": int(len(sizes)), "reported": rows, "topic_words": {f"topic_{i}": ", ".join(w[:5]) for i, w in enumerate(m.top_words(5))}, "written_to": save_frame(means.loc[keep].reset_index(), output_path, "topics_by_group.csv") if output_path else None}


@tool
def extract_document_keywords(data: str, text_column: str, top: int = 5, id_column: str | None = None, ngram_max: int = 2, min_df: int = 2, max_rows: int = 20000, seed: int = 42, output_path: str | None = None) -> dict:
    """The most characteristic terms of each individual document (highest TF-IDF within it, words and bigrams) — a topic-free description of every text, for tagging, search facets or labelling, and a way to check what a topic model is picking up.

    Args:
        text_column: Column holding each document's text.
        top: Keywords per document.
        id_column: Column carried through to the output.
        ngram_max: Longest phrase.
        min_df: Terms in fewer documents are ignored.
        max_rows: Documents sampled.
        output_path: Where to write the keywords.
    """
    frame, texts = load_corpus(data, text_column, id_column, None, None, max_rows, seed)
    from unsupervised.topic_modeling.tools._common import vectorise

    vec, X = vectorise(texts, "tfidf", min_df, 0.7, ngram_max)
    terms = vec.get_feature_names_out()
    X = X.tocsr()
    kw = [", ".join(str(terms[j]) for j in X[i].indices[np.argsort(-X[i].data)[:top]]) for i in range(X.shape[0])]
    out = pd.DataFrame({"keywords": kw})
    if id_column and id_column in frame.columns:
        out.insert(0, id_column, frame[id_column].to_numpy())
    return {"documents": len(out), "sample": out.head(8), "most_common_keywords": pd.Series([w for k in kw for w in k.split(", ") if w]).value_counts().head(15).to_dict(), "written_to": save_frame(out, output_path, "document_keywords.csv") if output_path else None}


@tool
def score_topic_coherence(model_path: str, data: str, text_column: str, n_words: int = 10, max_rows: int = 30000, seed: int = 42) -> dict:
    """Coherence of a saved model's topics measured on *any* corpus (e.g. a held-out sample, or the same topics on a later period): NPMI and UMass per topic, the best and worst topics, and a plain reading — the check that a model still describes the data it is applied to.

    Args:
        model_path: Path of the saved topic model (joblib).
        text_column: Column holding each document's text.
        n_words: Top words per topic used.
        max_rows: Documents sampled.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    m = _model(model_path)
    frame, texts = load_corpus(data, text_column, None, None, None, max_rows, seed)
    vocab = m.vec.vocabulary_
    binary = CountVectorizer(vocabulary=vocab, binary=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b").fit_transform(texts)
    tops = m.top_words(n_words)
    npmi, umass = coherence(tops, binary, vocab, "npmi", n_words), coherence(tops, binary, vocab, "umass", n_words)
    order = np.argsort(npmi)
    return {"documents": len(texts), "mean_npmi": round(float(np.mean(npmi)), 4), "mean_umass": round(float(np.mean(umass)), 3), "worst_topics": [{"topic": int(i), "npmi": round(npmi[i], 3), "words": tops[i][:6]} for i in order[:3]],
            "best_topics": [{"topic": int(i), "npmi": round(npmi[i], 3), "words": tops[i][:6]} for i in order[::-1][:3]],
            "reading": "NPMI above ~0.1 is coherent, around 0 means the words rarely co-occur (a junk topic), negative means they avoid each other"}


@tool
def predict_topics_for_new_text(model_path: str, texts: list[str], top: int = 3) -> dict:
    """Topic mixture for a few new texts straight from a saved model: the top topics with their shares and words — for scoring a live document or spot-checking a model on examples.

    Args:
        model_path: Path of the saved topic model (joblib).
        texts: The documents to score.
        top: Topics listed per document.
    """
    m = _model(model_path)
    dt = m.transform(texts)
    words = m.top_words(6)
    return {"model": m.kind, "predictions": [{"text": t[:100], "topics": [{"topic": int(k), "share": round(float(dt[i, k]), 3), "words": words[k]} for k in np.argsort(-dt[i])[:top]]} for i, t in enumerate(texts)]}
