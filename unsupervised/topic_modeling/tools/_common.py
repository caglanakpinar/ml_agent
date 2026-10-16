"""Topic-modelling machinery the tool modules share: turning a text column into a document-term matrix, the models (LDA, NMF, LSA,
clustering with class-based TF-IDF), and the measures that decide whether topics are any good — NPMI and UMass coherence, topic diversity,
redundancy between topics, and stability across random seeds. A topic model has no ground truth, so these measures *are* the evidence."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns
from ml_core.ml import sk

_URL = re.compile(r"https?://\S+|www\.\S+")
_NUM = re.compile(r"\b\d+\b")


def clean(text: str) -> str:
    return _NUM.sub(" ", _URL.sub(" ", str(text).lower()))


def load_corpus(data: Any, text_column: str, id_column: str | None = None, label_column: str | None = None, time_column: str | None = None, max_rows: int | None = None, seed: int = 42) -> tuple[pd.DataFrame, list[str]]:
    frame = load_frame(data)
    require_columns(frame, [text_column] + [c for c in (id_column, label_column, time_column) if c])
    frame = frame.dropna(subset=[text_column])
    frame = frame[frame[text_column].astype(str).str.strip() != ""]
    if max_rows and len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=seed)
    frame = frame.reset_index(drop=True)
    return frame, [clean(t) for t in frame[text_column]]


def vectorise(texts: list[str], kind: str = "counts", min_df: int = 3, max_df: float = 0.5, ngram_max: int = 1, max_features: int = 20000, stop_words: list[str] | str | None = "english") -> tuple[Any, Any]:
    sk()
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    cls = CountVectorizer if kind == "counts" else TfidfVectorizer
    kw = dict(min_df=min_df, max_df=max_df, ngram_range=(1, ngram_max), max_features=max_features, stop_words=stop_words, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b")
    if kind != "counts":
        kw["sublinear_tf"] = True
    vec = cls(**kw)
    X = vec.fit_transform(texts)
    if X.shape[1] < 5:
        raise ValueError(f"only {X.shape[1]} terms survive min_df={min_df}, max_df={max_df}; relax them")
    return vec, X


def normalise_rows(M: np.ndarray) -> np.ndarray:
    s = M.sum(axis=1, keepdims=True)
    return M / np.where(s > 0, s, 1)


class TopicModel:
    """Vectoriser + model; `topic_word` is the (topics × terms) matrix of non-negative weights, `transform(texts)` the document-topic shares."""

    def __init__(self, kind: str, vectoriser: Any, model: Any, topic_word: np.ndarray, terms: np.ndarray, centroids: Any = None, proj: Any = None):
        self.kind, self.vec, self.model, self.topic_word, self.terms, self.centroids, self.proj = kind, vectoriser, model, topic_word, terms, centroids, proj

    @property
    def n_topics(self) -> int:
        return self.topic_word.shape[0]

    def transform(self, texts: list[str]) -> np.ndarray:
        X = self.vec.transform([clean(t) for t in texts])
        if self.kind in ("kmeans", "embedding_clusters"):
            Z = self.proj.transform(X) if self.proj is not None else X
            Z = np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z)
            Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
            d = Z @ self.centroids.T
            e = np.exp((d - d.max(axis=1, keepdims=True)) * 8)
            return e / e.sum(axis=1, keepdims=True)
        if self.kind == "lsa":
            raw = np.abs(self.model.transform(X))
            return normalise_rows(raw)
        return normalise_rows(np.asarray(self.model.transform(X)))

    def top_words(self, n: int = 10) -> list[list[str]]:
        return [[str(self.terms[i]) for i in np.argsort(-row)[:n]] for row in self.topic_word]


def ctfidf(counts_by_topic: np.ndarray) -> np.ndarray:
    """Class-based TF-IDF (BERTopic): treat each cluster as one document; weights are tf · log(1 + average words per class / term frequency)."""
    tf = counts_by_topic / np.maximum(counts_by_topic.sum(axis=1, keepdims=True), 1)
    avg = counts_by_topic.sum() / counts_by_topic.shape[0]
    return tf * np.log(1 + avg / np.maximum(counts_by_topic.sum(axis=0, keepdims=True), 1))


# ---- evaluation ---------------------------------------------------------------------------------
def coherence(top_words: list[list[str]], binary_dtm: Any, vocab: dict[str, int], measure: str = "npmi", n: int = 10) -> list[float]:
    """Per-topic coherence from document co-occurrence of the topic's top words. NPMI in [−1, 1] tracks human judgement best; UMass is ≤ 0, higher = better."""
    B = binary_dtm.tocsc() if hasattr(binary_dtm, "tocsc") else binary_dtm
    D = B.shape[0]
    out = []
    cache: dict[int, np.ndarray] = {}

    def col(i: int) -> np.ndarray:
        if i not in cache:
            c = B[:, i]
            cache[i] = np.asarray(c.todense()).ravel().astype(bool) if hasattr(c, "todense") else np.asarray(c).ravel().astype(bool)
        return cache[i]

    for words in top_words:
        ids = [vocab[w] for w in words[:n] if w in vocab]
        scores = []
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                ca, cb = col(ids[a]), col(ids[b])
                dab, da, db = int((ca & cb).sum()), int(ca.sum()), int(cb.sum())
                if measure == "umass":
                    scores.append(np.log((dab + 1) / max(db, 1)))
                else:
                    pab = dab / D
                    scores.append(-1.0 if dab == 0 else (np.log(pab / ((da / D) * (db / D))) / -np.log(pab) if pab < 1 else 1.0))
        out.append(float(np.mean(scores)) if scores else 0.0)
    return out


def diversity(top_words: list[list[str]], n: int = 10) -> float:
    """Share of distinct words across all topics' top-n lists (1 = no word repeats; low = topics are near-duplicates)."""
    flat = [w for t in top_words for w in t[:n]]
    return len(set(flat)) / max(len(flat), 1)


def topic_similarity(topic_word: np.ndarray) -> np.ndarray:
    n = topic_word / np.maximum(np.linalg.norm(topic_word, axis=1, keepdims=True), 1e-12)
    return n @ n.T


def dominant(dt: np.ndarray) -> np.ndarray:
    return dt.argmax(axis=1)


def stability(fit_fn: Any, seeds: list[int]) -> float:
    """Mean best-match cosine similarity of topics fitted with different seeds (Hungarian matching) — 1 means the same topics every time."""
    from scipy.optimize import linear_sum_assignment

    mats = [m for m in (fit_fn(s) for s in seeds)]
    sims = []
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            a = mats[i] / np.maximum(np.linalg.norm(mats[i], axis=1, keepdims=True), 1e-12)
            b = mats[j] / np.maximum(np.linalg.norm(mats[j], axis=1, keepdims=True), 1e-12)
            S = a @ b.T
            r, c = linear_sum_assignment(-S)
            sims.append(S[r, c].mean())
    return float(np.mean(sims))


def summarise_model(tm: TopicModel, dt: np.ndarray, binary: Any, vocab: dict[str, int], labels: np.ndarray | None = None, n: int = 10) -> dict[str, Any]:
    tops = tm.top_words(n)
    npmi = coherence(tops, binary, vocab, "npmi", n)
    umass = coherence(tops, binary, vocab, "umass", n)
    dom = dominant(dt)
    sizes = np.bincount(dom, minlength=tm.n_topics)
    out: dict[str, Any] = {"topics": tm.n_topics, "mean_npmi_coherence": round(float(np.mean(npmi)), 4), "mean_umass_coherence": round(float(np.mean(umass)), 3), "topic_diversity": round(diversity(tops, n), 4),
                           "topic_sizes": sizes.tolist(), "empty_topics": int((sizes == 0).sum()), "documents_with_clear_topic": round(float((dt.max(axis=1) >= 0.5).mean()), 4),
                           "max_topic_similarity": round(float(np.max(topic_similarity(tm.topic_word) - np.eye(tm.n_topics))), 4)}
    out["topic_summaries"] = [{"topic": i, "size": int(sizes[i]), "top_words": tops[i], "npmi": round(npmi[i], 3)} for i in np.argsort(-sizes)]
    if labels is not None and len(set(labels)) > 1:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        out["agreement_with_known_categories"] = {"normalized_mutual_info": round(float(normalized_mutual_info_score(labels, dom)), 4), "adjusted_rand_index": round(float(adjusted_rand_score(labels, dom)), 4)}
    return out
