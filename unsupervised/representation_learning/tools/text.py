"""Representations of text: latent semantic analysis, topic-style and character embeddings, word vectors learned from co-occurrence (PPMI + SVD), pretrained sentence encoders, and similarity search."""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.ml import save_model
from ml_core.tooling import tool
from unsupervised.representation_learning.tools._common import (
    cooccurrence,
    effective_rank,
    isotropy,
    neighbour_purity,
    ppmi_vectors,
    probe,
)

_TOKEN = re.compile(r"[a-z0-9']+")


def _docs(data, text_column, label_column=None, max_rows=20000, seed=42):
    frame = load_frame(data)
    require_columns(frame, [text_column] + ([label_column] if label_column else []))
    frame = frame.dropna(subset=[text_column])
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=seed)
    frame = frame.reset_index(drop=True)
    y = pd.factorize(frame[label_column].astype(str))[0] if label_column else None
    return frame, frame[text_column].astype(str).tolist(), y


class TextEmbedder:
    """Vectoriser + projection: `transform(texts)` for any list of strings."""

    def __init__(self, vectoriser, projection, normalise: bool = True):
        self.vec, self.proj, self.normalise = vectoriser, projection, normalise

    def transform(self, texts: list[str]) -> np.ndarray:
        E = self.proj.transform(self.vec.transform(texts)) if self.proj is not None else self.vec.transform(texts)
        E = np.asarray(E.toarray() if hasattr(E, "toarray") else E, float)
        return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9) if self.normalise else E


def _report(label, frame, E, y, model, text_column, id_column, output_path, model_path, extra=None, kind="text"):
    out = {"method": label, "documents": int(len(E)), "dimensions": int(E.shape[1]), "effective_rank": round(effective_rank(E), 2), "isotropy_mean_abs_cosine": round(isotropy(E), 4), **(extra or {})}
    if y is not None and len(set(y)) > 1:
        out["probe"], out["neighbour_label_purity"] = probe(E, y), round(neighbour_purity(E, y), 4)
    table = pd.DataFrame(E, columns=[f"emb_{i + 1}" for i in range(E.shape[1])])
    if id_column and id_column in frame.columns:
        table.insert(0, id_column, frame[id_column].to_numpy())
    out["written_to"] = save_frame(table, output_path, f"embedding_{kind}.csv") if output_path else None
    out["model_path"] = save_model(model, model_path) if model is not None else None
    return out


@tool
def embed_text_lsa(data: str, text_column: str, n_components: int = 100, label_column: str | None = None, id_column: str | None = None, ngram_max: int = 2, min_df: int = 2, max_features: int = 50000, max_rows: int = 20000, seed: int = 42,
                   output_path: str | None = None, model_path: str | None = None) -> dict:
    """Latent semantic analysis: TF-IDF (word and bigram weights) compressed with truncated SVD, then length-normalised — the strong, fast, deterministic baseline for document embeddings. Documents that use related words land close even without sharing any, because the SVD merges words that co-occur.

    Args:
        text_column: Column holding each document's text.
        n_components: Embedding size.
        label_column: Column of known categories, kept aside to *probe* the embedding.
        id_column: Column copied next to the embedding.
        ngram_max: Longest word n-gram.
        min_df: A term must appear in at least this many documents.
        max_features: Vocabulary cap.
        max_rows: Documents sampled.
        output_path: Where to write the embedding.
        model_path: Where to save the embedder (joblib).
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    frame, texts, y = _docs(data, text_column, label_column, max_rows, seed)
    vec = TfidfVectorizer(ngram_range=(1, ngram_max), min_df=min_df, max_features=max_features, sublinear_tf=True, stop_words="english").fit(texts)
    X = vec.transform(texts)
    svd = TruncatedSVD(min(n_components, X.shape[1] - 1), random_state=seed).fit(X)
    m = TextEmbedder(vec, svd)
    return _report(f"lsa({svd.n_components})", frame, m.transform(texts), y, m, text_column, id_column, output_path, model_path, {"vocabulary": len(vec.vocabulary_), "variance_explained": round(float(svd.explained_variance_ratio_.sum()), 4)}, "lsa")


@tool
def embed_text_char_ngrams(data: str, text_column: str, n_components: int = 100, label_column: str | None = None, id_column: str | None = None, ngram_min: int = 3, ngram_max: int = 5, max_features: int = 100000, max_rows: int = 20000, seed: int = 42,
                           output_path: str | None = None, model_path: str | None = None) -> dict:
    """Character n-gram embedding: TF-IDF over character sequences (3–5 letters) compressed with SVD — robust to typos, spelling variants, inflections, names and codes, and needs no language knowledge or tokeniser; the right choice for short, messy strings (product titles, addresses, names).

    Args:
        text_column: Column holding each document's text.
        n_components: Embedding size.
        label_column: Column of known categories, kept aside to probe the embedding.
        id_column: Column copied next to the embedding.
        ngram_min: Shortest character n-gram.
        ngram_max: Longest character n-gram.
        max_features: Vocabulary cap.
        max_rows: Documents sampled.
        output_path: Where to write the embedding.
        model_path: Where to save the embedder.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    frame, texts, y = _docs(data, text_column, label_column, max_rows, seed)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(ngram_min, ngram_max), max_features=max_features, sublinear_tf=True, lowercase=True).fit(texts)
    X = vec.transform(texts)
    svd = TruncatedSVD(min(n_components, X.shape[1] - 1), random_state=seed).fit(X)
    m = TextEmbedder(vec, svd)
    return _report(f"char_ngrams({ngram_min}-{ngram_max})+svd({svd.n_components})", frame, m.transform(texts), y, m, text_column, id_column, output_path, model_path, {"ngram_features": len(vec.vocabulary_)}, "char_ngrams")


@tool
def embed_text_hashing(data: str, text_column: str, n_components: int = 100, n_features: int = 2**16, label_column: str | None = None, id_column: str | None = None, max_rows: int = 50000, seed: int = 42, output_path: str | None = None,
                       model_path: str | None = None) -> dict:
    """Hashing embedding: words (and bigrams) are hashed into a fixed number of buckets, TF-IDF weighted and compressed with SVD — no vocabulary to store, constant memory and streamable, so it scales to corpora too large to hold a dictionary, at the cost of rare collisions.

    Args:
        text_column: Column holding each document's text.
        n_components: Embedding size.
        n_features: Number of hash buckets.
        label_column: Column of known categories, kept aside to probe the embedding.
        id_column: Column copied next to the embedding.
        max_rows: Documents sampled.
        output_path: Where to write the embedding.
        model_path: Where to save the embedder.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.pipeline import make_pipeline

    frame, texts, y = _docs(data, text_column, label_column, max_rows, seed)
    vec = make_pipeline(HashingVectorizer(n_features=n_features, ngram_range=(1, 2), alternate_sign=False, stop_words="english"), TfidfTransformer(sublinear_tf=True)).fit(texts)
    X = vec.transform(texts)
    svd = TruncatedSVD(min(n_components, X.shape[1] - 1), random_state=seed).fit(X)
    m = TextEmbedder(vec, svd)
    return _report(f"hashing({n_features})+svd({svd.n_components})", frame, m.transform(texts), y, m, text_column, id_column, output_path, model_path, {"hash_buckets": n_features}, "hashing")


@tool
def embed_text_word_vectors(data: str, text_column: str, n_components: int = 50, window: int = 5, min_count: int = 3, label_column: str | None = None, id_column: str | None = None, weighting: Literal["idf", "mean"] = "idf", max_rows: int = 20000,
                            seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Learn word vectors from the corpus itself — a window co-occurrence matrix, shifted positive pointwise mutual information, SVD (the matrix-factorisation form of word2vec's skip-gram, Levy & Goldberg) — and represent each document as the (IDF-weighted) average of its word vectors. Captures word meaning from *this* corpus' usage, with no external model; the vectors also answer 'which words are similar'.

    Args:
        text_column: Column holding each document's text.
        n_components: Word-vector size.
        window: Words either side counted as context.
        min_count: Words rarer than this are ignored.
        label_column: Column of known categories, kept aside to probe the embedding.
        id_column: Column copied next to the embedding.
        weighting: idf (down-weight common words when averaging) or mean.
        max_rows: Documents sampled.
        output_path: Where to write the document embedding.
        model_path: Where to save the word-vector model.
    """
    frame, texts, y = _docs(data, text_column, label_column, max_rows, seed)
    toks = [_TOKEN.findall(t.lower()) for t in texts]
    from collections import Counter

    counts = Counter(w for t in toks for w in t)
    vocab = sorted(w for w, c in counts.items() if c >= min_count)
    if len(vocab) < n_components + 2:
        raise ValueError(f"only {len(vocab)} words occur at least {min_count} times; lower `min_count` or `n_components`")
    known = set(vocab)
    C, vocab = cooccurrence([[w for w in t if w in known] for t in toks], window, vocab)
    V = ppmi_vectors(C, n_components)
    df = Counter(w for t in toks for w in set(t))
    model = WordVectorModel(vocab, V, {w: float(np.log(len(toks) / (1 + df[w])) + 1) for w in vocab}, weighting == "idf")
    E = model.transform(texts)
    return _report(f"ppmi_svd_word_vectors({V.shape[1]},window={window})", frame, E, y, model, text_column, id_column, output_path, model_path, {"vocabulary": len(vocab), "documents_without_known_words": int((np.linalg.norm(E, axis=1) == 0).sum())}, "word_vectors")


class WordVectorModel:
    def __init__(self, vocab: list[str], vectors: np.ndarray, idf: dict[str, float], use_idf: bool):
        self.vocab, self.vectors, self.idf, self.use_idf = vocab, vectors, idf, use_idf
        self.index = {w: i for i, w in enumerate(vocab)}

    def transform(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.vectors.shape[1]))
        for n, t in enumerate(texts):
            ids = [self.index[w] for w in _TOKEN.findall(t.lower()) if w in self.index]
            if ids:
                w = np.array([self.idf[self.vocab[i]] if self.use_idf else 1.0 for i in ids])
                out[n] = (self.vectors[ids] * w[:, None]).sum(axis=0) / w.sum()
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

    def neighbours(self, word: str, k: int = 10) -> list[tuple[str, float]]:
        v = self.vectors / np.maximum(np.linalg.norm(self.vectors, axis=1, keepdims=True), 1e-9)
        sims = v @ v[self.index[word]]
        return [(self.vocab[i], float(sims[i])) for i in np.argsort(-sims)[1 : k + 1]]


@tool(requires=("sentence_transformers",))
def embed_text_sentence_transformer(data: str, text_column: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", label_column: str | None = None, id_column: str | None = None, batch_size: int = 64, max_rows: int = 20000, seed: int = 42,
                                    output_path: str | None = None) -> dict:
    """Pretrained sentence encoder (sentence-transformers): a transformer trained on large corpora maps each text to a dense vector where *meaning*, not shared words, determines closeness — the strongest general-purpose text embedding, at the cost of a model download and slower encoding. Reports the same probe as the cheap methods so the gain can be judged.

    Args:
        text_column: Column holding each document's text.
        model_name: Hugging Face id of a sentence-transformers model.
        label_column: Column of known categories, kept aside to probe the embedding.
        id_column: Column copied next to the embedding.
        batch_size: Texts per forward pass.
        max_rows: Documents sampled.
        output_path: Where to write the embedding.
    """
    from sentence_transformers import SentenceTransformer

    frame, texts, y = _docs(data, text_column, label_column, max_rows, seed)
    E = np.asarray(SentenceTransformer(model_name).encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False), float)
    return _report(f"sentence_transformer({model_name.split('/')[-1]})", frame, E, y, None, text_column, id_column, output_path, None, {"model": model_name}, "sentence_transformer")


@tool
def find_similar_documents(data: str, text_column: str, query: str, top: int = 10, method: Literal["lsa", "char_ngrams", "tfidf"] = "lsa", n_components: int = 100, id_column: str | None = None, max_rows: int = 50000, seed: int = 42) -> dict:
    """Semantic search over a table of documents: embed every document and the query the same way, and return the nearest documents by cosine similarity — and, as a sanity check, how the plain TF-IDF ranking differs (documents the embedding finds that share no query word are the semantic matches).

    Args:
        text_column: Column holding each document's text.
        query: The text to search for.
        top: Documents returned.
        method: lsa, char_ngrams or tfidf (no compression).
        n_components: Embedding size for lsa / char_ngrams.
        id_column: Column shown beside each result.
        max_rows: Documents indexed.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    frame, texts, _ = _docs(data, text_column, None, max_rows, seed)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True) if method == "char_ngrams" else TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=2)
    X = vec.fit_transform(texts)
    q = vec.transform([query])
    plain = np.asarray((X @ q.T).todense()).ravel() / np.maximum(np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel()) * np.sqrt(q.multiply(q).sum()), 1e-9)
    if method == "tfidf":
        score = plain
    else:
        svd = TruncatedSVD(min(n_components, X.shape[1] - 1), random_state=seed).fit(X)
        E = svd.transform(X)
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
        qe = svd.transform(q)
        score = E @ (qe / max(np.linalg.norm(qe), 1e-9)).ravel()
    order = np.argsort(-score)[:top]
    rows = [{"rank": r + 1, "similarity": round(float(score[i]), 4), "tfidf_similarity": round(float(plain[i]), 4), **({id_column: frame.loc[i, id_column]} if id_column and id_column in frame.columns else {"row": int(i)}), "text": texts[i][:140]} for r, i in enumerate(order)]
    return {"query": query, "method": method, "documents_indexed": len(texts), "results": rows, "semantic_only_matches": int(sum(1 for r in rows if r["tfidf_similarity"] < 0.05))}


@tool
def nearest_words(model_path: str, word: str, k: int = 10) -> dict:
    """The words closest to a given word in a saved word-vector model (from `embed_text_word_vectors`), by cosine similarity — the quickest check that learned vectors capture meaning in this corpus (neighbours should be related words), and a way to discover the vocabulary around a concept.

    Args:
        model_path: Path of the saved word-vector model (joblib).
        word: The word to look up.
        k: Neighbours returned.
    """
    import joblib

    m = joblib.load(model_path)
    if not hasattr(m, "neighbours"):
        raise ValueError("this model is not a word-vector model; use the file saved by embed_text_word_vectors")
    w = word.lower()
    if w not in m.index:
        raise KeyError(f"{word!r} is not in the vocabulary of {len(m.vocab)} words")
    return {"word": w, "neighbours": [{"word": n, "cosine": round(s, 4)} for n, s in m.neighbours(w, k)]}
