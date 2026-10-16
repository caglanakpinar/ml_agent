"""Rank documents against a text query when there are no learned features: BM25, TF-IDF similarity, and blending with a learned score."""

from __future__ import annotations

import re
from collections import Counter
from typing import Literal

import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.ml import sk
from ml_core.tooling import tool
from supervised.ranking.tools._common import per_group_metrics, summarise

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: object) -> list[str]:
    return _TOKEN.findall(str(text).lower())


def bm25_scores(frame: pd.DataFrame, query_column: str, doc_column: str, k1: float, b: float) -> np.ndarray:
    docs = [_tokens(t) for t in frame[doc_column]]
    n = len(docs)
    avg = max(np.mean([len(d) for d in docs]), 1e-9)
    df = Counter(t for d in docs for t in set(d))
    idf = {t: np.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
    scores = np.zeros(n)
    for i, (doc, query) in enumerate(zip(docs, frame[query_column])):
        tf = Counter(doc)
        for term in set(_tokens(query)):
            if term in tf:
                f = tf[term]
                scores[i] += idf.get(term, 0.0) * f * (k1 + 1) / (f + k1 * (1 - b + b * len(doc) / avg))
    return scores


def _summary(frame: pd.DataFrame, scores: np.ndarray, group_column: str | None, relevance: str | None, k: int) -> dict | None:
    if not group_column or not relevance or group_column not in frame or relevance not in frame:
        return None
    return summarise(per_group_metrics(frame.assign(_s=scores), group_column, relevance, "_s", k), k)


@tool
def rank_with_bm25(
    data: str, query_column: str, doc_column: str, group_column: str | None = None, relevance: str | None = None,
    k1: float = 1.5, b: float = 0.75, k: int = 10, output_path: str | None = None,
) -> dict:
    """Rank documents against their query with BM25 — the classic lexical baseline that needs no training; evaluates it when relevance labels exist.

    Args:
        query_column: Column holding the query text for each row.
        doc_column: Column holding the document text for each row.
        group_column: Column identifying the query/group; enables evaluation and per-group ranks.
        relevance: Column of relevance labels; enables evaluation.
        k1: Term-frequency saturation.
        b: Length normalisation, 0 (none) to 1 (full).
        k: Cut-off for the evaluation metrics.
    """
    frame = load_frame(data)
    require_columns(frame, [query_column, doc_column])
    scores = bm25_scores(frame, query_column, doc_column, k1, b)
    out = frame.assign(bm25=scores)
    if group_column:
        out["rank"] = out.groupby(group_column)["bm25"].rank(ascending=False, method="first").astype(int)
    return {"k1": k1, "b": b, "evaluation": _summary(frame, scores, group_column, relevance, k),
            "written_to": save_frame(out, output_path, "bm25_ranked.csv") if output_path else None,
            "score_summary": pd.Series(scores).describe().round(4).to_dict()}


@tool
def rank_with_tfidf_similarity(
    data: str, query_column: str, doc_column: str, group_column: str | None = None, relevance: str | None = None,
    ngram_max: int = 2, sublinear_tf: bool = True, k: int = 10, output_path: str | None = None,
) -> dict:
    """Rank documents by TF-IDF cosine similarity to the query — a soft lexical match that handles word order a little (n-grams) and needs no training.

    Args:
        query_column: Column holding the query text for each row.
        doc_column: Column holding the document text for each row.
        group_column: Column identifying the query/group; enables evaluation and per-group ranks.
        relevance: Column of relevance labels; enables evaluation.
        ngram_max: Longest word n-gram.
        sublinear_tf: Use 1+log(tf).
        k: Cut-off for the evaluation metrics.
    """
    sk()
    from sklearn.feature_extraction.text import TfidfVectorizer

    frame = load_frame(data)
    require_columns(frame, [query_column, doc_column])
    vec = TfidfVectorizer(ngram_range=(1, ngram_max), sublinear_tf=sublinear_tf, stop_words="english").fit(pd.concat([frame[doc_column], frame[query_column]]).astype(str))
    D, Q = vec.transform(frame[doc_column].astype(str)), vec.transform(frame[query_column].astype(str))
    scores = np.asarray(D.multiply(Q).sum(axis=1)).ravel()
    out = frame.assign(tfidf=scores)
    if group_column:
        out["rank"] = out.groupby(group_column)["tfidf"].rank(ascending=False, method="first").astype(int)
    return {"vocabulary": len(vec.vocabulary_), "evaluation": _summary(frame, scores, group_column, relevance, k),
            "written_to": save_frame(out, output_path, "tfidf_ranked.csv") if output_path else None}


@tool
def blend_rankers(
    data: str, group_column: str, relevance: str, score_columns: list[str], weights: list[float] | None = None,
    normalise: Literal["zscore", "minmax", "rank"] = "zscore", k: int = 10, output_path: str | None = None,
) -> dict:
    """Blend several score columns (e.g. BM25 + a learned ranker) per group after normalising each, and report NDCG for every input and for the blend — to see whether combining beats the best single signal.

    Args:
        group_column: Column identifying the query/group each row belongs to.
        score_columns: Score columns to blend, higher = better.
        weights: One weight per score column; equal when omitted.
        normalise: Per-group normalisation before blending: zscore, minmax, or rank.
        k: Cut-off for NDCG.
    """
    frame = load_frame(data)
    require_columns(frame, [group_column, relevance, *score_columns])
    w = np.asarray(weights if weights else [1.0] * len(score_columns), float)
    if len(w) != len(score_columns):
        raise ValueError("`weights` must have one value per score column")
    z = pd.DataFrame(index=frame.index)
    for c in score_columns:
        grp = frame.groupby(group_column)[c]
        if normalise == "zscore":
            z[c] = (frame[c] - grp.transform("mean")) / grp.transform("std").replace(0, 1).fillna(1)
        elif normalise == "minmax":
            z[c] = (frame[c] - grp.transform("min")) / (grp.transform("max") - grp.transform("min")).replace(0, 1)
        else:
            z[c] = grp.rank(pct=True)
    blended = (z.to_numpy() * (w / w.sum())).sum(axis=1)
    out = frame.assign(blend=blended)
    report = {c: summarise(per_group_metrics(frame, group_column, relevance, c, k), k)[f"ndcg@{k}"] for c in score_columns}
    blend_ndcg = summarise(per_group_metrics(out, group_column, relevance, "blend", k), k)[f"ndcg@{k}"]
    best_single = max(report.values())
    return {"normalise": normalise, "weights": w.tolist(), f"ndcg@{k}_inputs": report, f"ndcg@{k}_blend": blend_ndcg,
            "blend_beats_best_input": bool(blend_ndcg is not None and best_single is not None and blend_ndcg > best_single),
            "written_to": save_frame(out, output_path, "blended.csv") if output_path else None}
