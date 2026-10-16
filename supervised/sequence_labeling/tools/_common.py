"""Sequence-labelling machinery the tool modules share: reading CoNLL / long-format corpora, tag-scheme
handling, span extraction and entity-level scoring, the token feature function, and the taggers
themselves (frequency baseline, HMM + Viterbi, averaged perceptron, token-level logistic regression and
a gazetteer). Everything is plain Python / numpy / scikit-learn, so a tagger pickles with joblib."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns
from ml_core.ml import save_model

Sentence = list[tuple[str, str]]  # [(token, tag), ...]


# ---- reading ------------------------------------------------------------------------------------
def load_sequences(data: Any, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id") -> list[Sentence]:
    """Read a corpus as a list of sentences of (token, tag). Accepts CoNLL text (token and tag per line, blank line between
    sentences; the tag is the last column) or a long-format table with one row per token."""
    if isinstance(data, list) and data and isinstance(data[0], list):
        return [[(str(t), str(g)) for t, g in sent] for sent in data]
    if isinstance(data, (str, Path)) and Path(str(data)).suffix.lower() in (".conll", ".iob", ".bio", ".txt", ".tsv", ".conllu"):
        sentences: list[Sentence] = []
        current: Sentence = []
        for line in Path(str(data)).expanduser().read_text().splitlines():
            if not line.strip() or line.startswith("-DOCSTART-") or line.startswith("#"):
                if current:
                    sentences.append(current)
                    current = []
                continue
            parts = line.split()
            current.append((parts[0], parts[-1]))
        if current:
            sentences.append(current)
        return sentences
    frame = load_frame(data)
    require_columns(frame, [token_column, tag_column])
    if sentence_column in frame.columns:
        return [list(zip(g[token_column].astype(str), g[tag_column].astype(str))) for _, g in frame.groupby(sentence_column, sort=False)]
    sentences, current = [], []
    for tok, tag in zip(frame[token_column], frame[tag_column]):
        if pd.isna(tok):
            if current:
                sentences.append(current)
                current = []
        else:
            current.append((str(tok), str(tag)))
    return sentences + ([current] if current else [])


def split_sentences(sentences: list[Sentence], test_size: float, seed: int) -> tuple[list[Sentence], list[Sentence]]:
    order = list(range(len(sentences)))
    random.Random(seed).shuffle(order)
    cut = max(1, int(len(order) * test_size))
    test = [sentences[i] for i in order[:cut]]
    train = [sentences[i] for i in order[cut:]]
    return train, test


# ---- tag schemes and spans ----------------------------------------------------------------------
def scheme_of(tags: list[str]) -> str:
    present = {t.split("-")[0] for t in tags if "-" in t}
    if {"E", "S"} & present:
        return "BIOES"
    if "B" in present:
        return "BIO"
    if "I" in present:
        return "IOB1"
    return "plain"


def to_bio(tags: list[str]) -> list[str]:
    """Normalise BIOES / IOB1 to BIO (IOB2). Plain tag sets are returned unchanged."""
    out, prev = [], "O"
    for tag in tags:
        if tag == "O" or "-" not in tag:
            out.append(tag)
            prev = tag
            continue
        prefix, kind = tag.split("-", 1)
        if prefix in ("S", "U"):
            new = f"B-{kind}"
        elif prefix in ("E", "L"):
            new = f"I-{kind}"
        elif prefix == "I":
            prev_kind = prev.split("-", 1)[1] if "-" in prev else None
            new = tag if prev_kind == kind and prev != "O" else f"B-{kind}"
        else:
            new = tag
        out.append(new)
        prev = new
    return out


def from_bio(tags: list[str], scheme: str) -> list[str]:
    if scheme == "BIO":
        return list(tags)
    spans = bio_spans(tags)
    out = ["O"] * len(tags)
    for start, end, kind in spans:
        if scheme == "BIOES":
            if end - start == 1:
                out[start] = f"S-{kind}"
            else:
                out[start], out[end - 1] = f"B-{kind}", f"E-{kind}"
                for i in range(start + 1, end - 1):
                    out[i] = f"I-{kind}"
        else:  # IOB1: B- only between two adjacent spans of the same type
            for i in range(start, end):
                out[i] = f"I-{kind}"
            if start > 0 and out[start - 1] != "O" and out[start - 1].endswith(kind):
                out[start] = f"B-{kind}"
    return out


def bio_spans(tags: list[str]) -> list[tuple[int, int, str]]:
    """(start, end_exclusive, type) for each entity in a BIO sequence; a stray I- starts a span."""
    spans, start, kind = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, _, t = tag.partition("-")
        if start is not None and (tag == "O" or prefix == "B" or t != kind):
            spans.append((start, i, kind))
            start = kind = None
        if tag != "O" and start is None and "-" in tag:
            start, kind = i, t
    return spans


# ---- scoring ------------------------------------------------------------------------------------
def span_scores(gold: list[list[str]], pred: list[list[str]]) -> dict[str, Any]:
    """Entity-level precision / recall / F1 per type, micro and macro (exact boundary and type match)."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred):
        gs, ps = set(bio_spans(to_bio(g))), set(bio_spans(to_bio(p)))
        for s in gs & ps:
            tp[s[2]] += 1
        for s in ps - gs:
            fp[s[2]] += 1
        for s in gs - ps:
            fn[s[2]] += 1
    kinds = sorted(set(tp) | set(fp) | set(fn))

    def prf(a: int, b: int, c: int) -> dict[str, float]:
        p, r = a / (a + b) if a + b else 0.0, a / (a + c) if a + c else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r) if p + r else 0.0, 4)}

    per = {k: {**prf(tp[k], fp[k], fn[k]), "support": tp[k] + fn[k]} for k in kinds}
    micro = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return {"entity_micro": micro, "entity_macro_f1": round(float(np.mean([v["f1"] for v in per.values()])), 4) if per else 0.0, "per_type": per}


def token_scores(gold: list[list[str]], pred: list[list[str]]) -> dict[str, Any]:
    flat_g = [t for s in gold for t in s]
    flat_p = [t for s in pred for t in s]
    acc = float(np.mean([a == b for a, b in zip(flat_g, flat_p)]))
    per = {}
    for tag in sorted(set(flat_g)):
        tp = sum(1 for a, b in zip(flat_g, flat_p) if a == b == tag)
        fp = sum(1 for a, b in zip(flat_g, flat_p) if b == tag and a != tag)
        fn = sum(1 for a, b in zip(flat_g, flat_p) if a == tag and b != tag)
        p, r = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
        per[tag] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r) if p + r else 0.0, 4), "support": tp + fn}
    return {"token_accuracy": round(acc, 4), "token_macro_f1": round(float(np.mean([v["f1"] for v in per.values()])), 4), "per_tag": per}


def evaluate_predictions(gold: list[list[str]], pred: list[list[str]]) -> dict[str, Any]:
    out = token_scores(gold, pred)
    if scheme_of([t for s in gold for t in s]) != "plain":
        out = {**span_scores(gold, pred), "token_accuracy": out["token_accuracy"]}
    else:
        out = {"token_accuracy": out["token_accuracy"], "token_macro_f1": out["token_macro_f1"], "per_tag": out["per_tag"]}
    return out


# ---- features -----------------------------------------------------------------------------------
def shape(word: str) -> str:
    s = re.sub("[A-Z]", "X", word)
    s = re.sub("[a-z]", "x", s)
    s = re.sub("[0-9]", "d", s)
    return re.sub(r"(.)\1{2,}", r"\1\1", s)


def token_features(words: list[str], i: int, window: int = 2) -> dict[str, float]:
    w = words[i]
    f: dict[str, float] = {"bias": 1.0, "w.lower=" + w.lower(): 1.0, "w.shape=" + shape(w): 1.0, "w.suffix2=" + w[-2:].lower(): 1.0,
                           "w.suffix3=" + w[-3:].lower(): 1.0, "w.prefix2=" + w[:2].lower(): 1.0, "w.prefix3=" + w[:3].lower(): 1.0}
    if w.isupper():
        f["w.isupper"] = 1.0
    if w.istitle():
        f["w.istitle"] = 1.0
    if w.isdigit():
        f["w.isdigit"] = 1.0
    if "-" in w:
        f["w.hyphen"] = 1.0
    for d in range(1, window + 1):
        for sign in (-1, 1):
            j = i + sign * d
            if 0 <= j < len(words):
                n = words[j]
                f[f"{sign * d}:lower={n.lower()}"] = 1.0
                if d == 1:
                    f[f"{sign * d}:shape={shape(n)}"] = 1.0
                    f[f"{sign * d}:istitle={n.istitle()}"] = 1.0
            else:
                f[f"{sign * d}:edge"] = 1.0
    return f


# ---- taggers ------------------------------------------------------------------------------------
class Tagger:
    kind = "base"

    def tag(self, words: list[str]) -> list[str]:
        raise NotImplementedError

    def tag_many(self, sentences: list[list[str]]) -> list[list[str]]:
        return [self.tag(s) for s in sentences]


class FrequencyTagger(Tagger):
    kind = "most_frequent"

    def __init__(self, sentences: list[Sentence], lowercase: bool = True):
        self.lowercase = lowercase
        counts: dict[str, Counter] = defaultdict(Counter)
        overall: Counter = Counter()
        for sent in sentences:
            for w, t in sent:
                counts[w.lower() if lowercase else w][t] += 1
                overall[t] += 1
        self.best = {w: c.most_common(1)[0][0] for w, c in counts.items()}
        self.default = overall.most_common(1)[0][0]

    def tag(self, words: list[str]) -> list[str]:
        return [self.best.get(w.lower() if self.lowercase else w, self.default) for w in words]


def unknown_symbol(word: str) -> str:
    if word.isdigit():
        return "<UNK-num>"
    if word.isupper() and len(word) > 1:
        return "<UNK-upper>"
    if word.istitle():
        return "<UNK-title>"
    return f"<UNK-{word[-3:].lower()}>"


class HMMTagger(Tagger):
    """Bigram HMM: add-k smoothed transitions and emissions, rare words pooled into shape/suffix classes, Viterbi decoding."""

    kind = "hmm"

    def __init__(self, sentences: list[Sentence], smoothing: float = 0.1, rare_below: int = 2):
        self.smoothing = smoothing
        freq = Counter(w.lower() for s in sentences for w, _ in s)
        self.known = {w for w, c in freq.items() if c >= rare_below}
        tags = sorted({t for s in sentences for _, t in s})
        self.tags, self.index = tags, {t: i for i, t in enumerate(tags)}
        trans = np.full((len(tags) + 1, len(tags)), smoothing)  # last row = start
        emit: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(tags)))
        for sent in sentences:
            prev = len(tags)
            for w, t in sent:
                trans[prev, self.index[t]] += 1
                emit[self.symbol(w)][self.index[t]] += 1
                prev = self.index[t]
        self.log_trans = np.log(trans / trans.sum(axis=1, keepdims=True))
        totals = sum(emit.values())
        vocab = len(emit) + 1
        self.log_emit = {s: np.log((v + smoothing) / (totals + smoothing * vocab)) for s, v in emit.items()}
        self.log_unseen = np.log(smoothing / (totals + smoothing * vocab))

    def symbol(self, word: str) -> str:
        return word.lower() if word.lower() in self.known else unknown_symbol(word)

    def tag(self, words: list[str]) -> list[str]:
        if not words:
            return []
        n, k = len(words), len(self.tags)
        score = np.full((n, k), -np.inf)
        back = np.zeros((n, k), dtype=int)
        for i, w in enumerate(words):
            e = self.log_emit.get(self.symbol(w), self.log_unseen)
            if i == 0:
                score[0] = self.log_trans[k] + e
            else:
                cand = score[i - 1][:, None] + self.log_trans[:k]
                back[i] = cand.argmax(axis=0)
                score[i] = cand.max(axis=0) + e
        path = [int(score[-1].argmax())]
        for i in range(n - 1, 0, -1):
            path.append(int(back[i][path[-1]]))
        return [self.tags[j] for j in reversed(path)]


class PerceptronTagger(Tagger):
    """Greedy left-to-right averaged perceptron with previous-tag features."""

    kind = "perceptron"

    def __init__(self, sentences: list[Sentence] | None = None, n_iter: int = 8, seed: int = 0, window: int = 2):
        self.window = window
        self.weights: dict[str, dict[str, float]] = {}
        self.tags: set[str] = set()
        if sentences:
            self.train(sentences, n_iter, seed)

    def _feats(self, words: list[str], i: int, prev: str, prev2: str) -> dict[str, float]:
        f = token_features(words, i, self.window)
        f["prev_tag=" + prev] = 1.0
        f["prev_tags=" + prev2 + "|" + prev] = 1.0
        return f

    def _predict(self, feats: dict[str, float]) -> str:
        scores: dict[str, float] = defaultdict(float)
        for name, value in feats.items():
            for tag, weight in self.weights.get(name, {}).items():
                scores[tag] += value * weight
        return max(self.tags, key=lambda t: (scores.get(t, 0.0), t))

    def train(self, sentences: list[Sentence], n_iter: int, seed: int) -> None:
        self.tags = {t for s in sentences for _, t in s}
        totals: dict[tuple[str, str], float] = defaultdict(float)
        stamps: dict[tuple[str, str], int] = defaultdict(int)
        step = 0
        rng = random.Random(seed)
        data = list(sentences)

        def update(feats: dict[str, float], truth: str, guess: str) -> None:
            for name, value in feats.items():
                for tag, delta in ((truth, value), (guess, -value)):
                    key = (name, tag)
                    w = self.weights.setdefault(name, {}).get(tag, 0.0)
                    totals[key] += (step - stamps[key]) * w
                    stamps[key] = step
                    self.weights[name][tag] = w + delta

        for _ in range(n_iter):
            rng.shuffle(data)
            for sent in data:
                words = [w for w, _ in sent]
                prev, prev2 = "<S>", "<S>"
                for i, (_, truth) in enumerate(sent):
                    step += 1
                    feats = self._feats(words, i, prev, prev2)
                    guess = self._predict(feats)
                    if guess != truth:
                        update(feats, truth, guess)
                    prev2, prev = prev, truth  # teacher forcing
        for name, tags in self.weights.items():
            for tag, w in list(tags.items()):
                key = (name, tag)
                total = totals[key] + (step - stamps[key]) * w
                avg = total / max(step, 1)
                if abs(avg) > 1e-6:
                    tags[tag] = avg
                else:
                    del tags[tag]

    def tag(self, words: list[str]) -> list[str]:
        out, prev, prev2 = [], "<S>", "<S>"
        for i in range(len(words)):
            t = self._predict(self._feats(words, i, prev, prev2))
            out.append(t)
            prev2, prev = prev, t
        return out


class LogisticTokenTagger(Tagger):
    """Independent per-token classifier on window features (no tag history) — fast, strong with good features."""

    kind = "logistic"

    def __init__(self, sentences: list[Sentence], C: float = 1.0, window: int = 2):
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression

        self.window = window
        self.vec = DictVectorizer()
        X = self.vec.fit_transform([token_features([w for w, _ in s], i, window) for s in sentences for i in range(len(s))])
        y = [t for s in sentences for _, t in s]
        self.clf = LogisticRegression(C=C, max_iter=1000).fit(X, y)

    def tag(self, words: list[str]) -> list[str]:
        if not words:
            return []
        return list(self.clf.predict(self.vec.transform([token_features(words, i, self.window) for i in range(len(words))])))


class GazetteerTagger(Tagger):
    """Dictionary matcher: longest-match of entity strings seen in training, tagged in BIO. A strong NER baseline."""

    kind = "gazetteer"

    def __init__(self, sentences: list[Sentence], min_count: int = 1):
        counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        for s in sentences:
            tags = to_bio([t for _, t in s])
            for a, b, kind in bio_spans(tags):
                counts[tuple(w.lower() for w, _ in s[a:b])][kind] += 1
        self.entries = {k: c.most_common(1)[0][0] for k, c in counts.items() if sum(c.values()) >= min_count}
        self.longest = max((len(k) for k in self.entries), default=1)

    def tag(self, words: list[str]) -> list[str]:
        out, i = ["O"] * len(words), 0
        low = [w.lower() for w in words]
        while i < len(words):
            for size in range(min(self.longest, len(words) - i), 0, -1):
                kind = self.entries.get(tuple(low[i : i + size]))
                if kind:
                    out[i] = f"B-{kind}"
                    for j in range(i + 1, i + size):
                        out[j] = f"I-{kind}"
                    i += size
                    break
            else:
                i += 1
        return out


class CRFTagger(Tagger):
    kind = "crf"

    def __init__(self, sentences: list[Sentence], c1: float = 0.1, c2: float = 0.1, max_iter: int = 100, window: int = 2):
        import sklearn_crfsuite

        self.window = window
        self.crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=max_iter, all_possible_transitions=True)
        self.crf.fit([[token_features([w for w, _ in s], i, window) for i in range(len(s))] for s in sentences], [[t for _, t in s] for s in sentences])

    def tag(self, words: list[str]) -> list[str]:
        return list(self.crf.predict_single([token_features(words, i, self.window) for i in range(len(words))])) if words else []


def run_tagger(builder: Any, label: str, data: Any, token_column: str, tag_column: str, sentence_column: str, test_size: float, seed: int, model_path: str | None) -> dict[str, Any]:
    """Split sentences, fit a tagger with `builder(train)`, score it on the held-out sentences, and report the hardest tags."""
    sentences = load_sequences(data, token_column, tag_column, sentence_column)
    if len(sentences) < 10:
        raise ValueError(f"only {len(sentences)} sentences; need at least 10 to split")
    train, test = split_sentences(sentences, test_size, seed)
    tagger = builder(train)
    pred = tagger.tag_many([[w for w, _ in s] for s in test])
    gold = [[t for _, t in s] for s in test]
    scores = evaluate_predictions(gold, pred)
    baseline = FrequencyTagger(train)
    base_scores = evaluate_predictions(gold, baseline.tag_many([[w for w, _ in s] for s in test]))
    headline = "entity_micro" if "entity_micro" in scores else "token_accuracy"
    return {"model": label, "sentences_train": len(train), "sentences_test": len(test), "tokens_train": sum(len(s) for s in train), "holdout": scores,
            "most_frequent_baseline": base_scores[headline], "model_path": save_model(tagger, model_path)}
