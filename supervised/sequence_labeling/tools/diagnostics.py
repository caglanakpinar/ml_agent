"""Read-only checks on a tagged corpus before any tagger is fitted: size, tag set and scheme, invalid transitions, ambiguous words, and out-of-vocabulary rate."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ml_core.tooling import tool
from supervised.sequence_labeling.tools._common import (
    bio_spans,
    load_sequences,
    scheme_of,
    split_sentences,
)


@tool
def profile_tagged_corpus(data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id") -> dict:
    """Profile a tagged corpus: sentences, tokens, sentence-length distribution, the tag set with counts, the tag scheme (BIO / BIOES / IOB1 / plain), the share of tokens outside any entity, and vocabulary size."""
    sentences = load_sequences(data, token_column, tag_column, sentence_column)
    tags = Counter(t for s in sentences for _, t in s)
    lengths = np.array([len(s) for s in sentences])
    words = Counter(w.lower() for s in sentences for w, _ in s)
    scheme = scheme_of(list(tags))
    return {"sentences": len(sentences), "tokens": int(lengths.sum()), "sentence_length": {"min": int(lengths.min()), "median": float(np.median(lengths)), "p95": float(np.quantile(lengths, .95)), "max": int(lengths.max())},
            "scheme": scheme, "tag_count": len(tags), "tags": dict(tags.most_common()), "outside_share": round(tags.get("O", 0) / lengths.sum(), 4) if "O" in tags else None,
            "vocabulary": len(words), "hapax_share": round(sum(1 for c in words.values() if c == 1) / len(words), 4),
            "task_guess": "entity extraction (span F1 is the metric)" if scheme != "plain" else "per-token tagging such as POS (token accuracy is the metric)"}


@tool
def validate_tag_sequences(data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id", k: int = 10) -> dict:
    """Find invalid tag sequences: I- tags that follow O or a different entity type, B/E/S misuse in BIOES, empty tags. Invalid sequences poison training and silently lower span F1.

    Args:
        k: How many offending sentences to list.
    """
    sentences = load_sequences(data, token_column, tag_column, sentence_column)
    scheme = scheme_of([t for s in sentences for _, t in s])
    bad = []
    for n, sent in enumerate(sentences):
        problems, prev = [], "O"
        for i, (word, tag) in enumerate(sent):
            if not tag.strip():
                problems.append((i, "empty tag"))
            elif scheme in ("BIO", "BIOES") and tag.startswith(("I-", "E-")):
                kind = tag.split("-", 1)[1]
                prev_ok = prev != "O" and prev.split("-", 1)[0] in ("B", "I") and prev.split("-", 1)[1] == kind
                if not prev_ok:
                    problems.append((i, f"{tag} does not continue an entity of the same type (previous: {prev})"))
            prev = tag
        if problems:
            bad.append({"sentence": n, "tokens": " ".join(w for w, _ in sent)[:90], "problems": [f"token {i}: {m}" for i, m in problems[:3]]})
    return {"scheme": scheme, "sentences_checked": len(sentences), "sentences_with_problems": len(bad), "examples": bad[:k],
            "fix": "run repair_tag_sequences to rewrite stray I- tags as B-" if bad else "none needed"}


@tool
def find_ambiguous_tokens(data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id", min_count: int = 5, k: int = 15) -> dict:
    """List words that carry more than one tag in the corpus with the split between them — the ceiling on what a word-only tagger can reach, and a pointer to annotation inconsistencies.

    Args:
        min_count: Only words seen at least this often.
        k: How many words to list.
    """
    sentences = load_sequences(data, token_column, tag_column, sentence_column)
    by_word: dict[str, Counter] = defaultdict(Counter)
    for s in sentences:
        for w, t in s:
            by_word[w.lower()][t] += 1
    rows = [{"word": w, "count": sum(c.values()), "tags": dict(c.most_common(4)), "majority_share": round(c.most_common(1)[0][1] / sum(c.values()), 3)}
            for w, c in by_word.items() if len(c) > 1 and sum(c.values()) >= min_count]
    rows.sort(key=lambda r: r["majority_share"])
    total = sum(sum(c.values()) for c in by_word.values())
    ceiling = sum(c.most_common(1)[0][1] for c in by_word.values()) / total
    return {"ambiguous_words": len(rows), "word_only_accuracy_ceiling": round(float(ceiling), 4), "most_ambiguous": rows[:k],
            "reading": "a word-only (most-frequent-tag) tagger cannot exceed the ceiling; context features must close the gap"}


@tool
def check_vocabulary_overlap(data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id", test_size: float = 0.2, seed: int = 42) -> dict:
    """Share of held-out tokens (and entity tokens) never seen in training — the out-of-vocabulary rate that decides whether memorising features (word identity, gazetteers) will generalise or whether shape/suffix/context features are needed."""
    sentences = load_sequences(data, token_column, tag_column, sentence_column)
    train, test = split_sentences(sentences, test_size, seed)
    seen = {w.lower() for s in train for w, _ in s}
    tokens = [(w, t) for s in test for w, t in s]
    oov = [(w, t) for w, t in tokens if w.lower() not in seen]
    entity = [(w, t) for w, t in tokens if t != "O"]
    entity_oov = [(w, t) for w, t in entity if w.lower() not in seen]
    rate = len(oov) / len(tokens)
    return {"train_sentences": len(train), "test_sentences": len(test), "oov_rate": round(rate, 4),
            "entity_token_oov_rate": round(len(entity_oov) / len(entity), 4) if entity else None,
            "advice": "high OOV: prefer a feature-based tagger (perceptron, CRF, logistic) over HMM/gazetteer, or a pretrained transformer" if rate > 0.2 else "low OOV: word identity is informative; even simple taggers will do well"}


@tool
def entity_statistics(data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id", k: int = 5) -> dict:
    """Entity counts per type, length distribution, the most frequent entity strings, and entities that appear with several types — the class balance and difficulty of an extraction task.

    Args:
        k: How many most frequent entity strings to list per type.
    """
    from supervised.sequence_labeling.tools._common import to_bio

    sentences = load_sequences(data, token_column, tag_column, sentence_column)
    per_type: dict[str, Counter] = defaultdict(Counter)
    lengths: dict[str, list[int]] = defaultdict(list)
    by_string: dict[str, set] = defaultdict(set)
    for s in sentences:
        for a, b, kind in bio_spans(to_bio([t for _, t in s])):
            text = " ".join(w for w, _ in s[a:b])
            per_type[kind][text] += 1
            lengths[kind].append(b - a)
            by_string[text.lower()].add(kind)
    return {"types": {kind: {"entities": sum(c.values()), "distinct": len(c), "mean_tokens": round(float(np.mean(lengths[kind])), 2), "max_tokens": max(lengths[kind]),
                             "most_frequent": dict(c.most_common(k))} for kind, c in per_type.items()},
            "strings_with_several_types": {s: sorted(t) for s, t in by_string.items() if len(t) > 1},
            "rare_types": [kind for kind, c in per_type.items() if sum(c.values()) < 30]}
