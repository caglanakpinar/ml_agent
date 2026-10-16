"""Judge a tagger: entity-level scores per type, tag confusion, the actual errors with context, a learning curve, and tagging new text."""

from __future__ import annotations

import re
from collections import Counter

import joblib
import pandas as pd

from ml_core.data import save_frame
from ml_core.tooling import tool
from supervised.sequence_labeling.tools._common import (
    GazetteerTagger,
    HMMTagger,
    LogisticTokenTagger,
    PerceptronTagger,
    bio_spans,
    evaluate_predictions,
    load_sequences,
    split_sentences,
    to_bio,
)

_WORD = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]")


@tool
def evaluate_tagger(model_path: str, data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id") -> dict:
    """Score a saved tagger on a labelled corpus: entity precision / recall / F1 per type (micro and macro) for BIO tags, or per-tag scores for plain tag sets, plus token accuracy.

    Args:
        model_path: Path of the saved tagger (joblib).
    """
    tagger = joblib.load(model_path)
    sentences = load_sequences(data, token_column, tag_column, sentence_column)
    pred = tagger.tag_many([[w for w, _ in s] for s in sentences])
    return {"model": tagger.kind, "sentences": len(sentences), **evaluate_predictions([[t for _, t in s] for s in sentences], pred)}


@tool
def tag_confusion(model_path: str, data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id", k: int = 12) -> dict:
    """The most frequent token-level confusions (gold tag → predicted tag) — shows whether errors are boundaries (B vs I), type mix-ups, or missed entities (anything → O).

    Args:
        model_path: Path of the saved tagger (joblib).
        k: How many confusions to list.
    """
    tagger = joblib.load(model_path)
    sentences = load_sequences(data, token_column, tag_column, sentence_column)
    pred = tagger.tag_many([[w for w, _ in s] for s in sentences])
    pairs = Counter((g, p) for s, ps in zip(sentences, pred) for (_, g), p in zip(s, ps) if g != p)
    kinds = Counter()
    for (g, p), c in pairs.items():
        if p == "O":
            kinds["missed entity (predicted O)"] += c
        elif g == "O":
            kinds["spurious entity (gold O)"] += c
        elif g.split("-")[-1] == p.split("-")[-1]:
            kinds["boundary (right type, wrong B/I)"] += c
        else:
            kinds["wrong type"] += c
    return {"errors": int(sum(pairs.values())), "by_kind": dict(kinds.most_common()), "top_confusions": [{"gold": g, "predicted": p, "count": c} for (g, p), c in pairs.most_common(k)]}


@tool
def tagging_error_analysis(model_path: str, data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id", k: int = 15, output_path: str | None = None) -> dict:
    """List actual entity errors with sentence context: missed entities (false negatives), spurious ones (false positives) and wrong boundaries — the fastest way to see what to fix in features or annotation.

    Args:
        model_path: Path of the saved tagger (joblib).
        k: How many errors to list.
    """
    tagger = joblib.load(model_path)
    sentences = load_sequences(data, token_column, tag_column, sentence_column)
    rows = []
    for n, s in enumerate(sentences):
        words, gold = [w for w, _ in s], to_bio([t for _, t in s])
        pred = to_bio(tagger.tag(words))
        g, p = set(bio_spans(gold)), set(bio_spans(pred))
        for a, b, kind in g - p:
            rows.append({"sentence": n, "error": "missed" if not any(x[0] < b and a < x[1] for x in p) else "boundary/type", "span": " ".join(words[a:b]), "gold": kind,
                         "predicted": next((f"{x[2]}:{' '.join(words[x[0]:x[1]])}" for x in p if x[0] < b and a < x[1]), "O"), "context": " ".join(words[max(0, a - 4) : b + 4])})
        for a, b, kind in p - g:
            if not any(x[0] < b and a < x[1] for x in g):
                rows.append({"sentence": n, "error": "spurious", "span": " ".join(words[a:b]), "gold": "O", "predicted": kind, "context": " ".join(words[max(0, a - 4) : b + 4])})
    table = pd.DataFrame(rows)
    return {"errors": len(table), "by_kind": table["error"].value_counts().to_dict() if len(table) else {}, "examples": table.head(k), "written_to": save_frame(table, output_path, "tagging_errors.csv") if output_path else None}


@tool
def tagger_learning_curve(data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id", model: str = "perceptron", fractions: list[float] | None = None, test_size: float = 0.25, seed: int = 42) -> dict:
    """Held-out score at growing amounts of training data — shows whether more labelled sentences would help, so you know whether to annotate more or change the model.

    Args:
        model: perceptron, hmm, logistic or gazetteer.
        fractions: Shares of the training set to try, e.g. [0.1, 0.25, 0.5, 1.0].
    """
    sentences = load_sequences(data, token_column, tag_column, sentence_column)
    train, test = split_sentences(sentences, test_size, seed)
    gold, words = [[t for _, t in s] for s in test], [[w for w, _ in s] for s in test]
    build = {"perceptron": lambda tr: PerceptronTagger(tr, 5, seed), "hmm": lambda tr: HMMTagger(tr), "logistic": lambda tr: LogisticTokenTagger(tr), "gazetteer": lambda tr: GazetteerTagger(tr)}[model]
    rows = []
    for f in fractions or [0.1, 0.25, 0.5, 0.75, 1.0]:
        part = train[: max(5, int(len(train) * f))]
        scores = evaluate_predictions(gold, build(part).tag_many(words))
        rows.append({"fraction": f, "sentences": len(part), "f1": (scores.get("entity_micro") or {}).get("f1", scores["token_accuracy"])})
    gain = rows[-1]["f1"] - rows[-2]["f1"] if len(rows) > 1 else 0
    return {"model": model, "curve": rows, "verdict": "still improving: more labelled data should help" if gain > 0.01 else "plateaued: change features or model rather than annotating more"}


@tool
def tag_text(model_path: str, texts: list[str], output_path: str | None = None) -> dict:
    """Tag new raw text with a saved tagger: tokenises on words and punctuation, returns each token's tag and the extracted entity spans.

    Args:
        model_path: Path of the saved tagger (joblib).
        texts: Sentences to tag.
    """
    tagger = joblib.load(model_path)
    rows, entities = [], []
    for n, text in enumerate(texts):
        words = _WORD.findall(text)
        tags = tagger.tag(words)
        rows += [{"text_id": n, "token": w, "tag": t} for w, t in zip(words, tags)]
        entities += [{"text_id": n, "type": kind, "text": " ".join(words[a:b])} for a, b, kind in bio_spans(to_bio(tags))]
    table = pd.DataFrame(rows)
    return {"texts": len(texts), "tokens": len(table), "entities": entities, "written_to": save_frame(table, output_path, "tagged_tokens.csv") if output_path else None}
