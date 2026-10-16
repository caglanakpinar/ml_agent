"""Fit one sequence tagger and score it on held-out sentences, against a most-frequent-tag baseline."""

from __future__ import annotations

from typing import Literal

from ml_core.tooling import tool
from supervised.sequence_labeling.tools._common import (
    CRFTagger,
    FrequencyTagger,
    GazetteerTagger,
    HMMTagger,
    LogisticTokenTagger,
    PerceptronTagger,
    evaluate_predictions,
    load_sequences,
    run_tagger,
    split_sentences,
)


@tool
def fit_most_frequent_tag_baseline(data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id", test_size: float = 0.2, seed: int = 42, model_path: str | None = None) -> dict:
    """Tag each word with the tag it most often had in training (unknown words get the corpus-majority tag) — the floor every real tagger must clear, and a measure of how ambiguous the words are."""
    return run_tagger(lambda tr: FrequencyTagger(tr), "most_frequent_tag", data, token_column, tag_column, sentence_column, test_size, seed, model_path)


@tool
def fit_gazetteer_tagger(data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id", min_count: int = 1, test_size: float = 0.2, seed: int = 42, model_path: str | None = None) -> dict:
    """Build a dictionary of every entity string seen in training and tag text by longest match — precise and instant for closed vocabularies (product names, countries), blind to unseen entities.

    Args:
        min_count: Keep only entity strings seen at least this often.
    """
    return run_tagger(lambda tr: GazetteerTagger(tr, min_count), f"gazetteer(min_count={min_count})", data, token_column, tag_column, sentence_column, test_size, seed, model_path)


@tool
def fit_hmm_tagger(data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id", smoothing: float = 0.1, rare_below: int = 2, test_size: float = 0.2, seed: int = 42, model_path: str | None = None) -> dict:
    """Fit a bigram hidden Markov model tagger with Viterbi decoding — a classic generative sequence model; fast and data-efficient, but it sees only the current word and the previous tag.

    Args:
        smoothing: Add-k smoothing for transitions and emissions.
        rare_below: Words seen fewer times are pooled into shape/suffix classes so the model learns how to tag unknowns.
    """
    return run_tagger(lambda tr: HMMTagger(tr, smoothing, rare_below), f"hmm(smoothing={smoothing})", data, token_column, tag_column, sentence_column, test_size, seed, model_path)


@tool
def fit_perceptron_tagger(data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id", n_iter: int = 8, window: int = 2, test_size: float = 0.2, seed: int = 42, model_path: str | None = None) -> dict:
    """Fit an averaged perceptron tagger: rich word-shape, affix and neighbour features plus the previous tags, greedy left-to-right. Quick to train and a strong feature-based baseline with no extra packages.

    Args:
        n_iter: Training epochs.
        window: Words of context on each side used as features.
    """
    return run_tagger(lambda tr: PerceptronTagger(tr, n_iter, seed, window), f"averaged_perceptron(iter={n_iter})", data, token_column, tag_column, sentence_column, test_size, seed, model_path)


@tool
def fit_logistic_token_tagger(data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id", C: float = 1.0, window: int = 2, test_size: float = 0.2, seed: int = 42, model_path: str | None = None) -> dict:
    """Fit a logistic-regression classifier on each token's window features, independently per token — simple and calibrated, but with no notion of tag consistency across a span.

    Args:
        C: Inverse regularisation strength.
        window: Words of context on each side used as features.
    """
    return run_tagger(lambda tr: LogisticTokenTagger(tr, C, window), f"logistic_token(C={C})", data, token_column, tag_column, sentence_column, test_size, seed, model_path)


@tool(requires=("sklearn_crfsuite",))
def fit_crf_tagger(data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id", c1: float = 0.1, c2: float = 0.1, max_iter: int = 100, window: int = 2, test_size: float = 0.2, seed: int = 42, model_path: str | None = None) -> dict:
    """Fit a linear-chain conditional random field (sklearn-crfsuite) — the classical strongest feature-based sequence model: it scores whole tag sequences, so tags stay consistent across an entity.

    Args:
        c1: L1 regularisation.
        c2: L2 regularisation.
        max_iter: Optimiser iterations.
        window: Words of context on each side used as features.
    """
    return run_tagger(lambda tr: CRFTagger(tr, c1, c2, max_iter, window), "crf", data, token_column, tag_column, sentence_column, test_size, seed, model_path)


@tool
def compare_taggers(
    data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id", models: list[str] | None = None,
    test_size: float = 0.25, seed: int = 42,
) -> dict:
    """Fit several taggers on the same split and rank them by entity micro-F1 (token accuracy for plain tag sets) — the quickest way to see whether context features and sequence structure are worth their cost.

    Args:
        models: Any of most_frequent, gazetteer, hmm, perceptron, logistic, crf. Defaults to all that are installed.
    """
    import importlib.util

    sentences = load_sequences(data, token_column, tag_column, sentence_column)
    train, test = split_sentences(sentences, test_size, seed)
    builders = {"most_frequent": lambda: FrequencyTagger(train), "gazetteer": lambda: GazetteerTagger(train), "hmm": lambda: HMMTagger(train),
                "perceptron": lambda: PerceptronTagger(train, 6, seed), "logistic": lambda: LogisticTokenTagger(train)}
    if importlib.util.find_spec("sklearn_crfsuite"):
        builders["crf"] = lambda: CRFTagger(train)
    gold = [[t for _, t in s] for s in test]
    words = [[w for w, _ in s] for s in test]
    board = []
    for name in models or list(builders):
        if name not in builders:
            board.append({"model": name, "error": f"unavailable; choose from {list(builders)}"})
            continue
        try:
            scores = evaluate_predictions(gold, builders[name]().tag_many(words))
            micro = scores.get("entity_micro")
            board.append({"model": name, **({"precision": micro["precision"], "recall": micro["recall"], "f1": micro["f1"]} if micro else {"f1": scores["token_accuracy"]}), "token_accuracy": scores["token_accuracy"]})
        except Exception as error:
            board.append({"model": name, "error": str(error)[:100]})
    board.sort(key=lambda r: -r.get("f1", -1))
    return {"metric": "entity micro-F1" if any("precision" in r for r in board) else "token accuracy", "sentences_train": len(train), "sentences_test": len(test), "leaderboard": board, "best": board[0]["model"]}


@tool(requires=("transformers", "torch"))
def tag_with_pretrained_transformer(texts: list[str], model_name: str = "dslim/bert-base-NER", aggregation: Literal["simple", "first", "average", "max", "none"] = "simple", output_path: str | None = None) -> dict:
    """Tag raw text with a pretrained Hugging Face token-classification model (default: BERT NER) — no training data needed, often the strongest start for common entity types; downloads the model on first use.

    Args:
        texts: Sentences or documents to tag.
        model_name: Hugging Face model id of a token-classification model.
        aggregation: How word pieces are merged into entities.
    """
    import pandas as pd
    from transformers import pipeline

    from ml_core.data import save_frame

    tagger = pipeline("token-classification", model=model_name, aggregation_strategy=None if aggregation == "none" else aggregation)
    rows = []
    for n, text in enumerate(texts):
        for e in tagger(text):
            rows.append({"text_id": n, "entity": e.get("word"), "type": e.get("entity_group") or e.get("entity"), "score": round(float(e["score"]), 4), "start": e["start"], "end": e["end"]})
    table = pd.DataFrame(rows)
    return {"model": model_name, "texts": len(texts), "entities": len(table), "by_type": table["type"].value_counts().to_dict() if len(table) else {}, "written_to": save_frame(table, output_path, "transformer_entities.csv"), "sample": table.head(8)}
