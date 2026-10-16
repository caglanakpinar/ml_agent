"""Prepare a tagged corpus: convert tag schemes, repair invalid sequences, split, and export entities."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from ml_core.data import save_frame
from ml_core.tooling import tool
from supervised.sequence_labeling.tools._common import (
    bio_spans,
    from_bio,
    load_sequences,
    scheme_of,
    split_sentences,
    to_bio,
)


def _write_conll(sentences: list[list[tuple[str, str]]], path: str) -> str:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n\n".join("\n".join(f"{w}\t{t}" for w, t in s) for s in sentences) + "\n")
    return str(target)


@tool
def convert_tag_scheme(data: str, to: Literal["BIO", "BIOES", "IOB1"] = "BIO", token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id", output_path: str | None = None) -> dict:
    """Rewrite a corpus's tags into another scheme (BIO ↔ BIOES ↔ IOB1) and save it as CoNLL. BIOES gives sequence models explicit end-of-entity cues; BIO is the most widely supported.

    Args:
        to: Target scheme.
    """
    sentences = load_sequences(data, token_column, tag_column, sentence_column)
    before = scheme_of([t for s in sentences for _, t in s])
    converted = [list(zip([w for w, _ in s], from_bio(to_bio([t for _, t in s]), to))) for s in sentences]
    return {"from": before, "to": to, "sentences": len(converted), "written_to": _write_conll(converted, output_path or f"corpus_{to.lower()}.conll"), "sample": " ".join(f"{w}/{t}" for w, t in converted[0][:10])}


@tool
def repair_tag_sequences(data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id", output_path: str | None = None) -> dict:
    """Repair invalid BIO sequences by rewriting any I- tag that does not continue an entity of the same type as B-, and save the corrected corpus as CoNLL."""
    sentences = load_sequences(data, token_column, tag_column, sentence_column)
    fixed, changes = [], 0
    for s in sentences:
        old = [t for _, t in s]
        new, prev = [], "O"
        for t in old:
            if t.startswith("I-") and not (prev != "O" and prev.split("-", 1)[1] == t.split("-", 1)[1]):
                t = "B-" + t.split("-", 1)[1]
            new.append(t)
            prev = t
        changes += sum(a != b for a, b in zip(old, new))
        fixed.append(list(zip([w for w, _ in s], new)))
    return {"tags_changed": changes, "sentences": len(fixed), "written_to": _write_conll(fixed, output_path or "corpus_repaired.conll")}


@tool
def split_tagged_corpus(data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id", test_size: float = 0.2, dev_size: float = 0.0, seed: int = 42, output_dir: str = "corpus_split") -> dict:
    """Split a corpus into train / (dev) / test CoNLL files by whole sentences, with the tag counts of each so rare entity types are not lost from one side.

    Args:
        dev_size: Share of sentences for a development set; 0 for none.
        output_dir: Directory to write train.conll / dev.conll / test.conll into.
    """
    from collections import Counter

    sentences = load_sequences(data, token_column, tag_column, sentence_column)
    train, test = split_sentences(sentences, test_size, seed)
    dev: list = []
    if dev_size:
        train, dev = split_sentences(train, dev_size / (1 - test_size), seed + 1)
    out = {"train": train, "test": test, **({"dev": dev} if dev else {})}
    paths = {name: _write_conll(part, str(Path(output_dir) / f"{name}.conll")) for name, part in out.items()}
    return {"sentences": {n: len(p) for n, p in out.items()}, "tag_counts": {n: dict(Counter(t for s in p for _, t in s).most_common(8)) for n, p in out.items()}, "written_to": paths}


@tool
def extract_entities(data: str, token_column: str = "token", tag_column: str = "tag", sentence_column: str = "sentence_id", output_path: str | None = None) -> dict:
    """Turn tagged tokens into an entity table — one row per entity span with its text, type, sentence and token offsets — for review or downstream use."""
    sentences = load_sequences(data, token_column, tag_column, sentence_column)
    rows = []
    for n, s in enumerate(sentences):
        for a, b, kind in bio_spans(to_bio([t for _, t in s])):
            rows.append({"sentence": n, "start": a, "end": b, "type": kind, "text": " ".join(w for w, _ in s[a:b])})
    table = pd.DataFrame(rows)
    return {"entities": len(table), "by_type": table["type"].value_counts().to_dict() if len(table) else {}, "written_to": save_frame(table, output_path, "entities.csv"), "sample": table.head(8)}
