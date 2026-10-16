"""Keeping a prompt inside what the model can take.

The local model rejects (or crawls on) a prompt that is too long, and nothing in `agent_builder` trims one. Two things use this module, both
from `console.run.run_agent`:

  - before a call, a prompt whose estimated size is over `PROMPT_TOKEN_BUDGET` is shrunk first;
  - after a call, an error that looks like a context-length rejection shrinks the prompt and retries once.

"Shrinking" is map-reduce over the variable parts of the prompt (the context and each upstream agent's output): the text is split at blank lines
into chunks, the model condenses each chunk on its own, the answers are collected, and if they are still too big the same is done to them
again. Lines that must survive word for word — the `Current data path:` line, above all — are never sent through the model.

Tune with the environment: `ML_AGENT_PROMPT_TOKENS` (budget for a whole prompt, default 8000) and `ML_AGENT_CHUNK_TOKENS` (default 2500).
"""

from __future__ import annotations

import os
import re
from typing import Callable

PROMPT_TOKEN_BUDGET = int(os.environ.get("ML_AGENT_PROMPT_TOKENS", "8000"))
CHUNK_TOKENS = int(os.environ.get("ML_AGENT_CHUNK_TOKENS", "2500"))
MAX_ROUNDS = 3

# Lines kept verbatim, never summarised: a path or a name the model would only be able to garble.
PINNED = re.compile(r"^(Current data path:|Problem:|Data details as given:)", re.I)
OVERFLOW = re.compile(r"context (length|window)|maximum (context|sequence)|too (long|many tokens)|exceed\w* .{0,40}(token|length|limit)|token limit|sequence length|out of memory", re.I)

CONDENSE_PROMPT = """Condense the text below. It is part {part} of {parts} of the background for this task:

TASK: {question}

Keep, word for word, every file path, column name, number, metric and name that could matter for the task; drop repetition and anything irrelevant to it.
Write plain notes, at most {limit} words, and nothing else.

TEXT:
{chunk}
"""


def estimate_tokens(text: str) -> int:
    """A cheap upper-ish estimate (about 3.5 characters a token) — enough to decide whether to shrink, not an exact count."""
    return int(len(text) / 3.5) + 1


def is_overflow_error(error: BaseException) -> bool:
    """Whether an exception reads like the model refusing a prompt for its size."""
    return bool(OVERFLOW.search(f"{type(error).__name__}: {error}"))


def split_chunks(text: str, chunk_tokens: int = CHUNK_TOKENS) -> list[str]:
    """Split at blank lines into pieces of at most `chunk_tokens`; one block bigger than that is cut at line (then character) boundaries."""
    limit = chunk_tokens * 3
    pieces: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        while len(block) > limit:
            cut = block.rfind("\n", 0, limit)
            cut = cut if cut > limit // 2 else limit
            pieces.append(block[:cut])
            block = block[cut:].lstrip("\n")
        if block.strip():
            pieces.append(block)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > limit:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def condense(text: str, question: str, generate: Callable[[str], str], target_tokens: int, chunk_tokens: int = CHUNK_TOKENS) -> str:
    """Shrink `text` to about `target_tokens` by condensing it chunk by chunk, collecting the answers and, if needed, doing it again.

    `generate` is one model call (prompt in, answer out). Pinned lines are lifted out first and put back at the top, untouched. If a round
    does not make the text smaller, or `MAX_ROUNDS` is reached, what is left is cut to the budget rather than looping on.
    """
    pinned = [line for line in text.splitlines() if PINNED.match(line.strip())]
    body = "\n".join(line for line in text.splitlines() if not PINNED.match(line.strip()))
    budget = max(target_tokens - estimate_tokens("\n".join(pinned)), 200)
    for _ in range(MAX_ROUNDS):
        if estimate_tokens(body) <= budget:
            break
        chunks = split_chunks(body, chunk_tokens)
        words = max(60, min(400, int(budget * 0.5 / max(len(chunks), 1))))
        notes = [generate(CONDENSE_PROMPT.format(part=i, parts=len(chunks), question=question[:600], limit=words, chunk=chunk)).strip() for i, chunk in enumerate(chunks, 1)]
        shrunk = "\n\n".join(note for note in notes if note)
        if len(shrunk) >= len(body):
            break
        body = shrunk
    if estimate_tokens(body) > budget:
        body = body[: budget * 3] + "\n… [cut to fit the model's limit]"
    return "\n".join([*pinned, body]).strip()
