"""Reading a dataset in, and writing one back.

Every tool takes `data` — a path, an already-loaded DataFrame, or a list of row dicts — so a tool can be
called by an agent (a path in a prompt) and by code (a frame in memory) with the same signature.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_frame(data: Any) -> pd.DataFrame:
    """Return `data` as a DataFrame. A path may be csv, tsv, parquet, json, jsonl or xlsx."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, (list, tuple)):
        return pd.DataFrame(list(data))
    if isinstance(data, dict):
        return pd.DataFrame(data)
    if data is None or str(data).strip() == "":
        raise ValueError("no data was given — pass `data` as the path from the `Current data path:` line")

    path = Path(str(data)).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")

    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.DataFrame(json.loads(path.read_text()))
    raise ValueError(f"unsupported file type {suffix!r}; use csv, tsv, parquet, json, jsonl or xlsx")


def save_frame(frame: pd.DataFrame, path: str | None, default: str) -> str:
    """Write `frame` to `path` (or `default`) as csv/parquet by extension; returns where it went."""
    target = Path(path or default).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".parquet":
        frame.to_parquet(target, index=False)
    else:
        frame.to_csv(target, index=False)
    return str(target)


def as_list(value: Any) -> list[str] | None:
    """Accept a list, a comma-separated string, a single name or None."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(v) for v in value]


def require_columns(frame: pd.DataFrame, columns: list[str] | str | None) -> list[str]:
    """Check the named columns exist; the error lists what does, so a model can correct itself."""
    wanted = as_list(columns) or []
    absent = [c for c in wanted if c not in frame.columns]
    if absent:
        raise KeyError(f"column(s) {absent} not in the data; columns are {list(frame.columns)[:60]}")
    return wanted
