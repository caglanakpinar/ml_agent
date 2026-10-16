"""The registry of problem topics the console can orchestrate.

Every folder `<category>/<problem_type>/` that holds an `agentic_configuration.yaml` is a *topic*: a decider agent, a worker agent
and the tools they call (see `scaffold.py` for how those files are produced). This module finds them, loads their agent handles,
describes them to the model that has to choose between them, and prepares the data context they are run with.

The topic folder says what a topic *is*; what the console *runs* is `console/agentic_configurations.yaml`, which `scaffold.py console`
builds from all of them — so the handles and the tool lists below come from that one file, the same one `ml_decider_agent` lives in.

    topics = discover_topics()                       # {"supervised": ["classification", ...], ...}
    decider, worker = load_handles("supervised", "classification")
    context = data_context("customers.csv")          # the `Current data path:` line and a column listing
"""

from __future__ import annotations

import importlib
import re
import warnings
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES: tuple[str, ...] = ("supervised", "semisupervised", "unsupervised")
CONFIG_NAME = "agentic_configuration.yaml"
CONSOLE_CONFIG = ROOT / "console" / "agentic_configurations.yaml"
DATA_SUFFIXES = ("csv", "tsv", "parquet", "json", "jsonl", "xlsx", "xls", "txt", "conll", "iob", "bio")


def topic_dir(category: str, problem_type: str, root: Path = ROOT) -> Path:
    return root / category / problem_type


def discover_topics(root: Path = ROOT) -> dict[str, list[str]]:
    """Every runnable problem type per category: a subfolder that has an `agentic_configuration.yaml` (an empty placeholder folder is not one)."""
    found: dict[str, list[str]] = {}
    for category in CATEGORIES:
        base = root / category
        found[category] = (
            sorted(p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith((".", "_")) and (p / CONFIG_NAME).is_file()) if base.is_dir() else []
        )
    return found


def parse_topic(spec: str, root: Path = ROOT) -> tuple[str, str]:
    """`supervised/classification` or `supervised.classification` (or just `classification`, if it is unambiguous) → (category, type)."""
    spec = spec.strip().strip("/").replace(".", "/")
    topics = discover_topics(root)
    if "/" in spec:
        category, problem_type = spec.split("/", 1)
        if problem_type in topics.get(category, []):
            return category, problem_type
    else:
        matches = [(c, spec) for c, names in topics.items() if spec in names]
        if len(matches) == 1:
            return matches[0]
    known = [f"{c}/{n}" for c, names in topics.items() for n in names]
    raise ValueError(f"unknown topic {spec!r}; the runnable topics are: {', '.join(known)}")


def load_config(category: str, problem_type: str, root: Path = ROOT) -> dict[str, Any]:
    return yaml.safe_load((topic_dir(category, problem_type, root) / CONFIG_NAME).read_text())


def load_console_config(path: Path = CONSOLE_CONFIG) -> dict[str, Any]:
    """The console's config: its own agents and every topic's tools, decider and worker."""
    return yaml.safe_load(path.read_text())


def agent_names(category: str, problem_type: str, root: Path = ROOT) -> tuple[str, str]:
    """(decider name, worker name) of a topic, as its own config names them."""
    names = list(load_config(category, problem_type, root)["agents"])
    decider = next(name for name in names if name.endswith("_decider_agent"))
    worker = next(name for name in names if name != decider)
    return decider, worker


def load_handles(category: str, problem_type: str) -> tuple[Any, Any]:
    """(decider handle, worker handle) — built from the console config, where both agents are declared with their tools and upstream agents."""
    from console.agentic import sub_agent

    decider, worker = agent_names(category, problem_type)
    return sub_agent(decider), sub_agent(worker)


def tool_names(category: str, problem_type: str) -> dict[str, set[str]]:
    """Which tool names the decider and the worker are allowed to use (what a plan may mention), as the console config gives them."""
    agents = load_console_config()["agents"]
    return {name: set(agents[name].get("tools") or []) for name in agent_names(category, problem_type)}


def topic_blurb(category: str, problem_type: str, root: Path = ROOT) -> str:
    """What the topic is for and how to tell it from its neighbours — read from its decider prompt, so there is one source of truth."""
    text = (topic_dir(category, problem_type, root) / "decider_prompt.md").read_text()
    head = text.split("\n", 1)[0].strip()  # "You are the decider for <title> problems. <what the problem is>"
    head = re.sub(r"^You are the decider for (.*?) problems\.\s*", r"\1: ", head)
    section = re.search(r"## Is this the right topic\?\n\n(.*?)\n\n## Your tools", text, re.S)
    return f"{head}\n{section.group(1).strip()}" if section else head


def topic_brief(category: str, problem_type: str, root: Path = ROOT) -> str:
    """One sentence on what the topic is for — the opening of its decider prompt, without the confusable-neighbour notes `topic_blurb` adds."""
    text = (topic_dir(category, problem_type, root) / "decider_prompt.md").read_text()
    head = re.sub(r"^You are the decider for (.*?) problems\.\s*", r"\1: ", text.split("\n", 1)[0].strip())
    return re.split(r"(?<=[.!?])\s", head, maxsplit=1)[0]


def describe_available(topics: dict[str, list[str]] | None = None, category: str | None = None, brief: bool = False) -> str:
    """The list the deciding agent chooses from, with a short account of each — exact names first, so a verdict can be matched to them.

    For one category it is a flat list; for all of them the entries are grouped under a `### <category>` heading, which is also the
    `category` the decider must give back. `brief=True` gives one `name [category] — what it is for` line per sub agent instead of the full blurb: the
    name to answer with comes first, the family is a tag after it, and it does not look like the JSON the decider has to write.
    """
    topics = topics or discover_topics()
    if brief:
        return "\n".join(f"- {name} [{cat}] — {topic_brief(cat, name)}" for cat in ([category] if category else CATEGORIES) for name in topics.get(cat, []))
    if category:
        return "\n\n".join(f"- `{name}` — {topic_blurb(category, name)}" for name in topics.get(category, []))
    blocks = []
    for cat in CATEGORIES:
        entries = [f"- `{name}` — {topic_blurb(cat, name)}" for name in topics.get(cat, [])]
        if entries:
            blocks.append(f"### {cat}\n\n" + "\n\n".join(entries))
    return "\n\n".join(blocks)


def extract_data_path(*texts: str) -> str | None:
    """The first file path named in the text that exists on disk (absolute, `~`, or relative to the working directory or the repository)."""
    suffixes = "|".join(DATA_SUFFIXES)
    pattern = re.compile(r"""["'`]?((?:~|\.{1,2})?[\w@+=:,./\\-]*\.(?:""" + suffixes + r"""))\b["'`]?""", re.I)
    for text in texts:
        for candidate in pattern.findall(text or ""):
            for base in (Path.cwd(), ROOT):
                path = (Path(candidate).expanduser() if candidate.startswith(("~", "/")) else base / candidate).resolve()
                if path.is_file():
                    return str(path)
    return None


def data_context(path: str | None, max_columns: int = 60) -> str:
    """The `Current data path:` line the topic prompts rely on, followed by what is in the file when it is a table — so the decider
    names columns that exist instead of guessing them."""
    lines = [f"Current data path: {path or ''}"]
    if not path:
        return lines[0]
    try:
        import pandas as pd

        suffix = Path(path).suffix.lower()
        if suffix in (".csv", ".txt"):
            frame = pd.read_csv(path, nrows=2000)
        elif suffix == ".tsv":
            frame = pd.read_csv(path, sep="\t", nrows=2000)
        elif suffix == ".parquet":
            frame = pd.read_parquet(path)
        elif suffix in (".json",):
            frame = pd.read_json(path)
        elif suffix == ".jsonl":
            frame = pd.read_json(path, lines=True, nrows=2000)
        elif suffix in (".xlsx", ".xls"):
            frame = pd.read_excel(path, nrows=2000)
        else:
            return lines[0]
        total = sum(1 for _ in open(path, "rb")) - 1 if suffix in (".csv", ".tsv", ".txt", ".jsonl") else len(frame)
        cols = ", ".join(f"{c} ({frame[c].dtype}{', ' + str(frame[c].nunique()) + ' distinct' if frame[c].nunique() <= 12 else ''})" for c in list(frame.columns)[:max_columns])
        more = f" … and {len(frame.columns) - max_columns} more" if len(frame.columns) > max_columns else ""
        lines.append(f"The file has about {total} rows and {len(frame.columns)} columns: {cols}{more}.")
    except Exception as error:  # a context line is a convenience; never let it stop a run
        lines.append(f"(the file could not be previewed: {error})")
    return "\n".join(lines)


def data_signals(path: str | None, *texts: str, max_lines: int = 8) -> str:
    """Plain facts read off the data file that bear on which kind of problem it is, for the decider to read next to the column list.

    A small model reads a column list poorly. What it needs is the answer to the questions that decide the family and the type: is the
    column the problem talks about filled in on every row, on a few, or is it a yes/no, a set of categories or a number; is there more than one 0/1 label
    column; is there a date, a long free-text column, a group id. Only counts and names that are in the file are stated, nothing is guessed about the
    meaning, and a file that cannot be read gives an empty string.
    """
    if not path:
        return ""
    try:
        import pandas as pd

        suffix = Path(path).suffix.lower()
        readers = {".csv": lambda: pd.read_csv(path, nrows=20000), ".txt": lambda: pd.read_csv(path, nrows=20000), ".tsv": lambda: pd.read_csv(path, sep="\t", nrows=20000),
                   ".parquet": lambda: pd.read_parquet(path), ".jsonl": lambda: pd.read_json(path, lines=True, nrows=20000)}
        if suffix not in readers:
            return ""
        frame = readers[suffix]()
    except Exception:
        return ""
    if frame.empty:
        return ""

    text = " ".join(texts)
    lines: list[str] = []
    n = len(frame)

    def mentioned(col: str) -> bool:
        return bool(re.search(rf"(?<![\w]){re.escape(str(col))}(?![\w])", text))

    for col in [c for c in frame.columns if mentioned(c)][:4]:
        series = frame[col].replace("", pd.NA)
        filled = series.notna().mean()
        values = series.dropna()
        distinct = values.nunique()
        shown = ", ".join(str(v) for v in values.drop_duplicates().head(6))
        if filled < 0.98:
            extra = f"; the only value ever present is `{shown}`, nothing is marked as the opposite" if distinct == 1 else f"; {distinct} distinct labels ({shown})"
            lines.append(f"`{col}` has a value on only {filled:.1%} of rows and is blank on the rest{extra}")
        elif distinct == 2:
            lines.append(f"`{col}` is filled on every row and takes two values ({shown}) — a yes/no column")
        elif pd.api.types.is_numeric_dtype(values) and distinct <= 12 and (values % 1 == 0).all():
            lines.append(f"`{col}` is filled on every row and takes {distinct} whole-number levels ({', '.join(str(v) for v in sorted(values.unique())[:12])})")
        elif pd.api.types.is_numeric_dtype(values):
            lines.append(f"`{col}` is filled on every row and is a number with {distinct} distinct values (from {values.min():g} to {values.max():g})")
        elif distinct <= 12:
            lines.append(f"`{col}` is filled on every row and holds {distinct} categories ({shown})")

    zero_one = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and set(frame[c].dropna().unique()) <= {0, 1} and frame[c].nunique() == 2]
    prefixes: dict[str, list[str]] = {}
    for c in zero_one:
        if "_" in str(c):
            prefixes.setdefault(str(c).split("_", 1)[0] + "_", []).append(str(c))
    for prefix, cols in prefixes.items():
        if len(cols) >= 3:
            lines.append(f"{len(cols)} columns share the prefix `{prefix}` and are all 0/1 ({', '.join(cols[:5])}…) — one row can carry several of them")

    def parses_as_dates(col: str) -> bool:
        if pd.api.types.is_numeric_dtype(frame[col]) or not re.search(r"date|time|timestamp|publish|_at$|^ts$", str(col), re.I):
            return False
        sample = frame[col].dropna().astype(str).head(200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return len(sample) > 0 and pd.to_datetime(sample, errors="coerce").notna().mean() > 0.9

    dates = [c for c in frame.columns if parses_as_dates(c)]
    if dates:
        lines.append(f"`{dates[0]}` is a date/time column")
    long_text = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c]) and frame[c].dropna().astype(str).str.len().median() > 80]
    if long_text:
        lines.append(f"`{long_text[0]}` is free text (median {int(frame[long_text[0]].dropna().astype(str).str.len().median())} characters per row)")
    groups = [c for c in frame.columns if re.search(r"(query|session|basket|transaction|group|series|engine|user|customer|card|store)_?id$|^(query|session|basket|group)$", str(c), re.I) and frame[c].nunique() < 0.5 * n]
    if groups:
        lines.append(f"`{groups[0]}` repeats across rows ({frame[groups[0]].nunique():,} distinct values in {n:,} rows) — rows belong to groups")

    return (f"What the file shows (counts read from the first {n:,} rows):\n" + "\n".join(f"- {line}" for line in lines[:max_lines])) if lines else ""
