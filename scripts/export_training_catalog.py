"""Snapshot what the console really sends to its model, for the fine-tuning data generator (`qwen_customized_with_tiny_lora`, v5).

    python scripts/export_training_catalog.py --out /path/to/data/synthetic/ml_agent_v5 [--seeds 6]

Writes two files into `--out`:

    catalog.json        the console's prompts (chat, requirements, decider, reporter, the condense prompt), every topic's decider and worker prompt
                        with its category / problem type / one-line brief, and every tool (name, topic, module, description, arguments, whether it is a
                        read-only diagnostic) — the exact strings the model is shown at inference, so training records can be rendered identically
    results_pool.jsonl  real outputs of the real tools: every tool of every topic called on the repository's own synthetic datasets (`tests/datasets.py`),
                        `--seeds` times with different seeds, one JSON object per call: {tool, topic, dataset, seed, arguments, status, result}.
                        Training answers are written from these results, so a number in a write-up is a number a tool returned.

Nothing is invented here: the catalogue is read off the files and the pool off running the tools. Re-run it whenever a prompt or a tool changes, then rebuild the data.
"""

from __future__ import annotations

import argparse
import importlib
import json
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CALL_SECONDS = 25


class Timeout(Exception):
    pass


def _alarm(signum, frame):  # noqa: ARG001
    raise Timeout()


def export_catalog(out: Path) -> dict:
    from console.budget import CONDENSE_PROMPT
    from console.topics import agent_names, discover_topics, load_console_config, topic_brief

    config = load_console_config()
    tools = {}
    for spec in config["tools"]:
        package = ".".join(spec["caller"].split(".")[:2])
        module = spec["caller"].split(".")[3] if len(spec["caller"].split(".")) > 3 else ""
        tools[spec["name"]] = {
            "name": spec["name"], "topic": package, "module": module, "description": spec["description"], "caller": spec["caller"],
            "diagnostic": module == "diagnostics", "args": spec["args"],
        }
    topics = {}
    for category, names in discover_topics().items():
        for problem_type in names:
            folder = ROOT / category / problem_type
            decider, worker = agent_names(category, problem_type)
            topics[f"{category}/{problem_type}"] = {
                "category": category, "problem_type": problem_type, "brief": topic_brief(category, problem_type),
                "decider_agent": decider, "worker_agent": worker,
                "decider_prompt": (folder / "decider_prompt.md").read_text().strip(), "worker_prompt": (folder / "prompt.md").read_text().strip(),
                "decider_tools": config["agents"][decider].get("tools") or [], "worker_tools": config["agents"][worker].get("tools") or [],
            }
    prompts = {name: (ROOT / "console" / "prompts" / f"{name}.md").read_text().strip() for name in ("chat", "requirements", "decider", "reporter")}
    prompts["condense"] = CONDENSE_PROMPT
    catalog = {"prompts": prompts, "topics": topics, "tools": tools, "categories": ["supervised", "semisupervised", "unsupervised"]}
    (out / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False))
    return catalog


def file_columns(path) -> list[str]:
    """The column names of the data file a tool was called on (so the generator can rename them consistently), or [] when there is none."""
    import pandas as pd

    try:
        if isinstance(path, str) and Path(path).is_file():
            return [str(c) for c in pd.read_csv(path, nrows=0).columns]
    except Exception:  # noqa: BLE001
        pass
    return []


def export_results(out: Path, seeds: int, only: str | None) -> int:
    from ml_core.specs import discover_tools
    from tests import datasets
    from tests.cases import CASES, case_kwargs

    original = np.random.default_rng
    path = out / "results_pool.jsonl"
    written = skipped = 0
    reasons: dict[str, int] = {}
    with path.open("w", encoding="utf-8") as handle:
        for seed in range(seeds):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                # the datasets seed themselves with 0: shift that seed while they are built, and only then (tools call default_rng too)
                np.random.default_rng = lambda s=None, _o=original, _b=seed * 7919: _o((s + _b) if isinstance(s, int) else s)
                try:
                    data = datasets.build(tmp_path / "data")
                finally:
                    np.random.default_rng = original
                for package, case in CASES.items():
                    if only and only not in package:
                        continue
                    for info in discover_tools(package):
                        folder = tmp_path / f"{package}_{info.name}"
                        folder.mkdir(exist_ok=True)
                        try:
                            kwargs = case_kwargs(package, info, data, folder)
                            module = importlib.import_module(info.caller.rsplit(".", 1)[0])
                            function = getattr(module, info.name)
                            signal.signal(signal.SIGALRM, _alarm)
                            signal.alarm(CALL_SECONDS)
                            started = time.time()
                            try:
                                result = function(**kwargs)
                            finally:
                                signal.alarm(0)
                        except Timeout:
                            skipped += 1
                            reasons["timeout"] = reasons.get("timeout", 0) + 1
                            continue
                        except Exception as error:  # noqa: BLE001
                            skipped += 1
                            key = f"{type(error).__name__}: {str(error)[:60]}"
                            reasons[key] = reasons.get(key, 0) + 1
                            continue
                        if result.get("install") or result.get("status") != "ok":
                            skipped += 1
                            key = "install" if result.get("install") else f"status: {str(result.get('message'))[:60]}"
                            reasons[key] = reasons.get(key, 0) + 1
                            continue
                        shown = {k: (v if not str(v).startswith(str(tmp_path)) else Path(str(v)).name) for k, v in kwargs.items()}
                        columns = file_columns(kwargs.get("data"))
                        handle.write(json.dumps({"tool": info.tool_name, "topic": package, "seed": seed, "arguments": shown, "columns": columns,
                                                 "seconds": round(time.time() - started, 2), "result": result}, ensure_ascii=False, default=str) + "\n")
                        handle.flush()
                        written += 1
            print(f"  seed {seed}: {written:,} results so far ({skipped:,} calls skipped)", flush=True)
    np.random.default_rng = original
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  skipped {n:>5}  {reason}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=6, help="how many times to call every tool, each on freshly seeded data")
    parser.add_argument("--only", default=None, help="only topics whose package contains this text (for a quick trial)")
    parser.add_argument("--catalog-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    catalog = export_catalog(args.out)
    print(f"catalog: {len(catalog['topics'])} topics, {len(catalog['tools'])} tools, {len(catalog['prompts'])} prompts -> {args.out / 'catalog.json'}")
    if not args.catalog_only:
        count = export_results(args.out, args.seeds, args.only)
        print(f"results pool: {count:,} real tool outputs -> {args.out / 'results_pool.jsonl'}")


if __name__ == "__main__":
    main()
