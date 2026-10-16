"""Running an approved plan: the console calls each tool itself, and a model only writes up what came back.

The deciders produce a plan — a list of `{"tool", "arguments", "why"}` — and until now a worker *model* was asked to carry it out by emitting tool
calls in its own format. The small model that runs here never did (every worker finished "after 0 tool call(s)" and wrote a generic list of steps),
so nothing was ever measured. The tools are plain Python functions that return `{"status": "ok", ...}` or `{"status": "error", "message": ...}`
and never raise, so there is no reason to route them through a model:

    plan step ──► resolve the tool by name (console/agentic_configurations.yaml `tools:`; `caller` is the function)
              ──► keep the arguments the function takes, add what the plan left out (`data`, and the columns the problem is about)
              ──► call it in a child process with a time limit, in the run's own folder
              ──► keep the result dict, or why the step could not run

Nothing here calls a model. Every result is kept verbatim for the saved run; `digest` makes a compact version for the model that writes the report.
"""

from __future__ import annotations

import importlib
import inspect
import json
import multiprocessing
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STEP_TIMEOUT_SECONDS = 600
DIGEST_CHARS_PER_STEP = 1800
DIGEST_CHARS_TOTAL = 9000

# Arguments that name a column; when the plan leaves one out, the column the problem is about (see `router.column_hints`) is used.
COLUMN_ARGUMENTS = ("target", "label_column", "positive_label", "labels", "duration", "event", "date_column", "value_column", "text_column", "group_column", "relevance", "item_column", "transaction_column")
# Arguments that are files this run writes: given a place in the run's folder so nothing is left in the repository.
OUTPUT_ARGUMENTS = ("output_path", "model_path")


@dataclass
class StepResult:
    index: int
    tool: str
    arguments: dict[str, Any]
    status: str  # ok | error | skipped | timeout
    result: dict[str, Any] | None = None
    note: str = ""
    seconds: float = 0.0
    filled: dict[str, Any] = field(default_factory=dict)  # arguments the executor added, so the report can say where they came from
    dropped: list[str] = field(default_factory=list)  # arguments the plan gave that the function does not take
    role: str = ""  # what the step is for in a recipe (`check`, `baseline`, `candidate`, `model`...), so a score can find it; empty for a decider's plan

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_json(self) -> dict[str, Any]:
        return {"index": self.index, "tool": self.tool, "arguments": self.arguments, "status": self.status, "result": self.result, "note": self.note,
                "seconds": self.seconds, "filled": self.filled, "dropped": self.dropped, "role": self.role}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "StepResult":
        return cls(**{k: data[k] for k in ("index", "tool", "arguments", "status", "result", "note", "seconds", "filled", "dropped", "role") if k in data})


RESULTS_FILE = "results.json"


def save_results(results: list[StepResult], folder: Path, info: dict[str, Any] | None = None) -> Path:
    """Every step of a run, verbatim, next to the files the tools wrote — what a score is computed from afterwards. `info` is what a recipe decided on the way."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / RESULTS_FILE
    path.write_text(json.dumps({"steps": [r.to_json() for r in results], "info": info or {}}, indent=1, default=str))
    return path


def load_results(folder: Path) -> tuple[list[StepResult], dict[str, Any]]:
    saved = json.loads((folder / RESULTS_FILE).read_text())
    return [StepResult.from_json(item) for item in saved["steps"]], saved.get("info", {})


def data_columns(path: str | None) -> set[str]:
    """The column names of the data file, or an empty set when it cannot be read (then nothing is checked against it)."""
    from console.router import read_frame

    frame = read_frame(path, max_rows=5)
    return {str(c) for c in frame.columns} if frame is not None else set()


def tool_table() -> dict[str, dict[str, Any]]:
    """Every tool of every topic by the name agents call it, as the console config declares it (`caller`, `args`)."""
    from console.topics import load_console_config

    return {tool["name"]: tool for tool in load_console_config()["tools"]}


def resolve(tool_name: str, table: dict[str, dict[str, Any]] | None = None):
    """The Python function behind a tool name, or None when no tool has that name."""
    spec = (table or tool_table()).get(tool_name)
    if not spec:
        return None
    module_path, function = spec["caller"].rsplit(".", 1)
    return getattr(importlib.import_module(module_path), function)


def _child(module_path: str, function: str, kwargs: dict[str, Any], folder: str, queue) -> None:
    """Runs in its own process: the tool, in the run's folder, with its answer sent back as JSON-safe data."""
    try:
        os.chdir(folder)
        from ml_core.tooling import jsonable

        queue.put(("ok", jsonable(getattr(importlib.import_module(module_path), function)(**kwargs))))
    except BaseException as error:  # a tool never raises, but an import or a crash in the child must still come back
        queue.put(("error", f"{type(error).__name__}: {error}"))


def call_with_timeout(spec: dict[str, Any], kwargs: dict[str, Any], folder: Path, timeout: float) -> tuple[str, Any]:
    """Call one tool in a child process; `("ok", result)`, `("error", message)` or `("timeout", message)`. A spawned process, not a fork: the
    parent may hold a model on the GPU, which does not survive a fork."""
    module_path, function = spec["caller"].rsplit(".", 1)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_child, args=(module_path, function, kwargs, str(folder), queue), daemon=True)
    process.start()
    try:
        status, payload = queue.get(timeout=timeout)
    except Exception:
        status, payload = "timeout", f"no answer after {timeout:g} seconds; the step was stopped"
    finally:
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join(2)
    return status, payload


def prepare(step: dict[str, Any], table: dict[str, dict[str, Any]], data_path: str | None, hints: dict[str, Any], folder: Path, index: int,
            earlier: list[StepResult], shared: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str], str]:
    """The arguments to call the step's function with: `(kwargs, filled, dropped, problem)`; a non-empty `problem` means the step cannot run."""
    name = step.get("tool")
    function = resolve(name, table)
    if function is None:
        return {}, {}, [], f"no tool named {name!r}"
    parameters = inspect.signature(function).parameters
    given = step.get("arguments") if isinstance(step.get("arguments"), dict) else {}
    kwargs = {k: v for k, v in given.items() if k in parameters}
    dropped = [k for k in given if k not in parameters]
    filled: dict[str, Any] = {}

    if "data" in parameters and (not kwargs.get("data") or (isinstance(kwargs["data"], str) and not Path(kwargs["data"]).expanduser().exists())):
        if not data_path:
            return kwargs, filled, dropped, "this tool needs `data` and no data file is attached to the run"
        kwargs["data"] = data_path
        filled["data"] = data_path

    columns = data_columns(data_path)
    for argument in COLUMN_ARGUMENTS:
        if argument in parameters and kwargs.get(argument) and columns and argument not in ("positive_label",):
            named = kwargs[argument] if isinstance(kwargs[argument], list) else [kwargs[argument]]
            if all(isinstance(n, str) for n in named) and not set(named) <= columns and (shared.get(argument) or hints.get(argument)):
                kwargs.pop(argument)  # the plan named a column the file does not have (once a file name): use the one the problem is about
                dropped.append(f"{argument}={named[0]!r} (not a column of the data)")
        if argument in parameters and not kwargs.get(argument):
            value = shared.get(argument) or hints.get(argument)
            if value:
                kwargs[argument] = value
                filled[argument] = value

    for argument in OUTPUT_ARGUMENTS:
        if argument in parameters and not kwargs.get(argument):
            if argument == "model_path" and parameters[argument].default is inspect.Parameter.empty:
                saved = next((r.arguments["model_path"] for r in reversed(earlier) if r.ok and r.arguments.get("model_path")), None)
                if saved:
                    kwargs[argument] = saved
                    filled[argument] = saved
            elif parameters[argument].default is None:
                suffix = ".joblib" if argument == "model_path" else ".csv"
                kwargs[argument] = str(folder / f"step{index:02d}_{name.removesuffix('_tool')}{suffix}")
                filled[argument] = kwargs[argument]

    missing = [p.name for p in parameters.values() if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.name not in kwargs]
    if missing:
        return kwargs, filled, dropped, f"required argument(s) {', '.join(f'`{m}`' for m in missing)} were not in the plan and could not be worked out from the problem or the file"
    for argument in COLUMN_ARGUMENTS:  # the plan's own choice of a column carries over to the steps that did not name one
        if argument in kwargs and argument not in shared and argument not in filled:
            shared[argument] = kwargs[argument]
    return kwargs, filled, dropped, ""


def execute_plan(steps: list[dict[str, Any]], data_path: str | None, hints: dict[str, Any], folder: Path, timeout: float = STEP_TIMEOUT_SECONDS, echo=print,
                 start_index: int = 1, earlier: list[StepResult] | None = None, retry: bool = True) -> list[StepResult]:
    """Run the plan's steps in order. A step that cannot run (unknown tool, a required argument nobody can supply) is recorded, not fatal.

    A plan may be run in stages (a recipe decides its next steps from what the last ones returned): `start_index` continues the numbering and `earlier` is what the
    stages before it returned, so a step that needs an earlier step's saved model finds it. Only the steps of this stage are returned.

    A step that fails is tried once more without the plan's optional settings, because a decider's settings are often wrong; a recipe chose its settings on purpose,
    so it passes `retry=False` and sees the failure.
    """
    folder.mkdir(parents=True, exist_ok=True)
    table = tool_table()
    results: list[StepResult] = []
    shared: dict[str, Any] = {}
    seen: dict[str, int] = {}  # a call identical to an earlier one (after unknown arguments are dropped) would only repeat it
    for index, step in enumerate(steps, start=start_index):
        name = str(step.get("tool")) if isinstance(step, dict) else "?"
        started = time.time()
        if not isinstance(step, dict):
            results.append(StepResult(index, name, {}, "skipped", note="the step is not an object"))
            continue
        role = str(step.get("role") or "")
        kwargs, filled, dropped, problem = prepare(step, table, data_path, hints, folder, index, [*(earlier or []), *results], shared)
        key = json.dumps([name, {k: v for k, v in kwargs.items() if k not in OUTPUT_ARGUMENTS}], sort_keys=True, default=str)
        if not problem and key in seen:
            echo(f"  step {index}: {name} — not run again (same call as step {seen[key]})")
            continue
        if not problem:
            seen[key] = index
        if problem:
            results.append(StepResult(index, name, kwargs, "skipped", note=problem, filled=filled, dropped=dropped, role=role))
            echo(f"  step {index}: {name} — skipped: {problem}")
            continue
        status, payload = call_with_timeout(table[name], kwargs, folder, timeout)
        retried = ""
        if status == "ok" and isinstance(payload, dict) and payload.get("status") == "error" or status == "error":
            # The plan's optional settings are the likeliest cause of a failure (the decider once passed a column name as `order`): try once without them.
            parameters = inspect.signature(resolve(name, table)).parameters
            optional = [k for k in (step.get("arguments") or {}) if k in kwargs and k in parameters and parameters[k].default is not inspect.Parameter.empty and k not in COLUMN_ARGUMENTS + OUTPUT_ARGUMENTS and k != "data"]
            if optional and retry:
                again = {k: v for k, v in kwargs.items() if k not in optional}
                status2, payload2 = call_with_timeout(table[name], again, folder, timeout)
                if status2 == "ok" and not (isinstance(payload2, dict) and payload2.get("status") == "error"):
                    status, payload, kwargs, retried = status2, payload2, again, f"retried without the plan's optional setting(s) {', '.join(optional)}, which made the first call fail"
                    dropped = dropped + optional
        elapsed = time.time() - started
        if status == "ok" and isinstance(payload, dict) and payload.get("status") == "error":
            status, note = "error", str(payload.get("message", ""))
        else:
            note = payload if isinstance(payload, str) else ""
        note = f"{retried}. {note}".strip(". ") if retried else note
        results.append(StepResult(index, name, kwargs, status, payload if isinstance(payload, dict) else None, note, round(elapsed, 1), filled, dropped, role))
        echo(f"  step {index}: {name} — {status}" + (f": {note[:160]}" if note else "") + f" ({elapsed:.1f}s)")
    return results


# --------------------------------------------------------------------------------------------------------------------------------------
# what the report is written from, and checking it
# --------------------------------------------------------------------------------------------------------------------------------------
def compact(value: Any, limit: int) -> str:
    """JSON of `value` within `limit` characters: long lists are cut to their first items before the text itself is cut."""
    def trim(obj: Any, items: int) -> Any:
        if isinstance(obj, dict):
            return {k: trim(v, items) for k, v in obj.items()}
        if isinstance(obj, list):
            kept = [trim(v, items) for v in obj[:items]]
            return kept + [f"… {len(obj) - items} more"] if len(obj) > items else kept
        if isinstance(obj, float):
            return round(obj, 5)
        return obj

    for items in (20, 8, 4, 2):
        text = json.dumps(trim(value, items), separators=(",", ":"), default=str)
        if len(text) <= limit:
            return text
    return text[:limit] + "…"


def digest(results: list[StepResult]) -> str:
    """The results as a model should read them: one block per step, each cut to fit, the whole inside `DIGEST_CHARS_TOTAL`."""
    share = max(min(DIGEST_CHARS_PER_STEP, DIGEST_CHARS_TOTAL // max(len(results), 1)), 300)
    blocks = []
    for r in results:
        head = f"Step {r.index}: {r.tool} — {r.status}"
        if r.filled:
            head += f" (supplied by the console: {', '.join(k for k in r.filled if k not in OUTPUT_ARGUMENTS)})" if any(k not in OUTPUT_ARGUMENTS for k in r.filled) else ""
        if r.dropped:
            head += f" (the plan's arguments {', '.join(r.dropped)} are not arguments of this tool and were ignored)"
        body = compact(r.result, share) if r.result is not None else r.note
        blocks.append(f"{head}\n{body}")
    return "\n\n".join(blocks)


def numbers_in(text: str) -> list[str]:
    return re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?", text)


def ungrounded_numbers(report: str, results: list[StepResult]) -> list[str]:
    """Numbers in the written report that no tool result contains (as written, as a percentage of a result, or rounded from one). Small whole
    numbers are ignored: they are counts of steps and list numbers, and a check that cried wolf at them would be ignored."""
    pool: list[float] = []

    def collect(obj: Any) -> None:
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            pool.append(float(obj))
        elif isinstance(obj, dict):
            for key, value in obj.items():
                collect(value)
                collect(key if isinstance(key, (int, float)) else None)
        elif isinstance(obj, list):
            for value in obj:
                collect(value)
        elif isinstance(obj, str):
            pool.extend(float(m.replace(",", "")) for m in re.findall(r"-?\d[\d,]*(?:\.\d+)?", obj) if m.replace(",", "").replace(".", "", 1).replace("-", "", 1).isdigit())

    for r in results:
        collect(r.result)
        collect(r.arguments)
    unmatched = []
    for token in numbers_in(report):
        raw = token.rstrip("%").replace(",", "")
        try:
            x = float(raw)
        except ValueError:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if decimals == 0 and abs(x) <= 20:
            continue
        scale = [1, 100] if token.endswith("%") or "." in raw else [1]
        if not any(round(p * s, decimals) == round(x, decimals) or abs(p * s - x) <= 0.5 * 10 ** -decimals for p in pool for s in scale + [1]):
            unmatched.append(token)
    return unmatched


def render_results(results: list[StepResult], limit: int = 6000) -> str:
    """Every step and its result for the saved run — the evidence the report stands on, in full (up to `limit` characters each)."""
    parts = []
    for r in results:
        lines = [f"### Step {r.index}: `{r.tool}` — {r.status}" + (f" ({r.seconds}s)" if r.seconds else "")]
        if r.filled or r.dropped:
            notes = [f"supplied: {k}={v}" for k, v in r.filled.items()] + [f"ignored: {k}" for k in r.dropped]
            lines.append("_" + "; ".join(notes) + "_")
        if r.note:
            lines.append(r.note)
        if r.result is not None:
            text = json.dumps(r.result, indent=1, default=str)
            lines.append("```json\n" + (text if len(text) <= limit else text[:limit] + "\n… [cut]") + "\n```")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------------------------------------------------------------------
# the report: accepted only when it stands on the results, otherwise the console writes its own
# --------------------------------------------------------------------------------------------------------------------------------------
def check_report(report: str, results: list[StepResult]) -> str:
    """Why a model's write-up cannot be used, or an empty string when it can.

    The small model that writes it has, given real results, written a list of twenty tool names that were never run, a path that does not exist and the
    same line over and over. A write-up is accepted only if it names no tool that was not run, repeats no line more than twice, stays short, mentions no file
    path the run did not use, and — when the results hold numbers — uses at least one of them.
    """
    if not report.strip():
        return "it is empty"
    if len(report) > 2500:
        return f"it is {len(report):,} characters long"
    ran = {r.tool for r in results}
    invented = sorted({t for t in re.findall(r"\b[a-z][a-z0-9_]*_tool\b", report) if t not in ran})
    if invented:
        return f"it names tools that were not run ({', '.join(invented[:3])})"
    lines = [line.strip() for line in report.splitlines() if len(line.strip()) > 12]
    if lines and max((lines.count(line) for line in set(lines)), default=0) > 2:
        return "it repeats itself"
    known_paths = " ".join(str(v) for r in results for v in [*r.arguments.values(), r.result or ""])
    stray_paths = [p for p in re.findall(r"(?:/[\w.-]+){3,}", report) if p not in known_paths]
    if stray_paths:
        return f"it mentions a path the run never used ({stray_paths[0]})"
    has_numbers = any(r.result and any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in _walk(r.result)) for r in results)
    if has_numbers and not [t for t in numbers_in(report) if "." in t or len(t.strip("%").replace(",", "")) > 2]:
        return "it uses none of the numbers the steps returned"
    return ""


def _walk(obj: Any):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _walk(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk(value)
    else:
        yield obj


def _plain(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, dict):
        return ", ".join(f"{k}={_plain(v)}" for k, v in value.items() if isinstance(v, (str, int, float, bool)) and len(str(v)) < 60)
    return str(value)


def plain_summary(result: StepResult, limit: int = 900) -> str:
    """One step's output as a short plain sentence-list, with nothing added: the scalars, the first items of each list and the tool's own advice."""
    if result.result is None:
        return result.note or result.status
    parts = []
    for key, value in result.result.items():
        if key == "status":
            continue
        if isinstance(value, (int, float, bool)) or (isinstance(value, str) and len(value) <= 300):
            parts.append(f"{key.replace('_', ' ')}: {_plain(value)}")
        elif isinstance(value, list) and value:
            shown = "; ".join(_plain(item) for item in value[:4])
            parts.append(f"{key.replace('_', ' ')}: {shown}" + (f" (+{len(value) - 4} more)" if len(value) > 4 else ""))
        elif isinstance(value, dict) and value:
            parts.append(f"{key.replace('_', ' ')}: {_plain(value)}"[:260])
    text = "; ".join(parts)
    return text if len(text) <= limit else text[:limit] + "…"


def plain_report(results: list[StepResult]) -> str:
    """The console's own write-up, built from the results alone: what ran, then each step in a line or two."""
    ok = sum(r.ok for r in results)
    lines = [f"{ok} of {len(results)} planned step(s) ran. What each returned (summarised by the console, not by a model):"]
    for r in results:
        if r.ok and r.tool.endswith("catalogue_tool"):
            continue  # a listing of the topic's tools, not a finding
        lines.append(f"- {r.tool} ({r.status}): {plain_summary(r)}")
    return "\n".join(lines)


def diagnostic_playbook(category: str, problem_type: str, hints: dict[str, Any]) -> list[dict[str, Any]]:
    """The topic's own read-only checks, as a plan: every tool in its `diagnostics` module whose required arguments are the data and columns this problem
    supplies (`hints`), in the order the topic declares them, at most eight.

    What the console runs when a sub agent's decider cannot produce a plan: profiles, checks and baselines — real measurements, but not a fitted model,
    and the saved run says so. Nothing is chosen by a model, and nothing is written but the files each tool writes into the run's folder.
    """
    from console.topics import agent_names, load_console_config

    config = load_console_config()
    table = {tool["name"]: tool for tool in config["tools"]}
    worker = agent_names(category, problem_type)[1]
    steps = []
    for name in config["agents"][worker].get("tools") or []:
        spec = table.get(name)
        if not spec or ".tools.diagnostics." not in spec["caller"] or name.endswith("catalogue_tool"):
            continue
        required = [a["name"] for a in spec["args"] if a.get("required") and a["name"] != "data"]
        if all(hints.get(argument) for argument in required):
            steps.append({"tool": name, "arguments": {}, "why": "a read-only check from the topic's default playbook"})
    return steps[:8]
