"""Runs one sub agent pair: the topic's decider agent proposes how to solve the problem, a human approves the plan, and its worker agent
carries it out with the topic's tools. Both agents are declared in `console/agentic_configurations.yaml`.

    console/run.py                          console/orchestrate.py                      console/agentic_configurations.yaml
    ml_decider_agent names the sub agents → run_topic_workflow()  (once per sub agent) → <topic>_decider_agent  (read-only checks, a JSON plan)
                                              ├ the data path, resolved once up front    <topic>_agent          (every tool, runs the plan)
                                              ├ decider → plan → human approval
                                              └ worker with the plan as `{agent_output}`

Everything known so far flows into both agents: the problem, the data details, the data path and its columns, the conversation, and — for a
sub agent that is not the first — what the earlier ones found (`topic_context`); plus the outputs of the console agents upstream
(`requirement_agent`, `ml_decider_agent`), which `dependency_agent` in the YAML routes into `{agent_output}`.

The plan is validated before anyone is asked to approve it: it must be a JSON object with a non-empty `plan`, and every `tool` it names
must be one the worker really has. A plan naming a tool that does not exist is rejected automatically, the same way `run.py` rejects a
problem type that is not one of this repository's folders — there is nothing to execute from a name that names nothing.

Non-interactive use (`python cli.py solve ...`) passes `interactive=False`: no questions are asked, the first valid plan is accepted
and a missing data path is an error.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import click

from console.executor import diagnostic_playbook
from console.recipes import has_recipe
from console.router import column_hints
from console.topics import (
    ROOT,
    data_context,
    extract_data_path,
    load_handles,
    tool_names,
    topic_dir,
)

MAX_PLAN_ATTEMPTS = 3
# Whether a sub agent's worker *model* carries the approved plan out (`--worker`), as it did before `console/executor.py`. By default the console runs
# the plan's tool calls itself and a model only writes up the results; the worker model never made a tool call on the benchmark.
USE_WORKER = False
MAX_PLAN_STEPS = 12


def dedupe_steps(steps: list[Any]) -> list[Any]:
    """The plan's steps without exact repeats (same tool, same arguments — the `why` is ignored), at most `MAX_PLAN_STEPS`, in first-seen order."""
    seen: set[str] = set()
    kept: list[Any] = []
    for step in steps:
        key = json.dumps([step.get("tool"), step.get("arguments")], sort_keys=True, default=str) if isinstance(step, dict) else json.dumps(step, default=str)
        if key not in seen:
            seen.add(key)
            kept.append(step)
    return kept[:MAX_PLAN_STEPS]


def recover_plan(output: str, plan: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str, str]:
    """Make the most of a decider reply that did not come back as one clean JSON object. Returns `(plan, text, note)`; `note` is empty when
    nothing had to be done.

    The small model that runs the deciders tends to write a few good steps and then repeat them until it hits its token ceiling, so the reply is
    cut off in the middle of the JSON and the whole thing is thrown away. The steps it did write are usable, so they are read one by one
    out of the `"plan": [` list (each one must parse completely), exact repeats are dropped, and the plan is rebuilt around them. The
    result still goes through `validate_plan`, so a step that names a tool the worker does not have is still rejected. A plan that did parse
    is only de-duplicated.
    """
    if plan and isinstance(plan.get("plan"), list) and plan["plan"]:
        deduped = dedupe_steps(plan["plan"])
        if len(deduped) == len(plan["plan"]):
            return plan, output, ""
        dropped = len(plan["plan"]) - len(deduped)
        plan = {**plan, "plan": deduped}
        return plan, json.dumps(plan, indent=2), f"dropped {dropped} repeated or surplus step(s)"
    start = re.search(r'"plan"\s*:\s*\[', output)
    if not start:
        return plan, output, ""
    decoder, position, steps = json.JSONDecoder(), start.end(), []
    while True:
        match = re.compile(r"[\s,]*").match(output, position)
        position = match.end() if match else position
        if position >= len(output) or output[position] != "{":
            break
        try:
            step, position = decoder.raw_decode(output, position)
        except ValueError:  # the reply was cut off in the middle of this step
            break
        if isinstance(step, dict) and step.get("tool"):
            steps.append(step)
    steps = dedupe_steps(steps)
    if not steps:
        return plan, output, ""
    approach = re.search(r'"approach"\s*:\s*"([^"]*)"', output)
    metric = re.search(r'"primary_metric"\s*:\s*"([^"]*)"', output)
    rebuilt = {"approach": approach.group(1) if approach else "", "plan": steps, "primary_metric": metric.group(1) if metric else "", "risks": [], "escalation": None}
    return rebuilt, json.dumps(rebuilt, indent=2), f"the reply was cut off or repeating itself; kept its {len(steps)} distinct step(s)"


def validate_plan(plan: dict[str, Any] | None, allowed: set[str]) -> list[str]:
    """What is wrong with a decider's plan; an empty list means it can be run."""
    if not plan:
        return ["the reply was not a JSON object"]
    steps = plan.get("plan")
    if not isinstance(steps, list) or not steps:
        return ["`plan` must be a non-empty list of tool calls"]
    problems = []
    for index, step in enumerate(steps, start=1):
        name = step.get("tool") if isinstance(step, dict) else None
        if name not in allowed:
            problems.append(f"step {index} names the tool {name!r}, which the worker does not have")
    return problems


def resolve_data_path(data_path: str | None, *texts: str, interactive: bool = True) -> str | None:
    """The explicit path if given and real; else one found in the conversation; else (interactively) ask for it."""
    from console.run import ask

    if data_path:
        path = Path(data_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"no such data file: {data_path}")
        return str(path.resolve())
    found = extract_data_path(*texts)
    if found or not interactive:
        return found
    while True:
        answer = ask("\nI could not find a data file in what you told me. Path to the data (blank to continue without one): ")
        if answer is None or not answer.strip():
            return None
        path = Path(answer.strip().strip("'\"")).expanduser()
        if path.is_file():
            return str(path.resolve())
        click.secho(f"  {path} does not exist.", fg="yellow")


def topic_context(base: str, category: str, problem_type: str, problem: str, data_details: str, path: str | None, earlier: str = "") -> str:
    """Everything the topic agents are told: the conversation, what was decided, the data (path first, as the prompts expect) and what any
    sub agent that ran before this one found."""
    parts = [data_context(path), f"Category: {category}\nProblem type: {problem_type}"]
    if problem:
        parts.append(f"Problem: {problem}")
    if data_details:
        parts.append(f"Data details as given: {data_details}")
    if base:
        parts.append(f"Conversation so far:\n{base}")
    if earlier:
        parts.append(f"Sub agents that ran before this one, and what they found (their result files are named in it):\n{earlier}")
    return "\n\n".join(parts)


def run_plan_stage(
    decider: Any, question: str, context: str, allowed: set[str], problem: str, interactive: bool = True, agent_outputs: dict[str, str] | None = None,
    fallback: bool = False,
) -> tuple[str, dict[str, Any]] | None:
    """Run the decider until it produces a valid plan a human approves. `agent_outputs` is what the agents upstream of it produced.
    Returns `(raw_text, plan)`, or None when the user ends the session or no valid plan could be obtained."""
    from console.run import approve, as_json, rejection_prefix, run_agent

    prefix, invalid = "", 0
    while True:
        click.secho(f"\n[{decider.NAME}] deciding how to solve it...", fg="yellow")
        try:
            output = run_agent(decider, question, context, dict(agent_outputs or {}), prefix)
        except ValueError as error:
            click.secho(f"{decider.NAME} could not run: {error}", fg="red")
            return None

        click.secho(f"\n[{decider.NAME}]", fg="green", bold=True)
        click.echo(output)
        plan = as_json(output)
        plan, output, note = recover_plan(output, plan)
        if note:
            click.secho(f"  ({note})", fg="yellow")
        problems = validate_plan(plan, allowed)

        if plan and plan.get("escalation"):
            click.secho(f"\nThe decider escalated instead of planning: {plan['escalation']}", fg="yellow", bold=True)
            return output, plan

        if problems:
            invalid += 1
            click.secho(f"  (the plan is not usable — {'; '.join(problems)})", fg="yellow")
            if invalid >= MAX_PLAN_ATTEMPTS:
                if fallback:
                    click.secho("The decider did not produce a usable plan.", fg="red")
                    return output, {"fallback": True}
                click.secho("The decider did not produce a usable plan; stopping here.", fg="red")
                return None
            prefix = rejection_prefix(decider, problem, output + "\n\nProblems: " + "; ".join(problems))
            continue

        approved = True if not interactive else approve(decider.NAME)
        if approved is None:
            return None
        if approved:
            return output, plan or {}
        click.secho(f"Not approved — {decider.NAME} will search and go deeper.", fg="yellow")
        prefix = rejection_prefix(decider, problem, output)


def shown(path: Path) -> str:
    """A path relative to the repository when it is inside it, absolute otherwise."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def save_run(category: str, problem_type: str, question: str, context: str, plan: str, result: str, stamp: str | None = None) -> Path:
    """Keep the run next to the topic that produced it: `<category>/<problem_type>/runs/<timestamp>.md` (files the tools wrote are in `<timestamp>_files/`)."""
    folder = topic_dir(category, problem_type) / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stamp or time.strftime('%Y%m%d-%H%M%S')}.md"
    path.write_text(f"# {category}/{problem_type}\n\n## Request\n\n{question}\n\n## Context\n\n{context}\n\n## Plan\n\n```json\n{plan.strip()}\n```\n\n## Result\n\n{result.strip()}\n")
    return path


def run_topic_workflow(
    category: str, problem_type: str, question: str, base_context: str = "", problem: str = "", data_details: str = "", data_path: str | None = None, interactive: bool = True,
    upstream: dict[str, str] | None = None, earlier: str = "", data_resolved: bool = False, earlier_steps: list[Any] | None = None,
) -> dict[str, Any] | None:
    """Decide → approve → execute for one sub agent pair. Returns `{"plan", "result", "saved_to", "data_path"}`, or None when it was ended early.

    Args:
        upstream: Outputs of the agents before this pair, by name (`requirement_agent`, `ml_decider_agent`) — both agents get them as `{agent_output}`.
        earlier: What the sub agents that ran before this one found, shown to both agents in their context.
        data_resolved: The caller has already looked for (and, interactively, asked for) the data file, so a missing `data_path` is final.
        earlier_steps: The steps (`executor.StepResult`) the sub agents before this one ran, so a recipe can use what they fitted (the groups an earlier clustering found).
    """
    from console.run import run_agent

    decider, worker = load_handles(category, problem_type)
    allowed = tool_names(category, problem_type)[worker.NAME]
    upstream = dict(upstream or {})

    path = data_path if data_resolved and not data_path else resolve_data_path(data_path, data_details, problem, base_context, interactive=interactive)
    if path:
        click.secho(f"\nData: {path}", fg="cyan")
    else:
        click.secho("\nNo data file is attached to this run; the agents will have to say what they need.", fg="yellow")
    context = topic_context(base_context, category, problem_type, problem, data_details, path, earlier)

    click.secho(f"\n=== {category}/{problem_type}: decide ===", fg="cyan", bold=True)
    hints = column_hints(" ".join([question, problem, data_details, base_context]), path)
    planned = run_plan_stage(decider, question, context, allowed, problem or question, interactive, upstream, fallback=not USE_WORKER)
    if planned is None:
        click.secho("\nEnding the run here.", fg="yellow")
        return None
    plan_text, plan = planned
    if plan.get("fallback") and path and has_recipe(category, problem_type):
        plan = {"approach": "the topic's default recipe (the decider produced no usable plan): its read-only checks, a baseline, the candidates compared, the cross-validation winner fitted and scored",
                "plan": [], "recipe": True, "primary_metric": "", "risks": [], "escalation": None}
        plan_text = json.dumps(plan, indent=2)
        click.secho("\nFalling back to the topic's default recipe: checks → baseline → candidates → fit the winner → evaluate.", fg="yellow", bold=True)
        if interactive and not approve("the default recipe"):
            click.secho("\nEnding the run here.", fg="yellow")
            return None
    elif plan.get("fallback"):
        steps = diagnostic_playbook(category, problem_type, hints)
        if not steps:
            click.secho("\nNo read-only check of this topic can be run with what the problem and the file say; ending the run here.", fg="yellow")
            return None
        plan = {"approach": "the topic's read-only diagnostics (the decider produced no usable plan)", "plan": steps, "primary_metric": "", "risks": ["diagnostics only: no model was fitted"], "escalation": None}
        plan_text = json.dumps(plan, indent=2)
        click.secho(f"\nFalling back to the topic's own read-only checks ({len(steps)} of them):", fg="yellow", bold=True)
        click.echo("\n".join(f"  - {step['tool']}" for step in steps))
        if interactive:
            approved = approve("the diagnostics playbook")
            if not approved:
                click.secho("\nEnding the run here.", fg="yellow")
                return None
    if plan.get("escalation"):
        saved = save_run(category, problem_type, question, context, plan_text, f"Escalated by the decider: {plan['escalation']}")
        click.secho(f"\nSaved {shown(saved)}.", fg="green")
        return {"plan": plan, "result": None, "escalation": plan["escalation"], "saved_to": str(saved), "data_path": path}

    click.secho(f"\n=== {category}/{problem_type}: solve ===", fg="cyan", bold=True)
    if USE_WORKER:
        click.secho(f"\n[{worker.NAME}] working through the plan...", fg="yellow")
        try:
            result = run_agent(worker, question, context, {**upstream, decider.NAME: plan_text})
        except ValueError as error:
            click.secho(f"{worker.NAME} could not run: {error}", fg="red")
            return None
        click.secho(f"\n[{worker.NAME}]", fg="green", bold=True)
        click.echo(result)
        saved = save_run(category, problem_type, question, context, plan_text, result)
        click.secho(f"\nSaved {shown(saved)}.", fg="green", bold=True)
        return {"plan": plan, "result": result, "saved_to": str(saved), "data_path": path}

    result, full, ran = execute_and_report(category, problem_type, question, "", plan, path, hints, earlier_steps)
    saved = save_run(category, problem_type, question, context, plan_text, full, stamp=ran["stamp"])
    click.secho(f"\nSaved {shown(saved)}.", fg="green", bold=True)
    return {"plan": plan, "result": result, "saved_to": str(saved), "data_path": path, "steps": ran["counts"], "step_results": ran["results"]}


def execute_and_report(category: str, problem_type: str, question: str, words: str, plan: dict[str, Any], path: str | None, hints: dict[str, Any] | None = None,
                       earlier_steps: list[Any] | None = None) -> tuple[str, str, dict[str, Any]]:
    """Run the approved plan's tools in the console and have a model write up what they returned.

    A plan with `"recipe": true` is the topic's default recipe (`console/recipes.py`): its steps are chosen as it goes, from what the earlier ones returned.
    Either way the steps are saved verbatim as `results.json` beside the files the tools wrote, which is what `benchmarks/score.py` reads.

    Returns `(result, full, ran)`: `result` is the report with a one-line account of each step (what the next sub agent is told), `full` adds every
    step's complete output (what is saved), and `ran` holds the run's timestamp and the count of steps by outcome.
    """
    from console.agentic import ReporterAgent
    from console.executor import check_report, digest, execute_plan, plain_report, render_results, ungrounded_numbers
    from console.run import run_agent

    from console.executor import save_results
    from console.recipes import run_recipe

    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = topic_dir(category, problem_type) / "runs" / f"{stamp}_files"
    hints = hints if hints is not None else column_hints(words, path)
    info: dict[str, Any] = {}
    if plan.get("recipe") and path:
        click.secho("\n[console] running the topic's default recipe itself", fg="yellow")
        results, info = run_recipe(category, problem_type, path, hints, folder, echo=click.echo, upstream=list(earlier_steps or []))
    else:
        steps = [step for step in plan.get("plan", []) if isinstance(step, dict)]
        click.secho(f"\n[console] running the plan's {len(steps)} step(s) itself", fg="yellow")
        results = execute_plan(steps, path, hints, folder, echo=click.echo)
        save_results(results, folder)
    counts = {status: sum(r.status == status for r in results) for status in ("ok", "error", "skipped", "timeout")}
    click.secho(f"[console] ran {len(results)} step(s): " + ", ".join(f"{n} {status}" for status, n in counts.items() if n), fg="green" if counts["ok"] == len(results) else "yellow", bold=True)

    click.secho("\n[report_writer_agent] writing up the results...", fg="yellow")
    try:
        report = run_agent(ReporterAgent, question, digest(results), {}).strip()
        why_not = check_report(report, results)
    except Exception as error:  # a model that cannot write the report must not lose the measurements
        report, why_not = "", f"the model could not write it ({error})"
    if why_not:
        click.secho(f"  (the model's write-up was not used: {why_not}; the console's own summary of the results follows)", fg="yellow")
        report = plain_report(results)
    click.secho("\n[report]", fg="green", bold=True)
    click.echo(report)
    stray = ungrounded_numbers(report, results)
    warning = f"\n\n_Numbers in this write-up that no step returned: {', '.join(stray)}. Check them against the steps below._" if stray else ""
    summary = "\n".join(f"- step {r.index} `{r.tool}`: {r.status}" + (f" — {r.note[:120]}" if r.note else "") for r in results)
    return f"{report}{warning}\n\nSteps:\n{summary}", f"{report}{warning}\n\n## Steps run\n\n{render_results(results)}", {"stamp": stamp, "counts": counts, "results": results, "info": info}
