"""The pipeline `cli.py generate` drives: chat, a requirements gate, one decision, then the chosen sub agents.

    chat -> requirement_agent -> ml_decider_agent -> sub agent 1 -> sub agent 2 -> ...
                |                       |                  └ each: <topic>_decider_agent -> plan -> you approve -> <topic>_agent
                |                       └ names one or more of the 22 sub agents (console/orchestrate.py runs them in that order)
                └── requirements: false ── back to chat for what is missing    [--script] code_generator_agent writes pipeline.py

Every agent named here is declared in one file, `console/agentic_configurations.yaml` — the console's own four and the two agents of each
problem type (`scaffold.py console` builds that file; `console/topics.py` reads it).

Nothing starts until `requirement_agent` says both of its rules are satisfied — a problem statement, and
a data source with a way to connect to it (see `prompts/requirements.md`). Until then the console keeps
talking to the user, carrying the whole conversation forward as context, because the two halves usually
arrive in different turns ("predict churn" now, "it's in customers.csv" next).

Once the gate passes:

  - The data file is found once, up front (in the conversation, or by asking), so the decider sees what every sub agent will see: the
    `Current data path:` line and the real columns of the file.
  - `ml_decider_agent` reads the problem, the data and what each sub agent is for — `describe_available` lists the ones this repository
    really has, read off the filesystem rather than a hardcoded list — and names one or several `(category, problem_type)` pairs. A
    verdict that names a pair that does not exist, or names one twice, is treated as an automatic rejection.
  - Each chosen sub agent then runs through `console/orchestrate.py`: its decider runs read-only checks on the data and shows the plan it
    proposes for a human verdict, then its worker carries the plan out with the topic's tools. This is where the actual machine learning
    happens. Everything flows down: each sub agent is given the problem, the data, the conversation, the outputs of `requirement_agent` and
    `ml_decider_agent`, and what the sub agents before it found.
  - `code_generator_agent` writes a starter script for each chosen sub agent into this repository's own folder for it — only when asked
    for (`cli.py generate --script`), since the workers now do the work.

The decision stops at the console for a human verdict, the same way every stage in this project always has:

  - **approved** — its verdict is recorded, and the sub agents start.
  - **not approved** — the console runs a web search on that agent's own topic, then the same agent runs
    again with `prompts/not_approved.md` prepended to its rendered prompt, carrying both its rejected
    attempt and what the search found. It keeps its turn until it is approved.
  - **a verdict outside what is actually available** is treated as an automatic rejection — there is
    nothing downstream to build from a name that names nothing real.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Callable

import click

from console.agentic import (
    ChatAgent,
    CodeGeneratorAgent,
    MlDeciderAgent,
    RequirementAgent,
)
from console.budget import PROMPT_TOKEN_BUDGET, condense, estimate_tokens, is_overflow_error
from console.code_tracker import DecisionTracker
from console import orchestrate
from console.orchestrate import resolve_data_path, run_topic_workflow
from console.router import Route, route_problem
from console.topics import ROOT, data_context, data_signals, describe_available, discover_topics
from console.web_search import WebSearchCache, handle_prompt

# Where a rejected agent's web search is cached — one shared cache for every agent.
WEB_SEARCH_CACHE_DIR = Path(__file__).resolve().parent / "dbs" / "web_search_chroma"

# What a rejected agent searches the web for, by agent name. An agent's own name makes a poor query, so
# each one names the subject it would actually want to read about; the problem statement is appended to
# it at search time.
SEARCH_TOPICS: dict[str, str] = {
    MlDeciderAgent.NAME: "choosing the right machine learning problem type for a use case, supervised vs semisupervised vs unsupervised",
}

# The note prepended to a rejected agent's own prompt, with `{rejected}` and `{web_search}` filled in.
NOT_APPROVED_PROMPT = Path(__file__).resolve().parent / "prompts" / "not_approved.md"

# Where `code_generator_agent`'s script is written, relative to `ROOT` — `<category>/<problem_type>/`.
SCRIPT_NAME = "pipeline.py"


# Set by `chat(auto=True)` (`cli.py generate --yes`): every approval is granted and nothing waits on the keyboard. Used to drive the
# whole console unattended, e.g. by `benchmarks/run.py chat`.
AUTO = False
# Whether the rule-based router (`console/router.py`) reads the problem before `ml_decider_agent` does. `cli.py generate --no-router` turns it off.
ROUTER = True
# Unattended, a verdict that keeps failing would otherwise loop forever (a human would eventually give up); stop after this many rejections.
AUTO_MAX_REJECTIONS = 3


def ask(prompt: str) -> str | None:
    """Read one line from the user, or None when they end the session with Ctrl-C/Ctrl-D (or when running unattended)."""
    if AUTO:
        return None
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        click.echo()
        return None


def as_json(text: str) -> dict[str, Any] | None:
    """Parse an agent's reply as the JSON object its prompt asks for.

    A prompt is explicit about returning bare JSON, but a model will still wrap it in a markdown fence
    or a sentence often enough to matter, and one stray fence is not a reason to stall the run — so the
    outermost `{...}` is what gets parsed rather than the whole reply.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError:
        return None

    return parsed if isinstance(parsed, dict) else None


def truthy(value: Any) -> bool:
    """Read a JSON field's boolean intent, whether the model sent an actual bool or the word "true"."""
    return value is True or str(value).strip().lower() == "true"


def is_met(verdict: dict[str, Any]) -> bool:
    """Read the gate's `requirements` field."""
    return truthy(verdict.get("requirements"))


def discover_problem_types(root: Path = ROOT) -> dict[str, list[str]]:
    """The problem types this repository can actually run: one subdirectory per type under each of `supervised/`, `semisupervised/`
    and `unsupervised/` that holds an `agentic_configuration.yaml` (its decider, worker and tools — see `console/topics.py`). Read straight
    off the filesystem, so `ml_decider_agent` is always choosing from what is really here rather than a list that could drift out of
    sync with it, and a folder with no agents behind it is never offered.
    """
    return discover_topics(root)


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web through the shared cache, rendered as the block a prompt can read.

    Best effort by design: the caller is already handling a rejection, and a driver that is not
    installed, a cache that will not open or a search that times out are all reasons to retry with no
    search results rather than to fail the run. Every failure comes back as an empty string.
    """
    try:
        cache = WebSearchCache.open(WEB_SEARCH_CACHE_DIR)
        results, from_cache = asyncio.run(handle_prompt(query, cache, max_results=max_results))
    except Exception as error:  # a missing driver, an unopenable cache, a network failure
        click.secho(f"  (web search unavailable: {error})", fg="yellow")
        return ""

    if not results:
        return ""

    click.secho(
        f"  (web search: {len(results)} result(s){' from cache' if from_cache else ''})", fg="yellow"
    )
    return "\n\n".join(
        f"{position}. {result.title}\n   {result.url}\n   {result.snippet}"
        for position, result in enumerate(results, start=1)
    )


MAX_REJECTED_CHARS = 2000


def rejection_prefix(handle: Any, problem: str, rejected: str) -> str:
    """Build the block prepended to a rejected agent's prompt: the note, its attempt, and a search.

    The attempt is cut to `MAX_REJECTED_CHARS`: what makes a rejected answer wrong is visible in its opening, and a runaway one (a small model that
    never stopped) would otherwise swamp the retry's prompt.
    """
    if len(rejected) > MAX_REJECTED_CHARS:
        rejected = rejected[:MAX_REJECTED_CHARS] + f"\n… [{len(rejected) - MAX_REJECTED_CHARS} more characters cut]"
    if handle.NAME in SEARCH_TOPICS:
        topic = SEARCH_TOPICS[handle.NAME]
    elif handle.NAME.endswith("_decider_agent"):  # a topic's decider: search for how that kind of problem is best approached
        topic = f"{handle.NAME.removesuffix('_decider_agent').replace('_', ' ')} machine learning best practices methods pitfalls"
    else:
        topic = handle.NAME.replace("_", " ")
    found = web_search(f"{topic} {problem}".strip())

    # `.replace` rather than `.format`: the note is a markdown file that may hold literal braces, and
    # only these two placeholders are meant to be substituted.
    return (
        NOT_APPROVED_PROMPT.read_text()
        .replace("{rejected}", rejected)
        .replace("{web_search}", found or "(no results — the search returned nothing or was unavailable)")
    )


def run_agent(handle: Any, question: str, context: str, agent_outputs: dict[str, str], prefix: str = "") -> str:
    """Run one agent — through `.run()`, the same one-call interface every handle exposes — prepending
    `prefix` ahead of its own prompt on a retry.

    `.run()` renders and generates in one call, with nowhere to inject `prefix` ahead of the agent's own
    instructions: the only hook it exposes is `context`, which lands wherever `{context}` happens to sit
    in that agent's prompt — not necessarily first. A rejection needs it first, so a retry builds the
    prompt itself instead of going through `.run()`.
    """
    agent = handle.build()
    prompt = agent.build_prompt(question, context, agent_outputs)
    if not prompt.strip():
        raise ValueError(
            f"{handle.NAME} rendered an empty prompt — its configured `prompt_path` does not exist. "
            "Fix that entry in `agentic_configurations.yaml` before running it."
        )

    def shrink(question: str, context: str, agent_outputs: dict[str, str], prefix: str) -> tuple[str, dict[str, str], str]:
        """Split the variable parts of the prompt, have the model condense each piece, and collect the answers — see `console/budget.py`."""
        over = estimate_tokens(prefix + agent.build_prompt(question, context, agent_outputs)) - PROMPT_TOKEN_BUDGET
        click.secho(f"  ({handle.NAME}: the prompt is about {over + PROMPT_TOKEN_BUDGET} tokens, over the budget of {PROMPT_TOKEN_BUDGET}; condensing it in parts)", fg="yellow")
        share = max(PROMPT_TOKEN_BUDGET // 4, 500)
        condensed_outputs = {name: condense(text, question, agent.generate, share) if estimate_tokens(text) > share else text for name, text in agent_outputs.items()}
        context = condense(context, question, agent.generate, share * 2) if estimate_tokens(context) > share * 2 else context
        prefix = condense(prefix, question, agent.generate, share) if estimate_tokens(prefix) > share else prefix
        return context, condensed_outputs, prefix

    def attempt(question: str, context: str, agent_outputs: dict[str, str], prefix: str) -> str:
        if not prefix:
            return agent.run(question, context, agent_outputs)
        return agent.generate(f"{prefix}\n\n{agent.build_prompt(question, context, agent_outputs)}")

    shrunk = False
    if estimate_tokens(prefix + prompt) > PROMPT_TOKEN_BUDGET:
        context, agent_outputs, prefix = shrink(question, context, agent_outputs, prefix)
        shrunk = True
    try:
        return attempt(question, context, agent_outputs, prefix)
    except Exception as error:
        if shrunk or not is_overflow_error(error):
            raise
        click.secho(f"  ({handle.NAME}: the model rejected the prompt for its size — {error})", fg="yellow")
        context, agent_outputs, prefix = shrink(question, context, agent_outputs, prefix)
        return attempt(question, context, agent_outputs, prefix)


def approve(name: str) -> bool | None:
    """Ask the human to approve what an agent just produced. None when they end the session."""
    if AUTO:
        click.secho(f"\nApprove {name}? [y/n]: y (--yes)", fg="yellow")
        return True
    while True:
        answer = ask(f"\nApprove {name}? [y/n]: ")
        if answer is None:
            return None

        answer = answer.strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False

        click.echo("Please answer y or n.")


def check_sub_agents(verdict: dict[str, Any] | None, available: dict[str, list[str]]) -> tuple[list[tuple[str, str]], str | None]:
    """The `(category, problem_type)` pairs `ml_decider_agent` named, in order — or why the verdict cannot be used.

    The problem type is what the decider is asked for. Its category is looked up here — every problem type belongs to exactly one category — and
    a category the decider wrote that disagrees is simply replaced: the small model that runs this wrote `unsupervised/` in front of every
    answer, rejecting those only taught it nothing. What is still rejected is a problem type that does not exist (there is nothing to run for a
    name that names nothing real), one that sits in more than one category and was not given one of them, and one named twice (a second run of
    the same sub agent would only repeat the first).
    """
    entries = verdict.get("sub_agents") if verdict else None
    if not isinstance(entries, list) or not entries:
        return [], "`sub_agents` must be a non-empty list"
    chosen: list[tuple[str, str]] = []
    for entry in entries:
        if isinstance(entry, str):  # "supervised/classification" or just "classification"
            entry = {"problem_type": entry}
        category = str(entry.get("category", "")).strip() if isinstance(entry, dict) else ""
        problem_type = str(entry.get("problem_type", "")).strip() if isinstance(entry, dict) else ""
        if "/" in problem_type:
            category, problem_type = (part.strip() for part in problem_type.rsplit("/", 1))
        homes = [c for c, types in available.items() if problem_type in types]
        if not homes or (len(homes) > 1 and category not in homes):
            known = ", ".join(t for types in available.values() for t in types)
            return [], f"`{problem_type or '?'}` is not one of the sub agents (the problem types are: {known})"
        category = category if category in homes else homes[0]
        if (category, problem_type) in chosen:
            return [], f"{category}/{problem_type} is named twice"
        chosen.append((category, problem_type))
    return chosen, None


def canonical_verdict(verdict: dict[str, Any] | None, chosen: list[tuple[str, str]], raw: str) -> str:
    """The decider's verdict as the sub agents should read it: unchanged when every category it wrote was right, otherwise the same reasoning with
    each category the one this repository really files the problem type under. A verdict that cannot be parsed is passed on as written."""
    if not verdict:
        return raw
    entries = verdict.get("sub_agents") if isinstance(verdict.get("sub_agents"), list) else []
    written = [(str(e.get("category", "")).strip(), str(e.get("problem_type", "")).strip()) if isinstance(e, dict) else None for e in entries]
    if written == list(chosen):
        return raw  # nothing had to be corrected: pass it on exactly as written
    fixed = []
    for (category, problem_type), entry in zip(chosen, entries):
        fixed.append({"category": category, "problem_type": problem_type, "reasoning": str(entry.get("reasoning", "")) if isinstance(entry, dict) else ""})
    return json.dumps({**verdict, "sub_agents": fixed}, indent=2)


def run_decision_stage(
    handle: Any,
    check: Callable[[dict[str, Any] | None], tuple[Any, str | None]],
    question: str,
    context: str,
    agent_outputs: dict[str, str],
    problem: str,
) -> tuple[Any, str] | None:
    """Run one decision agent to a human-approved verdict, retrying with a web search on rejection — the
    same approve/reject loop every stage in this project has always used. `check` reads the parsed verdict and returns
    `(value, None)`, or `(None, why_not)` when it names something that does not exist, which is treated as an automatic
    rejection: there is nothing downstream can build from a category or problem type this repository does not actually have.

    Returns `(value, raw_text)` once approved, or None when the user ends the session.
    """
    prefix, rejections = "", 0
    while True:
        click.secho(f"\n[{handle.NAME}] working...", fg="yellow")
        try:
            output = run_agent(handle, question, context, agent_outputs, prefix)
        except ValueError as error:
            click.secho(f"{handle.NAME} could not run: {error}", fg="red")
            return None

        click.secho(f"\n[{handle.NAME}]", fg="green", bold=True)
        click.echo(output)

        value, reason = check(as_json(output))
        if reason:
            click.secho(f"  ({handle.NAME}'s verdict cannot be used — {reason} — treating as not approved)", fg="yellow")
            approved: bool | None = False
        else:
            approved = approve(handle.NAME)
            if approved is None:
                return None

        if approved:
            return value, output

        rejections += 1
        if AUTO and rejections >= AUTO_MAX_REJECTIONS:
            click.secho(f"{handle.NAME} was rejected {rejections} times in a row; giving up (unattended run).", fg="red")
            return None
        click.secho(f"Not approved — {handle.NAME} will search and go deeper.", fg="yellow")
        prefix = rejection_prefix(handle, problem, f"{output}\n\nWhy it cannot be used: {reason}" if reason else output)


def default_script_path(category: str, problem_type: str) -> Path:
    """Where a script for this category/problem type belongs — this repository's own folder for it."""
    return ROOT / category / problem_type / SCRIPT_NAME


def write_script(content: str, path: Path) -> Path:
    """Write `content` to `path`, creating its parent directory if it does not exist yet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def generate_code(
    question: str,
    agent_outputs: dict[str, str],
    tracker: DecisionTracker,
    problem: str,
    data_details: str,
    category: str,
    problem_type: str,
) -> None:
    """Write a starter script for `category`/`problem_type`, one of the sub agents `ml_decider_agent` chose and the console approved.

    `code_generator_agent` is asked to write the script itself — see `prompts/code_generator.md` — so
    its narrative can come from what every earlier stage actually decided (`agent_outputs`, rendered
    through its own `dependency_agent` list as `{agent_output}`). Its answer is trusted only as far as
    `DecisionTracker.covers()` can check: the script has to actually name the category and problem type
    that were decided, or this falls back to `tracker.fallback_script()` instead — a plainer script is a
    smaller loss than one that silently drifts from what was decided.

    Best effort, like the rest of a pipeline's tail end: a model failure here does not undo an already
    approved run, so it is logged to the console and swallowed rather than raised.
    """
    fallback = tracker.fallback_script(problem, data_details, category, problem_type)
    path = default_script_path(category, problem_type)
    script = fallback

    try:
        reply = CodeGeneratorAgent.run(
            question=f"{question}\n\nWrite the starter script for the sub agent {category}/{problem_type}.", agent_outputs=agent_outputs
        )
        parsed = as_json(reply)
        candidate = str(parsed["script"]) if parsed and parsed.get("script") else ""
        if candidate and tracker.covers(candidate, category, problem_type):
            script = candidate
            given_path = str(parsed.get("path") or "") if parsed else ""
            if given_path:
                candidate_path = (ROOT / given_path).resolve()
                if str(candidate_path).startswith(str(ROOT / category / problem_type)):
                    path = candidate_path
        else:
            click.secho(
                "  (code_generator_agent's script did not ground itself in what was decided; "
                "using the deterministic starter instead)",
                fg="yellow",
            )
    except Exception as error:
        click.secho(f"  (code generation unavailable: {error})", fg="yellow")

    written = write_script(script, path)
    click.secho(f"\nWrote {written.relative_to(ROOT)}.", fg="green", bold=True)


def decider_context(context: str, problem: str, data_details: str, path: str | None, available: dict[str, list[str]], hint: str = "") -> str:
    """What `ml_decider_agent` is told: the data path and its real columns, the problem, and every sub agent there is — what it is for and
    what it is easily confused with. The conversation comes last, since the first three are what the choice is made from."""
    return "\n\n".join(
        part for part in [
            data_context(path),
            data_signals(path, problem, data_details, context),
            hint,
            f"Problem: {problem}",
            f"Data details as given: {data_details}",
            f"The sub agents that exist — each line is `name [family] — what it is for`. Answer with a `name` from this list, spelled exactly:\n\n{describe_available(available, brief=True)}",
            f"Conversation so far:\n{context}",
        ] if part
    )


def hand_to_sub_agents(
    chosen: list[tuple[str, str]], question: str, context: str, problem: str, data_details: str, path: str | None, agent_outputs: dict[str, str], tracker: DecisionTracker,
) -> list[tuple[str, str]]:
    """Run each chosen sub agent, in order. A sub agent after the first is handed what the ones before it found. Returns the pairs that finished."""
    finished: list[tuple[str, str]] = []
    earlier: list[str] = []
    earlier_steps: list[Any] = []  # the steps themselves, for a sub agent that can use what an earlier one fitted
    for category, problem_type in chosen:
        click.secho(f"\n=== sub agent {len(finished) + 1} of {len(chosen)}: {category}/{problem_type} ===", fg="cyan", bold=True)
        outcome = run_topic_workflow(
            category, problem_type, question, context, problem, data_details, path,
            upstream=agent_outputs, earlier="\n\n".join(earlier), data_resolved=True, interactive=not AUTO, earlier_steps=earlier_steps,
        )
        if outcome is None:
            return finished
        tracker.record("sub agent", f"{category}.{problem_type}", {"saved_to": outcome["saved_to"], "data_path": outcome["data_path"], "escalated": bool(outcome.get("escalation"))})
        finished.append((category, problem_type))
        earlier_steps.extend(outcome.get("step_results") or [])
        found = outcome.get("result") or f"Escalated by its decider: {outcome.get('escalation')}"
        earlier.append(f"### {category}/{problem_type}\n\nSaved to {outcome['saved_to']}\n\n{found}")
    return finished


def user_turns(question: str, context: str) -> str:
    """What the user has said so far, without the assistant's replies, the requirement agent's restatement or anything else a model wrote."""
    turns = [chunk.strip() for chunk in re.split(r"(?m)^(?=(?:User|Assistant): )", context) if chunk.startswith("User: ")]
    spoken = "\n".join(chunk[len("User: "):] for chunk in turns)
    return spoken if question in spoken else f"{spoken}\n{question}".strip()


def routed_note(route: Route) -> str:
    return "Rule check (read from what the user said and from the file):\n" + route.explain()


def choose_sub_agents(question: str, context: str, problem: str, data_details: str, path: str | None, available: dict[str, list[str]], agent_outputs: dict[str, str]) -> tuple[list[tuple[str, str]], str] | None:
    """Which sub agents the problem goes to: the rule-based router reads it first, then `ml_decider_agent` decides what the router could not.

    - The router is clearly ahead → its reading is put to the approver exactly as the model's verdict would have been, and no model is called.
      An approver who refuses it sends the problem to the model, which is then free to name any sub agent.
    - It is not → the model chooses, from the router's shortlist (a name outside it is rejected with the shortlist as the reason).
    - Unattended, and the model still has not produced a verdict that can be used → the router's best reading is taken, so a run is not lost
      to a model that cannot follow the format.
    """
    route = route_problem(user_turns(question, context), path, available) if ROUTER else None
    restrict = None
    if route is not None:
        click.secho("\n[router]", fg="green", bold=True)
        click.echo(route.explain())
        decided = route.decided
        if decided:
            text = json.dumps(route.verdict(), indent=2)
            click.secho(f"\nThe router is clearly ahead: {', '.join(f'{c}/{t}' for c, t in decided)}.", fg="green")
            approved = approve("the router's choice")
            if approved is None:
                return None
            if approved:
                return decided, text
            click.secho("Not approved — handing the choice to ml_decider_agent.", fg="yellow")
        else:
            shortlist = route.shortlist
            restrict = shortlist or None

    def check(parsed: dict[str, Any] | None) -> tuple[Any, str | None]:
        chosen, reason = check_sub_agents(parsed, available)
        if not reason and restrict and any(pair not in restrict for pair in chosen):
            names = ", ".join(f"{c}/{t}" for c, t in restrict)
            return [], f"the rule check narrows the choice to: {names}; choose among those"
        return chosen, reason

    result = run_decision_stage(
        MlDeciderAgent,
        check,
        question,
        decider_context(context, problem, data_details, path, available, routed_note(route) if route is not None and route.jobs and not route.decided else ""),
        agent_outputs,
        problem,
    )
    if result is None and AUTO and route is not None and route.shortlist:
        top = route.shortlist[0]
        click.secho(f"\nml_decider_agent gave no usable verdict; unattended, so taking the router's best reading: {top[0]}/{top[1]}.", fg="yellow")
        best = next((c for job in route.jobs for c in job.candidates if c.problem_type == top[1]), None)
        verdict = {"sub_agents": [{"category": top[0], "problem_type": top[1], "reasoning": "; ".join(best.reasons[:3]) if best else ""}], "reasoning": "Chosen by the rule-based router after the decider gave no usable verdict."}
        return [top], json.dumps(verdict, indent=2)
    return result


def run_pipeline(question: str, context: str, verdict: dict[str, Any], write_script: bool = False) -> bool:
    """Find the data, let `ml_decider_agent` choose the sub agents (stopping at the console for approval), run each of them, and write a
    starter script for each when asked.

    Returns True when the decision was approved and every chosen sub agent's workflow ran, False when the user ended the session partway or
    no problem-type folders exist at all.
    """
    problem = str(verdict.get("problem", ""))
    data_details = str(verdict.get("data_details", ""))
    agent_outputs: dict[str, str] = {
        RequirementAgent.NAME: f"Problem: {problem}\nData details: {data_details}"
    }
    tracker = DecisionTracker()

    available = {category: types for category, types in discover_problem_types().items() if types}
    if not available:
        click.secho("No problem-type folders with agents exist yet.", fg="red")
        return False

    path = resolve_data_path(None, data_details, problem, context, interactive=not AUTO)
    if path:
        click.secho(f"\nData: {path}", fg="cyan")

    click.secho("\n=== which sub agents ===", fg="cyan", bold=True)
    result = choose_sub_agents(question, context, problem, data_details, path, available, agent_outputs)
    if result is None:
        click.secho("\nEnding the run here.", fg="yellow")
        return False

    chosen, decider_output = result
    agent_outputs[MlDeciderAgent.NAME] = canonical_verdict(as_json(decider_output), chosen, decider_output)
    tracker.record("sub agents", MlDeciderAgent.NAME, {"sub_agents": [f"{c}/{t}" for c, t in chosen]})
    click.secho(f"\nDecided: {', '.join(f'{c}/{t}' for c, t in chosen)}.", fg="green", bold=True)

    finished = hand_to_sub_agents(chosen, question, context, problem, data_details, path, agent_outputs, tracker)
    if write_script:
        for category, problem_type in finished:
            generate_code(question, agent_outputs, tracker, problem, data_details, category, problem_type)
    return len(finished) == len(chosen)


def chat(write_script: bool = False, message: str | None = None, auto: bool = False, router: bool = True, worker: bool = False) -> bool:
    """Talk to the user until the requirements gate passes, then run the pipeline.

    Args:
        message: A first user turn to send instead of reading one — the problem as the user would type it.
        worker: Have each sub agent's worker model carry its plan out (the old way) instead of the console running the plan's tools itself.
        router: Let the rule-based router read the problem before `ml_decider_agent` does (`--no-router` turns it off).
        auto: Run unattended: approve every stage and never wait on the keyboard. If the requirements gate is not met by `message`, the
            session ends (there is nobody to answer the follow-up question). Returns whether the pipeline ran to the end.
    """
    global AUTO, ROUTER
    AUTO = auto
    ROUTER = router
    orchestrate.USE_WORKER = worker
    click.echo("Hey, I am ML Agent. How can I help you?")
    transcript: list[str] = []
    pending = message

    while True:
        if pending is not None:
            question, pending = pending, None
            click.echo(f"You: {question}")
        else:
            question = ask("You: ")
        if question is None:
            return False

        if not question.strip():
            continue

        transcript.append(f"User: {question}")
        context = "\n".join(transcript)

        try:
            verdict = as_json(RequirementAgent.run(question=question, context=context))
        except ValueError as error:  # an unresolvable key or model — nothing to retry against
            click.secho(f"ML Agent: ({error})", fg="red")
            return False

        # A reply that would not parse is treated as "not yet": the chat agent asks for what is
        # missing, which is the same thing that happens on an honest `requirements: false`.
        if verdict and is_met(verdict):
            click.secho("\nRequirements met — starting the pipeline.", fg="green", bold=True)
            click.echo(f"  problem       {verdict.get('problem', '')}")
            click.echo(f"  data details  {verdict.get('data_details', '')}")
            return run_pipeline(question, context, verdict, write_script)

        gate = json.dumps(verdict) if verdict else "the requirements check did not return a verdict"
        reply = ChatAgent.run(
            question=question,
            context=f"{context}\n\nRequirements check: {gate}\n\n"
            "Requirements are not met yet. Ask the user for whatever is missing — the problem "
            "statement, or where the data is and how to connect to it — in one short question.",
        )
        transcript.append(f"Assistant: {reply}")
        click.echo(f"ML Agent: {reply}")
