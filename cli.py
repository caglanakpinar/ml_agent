"""ml-agent: describe a machine-learning problem, and the right team of agents and tools solves it.

    python cli.py generate                      # chat → the decider picks the sub agents → each one's decider and worker run
    python cli.py solve supervised/regression --data prices.csv --question "predict price"   # skip the chat
    python cli.py topics                        # every problem type, with its tools
    python cli.py tools supervised/ranking      # the tools of one topic, with arguments
    python cli.py configure                     # edit the agent configs in a browser

Installed as the `ml-agent` console script (see `[tool.poetry.scripts]` in `pyproject.toml`).
`agent-builder` — the git dependency this project's agents are built on — installs its own command as a
top-level `cli` module too (`agentic-ai`'s entry point is `cli:cli`; see its `dist-info`). Both packages
placing a module at that same top-level name means whichever installs second overwrites the other's
`cli.py` in `site-packages`, and `import cli` resolves to whichever is left — this repo's own `generate`
command, confirmed by running `python -c "import cli"` from here, not the vendored one. Harmless for
this project's own use, since nothing here calls the vendored `agentic-ai` command; worth knowing if
that command stops resolving to what `agent-builder` shipped.
"""

import click

from console.run import chat


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """ml-agent."""


@cli.command("generate")
@click.option("--script", "write_script", is_flag=True, help="Also write a starter pipeline.py for each chosen sub agent (the sub agents already do the work).")
@click.option("--message", default=None, help="Send this as the first thing the user says, instead of typing it.")
@click.option("--yes", "assume_yes", is_flag=True, help="Run unattended: approve every stage (decider, each plan) and never wait for input. Exit 1 if the pipeline did not finish.")
@click.option("--worker", "use_worker", is_flag=True, help="Have each sub agent's worker model carry its plan out, as before; by default the console runs the plan's tools itself and a model only writes up the results.")
@click.option("--no-router", is_flag=True, help="Skip the rule-based router: ml_decider_agent alone chooses the sub agents.")
def generate(write_script: bool, message: str | None, assume_yes: bool, no_router: bool, use_worker: bool) -> None:
    """Start an interactive session: chat until the requirements are met, then `ml_decider_agent` names the
    sub agents (problem types) the problem goes to; each one's decider proposes a plan and, once you approve
    it, its worker carries it out. Every sub agent is given the problem, the data and what earlier ones found.
    """
    finished = chat(write_script, message=message, auto=assume_yes, router=not no_router, worker=use_worker)
    if assume_yes and not finished:
        raise SystemExit(1)


@cli.command("topics")
def topics() -> None:
    """List every problem type the console can orchestrate, with how many tools its agents have."""
    from console.topics import discover_topics, load_config

    for category, names in discover_topics().items():
        click.secho(category, fg="cyan", bold=True)
        for name in names:
            cfg = load_config(category, name)
            click.echo(f"  {name:<28} {len(cfg['tools']) - 1:>3} tools   {', '.join(cfg['agents'])}")


@cli.command("tools")
@click.argument("topic")
@click.option("--module", "group", default=None, help="Only this tool module, e.g. models or evaluation.")
def tools(topic: str, group: str | None) -> None:
    """List the tools of one topic — `supervised/classification` or just `classification` — with their arguments."""
    from console.topics import parse_topic
    from ml_core.catalogue import list_tools

    category, problem_type = parse_topic(topic)
    listing = list_tools(f"{category}.{problem_type}", group)
    click.secho(f"{category}/{problem_type}: {listing['count']} tools in {', '.join(listing['modules'])}", fg="cyan", bold=True)
    current = None
    for tool in listing["tools"]:
        if tool["module"] != current:
            current = tool["module"]
            click.secho(f"\n[{current}]", fg="yellow")
        needs = f"  (needs {', '.join(tool['requires'])})" if tool["requires"] else ""
        click.echo(f"  {tool['name']}{needs}\n      {tool['description'][:150]}")


@cli.command("solve")
@click.argument("topic")
@click.option("--data", "data_path", required=True, type=click.Path(exists=True, dir_okay=False), help="The data file the agents work on.")
@click.option("--question", required=True, help="What you want done, in plain words.")
@click.option("--problem", default="", help="One-sentence problem statement (defaults to the question).")
@click.option("--yes", "assume_yes", is_flag=True, help="Accept the decider's plan without asking.")
@click.option("--worker", "use_worker", is_flag=True, help="Have the worker model carry the plan out, as before; by default the console runs the plan's tools itself.")
def solve(topic: str, data_path: str, question: str, problem: str, assume_yes: bool, use_worker: bool) -> None:
    """Run one sub agent's decider and worker directly, with no chat and no `ml_decider_agent`.

    TOPIC is `<category>/<problem_type>` (or just the problem type when it is unambiguous), e.g. `supervised/classification`.
    The decider proposes a plan for you to approve (or `--yes`); the worker then carries it out with the topic's tools, and the
    result is saved under `<category>/<problem_type>/runs/`.
    """
    from console import orchestrate
    from console.orchestrate import run_topic_workflow
    from console.topics import parse_topic

    orchestrate.USE_WORKER = use_worker

    try:
        category, problem_type = parse_topic(topic)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    outcome = run_topic_workflow(category, problem_type, question, "", problem or question, "", data_path, interactive=not assume_yes)
    if outcome is None:
        raise SystemExit(1)


@cli.command("configure")
@click.option("--port", default=8787, show_default=True, help="Port to serve the editor on.")
@click.option("--no-browser", is_flag=True, help="Print the URL instead of opening a browser.")
def configure(port: int, no_browser: bool) -> None:
    """Edit the `llms:` and `agents:` blocks of `console/agentic_configurations.yaml` — the console's agents and every sub agent.

    Opens a local page — bound to 127.0.0.1, nothing reachable off this machine — that shows what
    editing that file drives, and edits its LLMs and agents in place. Everything else in the file,
    comments included, is written back untouched. `console.config_ui` has the details.
    """
    from console.config_ui import serve  # imported here: `generate` should not pay for a web server

    serve(port=port, open_browser=not no_browser)


if __name__ == "__main__":
    cli()
