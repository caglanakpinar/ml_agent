"""Thin, named handles onto the agents `agentic_configurations.yaml` declares under `agents:`.

Building one is the two-line pattern `agent_builder`'s own docstring shows —

    configs = load_configs(CONFIG_DIR)
    agent = build_agent("chat_agent", configs)

— wrapped so a caller writes `ChatAgent.run(question=...)` rather than resolving the name and the
config directory itself:

    from console.agentic import ChatAgent, RequirementAgent

    reply = ChatAgent.run(question="what can you help me with?")
    gate = RequirementAgent.run(question=reply)   # a JSON string — see requirements.md's own contract

These are the console's own four agents, which `console/run.py` drives: `chat_agent` talks to the user,
`requirement_agent` gates the run on a problem statement and a data source, `ml_decider_agent` decides
which sub agents the problem is handed to, and `code_generator_agent` writes a starter script for each
one chosen. The sub agents themselves — a decider and a worker per problem type, 44 in all — are declared
in the same YAML, and `sub_agent()` below gives a handle onto any of them:

    from console.agentic import sub_agent

    decider = sub_agent("classification_decider_agent")
    plan = decider.run(question=..., context=..., agent_outputs={"requirement_agent": ..., "ml_decider_agent": ...})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from agent_builder import build_agent

# `agentic_configurations.yaml` lives at the root of this package.
CONFIG_DIR = Path(__file__).resolve().parent


class AgentHandle:
    """Base for a named handle onto one `agents:` entry. A subclass sets `NAME` and nothing else.

    Args (class attributes a subclass sets):
        NAME: The key this agent is registered under in `agentic_configurations.yaml`'s `agents:`
            block — exactly as written there, since that is what `build_agent` looks it up by.
    """

    NAME: ClassVar[str]
    _agent: ClassVar[Any] = None

    @classmethod
    def build(cls, configs: Any = None, **overrides: Any) -> Any:
        """Build (or return the cached) `BaseAgent` this handle names, via `agent_builder.build_agent`."""
        if cls._agent is None or configs is not None or overrides:
            cls._agent = build_agent(cls.NAME, configs or CONFIG_DIR, **overrides)
        return cls._agent

    @classmethod
    def run(
        cls,
        question: str,
        context: str = "",
        agent_outputs: dict[str, str] | None = None,
        configs: Any = None,
        **kwargs: Any,
    ) -> str:
        """Build this agent if it is not already built, and run it — the one-call form most callers want."""
        agent = cls.build(configs)
        return agent.run(question, context, agent_outputs or {}, **kwargs)


class ChatAgent(AgentHandle):
    """`chat_agent` — the front door: the agent `cli.py`'s `generate` command talks to."""

    NAME = "chat_agent"


class RequirementAgent(AgentHandle):
    """`requirement_agent` — the gate before any pipeline starts.

    Checks that a problem statement and a data location/connection are both on the record, and reports
    the verdict as one JSON object — see `prompts/requirements.md` for the exact contract:

        {"requirements": true, "problem": "...", "data_details": "..."}

    `.run()` still returns the raw model text (a `BaseAgent` always does); parse it as JSON to get the
    dict this docstring describes, e.g. `json.loads(RequirementAgent.run(question=...))`.
    """

    NAME = "requirement_agent"


class MlDeciderAgent(AgentHandle):
    """`ml_decider_agent` — decides which sub agents the problem is handed to.

    Reads the requirements verdict, the data's own columns and what each sub agent is for (`console.topics.describe_available` lists the
    ones that really exist, read off the folders), and names one or several, in the order they run. See `prompts/decider.md` for the
    contract:

        {"sub_agents": [{"category": "supervised", "problem_type": "classification", "reasoning": "..."}], "reasoning": "..."}
    """

    NAME = "ml_decider_agent"


class ReporterAgent(AgentHandle):
    """`report_writer_agent` — writes up what an approved plan's tools returned, after `console/executor.py` ran them (no tool calls of its own)."""

    NAME = "report_writer_agent"


class CodeGeneratorAgent(AgentHandle):
    """`code_generator_agent` — runs once, after both decisions above are approved.

    Its `dependency_agent` in `agentic_configurations.yaml` lists `requirement_agent` and
    `ml_decider_agent`, so `{agent_output}` in `prompts/code_generator.md` renders each one's own
    verdict, labelled by who produced it. It writes a starter script into this repository's own
    `<category>/<problem_type>/` folder, once for every sub agent chosen — see `console/run.py`'s
    `generate_code()`.
    """

    NAME = "code_generator_agent"


_SUB_AGENTS: dict[str, type[AgentHandle]] = {}


def sub_agent(name: str) -> type[AgentHandle]:
    """A handle onto one sub agent — a topic's decider or worker — declared in the console's `agentic_configurations.yaml`.

    The same one-call interface as the console's own agents. Handles are cached by name, so a sub agent is built once per process.
    The name must be one the config declares; `build_agent` says so if it is not.
    """
    if name not in _SUB_AGENTS:
        _SUB_AGENTS[name] = type("".join(part.capitalize() for part in name.split("_")), (AgentHandle,), {"NAME": name, "__doc__": f"`{name}` — a sub agent."})
    return _SUB_AGENTS[name]
