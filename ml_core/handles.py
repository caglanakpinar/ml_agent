"""One named handle onto an agent a topic's `agentic_configuration.yaml` declares.

`agent_builder.build_agent(name, config_dir)` is the whole of the mechanism — it reads the YAML in
`config_dir`, resolves the agent's LLM, prompt and tools, and returns a runnable agent. A handle is a
name plus that directory, with the built agent cached, so callers write `Classifier.run(question=...)`
and the (expensive) build happens once per process.

    class Classifier(AgentHandle):
        NAME = "classification_agent"
        CONFIG_DIR = Path(__file__).resolve().parent
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar


def reset_agent_builder_registries() -> None:
    """Empty `agent_builder`'s class-level config registries, so only the config about to be read is in play.

    `agent_builder.utils.configs.Configs` stores every parsed `tools:`, `agents:` and `llms:` entry in dictionaries shared by *all*
    instances. Without this, two configs loaded in one process overwrite each other's entries of the same name — the console's own
    agents, or a second topic's identically named tools. Each `Configs(...)` re-reads its own file, so clearing here costs nothing.
    """
    try:
        from utils.configs import Configs
    except ImportError:  # agent_builder not installed: nothing to isolate
        return
    for registry in ("db_confgs", "llm_configs", "embeddings_configs", "agent_configs", "tool_configs"):
        getattr(Configs, registry, {}).clear()


class AgentHandle:
    """Base for a named handle onto one `agents:` entry. A subclass sets `NAME` and `CONFIG_DIR`."""

    NAME: ClassVar[str]
    CONFIG_DIR: ClassVar[Path]
    _agent: ClassVar[Any] = None

    @classmethod
    def build(cls, configs: Any = None, **overrides: Any) -> Any:
        """Build (or return the cached) agent this handle names. Any argument forces a rebuild."""
        if cls._agent is None or configs is not None or overrides:
            from agent_builder import build_agent  # imported here: tools never need the LLM stack

            reset_agent_builder_registries()
            cls._agent = build_agent(cls.NAME, configs or cls.CONFIG_DIR, **overrides)
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
        """Build this agent if needed and run it — the one-call form most callers want."""
        return cls.build(configs).run(question, context, agent_outputs or {}, **kwargs)
