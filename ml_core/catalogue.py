"""The runtime view of a topic's tools — what a topic's `<slug>_catalogue_tool` returns to an agent.

An agent can read its tool list in its prompt, but a decider that wants to re-check "is there a tool for
X?" mid-run, or a person debugging a run, can call the catalogue instead. It is built from the same
`ml_core.specs.discover_tools` the YAML and the prompt are rendered from, so the three cannot disagree.
"""

from __future__ import annotations

from typing import Any

from ml_core.specs import discover_tools


def list_tools(package: str, group: str | None = None) -> dict[str, Any]:
    """Every tool of `package` (e.g. "supervised.classification"), optionally one module's, with arguments."""
    tools = discover_tools(package)
    modules = sorted({t.module for t in tools})
    if group:
        if group not in modules:
            raise ValueError(f"no tool module {group!r}; the modules are {modules}")
        tools = [t for t in tools if t.module == group]
    return {
        "topic": package,
        "modules": modules,
        "count": len(tools),
        "tools": [
            {
                "name": t.tool_name,
                "module": t.module,
                "description": t.description,
                "requires": list(t.requires),
                "arguments": [
                    {"name": a.name, "type": a.type, "required": a.required, **({"choices": a.enum} if a.enum else {})}
                    for a in t.args
                ],
            }
            for t in tools
        ],
    }
