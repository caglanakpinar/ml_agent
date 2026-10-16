"""The runtime catalogue of this topic's tools."""

from __future__ import annotations

from ml_core.catalogue import list_tools
from ml_core.tooling import tool


@tool
def tool_catalogue(group: str | None = None) -> dict:
    """List every tool available for positive-unlabelled (PU) learning with its description, arguments and optional dependencies — call it to check whether a tool exists for a step before improvising one.

    Args:
        group: Restrict the listing to one tool module, e.g. "models" or "evaluation".
    """
    return list_tools("semisupervised.positive_unlabeled_learning", group)
