"""The decider for classification (binary and multiclass): `classification_decider_agent` in `agentic_configuration.yaml`.

It reads the problem statement and the data, runs the read-only checks in `tools/diagnostics.py`, and
answers one question — *which approach, with which tools, and why* — as a JSON object the worker agent
(`agents.py`) then executes. It never fits a model or changes data.

    from supervised.classification.decider import DECIDER

    plan = DECIDER.run(question="...", context="Current data path: data.csv")
"""

from __future__ import annotations

from pathlib import Path

from ml_core.handles import AgentHandle

CONFIG_DIR = Path(__file__).resolve().parent


class ClassificationDecider(AgentHandle):
    """`classification_decider_agent` — chooses the approach for classification (binary and multiclass) and names the tools to run."""

    NAME = "classification_decider_agent"
    CONFIG_DIR = CONFIG_DIR


DECIDER = ClassificationDecider
