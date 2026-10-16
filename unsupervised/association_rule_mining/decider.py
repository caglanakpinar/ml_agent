"""The decider for association rule mining (market-basket analysis): `association_rules_decider_agent` in `agentic_configuration.yaml`.

It reads the problem statement and the data, runs the read-only checks in `tools/diagnostics.py`, and
answers one question — *which approach, with which tools, and why* — as a JSON object the worker agent
(`agents.py`) then executes. It never fits a model or changes data.

    from unsupervised.association_rule_mining.decider import DECIDER

    plan = DECIDER.run(question="...", context="Current data path: data.csv")
"""

from __future__ import annotations

from pathlib import Path

from ml_core.handles import AgentHandle

CONFIG_DIR = Path(__file__).resolve().parent


class AssociationRuleMiningDecider(AgentHandle):
    """`association_rules_decider_agent` — chooses the approach for association rule mining (market-basket analysis) and names the tools to run."""

    NAME = "association_rules_decider_agent"
    CONFIG_DIR = CONFIG_DIR


DECIDER = AssociationRuleMiningDecider
