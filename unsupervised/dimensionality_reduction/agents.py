"""The worker for dimensionality reduction: `dimensionality_reduction_agent` in `agentic_configuration.yaml`.

It has every tool in `tools/` and runs after the decider (`decider.py`), reading the decider's plan as
`{agent_output}`. Run together, in order, they are this topic's workflow:

    from unsupervised.dimensionality_reduction.decider import DECIDER
    from unsupervised.dimensionality_reduction.agents import AGENT

    plan = DECIDER.run(question=question, context=context)
    result = AGENT.run(question=question, context=context, agent_outputs={DECIDER.NAME: plan})

`console/orchestrate.py` does exactly that, and also handles the human approval step in between.
"""

from __future__ import annotations

from pathlib import Path

from ml_core.handles import AgentHandle

CONFIG_DIR = Path(__file__).resolve().parent
WORKFLOW = ("dimensionality_reduction_decider_agent", "dimensionality_reduction_agent")


class DimensionalityReductionAgent(AgentHandle):
    """`dimensionality_reduction_agent` — solves dimensionality reduction problems with the tools in `tools/`."""

    NAME = "dimensionality_reduction_agent"
    CONFIG_DIR = CONFIG_DIR


AGENT = DimensionalityReductionAgent
