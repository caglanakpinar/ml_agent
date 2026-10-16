"""The worker for ordinal regression (predicting an ordered category): `ordinal_agent` in `agentic_configuration.yaml`.

It has every tool in `tools/` and runs after the decider (`decider.py`), reading the decider's plan as
`{agent_output}`. Run together, in order, they are this topic's workflow:

    from supervised.ordinal_regression.decider import DECIDER
    from supervised.ordinal_regression.agents import AGENT

    plan = DECIDER.run(question=question, context=context)
    result = AGENT.run(question=question, context=context, agent_outputs={DECIDER.NAME: plan})

`console/orchestrate.py` does exactly that, and also handles the human approval step in between.
"""

from __future__ import annotations

from pathlib import Path

from ml_core.handles import AgentHandle

CONFIG_DIR = Path(__file__).resolve().parent
WORKFLOW = ("ordinal_decider_agent", "ordinal_agent")


class OrdinalRegressionAgent(AgentHandle):
    """`ordinal_agent` — solves ordinal regression (predicting an ordered category) problems with the tools in `tools/`."""

    NAME = "ordinal_agent"
    CONFIG_DIR = CONFIG_DIR


AGENT = OrdinalRegressionAgent
