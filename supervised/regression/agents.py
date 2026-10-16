"""The worker for regression (predicting a continuous value): `regression_agent` in `agentic_configuration.yaml`.

It has every tool in `tools/` and runs after the decider (`decider.py`), reading the decider's plan as
`{agent_output}`. Run together, in order, they are this topic's workflow:

    from supervised.regression.decider import DECIDER
    from supervised.regression.agents import AGENT

    plan = DECIDER.run(question=question, context=context)
    result = AGENT.run(question=question, context=context, agent_outputs={DECIDER.NAME: plan})

`console/orchestrate.py` does exactly that, and also handles the human approval step in between.
"""

from __future__ import annotations

from pathlib import Path

from ml_core.handles import AgentHandle

CONFIG_DIR = Path(__file__).resolve().parent
WORKFLOW = ("regression_decider_agent", "regression_agent")


class RegressionAgent(AgentHandle):
    """`regression_agent` — solves regression (predicting a continuous value) problems with the tools in `tools/`."""

    NAME = "regression_agent"
    CONFIG_DIR = CONFIG_DIR


AGENT = RegressionAgent
