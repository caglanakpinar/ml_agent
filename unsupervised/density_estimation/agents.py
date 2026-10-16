"""The worker for density estimation: `density_estimation_agent` in `agentic_configuration.yaml`.

It has every tool in `tools/` and runs after the decider (`decider.py`), reading the decider's plan as
`{agent_output}`. Run together, in order, they are this topic's workflow:

    from unsupervised.density_estimation.decider import DECIDER
    from unsupervised.density_estimation.agents import AGENT

    plan = DECIDER.run(question=question, context=context)
    result = AGENT.run(question=question, context=context, agent_outputs={DECIDER.NAME: plan})

`console/orchestrate.py` does exactly that, and also handles the human approval step in between.
"""

from __future__ import annotations

from pathlib import Path

from ml_core.handles import AgentHandle

CONFIG_DIR = Path(__file__).resolve().parent
WORKFLOW = ("density_estimation_decider_agent", "density_estimation_agent")


class DensityEstimationAgent(AgentHandle):
    """`density_estimation_agent` — solves density estimation problems with the tools in `tools/`."""

    NAME = "density_estimation_agent"
    CONFIG_DIR = CONFIG_DIR


AGENT = DensityEstimationAgent
