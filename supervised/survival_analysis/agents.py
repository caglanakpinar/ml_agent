"""The worker for survival analysis (time-to-event with censoring): `survival_agent` in `agentic_configuration.yaml`.

It has every tool in `tools/` and runs after the decider (`decider.py`), reading the decider's plan as
`{agent_output}`. Run together, in order, they are this topic's workflow:

    from supervised.survival_analysis.decider import DECIDER
    from supervised.survival_analysis.agents import AGENT

    plan = DECIDER.run(question=question, context=context)
    result = AGENT.run(question=question, context=context, agent_outputs={DECIDER.NAME: plan})

`console/orchestrate.py` does exactly that, and also handles the human approval step in between.
"""

from __future__ import annotations

from pathlib import Path

from ml_core.handles import AgentHandle

CONFIG_DIR = Path(__file__).resolve().parent
WORKFLOW = ("survival_decider_agent", "survival_agent")


class SurvivalAnalysisAgent(AgentHandle):
    """`survival_agent` — solves survival analysis (time-to-event with censoring) problems with the tools in `tools/`."""

    NAME = "survival_agent"
    CONFIG_DIR = CONFIG_DIR


AGENT = SurvivalAnalysisAgent
