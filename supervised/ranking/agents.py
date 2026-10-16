"""The worker for ranking (learning to rank): `ranking_agent` in `agentic_configuration.yaml`.

It has every tool in `tools/` and runs after the decider (`decider.py`), reading the decider's plan as
`{agent_output}`. Run together, in order, they are this topic's workflow:

    from supervised.ranking.decider import DECIDER
    from supervised.ranking.agents import AGENT

    plan = DECIDER.run(question=question, context=context)
    result = AGENT.run(question=question, context=context, agent_outputs={DECIDER.NAME: plan})

`console/orchestrate.py` does exactly that, and also handles the human approval step in between.
"""

from __future__ import annotations

from pathlib import Path

from ml_core.handles import AgentHandle

CONFIG_DIR = Path(__file__).resolve().parent
WORKFLOW = ("ranking_decider_agent", "ranking_agent")


class RankingAgent(AgentHandle):
    """`ranking_agent` — solves ranking (learning to rank) problems with the tools in `tools/`."""

    NAME = "ranking_agent"
    CONFIG_DIR = CONFIG_DIR


AGENT = RankingAgent
