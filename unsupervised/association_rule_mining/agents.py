"""The worker for association rule mining (market-basket analysis): `association_rules_agent` in `agentic_configuration.yaml`.

It has every tool in `tools/` and runs after the decider (`decider.py`), reading the decider's plan as
`{agent_output}`. Run together, in order, they are this topic's workflow:

    from unsupervised.association_rule_mining.decider import DECIDER
    from unsupervised.association_rule_mining.agents import AGENT

    plan = DECIDER.run(question=question, context=context)
    result = AGENT.run(question=question, context=context, agent_outputs={DECIDER.NAME: plan})

`console/orchestrate.py` does exactly that, and also handles the human approval step in between.
"""

from __future__ import annotations

from pathlib import Path

from ml_core.handles import AgentHandle

CONFIG_DIR = Path(__file__).resolve().parent
WORKFLOW = ("association_rules_decider_agent", "association_rules_agent")


class AssociationRuleMiningAgent(AgentHandle):
    """`association_rules_agent` — solves association rule mining (market-basket analysis) problems with the tools in `tools/`."""

    NAME = "association_rules_agent"
    CONFIG_DIR = CONFIG_DIR


AGENT = AssociationRuleMiningAgent
