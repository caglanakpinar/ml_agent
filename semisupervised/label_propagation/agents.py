"""The worker for label propagation (graph-based semi-supervised learning): `label_propagation_agent` in `agentic_configuration.yaml`.

It has every tool in `tools/` and runs after the decider (`decider.py`), reading the decider's plan as
`{agent_output}`. Run together, in order, they are this topic's workflow:

    from semisupervised.label_propagation.decider import DECIDER
    from semisupervised.label_propagation.agents import AGENT

    plan = DECIDER.run(question=question, context=context)
    result = AGENT.run(question=question, context=context, agent_outputs={DECIDER.NAME: plan})

`console/orchestrate.py` does exactly that, and also handles the human approval step in between.
"""

from __future__ import annotations

from pathlib import Path

from ml_core.handles import AgentHandle

CONFIG_DIR = Path(__file__).resolve().parent
WORKFLOW = ("label_propagation_decider_agent", "label_propagation_agent")


class LabelPropagationAgent(AgentHandle):
    """`label_propagation_agent` — solves label propagation (graph-based semi-supervised learning) problems with the tools in `tools/`."""

    NAME = "label_propagation_agent"
    CONFIG_DIR = CONFIG_DIR


AGENT = LabelPropagationAgent
