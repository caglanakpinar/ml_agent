"""The worker for pseudo-labelling (labelling unlabelled data with a model): `pseudo_labeling_agent` in `agentic_configuration.yaml`.

It has every tool in `tools/` and runs after the decider (`decider.py`), reading the decider's plan as
`{agent_output}`. Run together, in order, they are this topic's workflow:

    from semisupervised.pseudo_labeling.decider import DECIDER
    from semisupervised.pseudo_labeling.agents import AGENT

    plan = DECIDER.run(question=question, context=context)
    result = AGENT.run(question=question, context=context, agent_outputs={DECIDER.NAME: plan})

`console/orchestrate.py` does exactly that, and also handles the human approval step in between.
"""

from __future__ import annotations

from pathlib import Path

from ml_core.handles import AgentHandle

CONFIG_DIR = Path(__file__).resolve().parent
WORKFLOW = ("pseudo_labeling_decider_agent", "pseudo_labeling_agent")


class PseudoLabelingAgent(AgentHandle):
    """`pseudo_labeling_agent` — solves pseudo-labelling (labelling unlabelled data with a model) problems with the tools in `tools/`."""

    NAME = "pseudo_labeling_agent"
    CONFIG_DIR = CONFIG_DIR


AGENT = PseudoLabelingAgent
