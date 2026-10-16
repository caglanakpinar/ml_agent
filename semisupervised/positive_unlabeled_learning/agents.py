"""The worker for positive-unlabelled (PU) learning: `pu_learning_agent` in `agentic_configuration.yaml`.

It has every tool in `tools/` and runs after the decider (`decider.py`), reading the decider's plan as
`{agent_output}`. Run together, in order, they are this topic's workflow:

    from semisupervised.positive_unlabeled_learning.decider import DECIDER
    from semisupervised.positive_unlabeled_learning.agents import AGENT

    plan = DECIDER.run(question=question, context=context)
    result = AGENT.run(question=question, context=context, agent_outputs={DECIDER.NAME: plan})

`console/orchestrate.py` does exactly that, and also handles the human approval step in between.
"""

from __future__ import annotations

from pathlib import Path

from ml_core.handles import AgentHandle

CONFIG_DIR = Path(__file__).resolve().parent
WORKFLOW = ("pu_learning_decider_agent", "pu_learning_agent")


class PositiveUnlabeledLearningAgent(AgentHandle):
    """`pu_learning_agent` — solves positive-unlabelled (PU) learning problems with the tools in `tools/`."""

    NAME = "pu_learning_agent"
    CONFIG_DIR = CONFIG_DIR


AGENT = PositiveUnlabeledLearningAgent
