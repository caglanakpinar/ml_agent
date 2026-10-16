"""The worker for self-training (semi-supervised learning): `self_training_agent` in `agentic_configuration.yaml`.

It has every tool in `tools/` and runs after the decider (`decider.py`), reading the decider's plan as
`{agent_output}`. Run together, in order, they are this topic's workflow:

    from semisupervised.self_training.decider import DECIDER
    from semisupervised.self_training.agents import AGENT

    plan = DECIDER.run(question=question, context=context)
    result = AGENT.run(question=question, context=context, agent_outputs={DECIDER.NAME: plan})

`console/orchestrate.py` does exactly that, and also handles the human approval step in between.
"""

from __future__ import annotations

from pathlib import Path

from ml_core.handles import AgentHandle

CONFIG_DIR = Path(__file__).resolve().parent
WORKFLOW = ("self_training_decider_agent", "self_training_agent")


class SelfTrainingAgent(AgentHandle):
    """`self_training_agent` — solves self-training (semi-supervised learning) problems with the tools in `tools/`."""

    NAME = "self_training_agent"
    CONFIG_DIR = CONFIG_DIR


AGENT = SelfTrainingAgent
