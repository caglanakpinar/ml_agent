"""The worker for topic modeling: `topic_modeling_agent` in `agentic_configuration.yaml`.

It has every tool in `tools/` and runs after the decider (`decider.py`), reading the decider's plan as
`{agent_output}`. Run together, in order, they are this topic's workflow:

    from unsupervised.topic_modeling.decider import DECIDER
    from unsupervised.topic_modeling.agents import AGENT

    plan = DECIDER.run(question=question, context=context)
    result = AGENT.run(question=question, context=context, agent_outputs={DECIDER.NAME: plan})

`console/orchestrate.py` does exactly that, and also handles the human approval step in between.
"""

from __future__ import annotations

from pathlib import Path

from ml_core.handles import AgentHandle

CONFIG_DIR = Path(__file__).resolve().parent
WORKFLOW = ("topic_modeling_decider_agent", "topic_modeling_agent")


class TopicModelingAgent(AgentHandle):
    """`topic_modeling_agent` — solves topic modeling problems with the tools in `tools/`."""

    NAME = "topic_modeling_agent"
    CONFIG_DIR = CONFIG_DIR


AGENT = TopicModelingAgent
