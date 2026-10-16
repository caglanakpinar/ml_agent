"""The worker for multi-label classification: `multilabel_agent` in `agentic_configuration.yaml`.

It has every tool in `tools/` and runs after the decider (`decider.py`), reading the decider's plan as
`{agent_output}`. Run together, in order, they are this topic's workflow:

    from supervised.multi_label_classification.decider import DECIDER
    from supervised.multi_label_classification.agents import AGENT

    plan = DECIDER.run(question=question, context=context)
    result = AGENT.run(question=question, context=context, agent_outputs={DECIDER.NAME: plan})

`console/orchestrate.py` does exactly that, and also handles the human approval step in between.
"""

from __future__ import annotations

from pathlib import Path

from ml_core.handles import AgentHandle

CONFIG_DIR = Path(__file__).resolve().parent
WORKFLOW = ("multilabel_decider_agent", "multilabel_agent")


class MultiLabelClassificationAgent(AgentHandle):
    """`multilabel_agent` — solves multi-label classification problems with the tools in `tools/`."""

    NAME = "multilabel_agent"
    CONFIG_DIR = CONFIG_DIR


AGENT = MultiLabelClassificationAgent
