"""The worker for sequence labeling (NER, POS tagging, chunking): `sequence_labeling_agent` in `agentic_configuration.yaml`.

It has every tool in `tools/` and runs after the decider (`decider.py`), reading the decider's plan as
`{agent_output}`. Run together, in order, they are this topic's workflow:

    from supervised.sequence_labeling.decider import DECIDER
    from supervised.sequence_labeling.agents import AGENT

    plan = DECIDER.run(question=question, context=context)
    result = AGENT.run(question=question, context=context, agent_outputs={DECIDER.NAME: plan})

`console/orchestrate.py` does exactly that, and also handles the human approval step in between.
"""

from __future__ import annotations

from pathlib import Path

from ml_core.handles import AgentHandle

CONFIG_DIR = Path(__file__).resolve().parent
WORKFLOW = ("sequence_labeling_decider_agent", "sequence_labeling_agent")


class SequenceLabelingAgent(AgentHandle):
    """`sequence_labeling_agent` — solves sequence labeling (NER, POS tagging, chunking) problems with the tools in `tools/`."""

    NAME = "sequence_labeling_agent"
    CONFIG_DIR = CONFIG_DIR


AGENT = SequenceLabelingAgent
