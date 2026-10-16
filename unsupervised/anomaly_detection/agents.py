"""The worker for anomaly detection (outlier detection): `anomaly_detection_agent` in `agentic_configuration.yaml`.

It has every tool in `tools/` and runs after the decider (`decider.py`), reading the decider's plan as
`{agent_output}`. Run together, in order, they are this topic's workflow:

    from unsupervised.anomaly_detection.decider import DECIDER
    from unsupervised.anomaly_detection.agents import AGENT

    plan = DECIDER.run(question=question, context=context)
    result = AGENT.run(question=question, context=context, agent_outputs={DECIDER.NAME: plan})

`console/orchestrate.py` does exactly that, and also handles the human approval step in between.
"""

from __future__ import annotations

from pathlib import Path

from ml_core.handles import AgentHandle

CONFIG_DIR = Path(__file__).resolve().parent
WORKFLOW = ("anomaly_detection_decider_agent", "anomaly_detection_agent")


class AnomalyDetectionAgent(AgentHandle):
    """`anomaly_detection_agent` — solves anomaly detection (outlier detection) problems with the tools in `tools/`."""

    NAME = "anomaly_detection_agent"
    CONFIG_DIR = CONFIG_DIR


AGENT = AnomalyDetectionAgent
