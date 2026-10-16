"""Shared plumbing for every topic's `tools/` — the agent-facing contract, data loading and the
helpers the ML tools have in common.

    ml_core.tooling   the `@tool` decorator and the `{"status": ...}` result shape every caller returns
    ml_core.data      reading a dataset from a path / frame / rows, and writing one back
    ml_core.ml        feature/target split, preprocessing, hold-out splits, model persistence
    ml_core.specs     reads a topic's tool functions and renders their `agentic_configuration.yaml` entries
    ml_core.handles   the `AgentHandle` every topic's `agents.py` / `decider.py` subclass
"""
