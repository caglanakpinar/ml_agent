"""The hand-written part of `console/agentic_configurations.yaml`: the console's own LLMs and agents.

`scaffold_console.py` puts these in front of everything it derives from the topics — every topic's tools, decider agent and worker agent —
so the console config is one file that holds the whole roster. Edit the text here and run `python scaffold.py console`, not the YAML.
"""

# The two console LLMs, then the two every topic agent uses. Same model today; named separately so a topic's models and the console's can
# be changed independently.
LLMS = '''llms:
  generator_llm_thinker:  # the workhorse behind the console's own agents
    model: "hf_local/Caglana/qwen0.5b-tinylora-ds-assistant"
    api_key: "HF_TOKEN"
    max_tokens: 16000
    type: "generator"
  generator_llm_thinker_2:  # the console deciders' own model, and everyone's fallback
    model: "hf_local/Caglana/qwen0.5b-tinylora-ds-assistant"
    api_key: "HF_TOKEN"
    max_tokens: 16000
    type: "generator"
  console_json_llm:  # the console's two agents that answer with one small JSON object (the requirement gate and ml_decider_agent). The same model,
    # but with a low ceiling: a small model that fails to stop would otherwise write 16000 tokens of repetition, and that text would be fed back to it
    model: "hf_local/Caglana/qwen0.5b-tinylora-ds-assistant"
    api_key: "HF_TOKEN"
    max_tokens: 1500
    type: "generator"
  console_report_llm:  # writes the short report on what a plan's tools returned (console/executor.py has already run them): same model, small ceiling
    model: "hf_local/Caglana/qwen0.5b-tinylora-ds-assistant"
    api_key: "HF_TOKEN"
    max_tokens: 1500
    type: "generator"
  # Ceilings: a sub agent's decider writes one JSON plan (about 1k tokens) and its worker a short report, so a small model that fails to stop is cut
  # off at 3000 / 6000 tokens, not 16000 (about 19 minutes of repeating itself).
  topic_worker_llm:  # the workhorse behind every sub agent's worker, and its decider's fallback
    model: "hf_local/Caglana/qwen0.5b-tinylora-ds-assistant"
    api_key: "HF_TOKEN"
    max_tokens: 6000
    type: "generator"
  topic_decider_llm:  # the model behind every sub agent's decider, and its worker's fallback
    model: "hf_local/Caglana/qwen0.5b-tinylora-ds-assistant"
    api_key: "HF_TOKEN"
    max_tokens: 3000
    type: "generator"
'''

# The console's own agents. `ml_decider_agent` is the one that decides which sub agents run.
AGENTS = '''  # ---- the console ---------------------------------------------------------------------
  chat_agent:  # the front door — no dependency_agent, `cli.py generate` is what talks to it
    type: "generator"
    llm: "generator_llm_thinker"
    substitute_llm: "generator_llm_thinker_2"
    prompt_path: "prompts/chat.md"

  requirement_agent:  # the gate that decides whether a run may start at all — no dependency_agent,
    # it is run straight off the conversation (`{context}`) before chat_agent ever gets a turn
    type: "generator"
    llm: "console_json_llm"
    substitute_llm: "generator_llm_thinker_2"
    prompt_path: "prompts/requirements.md"

  ml_decider_agent:  # decides WHICH sub agents (below) the problem is handed to — one or several, in order. It reads the
    # problem, the data details, the data's own columns and what each sub agent is for; console/run.py lists the
    # sub agents that really exist (read off the folders), and refuses a verdict that names one that does not.
    type: "generator"
    llm: "console_json_llm"
    substitute_llm: "generator_llm_thinker_2"
    prompt_path: "prompts/decider.md"
    dependency_agent: "requirement_agent"

  report_writer_agent:  # after console/executor.py has run an approved plan's tools itself, writes up what they returned — and only that
    type: "generator"
    llm: "console_report_llm"
    substitute_llm: "generator_llm_thinker_2"
    prompt_path: "prompts/reporter.md"

  code_generator_agent:  # runs once per chosen sub agent, after the decision is approved — see console/run.py's
    # generate_code(). `dependency_agent` lists every agent this one's prompt reads as `{agent_output}`.
    type: "generator"
    llm: "generator_llm_thinker"
    substitute_llm: "generator_llm_thinker_2"
    prompt_path: "prompts/code_generator.md"
    dependency_agent:
      - "requirement_agent"
      - "ml_decider_agent"
'''

# The agents every sub agent is told about, in this order, ahead of anything else it works from. They are what carries the problem, the
# data details and the console decider's verdict into the sub agent's `{agent_output}`.
UPSTREAM = ("requirement_agent", "ml_decider_agent")
