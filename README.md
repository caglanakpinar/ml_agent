# ml_agent

Multi-agent LLM framework for building autonomous ML pipelines.

You describe a machine-learning problem and point at some data. ML Agent's decider picks which of its
**sub agents** the problem belongs to — one or several — and hands it to them: 22 problem types, 565
tools, each type a pair of sub agents, a **decider** (looks at the data, proposes a plan) and a **worker**
(carries the plan out). All 44 are declared in the console's own config, `console/agentic_configurations.yaml`.

```
python cli.py generate
```

```
chat ─▶ requirement_agent ─▶ ml_decider_agent ─▶ sub agent 1 ─▶ sub agent 2 ─▶ …
 │         (is there a problem     (reads the problem, the      each one: <topic> decider ─▶ plan ─▶ you approve ─▶ <topic> worker
 │          and a data source?)     data's columns and what                    (read-only checks)     (JSON)                 (every tool of the topic)
 └─ asks for what is missing        each of the 22 is for;
                                    names one or more, in order)
```

The decider chooses; everything it and the gate learned flows on. Every sub agent is given the problem,
the data path with the file's real columns, the conversation, the outputs of `requirement_agent` and
`ml_decider_agent` (through `dependency_agent` in the YAML, as `{agent_output}`), and — for a sub agent
after the first — what the earlier ones found.

Every stage that decides something stops for a human verdict. A rejected answer is retried with a
web search and the rejected attempt in its prompt (`console/prompts/not_approved.md`); an answer that
names something that does not exist — a sub agent with no folder, a tool the worker does not have —
is rejected automatically.

## Workflow

What one `python cli.py generate` run does, in order. **⏸** marks a stop for your `y`/`n` (`--yes` accepts all of them).

```
 1 chat ─▶ requirement_agent ──── false ──▶ chat_agent asks for what is missing ─┐
 │              │ true                                                          │
 │              ▼                                                               │
 2          find the data file (in the conversation, else ask)  ◀───────────────┘
 │              ▼
 3          router ─▶ ml_decider_agent ─▶ ⏸ which sub agents, in what order
 │              ▼
 4          for each sub agent:
 │            <topic>_decider_agent ─▶ plan ─▶ ⏸ ─▶ run the plan's tools ─▶ report_writer_agent ─▶ runs/<timestamp>.md
 │              ▲                       │ rejected                                                     │
 │              └── web search ◀────────┘                                  what it found goes on to the next sub agent
 ▼
 5          [--script] code_generator_agent ─▶ <category>/<problem_type>/pipeline.py
```

1. **Chat and requirements gate.** Every message goes to `requirement_agent`, which needs two things: a problem
   statement and a data source. If either is missing, `chat_agent` asks for it in one short question and the whole
   conversation is carried forward ("predict churn" now, "it's in `customers.csv`" next). Nothing else starts before both
   are there.
2. **Find the data.** The file is located once, up front — in the conversation, or by asking — so every agent after this sees
   the same `Current data path:` line and the file's real columns.
3. **Choose the sub agents.** The rule-based router (`console/router.py`) reads the problem and the file first.
   * Clearly ahead → its choice is put to you ⏸ exactly as the model's verdict would be, and no model is called. Refuse it
     and `ml_decider_agent` chooses freely.
   * Not clear → `ml_decider_agent` chooses from the router's shortlist.

   Either way the answer is one or more problem types, in order. A type that does not exist, or one named twice, is
   rejected automatically. `--no-router` leaves the choice to `ml_decider_agent` alone.
4. **Run each chosen sub agent** (`console/orchestrate.py`), one after the other:
   1. *Decide.* `<topic>_decider_agent` runs read-only diagnostics on the data and returns a JSON plan: the approach, the
      steps (a tool and its arguments each), the metric that judges the result, the risks — or an escalation, if this topic
      is the wrong one. The plan is checked before you see it: it must have steps, and every tool it names must exist. A reply
      that was cut off or is repeating itself is salvaged (distinct steps kept, at most 12); after three unusable plans the
      topic's **default recipe** is offered instead (`console/recipes.py`: its read-only checks, a baseline, the candidates
      compared, the cross-validation winner fitted and scored; a topic without a recipe gets its read-only checks only).
   2. *Approve* ⏸. A rejected plan is retried with a web search and the rejected attempt in the prompt.
   3. *Solve.* The console runs the plan's tool calls itself (`console/executor.py`; each ends `ok`, `error`, `skipped` or
      `timeout`), then `report_writer_agent` writes up what came back. A write-up that names a tool that was not run, a path the
      run never used, or repeats itself — or one that cannot be produced — is replaced by the console's own summary; numbers
      no step returned are flagged. `--worker` has the
      sub agent's worker model carry the plan out instead.
   4. *Save.* `<category>/<problem_type>/runs/<timestamp>.md` holds the request, the context, the approved plan and the
      result; files the tools wrote go in `<timestamp>_files/`.

   Each sub agent after the first is told what the earlier ones found. A sub agent whose decider escalates runs nothing; its
   escalation is saved and handed on the same way.
5. **Scripts** (only with `--script`). `code_generator_agent` writes a starter `pipeline.py` for each sub agent that
   finished; if its script does not name the decided problem type, a plain deterministic one is written instead.

Unattended (`--yes`): every approval is granted, a stage rejected three times in a row ends the run, and if
`ml_decider_agent` gives no usable verdict the router's best reading is taken. `python cli.py solve <topic> --data … --question …`
skips steps 1–3 and runs step 4 for the one sub agent you name.

## The topics

| | topic | tools | modules |
|---|---|---:|---|
| **supervised** | `classification` | 34 | diagnostics, models, imbalance, evaluation |
| | `regression` | 36 | diagnostics, models, evaluation |
| | `ranking` | 22 | diagnostics, models, text, evaluation |
| | `survival_analysis` | 27 | diagnostics, estimators, models, evaluation |
| | `time_series_forecasting` | 31 | diagnostics, baselines, smoothing, statistical, machine_learning, evaluation |
| | `sequence_labeling` | 22 | diagnostics, corpus, models, evaluation |
| | `ordinal_regression` | 18 | diagnostics, models, evaluation |
| | `multi_label_classification` | 22 | diagnostics, models, evaluation |
| **semisupervised** | `self_training` | 21 | diagnostics, models, evaluation |
| | `pseudo_labeling` | 23 | diagnostics, generators, training, evaluation |
| | `co_training` | 18 | diagnostics, models, evaluation |
| | `consistency_regularization` | 19 | diagnostics, models, evaluation |
| | `label_propagation` | 22 | diagnostics, models, evaluation |
| | `active_learning` | 26 | diagnostics, queries, simulation, workflow |
| | `positive_unlabeled_learning` | 17 | diagnostics, models, evaluation |
| **unsupervised** | `clustering` | 33 | diagnostics, selection, algorithms, evaluation |
| | `dimensionality_reduction` | 30 | diagnostics, selection, linear, nonlinear, evaluation |
| | `anomaly_detection` | 34 | diagnostics, statistical, ml_models, timeseries, ensemble |
| | `density_estimation` | 28 | diagnostics, univariate, multivariate, evaluation |
| | `association_rule_mining` | 26 | diagnostics, mining, rules, usage |
| | `representation_learning` | 30 | diagnostics, tabular, text, entities, evaluation |
| | `topic_modeling` | 26 | diagnostics, models, selection, analysis |

`python cli.py topics` lists them; `python cli.py tools supervised/ranking` lists a topic's tools with
their arguments.

## What is in a topic folder

```
supervised/classification/
├── tools/                       the functions the agents call — hand-written
│   ├── diagnostics.py           read-only checks; the ONLY tools the decider gets
│   ├── models.py, evaluation.py, imbalance.py ...
│   ├── _common.py               helpers (leading underscore = not a tool module)
│   └── catalogue.py             <slug>_catalogue_tool: the tool list at run time (generated)
├── decider.py                   DECIDER — handle on <slug>_decider_agent (generated)
├── agents.py                    AGENT   — handle on <slug>_agent          (generated)
├── agentic_configuration.yaml   LLMs, both agents, every tool with its description and arguments,
│                                the orchestrator and the pipeline (generated)
├── prompt.md                    the worker's prompt, with the tool catalogue embedded (generated)
└── decider_prompt.md            the decider's prompt: is this the right topic, which tools, what plan (generated)
```

* **decider** — runs only `diagnostics.py` tools and the catalogue. It answers one question — *which
  approach, with which tools, in what order, judged by which metric* — as a JSON plan. It never fits
  anything.
* **worker** — has every tool. It reads the plan as `{agent_output}` and runs it, reporting only what
  the tools returned.

### Nothing is written twice

A tool is a plain function:

```python
@tool
def fit_random_forest(data: str, target: str, n_estimators: int = 300, ...) -> dict:
    """Fit a random forest — a strong default for tabular data ...

    Args:
        n_estimators: Number of trees.
    """
```

`@tool` (see `ml_core/tooling.py`) guarantees the contract an agent relies on: it returns
`{"status": "ok", ...}` or `{"status": "error", "message": ...}`, never raises, returns JSON-safe
values, and reports a missing optional package (`requires=("lightgbm",)`) with an `install` hint
instead of an `ImportError`.

`python scaffold.py` reads those functions — name, docstring, signature, `Literal[...]` choices,
`requires` — and writes the YAML, both prompts, the handles and the catalogue from them, plus the prose
in `scaffold_content.py` / `scaffold_parts/` (what the problem is, how to choose between tools, how to
judge a result, what to produce, the rules). Every argument must be documented, either in the
function's `Args:` block or in `ml_core.specs.COMMON_ARGS`, or generation fails.

To **add a tool**: write the function in a tools module, add it to a `choose` row in
`scaffold_parts/`, run `python scaffold.py`. `pytest tests/test_topics.py` fails if a generated file is
stale or a tool is not in the config.

## Commands

| | |
|---|---|
| `python cli.py generate` | chat → requirements gate → `ml_decider_agent` picks the sub agents → for each: its decider → your approval → its worker. `--script` also writes a starter `pipeline.py` for each. |
| `python cli.py solve supervised/regression --data prices.csv --question "predict price"` | skip the chat and `ml_decider_agent`; `--yes` accepts the plan without asking. |
| `python cli.py topics` | every topic with its tool count and agents. |
| `python cli.py tools <topic> [--module models]` | the topic's tools and their arguments. |
| `python -m benchmarks.run score all` | every benchmark case's default recipe, no model involved, then a table of baseline, model and the case's reference bar (see `benchmarks/README.md`). |
| `python cli.py configure` | local web editor for the LLMs and agents in `console/agentic_configurations.yaml` — the console's and every sub agent's. |
| `python scaffold.py [topic\|console] [--check]` | regenerate the YAML, prompts and handles from the code — including the console's one config. |

A run is saved to `<category>/<problem_type>/runs/<timestamp>.md` (request, context, the approved plan,
the result).

## The console

| file | |
|---|---|
| `console/run.py` | the chat loop and the one decision (`ml_decider_agent`); hands each chosen sub agent, in order, to `orchestrate`. |
| `console/orchestrate.py` | per sub agent: decide → validate the plan → human approval → worker → save. A plan that names a tool the worker lacks is never run. |
| `console/topics.py` | finds topics (a folder is one only if it holds an `agentic_configuration.yaml`), describes each to the model that must choose between them, finds the data file in the conversation and shows the agents the real column names. |
| `console/agentic_configurations.yaml` | **generated** (`python scaffold.py console`): the console's agents (`chat_agent`, `requirement_agent`, `ml_decider_agent`, `code_generator_agent`) and all 44 sub agents with their 587 tools, one config. Each topic folder keeps its own `agentic_configuration.yaml`, which this file is built from. |
| `console/prompts/decider.md` | what `ml_decider_agent` is asked: pick one or more sub agents from the list, as JSON. |

## Shared code

`ml_core/` — `tooling` (the `@tool` contract), `data` (reading a table from a path, frame or rows),
`ml` (feature/target split, preprocessing, version-aware scikit-learn helpers), `semi` and `semi_diag`
(the one protocol every semi-supervised tool is judged by: hold out labelled rows, compare with the
same learner trained on labels alone), `unsup`, `specs` (reads tools, renders YAML and prompts),
`handles` (agent handles), `catalogue`.

## Setup

```
poetry install                      # core: pandas, scikit-learn, scipy, agent-builder
poetry install -E forecasting       # statsmodels, prophet     (ARIMA / ETS / Prophet tools)
poetry install -E boosting          # lightgbm, xgboost
poetry install -E survival          # scikit-survival
poetry install -E nlp               # transformers, sentence-transformers, bertopic, CRF
poetry install -E manifold          # umap-learn
poetry install -E all
```

A tool whose package is missing says so in its description (`needs lightgbm`) and, if called, answers
with an `install` field instead of failing — the worker is told to pick another tool.

`poetry.lock` predates the new dependencies: run `poetry lock` once.

The LLMs are all declared in `console/agentic_configurations.yaml`: the console's own (`generator_llm_thinker*`) and the
sub agents' (`topic_worker_llm`, `topic_decider_llm`) — all currently `hf_local/Caglana/qwen0.5b-tinylora-ds-assistant`.
That file is generated, so a change made there with `python cli.py configure` lasts until the next `python scaffold.py`; make it
lasting in `scaffold_parts/console.py` (the LLMs and the console agents) or the topic's own config.

Tool names are unique across topics — the ten that more than one topic defines carry their topic
(`self_training_predict_with_semi_model_tool`, `topic_modeling_fit_lda_tool`; see `TOPIC_QUALIFIED` in
`ml_core/specs.py`) — because the console config keys one `tools:` block by name.

## Benchmarks

`benchmarks/` holds 13 seeded data-science problems, each with planted traps (leaking columns, drift, position bias, a labelling bias, gross outliers…) and a reference
solution that shows the trap is real. Two questions are asked of them; the answers below are from 13–15 Oct 2026. Details: [benchmarks/README.md](benchmarks/README.md),
[benchmarks/RESULTS.md](benchmarks/RESULTS.md) and [benchmarks/SCORES.md](benchmarks/SCORES.md).

### 1. Does the console send each problem to the right sub agent and run it?

`python -m benchmarks.run chat all --yes` hands every case to the whole console unattended, with the local `qwen0.5b-tinylora-ds-assistant` model.

| | model alone | with `console/router.py` and `console/executor.py` |
|---|---:|---:|
| sub agents chosen as expected | 0/13 | **13/13** |
| runs that reached a sub agent | 1/13 (and it chose the wrong two) | **13/13** |
| plan steps that ran with a result | 0/0 | **66/66** |

The model alone kept naming categories that do not exist (`unsupervised/classification`) and was rejected three times in a row on 12 of 13 cases. Those 66 steps were mostly
the topic's read-only checks, because the model's decider rarely wrote a plan, and its write-ups were never used (all 13 failed `check_report`). That is what the default
recipe (`console/recipes.py`) was written to fix: it fits something.

### 2. Does what it fits beat a baseline?

`python -m benchmarks.run score all` runs each case's default recipe (no model involved, about 12 minutes) and scores it against a baseline, the simplest model of the topic on
the same rows, and against the case's reference bar.

| case | metric | baseline | model | bar | |
|---|---|---:|---:|---:|---|
| c01 churn | PR-AUC | 0.056 (most frequent class) | 0.308 | 0.123¹ | beats |
| c02 demand forecast | WAPE ↓ | 0.255 (same weekday last week) | 0.199 | 0.145 | beats |
| c03 survival | concordance | 0.500 (random order of risk) | 0.656 | 0.680 | beats |
| c04 contextual fraud | PR-AUC | 0.046 (isolation forest, raw columns) | 0.232 | 0.274 | beats |
| c05 segments | ARI | 0.101 (k-means, raw columns) | 0.271 | 0.388 | beats |
| c06 search ranking | NDCG@10 | 0.308 (random order) | 0.819² | 0.813 | beats |
| c07 ticket tagging | macro-F1 | 0.059 (most common label) | 0.749 | 0.846 | beats |
| c08 credit grade | MAE ↓ | 1.090 (median grade) | 0.501 | 0.486 | beats |
| c09 registry (PU) | AUC vs hidden positives | 0.877 (labelled vs unlabelled) | 0.873 | — | **ties** (−0.004) |
| c09 registry (PU) | error of the class prior ↓ | 0.101 (all unlabelled negative) | 0.055 | 0.020 | beats |
| c10 machine faults | macro-F1 | 0.568 (labelled rows alone) | 0.668 | 0.707 | beats |
| c11 basket rules | top-10 rules that hold | 0.00 (all baskets, by confidence) | 0.80 | — | beats |
| c12 news topics | NMI | 0.514 (NMF defaults) | 0.524 | 0.560 | beats |
| c13 engines, clustering | ARI of the hidden regimes | 0.796 (k-means on every row) | 0.958 | 0.970 | beats |
| c13 engines, regression | MAE ↓ (held-out engines) | 37.0 (training mean) | 7.2 | 6.55 | beats |

It beats its baseline on **18 of 19 scored rows** (the exception is c09's ranking AUC, 0.004 below its baseline) and reaches the reference bar on 4 of 17. The scores also show where it falls short: c09's
prior estimate is still 39% low (0.086 against a true 0.141), because the "labelled at random" assumption is broken, and c07 misses its bar by 0.10 macro-F1.

How to read it:

* A baseline is the simplest sensible model, so beating it is a low bar. The bar is context, not a pass mark: the references use hand-built features and protocols a generic
  recipe does not have.
* ¹ c01's hold-out is a random split; the bar splits by signup month, which is harder because the effect of support tickets drifts. ² c06's ranker reads `old_rank_score`, which
  the reference leaves out (it does drop `shown_position`, which would not exist at serving time).
* Some choices were made after seeing the results (the cluster count from each algorithm's own criterion, a floor of 30 baskets per rule, the topic count from the steepest fall in
  explained variance). They are generic rules, but there is no held-out set of cases, so c05, c11 and c12 in particular were developed on this benchmark.
* The recipes have been run on their own, not yet end to end through the local model with `cli.py generate`; `python -m benchmarks.run chat … --yes` prints this same table for what
  the console saved.

## Tests

```
pytest tests                                   # 1060+ checks, ~6½ minutes
pytest tests/test_tools.py -k clustering       # one topic
PYTHONPATH=/path/to/agent_builder pytest tests/test_agent_builder_compat.py
```

* `test_tools.py` — every tool of every topic is called on synthetic data and must answer `ok`.
* `test_correctness.py` — hand-written algorithms against references or known answers (the three
  itemset miners agree exactly; Kaplan–Meier, concordance and NDCG match hand computations; the ordered
  logit recovers simulated coefficients).
* `test_topics.py` — generated files agree with the code; agents are wired to tools that exist; every
  prompt mentions every tool.
* `test_console.py` — topic discovery, data resolution and decide → approve → solve with fake agents.
* `test_recipes.py` — the default recipes name only tools that exist, find a planted leak and beat their baseline on a small table; the preparation they share; the score maths.
* `test_agent_builder_compat.py` — `agent_builder` accepts every topic config, builds every tool box,
  and runs tools through it the way a model's tool call would (skipped if it is not installed).

## Limits worth knowing

* **The prompts have only been run against a 0.5B local model.** The tools, the configs and the orchestration are tested, and
  the benchmarks ran the whole console with `qwen0.5b-tinylora-ds-assistant` (see Benchmarks): it chose no sub agent correctly
  on its own, wrote a usable plan in 3 of 13 cases, and its write-ups were never used. Expect to point the deciders and
  `topic_worker_llm` at a stronger model.
* `agent_builder` keeps parsed configs in class-level dictionaries, so two configs loaded in one process
  overwrite same-named entries. `AgentHandle.build()` clears them before building, and the topic LLMs
  have their own names, so a run is isolated — but a pipeline that mixes two topics' agents in one step
  would need that fixed upstream.
* The numpy neural pieces (autoencoder is scikit-learn's MLP; contrastive learning, Pi-model, Mean
  Teacher, FixMatch, nnPU are numpy random-feature models) show whether an idea helps on your data,
  not what a tuned deep network would reach.
