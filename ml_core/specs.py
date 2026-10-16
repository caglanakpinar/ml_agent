"""Reads a topic's tool functions and renders what an agent is told about them.

A topic's tools live in `<category>/<topic>/tools/*.py`, each a function wrapped in `@tool`. Writing
the same facts again by hand into the topic's `agentic_configuration.yaml` and `prompt.md` is how a tool
ends up in the code but not the config, or in the config with a description that no longer matches what
it does. So nothing is written twice: this module reads the functions — name, docstring, signature,
`Literal[...]` enums, `requires=` — and renders from them.

    discover_tools("supervised.classification")     -> [ToolInfo, ...] in module order, then source order
    render_tools_yaml(tools, "supervised.classification")
    render_tool_catalogue(tools)                     -> the markdown block `prompt.md` embeds

`scaffold.py` at the repository root is what calls these to (re)write every topic's generated files;
`tests/test_topics.py` calls them to prove the files on disk are not stale.

Module convention: `tools/diagnostics.py` holds the read-only inspection tools. They are the only tools
the topic's *decider* agent is given — it looks at the problem and the data and decides, and the worker
agent, which has every tool, acts on that decision.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Literal, get_args, get_origin

TOOL_SUFFIX = "_tool"
DIAGNOSTIC_MODULE = "diagnostics"

# Function names that more than one topic defines, each with its own behaviour. The console config holds every topic's tools in one
# `tools:` block, and `agent_builder` keys that block by name, so a name two topics share would let one topic's tool silently replace
# the other's. These carry their topic in the name the agents see (`active_learning_predict_with_semi_model_tool`); every other tool is
# unique already. `tests/test_console_config.py` fails if a new collision appears, which is the cue to add the function name here.
TOPIC_QUALIFIED = frozenset({
    "profile_label_coverage",
    "check_labeled_unlabeled_shift",
    "baseline_supervised_only",
    "label_scarcity_curve",
    "check_base_learner_confidence",
    "suggest_confidence_threshold",
    "predict_with_semi_model",
    "pseudo_label_quality_report",
    "simulate_label_scarcity",
    "fit_lda",
})

# What an argument means wherever it appears, so a function's docstring only needs to document what is
# particular to it. A docstring's own `Args:` entry always wins over these.
COMMON_ARGS: dict[str, str] = {
    "data": "Path to the CSV/Parquet/JSON dataset (take it from the `Current data path:` line), or the rows themselves.",
    "target": "Name of the column being predicted.",
    "features": "Feature columns to use. Defaults to every column except the target.",
    "columns": "Columns to use. Defaults to every numeric column.",
    "drop": "Columns to exclude (ids, leaky columns, free text).",
    "test_size": "Fraction of rows held out to score the model on.",
    "seed": "Random seed, so a run can be repeated.",
    "model_path": "Where to save the fitted model (joblib). Nothing is saved when omitted.",
    "output_path": "Where to write the result file. A default name in the working directory when omitted.",
    "cv": "Number of cross-validation folds.",
    "scoring": "Metric to optimise or report.",
    "n_clusters": "Number of clusters to find.",
    "n_components": "Number of components / dimensions to keep.",
    "n_iter": "How many candidates or iterations to try.",
    "scale": "Standardise numeric columns first (recommended for distance- and margin-based methods).",
    "contamination": "Expected share of anomalies in the data, between 0 and 0.5.",
    "k": "How many items to return or consider.",
    "label_column": "Column holding the label; missing values mean 'unlabeled'.",
    "unlabeled_value": "The value in the label column that marks a row as unlabeled, in addition to blanks.",
    "positive_label": "The value that counts as the positive class.",
    "date_column": "Column holding the timestamp.",
    "value_column": "Column holding the series values.",
    "horizon": "How many future steps to forecast.",
    "period": "Seasonal period in steps (e.g. 7 for daily data with a weekly cycle). Detected when omitted.",
    "text_column": "Column holding the text of each document.",
    "duration": "Column holding time-to-event or time-to-censoring.",
    "event": "Column holding 1 when the event was observed and 0 when the row is censored.",
    "relevance": "Column holding the graded relevance of each row to its group.",
    "labels": "Columns holding the label indicators (one 0/1 column per label).",
    "relevant_from": "Relevance at or above which an item counts as relevant for MAP/MRR/precision/recall.",
    "score_column": "Column holding the score each item was ranked by (higher ranks first).",
    "query_column": "Column holding the query text for each row.",
    "doc_column": "Column holding the document text for each row.",
    "n_estimators": "Number of trees / boosting rounds.",
    "learning_rate": "Shrinkage applied to each boosting round or gradient step; lower needs more rounds.",
    "max_depth": "Maximum tree depth; unset lets trees grow fully.",
    "alpha": "Regularisation strength; larger shrinks more.",
    "C": "Inverse regularisation strength; smaller is stronger.",
    "column": "The numeric column to analyse.",
    "min_df": "Terms appearing in fewer documents than this are dropped.",
    "max_df": "Terms appearing in more than this share of documents are dropped.",
    "stop_words": "Remove common English words (english) or keep every word (none).",
    "ngram_max": "Longest word n-gram (1 = single words, 2 = also pairs).",
    "n_topics": "Number of topics.",
    "transaction_column": "Column identifying the transaction (long format: one row per item, with `item_column`).",
    "item_column": "Column holding one item per row (long format).",
    "basket_column": "Alternatively, one column listing each transaction's items joined by `separator`.",
    "onehot_columns": "Alternatively, 0/1 columns, one per item.",
    "min_support": "Share of transactions an itemset or rule must appear in.",
    "min_confidence": "Smallest confidence: P(consequent | antecedent).",
    "min_lift": "Smallest lift: confidence relative to the consequent's base rate.",
    "prior": "Class prior — the share of ALL rows that are truly positive. Estimated when omitted; a wrong value shifts every threshold.",
    "max_rows": "Subsample to this many rows first (for methods that are quadratic in the number of rows).",
    "initial_size": "Number of labelled rows the simulation starts from.",
    "batch_size": "Rows queried (or labelled) per round.",
    "lr": "Step size of the optimiser.",
    "epochs": "Number of passes of gradient descent over the training data.",
    "n_neighbors": "Number of neighbours considered.",
    "epsilon": "Tolerance / margin width.",
    "l2": "L2 penalty (weight decay).",
    "max_iter": "Maximum number of iterations.",
    "covariates": "Covariate columns; defaults to every column except duration and event.",
    "id_column": "Column to carry through to the output so results can be joined back.",
    "n_groups": "Number of groups (quantile bins) to split into.",
    "bins": "Number of quantile bins used to cut a numeric column into groups.",
    "folds": "Number of cross-validation folds.",
    "times": "Times at which to report.",
    "horizons": "Times at which to evaluate; defaults to quartiles of the event times.",
    "n_bins": "Number of bins / intervals.",
    "freq": "Pandas frequency alias (D, W, MS, h ...); inferred from the dates when omitted.",
    "holdout": "Hide the last N points, forecast them and report errors; 0 forecasts the real future.",
    "n_splits": "Number of forecast origins / splits.",
    "params": "Extra arguments for the chosen method, as a JSON object.",
    "token_column": "Column holding the token (used for CSV corpora; CoNLL text files need no columns).",
    "tag_column": "Column holding the tag of each token (used for CSV corpora).",
    "sentence_column": "Column identifying the sentence each token belongs to (used for CSV corpora).",
    "truth_column": "Column holding the true label of every row, including the unlabelled ones — only for experiments where labels were hidden on purpose; used to audit pseudo-labels, never to train.",
    "base": "Base learner used inside the method.",
    "threshold": "Confidence a prediction needs before it is trusted as a pseudo-label.",
    "keep_fraction": "Share of labels to keep (the rest are hidden).",
    "separator": "Delimiter between the values inside one cell.",
    "series_column": "Column identifying which series each row belongs to.",
    "bootstrap": "Number of bootstrap resamples.",
    "group_column": "Column identifying the group each row belongs to (for survival: the column defining groups to compare).",
}


@dataclass
class ArgInfo:
    name: str
    type: str
    description: str
    required: bool
    enum: list[str] | None = None
    default: Any = None


@dataclass
class ToolInfo:
    name: str  # the function name
    module: str  # short module name, e.g. "models"
    caller: str  # dotted path to the function
    description: str
    group: str  # the module docstring's first line
    args: list[ArgInfo] = field(default_factory=list)
    requires: tuple[str, ...] = ()

    @property
    def tool_name(self) -> str:
        """What the agents call it: the function name plus `_tool`, with the topic in front when the name is shared (see `TOPIC_QUALIFIED`)."""
        if self.name in TOPIC_QUALIFIED:
            return f"{self.caller.split('.')[1]}_{self.name}{TOOL_SUFFIX}"
        return f"{self.name}{TOOL_SUFFIX}"

    @property
    def diagnostic(self) -> bool:
        return self.module == DIAGNOSTIC_MODULE


def _kind(annotation: Any) -> tuple[str, list[str] | None]:
    """Map a type annotation to the YAML `type:` and an `enum:` when it is a Literal."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return "str", None
    origin = get_origin(annotation)
    if origin is Literal:
        return "str", [str(a) for a in get_args(annotation)]
    if origin in (typing.Union, types.UnionType):
        members = [a for a in get_args(annotation) if a is not type(None)]
        kinds = [_kind(m) for m in members]
        for preferred in ("list", "dict"):
            for kind, enum in kinds:
                if kind == preferred:
                    return kind, enum
        for kind, enum in kinds:
            if enum:
                return kind, enum
        return (kinds[0] if kinds else ("str", None))
    if origin in (list, tuple, set, frozenset) or annotation in (list, tuple, set):
        return "list", None
    if origin is dict or annotation is dict:
        return "dict", None
    if annotation is bool:
        return "bool", None
    if annotation is int:
        return "int", None
    if annotation is float:
        return "float", None
    return "str", None


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a docstring into its summary paragraph and an `Args:` name -> description mapping."""
    doc = inspect.cleandoc(doc or "")
    head, _, rest = doc.partition("\nArgs:")
    summary = " ".join(head.strip().split("\n\n")[0].split())
    described: dict[str, str] = {}
    if rest:
        block = re.split(r"\n(?:Returns|Raises|Notes?|Example|Examples):", rest)[0]
        lines = [line for line in block.splitlines() if line.strip()]
        base = min((len(line) - len(line.lstrip()) for line in lines), default=0)
        current = None
        for line in lines:
            depth = len(line) - len(line.lstrip())
            match = re.match(r"^(\w+)\s*(?:\([^)]*\))?:\s*(.*)$", line.strip())
            if depth == base and match:
                current = match.group(1)
                described[current] = match.group(2).strip()
            elif current:
                described[current] += " " + line.strip()
    return summary, {k: " ".join(v.split()) for k, v in described.items()}


def describe(fn: Any, module_short: str, group: str, module_path: str) -> ToolInfo:
    inner = getattr(fn, "__wrapped__", fn)
    signature = inspect.signature(inner, eval_str=True)
    summary, documented = _parse_docstring(inner.__doc__ or "")
    if not summary:
        raise ValueError(f"{module_path}.{inner.__name__} has no docstring — an agent reads it as the tool's description")

    args = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        kind, enum = _kind(parameter.annotation)
        text = documented.get(name) or COMMON_ARGS.get(name)
        if not text:
            raise ValueError(
                f"{module_path}.{inner.__name__}: argument `{name}` is undocumented — add it to the "
                "docstring's Args: block or to ml_core.specs.COMMON_ARGS"
            )
        has_default = parameter.default is not inspect.Parameter.empty
        args.append(
            ArgInfo(name, kind, text, required=not has_default, enum=enum,
                    default=parameter.default if has_default else None)
        )
    return ToolInfo(
        name=inner.__name__,
        module=module_short,
        caller=f"{module_path}.{inner.__name__}",
        description=summary,
        group=group,
        args=args,
        requires=tuple(getattr(fn, "__requires__", ())),
    )


def discover_tools(package: str) -> list[ToolInfo]:
    """Every `@tool` function in `<package>.tools.*`, modules in alphabetical order, functions in source order.

    Modules named with a leading underscore are helpers and are skipped, as are functions a module merely
    imports from elsewhere — only what a module defines is that module's tool.
    """
    tools_package = importlib.import_module(f"{package}.tools")
    found: list[ToolInfo] = []
    for module_info in sorted(pkgutil.iter_modules(tools_package.__path__), key=lambda m: m.name):
        if module_info.name.startswith("_") or module_info.name == "catalogue":
            continue
        module_path = f"{package}.tools.{module_info.name}"
        module = importlib.import_module(module_path)
        group = (inspect.getdoc(module) or module_info.name).splitlines()[0].strip()
        members = [
            fn
            for _, fn in inspect.getmembers(module, inspect.isfunction)
            if getattr(fn, "__tool__", False) and getattr(fn, "__wrapped__", fn).__module__ == module_path
        ]
        members.sort(key=lambda fn: inspect.getsourcelines(getattr(fn, "__wrapped__", fn))[1])
        found.extend(describe(fn, module_info.name, group, module_path) for fn in members)

    names = [t.name for t in found]
    duplicated = {n for n in names if names.count(n) > 1}
    if duplicated:
        raise ValueError(f"{package}: tool names defined twice: {sorted(duplicated)}")
    return found


def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_tools_yaml(tools: list[ToolInfo], catalogue_caller: str, catalogue_description: str, catalogue_name: str = "tool_catalogue_tool") -> str:
    """The `tools:` block: the catalogue tool first, then every tool, grouped by module."""
    lines = ["tools:"]

    def emit(name: str, description: str, caller: str, args: list[ArgInfo]) -> None:
        lines.append(f"  - name: {_quote(name)}")
        lines.append(f"    description: {_quote(description)}")
        lines.append(f"    caller: {_quote(caller)}")
        if args:
            lines.append("    args:")
        for arg in args:
            lines.append(f"      - name: {_quote(arg.name)}")
            lines.append(f"        type: {_quote(arg.type)}")
            lines.append(f"        description: {_quote(arg.description)}")
            if arg.enum:
                lines.append("        enum: [" + ", ".join(_quote(e) for e in arg.enum) + "]")
            lines.append(f"        required: {'true' if arg.required else 'false'}")
        lines.append("")

    lines.append("  # ---- catalogue " + "-" * 70)
    emit(
        catalogue_name,
        catalogue_description,
        catalogue_caller,
        [ArgInfo("group", "str", "Restrict the listing to one tool module, e.g. 'models'.", False)],
    )
    current = None
    for info in tools:
        if info.module != current:
            current = info.module
            lines.append(f"  # ---- {info.module}: {info.group} ".ljust(100, "-"))
        description = info.description
        if info.requires:
            description += f" Requires: {', '.join(info.requires)}."
        emit(info.tool_name, description, info.caller, info.args)
    return "\n".join(lines).rstrip() + "\n"


def render_tool_catalogue(tools: list[ToolInfo]) -> str:
    """The markdown list `prompt.md` embeds: every tool, its description, grouped by module."""
    lines: list[str] = []
    current = None
    for info in tools:
        if info.module != current:
            current = info.module
            lines += ["", f"**{info.module}** — {info.group}", ""]
        needs = f" *(needs {', '.join(info.requires)})*" if info.requires else ""
        lines.append(f"- `{info.tool_name}` — {info.description}{needs}")
    return "\n".join(lines).strip()
