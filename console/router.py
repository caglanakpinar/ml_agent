"""A rule-based first reading of a problem: which sub agent (problem type) do the user's words and the data file point to?

`ml_decider_agent` is a small model, and on the benchmark it answered the same pair of names for most problems whatever they said. What a
file and a sentence can settle does not need a model: whether the column the user talks about is filled on every row (supervised) or on
a few (semisupervised) or absent (unsupervised); whether it is yes/no, a few categories, ordered levels or a number; whether several
0/1 columns share a prefix; whether there is a duration next to an event; whether rows are grouped by a query or a basket id. The router reads those
facts, scores every problem type with named rules, and keeps the reasons, so an approver sees why.

    route = route_problem(user_text, data_path, topics)
    route.decided          # [("supervised", "classification")] when one reading is clearly ahead, else None
    route.shortlist        # the best few, for the model to choose among when it is not

A reading is *decided* when its best score is at least `MIN_SCORE` and leads the next one by at least `MIN_MARGIN`. Some things a file cannot say
(label propagation or self-training; clustering or anomaly detection when the user says "unusual groups"): those come back as a shortlist.
Two jobs in one request ("first find the groups, then predict X") are read by splitting at "then" and routing each part.

Nothing here calls a model. The rules are plain data below, so they can be read, tested and changed.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MIN_SCORE = 6
MIN_MARGIN = 3
SHORTLIST = 3

PATH_LIKE = re.compile(r"(?:[\w.~-]*/)+[\w.-]*|[\w.-]+\.(?:csv|tsv|txt|parquet|json|jsonl|xlsx|xls)\b", re.I)


def words(*stems: str) -> re.Pattern[str]:
    """A case-insensitive pattern matching any of the stems at the start of a word (`segment` matches `segments`, `segmentation`)."""
    return re.compile(r"\b(?:" + "|".join(stems) + r")", re.I)


# What the user's words can say. Each pattern is a stem list, so inflections match.
FORECAST = words("forecast", "project(?:ion)?s? (?:for|of)", r"predict (?:the )?next", r"next \d+ (?:days?|weeks?|months?|hours?|quarters?|years?)", r"next (?:day|week|month|quarter|year)", r"future (?:demand|sales|values)")
SURVIVAL = words("how long", "time to", "time until", "survival", "censor", "hazard", r"not yet", r"still (?:in|well|active|alive|here)", r"have not (?:yet )?(?:churned|failed|happened|relapsed|died)", "remission", "lifetime", "until (?:they|it|the)")
ANOMALY = words("fraud", "anomal", "outlier", "unusual", "suspicious", "abnormal", "irregular", "odd ", "strange", "investigat", "fake", "forged", "counterfeit", "suspect", "dubious", "scam", "laundering", "spam", r"look(?:s)? (?:wrong|off|odd|fishy|wrong)", "misuse", "abuse")
CLUSTER = words(r"(?:divide|split|sort|bucket|partition|break|put)\w* (?:\w+ ){0,5}?into (?:a few |some |several |\d+ )?(?:\w+ )?(?:types|groups|kinds|buckets|clusters|segments|tiers)", r"types of (?:users|customers|clients|members)", "segment", "cluster", "regimes?", "personas?", "typolog", "natural groups?", r"groups? of \w+s\b", r"group (?:our|the|my|these|those) \w+", "behaviou?ral groups", "operating (?:modes|regimes)")
BASKET = words("bought together", "basket", "association", "co-?occur", "combinations? (?:of )?(?:products|items)", "product combinations", "frequently (?:bought|purchased)", "often bought", "market.basket", "cross.sell")
TOPIC = words("themes?", "topics?", r"what (?:are|is) (?:they|these|the \w+) about", "subjects?", "discuss", r"summari[sz]e (?:what|the|these|all)", r"what (?:\w+ ){1,3}(?:complain|talk|write|say|mention|ask)\w* about")
DENSITY = words("density", "distributed", "distribution of", "likelihood", "probability (?:that|of|a)\b", "how likely", "tails?\b", "percentile")
REDUCE = words("dimension", "compress", "visuali[sz]e", "pca\b", "t-?sne", "umap", r"too many (?:columns|features)", r"reduce the (?:number of )?(?:columns|features)", r"\d ?-?d (?:picture|plot|map|view|chart)", r"(?:picture|map|plot) of how", "low.dimensional", r"\bplot\b")
EMBED = words("embedding", "representation", "similarity search", "vector space")
RANK = words("rank", "ndcg", r"order (?:the|these|those|items|results|products|postings)", r"sort (?:the|these|those)", "search results?", "best order")
ORDERED = words("grade", "rating", "stars?\b", "severity", "ordered", r"order(?:ing)? matters", "worst", "best\b", "satisf", "tier", "off by (?:one|a)", r"1 ?(?:=|to|–|-) ?\d")
TAGGING = words("each (?:word|token)", "token", "named entit", r"\bner\b", "entity type", "part.of.speech", "tag each")
MULTI = words("every (?:genre|tag|label|category)", "all (?:the )?(?:tags|labels|genres)", "multi.?label", "tags? (?:that|which) apply", r"labels? that apply", r"(?:genre|tag|label)s? (?:that )?applies", "assign every")
PROPAGATE = words("propagat", "neighbo", "similar", "graph", "manifold", "curved", "smooth")
SELFTRAIN = words("confident", "self.?train", "iterat")
PSEUDO = words("pseudo", "relabel", "machine.made", "generate labels")
VIEWS = words(r"two (?:views|feature sets)", r"independent (?:views|sets)", "co.?train", "multi.?view")
AUGMENT = words("augment", "perturb", "consisten")
ANNOTATE = words("annotat", r"which (?:ones|rows|items|images|examples) (?:should|to)", r"label (?:next|more)", "labell?ing budget", "budget for label", r"pay (?:\w+ )?to label")
POSITIVE_ONLY = words("blocklist", "watchlist", "known (?:positives|cases|fraud)", "undiagnosed", "hidden positives", "everyone else is unlabell?ed", r"only (?:some|a few|a fraction)[\w ,]{0,40}(?:recorded|marked|on our|diagnosed|listed|confirmed)")
FILL_REST = words(r"(?:label|fill in|complete) the (?:rest|remaining|others)", "unlabell?ed")
NO_LABELS = words(r"no (?:\w+ )?labels", r"(?:nobody|no one) has (?:\w+ )?(?:marked|tagged|labell?ed|gone through|checked|reviewed|looked)", "unlabell?ed", r"without labels", r"no (?:verified|confirmed|known|marked|recorded|tagged) \w+", "never recorded")
HIDDEN_GROUPS = words(r"not one (?:population|group|kind)", r"nobody (?:has )?labell?ed (?:the |any )?[\w ]{0,25}(?:regime|group|type|segment|mode|kind)", r"(?:unlabell?ed|hidden|unknown) (?:groups?|regimes?|types?|segments?|modes?)",
                    r"(?:different|several|distinct) (?:kinds|types|ways|regimes|modes|populations)", "behave differently", r"discover (?:the |which )?[\w ]{0,20}(?:groups?|regimes?|types?|segments?)")
SEQUENCE = re.compile(r"\bthen\b", re.I)
PREDICT = words("predict", "estimate", "classif", "forecast", "model the")

# Column-name cues.
DURATION_NAME = re.compile(r"duration|tenure|time_to|survival|follow|lifetime|age_at|(?:seconds|minutes|hours|days|weeks|months|years)_?(?:to|until|in|of|since|open|alive)?(?:_|$)", re.I)
EVENT_NAME = re.compile(r"event|churn|died|death|fail|relaps|observed|status|cens|cancel|default", re.I)
GROUP_QUERY = re.compile(r"(?:query|search|session|request|page|list|impression|result)s?_?id$|^(?:query|search|session)$", re.I)
GROUP_BASKET = re.compile(r"(?:basket|order|transaction|invoice|receipt|cart|visit)_?id$|^(?:basket|order|transaction)$", re.I)
ITEM_NAME = re.compile(r"^(?:item|items|product|products|sku|article|good|goods|product_name|item_name)$", re.I)
RELEVANCE_NAME = re.compile(r"relevan|usefulness|click|grade|rating|score$|label$|purchas|engag", re.I)
TOKEN_NAME = re.compile(r"^(?:token|word|tok)$", re.I)
TAG_NAME = re.compile(r"^(?:tag|ner|ner_tag|pos|pos_tag|entity|bio)$", re.I)
SENTENCE_NAME = re.compile(r"sentence|utterance|^doc(?:ument)?_?id$", re.I)
LABELISH_NAME = re.compile(r"label|target|truth|annotation|diagnos", re.I)
ORDERED_NAME = re.compile(r"grade|rating|star|severity|level|stage|tier|satisf", re.I)
ID_NAME = re.compile(r"(?:^|_)id$|^id$|^index$", re.I)
DATE_NAME = re.compile(r"date|time|timestamp|publish|_at$|^ts$|^day$|^month$", re.I)


@dataclass
class Column:
    """What the file says about one column the user's words point at."""

    name: str
    filled: float
    distinct: int
    kind: str  # binary | levels | categories | number | text | id | other
    values: list[str]


@dataclass
class Candidate:
    problem_type: str
    score: int = 0
    reasons: list[str] = field(default_factory=list)
    hidden_groups: bool = False  # clustering scored because the text says the groups exist but nobody has labelled them

    def add(self, points: int, reason: str) -> None:
        self.score += points
        self.reasons.append(reason)


@dataclass
class Job:
    """The reading of one job in the request: every problem type that scored, best first."""

    text: str
    candidates: list[Candidate]

    @property
    def top(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def margin(self) -> int:
        return self.candidates[0].score - (self.candidates[1].score if len(self.candidates) > 1 else 0) if self.candidates else 0

    @property
    def decided(self) -> bool:
        return bool(self.candidates) and self.candidates[0].score >= MIN_SCORE and self.margin >= MIN_MARGIN


@dataclass
class Route:
    jobs: list[Job]
    topics: dict[str, list[str]]

    def pair(self, problem_type: str) -> tuple[str, str]:
        return next((c, problem_type) for c, types in self.topics.items() if problem_type in types)

    @property
    def decided(self) -> list[tuple[str, str]] | None:
        """The `(category, problem_type)` pairs, in run order, when every job has a clear leader; otherwise None."""
        if not self.jobs or not all(job.decided for job in self.jobs):
            return None
        types = [job.top.problem_type for job in self.jobs]  # type: ignore[union-attr]
        return [self.pair(t) for t in dict.fromkeys(types)]

    @property
    def shortlist(self) -> list[tuple[str, str]]:
        """The best few readings across the jobs, for a model to choose among when nothing is decided."""
        seen: dict[str, int] = {}
        for job in self.jobs:
            for candidate in job.candidates[:SHORTLIST]:
                seen[candidate.problem_type] = max(seen.get(candidate.problem_type, 0), candidate.score)
        ranked = sorted(seen, key=lambda t: -seen[t])[:SHORTLIST]
        return [self.pair(t) for t in ranked]

    def explain(self) -> str:
        """The evidence, one line per job and candidate, for the decider's context and for the approver."""
        lines = []
        for index, job in enumerate(self.jobs, start=1):
            if len(self.jobs) > 1:
                lines.append(f"Job {index}: {job.text.strip()[:120]}")
            for candidate in job.candidates[:SHORTLIST]:
                lines.append(f"- {candidate.problem_type} (score {candidate.score}): " + "; ".join(candidate.reasons[:4]))
            if not job.candidates:
                lines.append("- nothing in the text or the file points to a problem type")
        return "\n".join(lines)

    def verdict(self) -> dict[str, Any]:
        """The decided pairs as the JSON `ml_decider_agent` would have produced."""
        pairs = self.decided or []
        by_type = {job.top.problem_type: job.top for job in self.jobs if job.top}
        return {
            "sub_agents": [{"category": c, "problem_type": t, "reasoning": "; ".join(by_type[t].reasons[:3])} for c, t in pairs],
            "reasoning": "Chosen by the rule-based router from the problem text and the data file.",
        }


# --------------------------------------------------------------------------------------------------------------------------------------
# reading the file
# --------------------------------------------------------------------------------------------------------------------------------------
def read_frame(path: str | None, max_rows: int = 20000):
    """The first `max_rows` rows of a data file, or None when it cannot be read."""
    if not path:
        return None
    try:
        import pandas as pd

        suffix = Path(path).suffix.lower()
        if suffix in (".csv", ".txt"):
            return pd.read_csv(path, nrows=max_rows)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t", nrows=max_rows)
        if suffix == ".parquet":
            return pd.read_parquet(path).head(max_rows)
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True, nrows=max_rows)
    except Exception:
        return None
    return None


def describe_column(frame, name: str) -> Column:
    import pandas as pd

    series = frame[name].replace("", pd.NA)
    values = series.dropna()
    distinct = int(values.nunique())
    filled = float(series.notna().mean())
    numeric = pd.api.types.is_numeric_dtype(values) and len(values) > 0
    long_text = bool(len(values)) and not numeric and bool(values.astype(str).str.len().median() > 80)
    if long_text:
        kind = "text"
    elif ID_NAME.search(name) or (distinct >= 0.95 * max(len(frame), 1) and not numeric):
        kind = "id"
    elif distinct == 2:
        kind = "binary"
    elif numeric and distinct <= 12 and bool((values % 1 == 0).all()):
        kind = "levels"
    elif numeric:
        kind = "number"
    elif distinct <= 12:
        kind = "categories"
    else:
        kind = "other"
    return Column(name, filled, distinct, kind, [str(v) for v in values.drop_duplicates().head(6)])


def mentions(text: str, name: str) -> bool:
    spoken = str(name).replace("_", " ")
    return any(re.search(rf"(?<![\w]){re.escape(candidate)}(?![\w])", text, re.I) for candidate in {str(name), spoken})


def zero_one_groups(frame) -> dict[str, list[str]]:
    import pandas as pd

    groups: dict[str, list[str]] = {}
    for column in frame.columns:
        values = frame[column].dropna()
        if "_" in str(column) and pd.api.types.is_numeric_dtype(values) and set(values.unique()) <= {0, 1} and values.nunique() == 2:
            groups.setdefault(str(column).split("_", 1)[0] + "_", []).append(str(column))
    return {prefix: cols for prefix, cols in groups.items() if len(cols) >= 3}


def date_columns(frame) -> list[str]:
    """Columns whose values read as dates/times — judged by the values, so `week` or `period` count as well as `date`."""
    import pandas as pd

    found = []
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            continue
        sample = frame[column].dropna().astype(str).head(200)
        if len(sample) == 0 or sample.str.len().median() > 40 or not sample.str.contains(r"\d", regex=True).mean() > 0.9:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if pd.to_datetime(sample, errors="coerce").notna().mean() > 0.9:
                found.append(str(column))
    return found


ORDINAL_SCALES = [
    ("low", "medium", "high", "critical", "very high", "moderate", "minor", "major", "severe", "mild", "none", "extreme", "med"),
    ("small", "medium", "large", "extra large", "xl", "xs", "s", "m", "l"),
    ("poor", "fair", "good", "very good", "excellent", "bad", "average", "great", "terrible"),
    ("never", "rarely", "sometimes", "often", "always", "seldom", "usually"),
    ("strongly disagree", "disagree", "neutral", "agree", "strongly agree"),
    ("beginner", "intermediate", "advanced", "expert"),
    ("junior", "mid", "senior", "lead"),
    ("bronze", "silver", "gold", "platinum"),
]


def ordered_vocabulary(values: list[str]) -> bool:
    """Whether three or more category values are all drawn from one familiar ordered scale (low < medium < high, poor < fair < good, ...)."""
    lowered = {v.strip().lower() for v in values}
    return len(lowered) >= 3 and any(lowered <= set(scale) for scale in ORDINAL_SCALES)


# --------------------------------------------------------------------------------------------------------------------------------------
# scoring one job
# --------------------------------------------------------------------------------------------------------------------------------------
def score_job(text: str, frame, topics: dict[str, list[str]]) -> list[Candidate]:
    """Score every problem type for one job's text and the file. Only types that exist in `topics` and scored above zero are returned."""
    import pandas as pd

    known = {t for types in topics.values() for t in types}
    table: dict[str, Candidate] = {}

    def add(problem_type: str, points: int, reason: str) -> None:
        if problem_type in known:
            table.setdefault(problem_type, Candidate(problem_type)).add(points, reason)

    def hit(pattern: re.Pattern[str]) -> str | None:
        found = pattern.search(text)
        return found.group(0).strip() if found else None

    columns = list(frame.columns) if frame is not None else []
    facts = {c: describe_column(frame, c) for c in columns} if frame is not None else {}

    # --- the column(s) the user's words point at: spoken of in the text, or named like a label
    spoken = [c for c in columns if mentions(text, c) and facts[c].kind != "id"]
    labelish = [c for c in columns if LABELISH_NAME.search(str(c)) and c not in spoken and facts[c].kind != "id" and 0.0 < facts[c].filled < 0.9]
    groups_01 = zero_one_groups(frame) if frame is not None else {}
    grouped = {c for cols in groups_01.values() for c in cols}
    targets = [facts[c] for c in spoken + labelish if c not in grouped]

    complete = [t for t in targets if t.filled >= 0.9]
    partial = [t for t in targets if 0.002 <= t.filled < 0.9 and t.kind != "text"]

    # --- supervised: the one target column filled on (almost) every row
    duration = [c for c in columns if DURATION_NAME.search(str(c)) and facts[c].kind in ("number", "levels")]
    cue = hit(SURVIVAL)
    event = [c for c in columns if facts[c].kind == "binary" and c not in duration and (EVENT_NAME.search(str(c)) or (cue and duration and c in spoken))]
    survival_shape = bool(duration and event and cue)  # a duration next to an event, and words about time to it: the event column is the event, not a class to predict

    def supervised(t: Column) -> list[tuple[str, int, str]]:
        if t.kind == "binary":
            return [("classification", 7, f"`{t.name}` is a yes/no column filled on every row")]
        if t.kind == "categories":
            if ordered_vocabulary(t.values):
                return [("classification", 3, f"`{t.name}` holds {t.distinct} categories on every row"),
                        ("ordinal_regression", 9, f"`{t.name}` holds {t.distinct} levels of one ordered scale ({', '.join(t.values[:4])})")]
            return [("classification", 6, f"`{t.name}` holds {t.distinct} categories on every row")]
        if t.kind == "levels":
            ordered = hit(ORDERED)
            return [("classification", 3, f"`{t.name}` takes {t.distinct} whole-number levels"),
                    ("ordinal_regression", 2 + (4 if ordered else 0) + (2 if ORDERED_NAME.search(t.name) else 0),
                     f"`{t.name}` takes {t.distinct} levels" + (f" and the text speaks of an order ('{ordered}')" if ordered else ""))]
        if t.kind == "number":
            return [("regression", 6, f"`{t.name}` is a number ({t.distinct} distinct values) on every row")]
        return []

    options = [supervised(t) for t in complete if not (survival_shape and t.name in event)]
    options = [o for o in options if o]
    if options:
        for problem_type, points, reason in max(options, key=lambda o: max(p for _, p, _ in o)):
            add(problem_type, points, reason)

    # --- semisupervised: the label column is filled on only some rows
    for t in partial[:1]:
        share = f"`{t.name}` has a value on only {t.filled:.1%} of rows"
        if t.distinct == 1:
            add("positive_unlabeled_learning", 10, f"{share} and the only value present is '{t.values[0]}': nothing is marked as the opposite")
        else:
            numeric_only = frame is not None and all(pd.api.types.is_numeric_dtype(frame[c]) for c in columns if c != t.name)
            add("label_propagation", 3 + (2 if numeric_only else 0), f"{share}; the other columns are all numeric, so a similarity graph is meaningful" if numeric_only else share)
            add("self_training", 3, share)
            add("pseudo_labeling", 3, share)
            if (word := hit(FILL_REST)):
                add("pseudo_labeling", 1, f"the text asks to label the rest ('{word}')")
                add("label_propagation", 1, f"the text asks to label the rest ('{word}')")
            if (word := hit(PROPAGATE)):
                add("label_propagation", 4, f"the text speaks of similarity/propagation ('{word}')")
            if (word := hit(SELFTRAIN)):
                add("self_training", 4, f"the text speaks of confident/iterated labelling ('{word}')")
            if (word := hit(PSEUDO)):
                add("pseudo_labeling", 4, f"the text speaks of pseudo-labels ('{word}')")
            if (word := hit(VIEWS)):
                add("co_training", 8, f"the text speaks of two views ('{word}')")
            if (word := hit(AUGMENT)):
                add("consistency_regularization", 8, f"the text speaks of augmentation/consistency ('{word}')")
        if (word := hit(POSITIVE_ONLY)):
            add("positive_unlabeled_learning", 4, f"the text says only some positives are recorded ('{word}')")
        if (word := hit(ANNOTATE)):
            add("active_learning", 9, f"the text is about choosing what to label ('{word}')")

    # --- structure in the file
    for prefix, cols in groups_01.items():
        add("multi_label_classification", 8, f"{len(cols)} columns share the prefix `{prefix}` and are all 0/1 — one row can carry several")
    if (word := hit(MULTI)):
        add("multi_label_classification", 3, f"the text asks for every label that applies ('{word}')")

    if duration and event:
        add("survival_analysis", 10 if cue else 3, f"`{duration[0]}` looks like a duration next to the 0/1 `{event[0]}`" + (f" and the text asks about time to an event ('{cue}')" if cue else ""))
    elif cue:
        add("survival_analysis", 4, f"the text asks about time to an event ('{cue}')")

    dates = date_columns(frame) if frame is not None else []
    if (word := hit(FORECAST)):
        add("time_series_forecasting", 10 if dates else 4, f"the text asks for a forecast ('{word}')" + (f" and `{dates[0]}` is a date column" if dates else ""))

    query = [c for c in columns if GROUP_QUERY.search(str(c))]
    relevance = [c for c in columns if RELEVANCE_NAME.search(str(c)) and facts[c].kind in ("levels", "binary", "number")]
    if query and relevance:
        add("ranking", 8, f"rows are grouped by `{query[0]}` and `{relevance[0]}` grades each item")
    if (word := hit(RANK)):
        add("ranking", 4, f"the text is about ordering items ('{word}')")
        repeating = [c for c in columns if ID_NAME.search(str(c)) and frame is not None and 1 < facts[c].distinct < 0.5 * len(frame)]
        if repeating and not query:
            add("ranking", 6, f"rows are grouped by `{repeating[0]}` ({facts[repeating[0]].distinct} groups) and the text asks for an order")

    basket = [c for c in columns if GROUP_BASKET.search(str(c))]
    items = [c for c in columns if ITEM_NAME.search(str(c))]
    if basket and items:
        add("association_rule_mining", 9, f"one row per item (`{items[0]}`) grouped by `{basket[0]}`")
    if (word := hit(BASKET)):
        add("association_rule_mining", 5, f"the text is about items bought together ('{word}')")

    if any(TOKEN_NAME.search(str(c)) for c in columns) and any(TAG_NAME.search(str(c)) for c in columns):
        add("sequence_labeling", 10, "the file has a token column and a tag column" + (" grouped by sentence" if any(SENTENCE_NAME.search(str(c)) for c in columns) else ""))
    if (word := hit(TAGGING)):
        add("sequence_labeling", 4, f"the text is about tagging tokens ('{word}')")

    long_text = [c for c in columns if facts[c].kind == "text"]
    if long_text:
        add("topic_modeling", 4, f"`{long_text[0]}` is free text")
    if (word := hit(TOPIC)):
        add("topic_modeling", 5 if long_text else 2, f"the text asks about themes/topics ('{word}')")

    # --- unsupervised: words about structure, only when no target column is in play
    open_ended = not complete and not partial
    if (word := hit(ANOMALY)):
        add("anomaly_detection", 6 if open_ended else 1, f"the text is about unusual rows ('{word}')")
        if open_ended and (none := hit(NO_LABELS)):
            add("anomaly_detection", 2, f"and there are no labels ('{none}')")
    if (word := hit(CLUSTER)):
        add("clustering", 6 if open_ended or not complete else 2, f"the text asks for groups/segments ('{word}')")
    if (word := hit(HIDDEN_GROUPS)):
        add("clustering", 7, f"the text says there are groups nobody has labelled ('{word}')")
        table["clustering"].hidden_groups = True
    if (word := hit(DENSITY)) and open_ended:
        add("density_estimation", 6, f"the text is about a distribution/probability ('{word}')")
    if (word := hit(REDUCE)):
        add("dimensionality_reduction", 7 + (2 if len(columns) > 30 else 0), f"the text asks to compress/visualise dimensions ('{word}')" + (f"; the file has {len(columns)} columns" if len(columns) > 30 else ""))
    if (word := hit(EMBED)):
        add("representation_learning", 6, f"the text asks for embeddings/representations ('{word}')")

    ranked = sorted(table.values(), key=lambda c: (-c.score, c.problem_type))
    return [c for c in ranked if c.score > 0]


def user_text(text: str) -> str:
    """The user's words without file paths: a folder or file name is not part of what they asked, and would only match words by accident."""
    return PATH_LIKE.sub(" ", text)


def split_jobs(text: str) -> list[str]:
    """Two jobs in one request ("first find the groups, then predict X") split at 'then'; a request with no 'then' is one job."""
    parts = [p for p in re.split(r"[,;]?\s*\bthen\b[,]?\s*", text, flags=re.I) if len(p.split()) >= 3]
    return parts if len(parts) > 1 else [text]


def route_problem(text: str, path: str | None, topics: dict[str, list[str]]) -> Route:
    """Read `text` (what the user said) and the file at `path` and return the reading of each job in the request."""
    text = user_text(text)
    frame = read_frame(path)
    parts = split_jobs(text)
    if len(parts) > 1:
        jobs = [Job(part, score_job(part, frame, topics)) for part in parts]
        tops = [job.top.problem_type for job in jobs if job.top]
        if len(tops) == len(jobs) and len(set(tops)) == len(tops) and all(job.top.score >= MIN_SCORE for job in jobs if job.top):
            return Route(jobs, topics)
    candidates = score_job(text, frame, topics)
    # A named target plus groups nobody has labelled ("they degrade in different ways, nobody labelled the regime"): find the groups first, then predict.
    hidden = next((c for c in candidates if c.problem_type == "clustering" and c.hidden_groups), None)
    supervised = [c for c in candidates if c.problem_type in topics.get("supervised", [])]
    if hidden and supervised and supervised[0].score >= MIN_SCORE:
        return Route([Job(text, [hidden]), Job(text, supervised)], topics)
    return Route([Job(text, candidates)], topics)


def column_hints(text: str, path: str | None) -> dict[str, Any]:
    """The columns the problem is about, by the argument name the tools use for them (`target`, `label_column`, `duration`, `event`, ...).

    Read the same way `score_job` reads them — the columns the user names, or a partly filled column named like a label — and used by the plan executor to
    fill in an argument the decider's plan left out. Only columns that exist in the file are returned.
    """
    import pandas as pd

    text = user_text(text)
    frame = read_frame(path)
    if frame is None:
        return {}
    columns = [str(c) for c in frame.columns]
    facts = {c: describe_column(frame, c) for c in columns}
    spoken = [c for c in columns if mentions(text, c) and facts[c].kind != "id"]
    labelish = [c for c in columns if LABELISH_NAME.search(c) and c not in spoken and facts[c].kind != "id" and 0.0 < facts[c].filled < 0.9]
    grouped = {c for cols in zero_one_groups(frame).values() for c in cols}
    hints: dict[str, Any] = {}

    duration = [c for c in columns if DURATION_NAME.search(c) and facts[c].kind in ("number", "levels")]
    cue = SURVIVAL.search(text)
    event = [c for c in columns if facts[c].kind == "binary" and c not in duration and (EVENT_NAME.search(c) or (cue and duration and c in spoken))]
    if duration and event:
        hints["duration"], hints["event"] = next((d for d in duration if d in spoken), duration[0]), event[0]

    candidates = [c for c in spoken + labelish if c not in grouped and c not in (hints.get("duration"), hints.get("event"))]
    partial = [c for c in candidates if 0.002 <= facts[c].filled < 0.9 and facts[c].kind != "text"]
    complete = [c for c in candidates if facts[c].filled >= 0.9 and facts[c].kind in ("binary", "categories", "levels", "number")]
    priority = {"binary": 0, "categories": 1, "levels": 2, "number": 3}
    if partial:
        hints["label_column"] = hints["target"] = partial[0]
        if facts[partial[0]].distinct == 1:  # only one value is ever present: that value is what "positive" means here
            hints["positive_label"] = facts[partial[0]].values[0]
    elif complete:
        best = min(complete, key=lambda c: (priority[facts[c].kind], columns.index(c)))
        hints["target"] = hints["label_column"] = best
    elif hints.get("event"):
        hints["target"] = hints["label_column"] = hints["event"]

    dates = date_columns(frame)
    if dates:
        hints["date_column"] = dates[0]
    numeric_spoken = [c for c in spoken if facts[c].kind == "number" and c not in dates]
    value = numeric_spoken[0] if numeric_spoken else next((c for c in columns if facts[c].kind == "number" and c not in dates and c not in hints.values()), None)
    if value:
        hints["value_column"] = value
    texts = [c for c in columns if facts[c].kind == "text"]
    if texts:
        hints["text_column"] = texts[0]
    groups = [c for c in columns if GROUP_QUERY.search(c)] or [c for c in columns if GROUP_BASKET.search(c)] or [c for c in columns if ID_NAME.search(c) and 1 < facts[c].distinct < 0.5 * len(frame)]
    if groups:
        hints["group_column"] = hints["transaction_column"] = groups[0]
    graded = [c for c in columns if RELEVANCE_NAME.search(c) and facts[c].kind in ("levels", "binary", "number")]
    if groups and graded:
        hints["relevance"] = hints["target"] if hints.get("target") in graded else graded[0]
    multi = max(zero_one_groups(frame).values(), key=len, default=[])
    if multi:
        hints["labels"] = list(multi)  # type: ignore[assignment]
    items = [c for c in columns if ITEM_NAME.search(c)]
    if items:
        hints["item_column"] = items[0]
    return hints
