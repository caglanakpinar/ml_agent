"""A sub agent's default run when its decider cannot produce a plan: diagnostics → baseline → candidates → fit the winner → evaluate.

`executor.diagnostic_playbook` stops at read-only checks, so a run that fell back to it never fitted anything. A recipe goes on: it reads what the checks
found (a leaking column, an id, a position the ranker will not have at serving time) and builds the next steps from it, the way an analyst would.

    run_recipe(category, problem_type, data_path, hints, folder)
        ├ the topic's read-only checks                       role `check`
        ├ a baseline: the simplest model of the topic         role `baseline`
        ├ candidates compared with the topic's own `compare_*` tool, on cross-validation     role `candidate`
        ├ the cross-validation winner fitted on a held-out split       role `model`
        └ where the problem has ground truth kept apart, the file of predictions that can be scored     role `predictions`

Nothing here calls a model. Every step is a call of a tool the topic's worker has, so the saved run reads like any other. The steps carry a `role` so that
`benchmarks/score.py` can find the baseline and the model without knowing which tool produced them; the recipe's own choices (what was dropped and why, which
file holds the validation rows) are kept in `info`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from console.executor import StepResult, diagnostic_playbook, execute_plan, save_results

ID_COLUMN = re.compile(r"(?:^|_)id$|^id$|^index$", re.I)
POSITION_COLUMN = re.compile(r"position|slot|(?:^|_)rank_?shown|shown_rank|impression_rank", re.I)
LEAK_SCORE = 0.95  # a single column that separates the classes this well, on its own, is almost certainly known only after the outcome


@dataclass
class Recipe:
    """One run: the steps so far, the data every later step reads, and what the recipe decided along the way."""

    category: str
    problem_type: str
    path: str
    hints: dict[str, Any]
    folder: Path
    echo: Callable[..., Any] = print
    upstream: list[StepResult] = field(default_factory=list)  # what the sub agents that ran before this one returned
    results: list[StepResult] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    data: str = ""  # the file later steps read; a derived copy of `path` when the recipe had to prepare one
    _frame: Any = None

    def __post_init__(self) -> None:
        self.data = self.data or self.path

    # ---- running steps ------------------------------------------------------------------------------------------------
    def step(self, tool: str, role: str, why: str = "", **arguments: Any) -> dict[str, Any]:
        arguments = {k: v for k, v in arguments.items() if v is not None and v != []}
        return {"tool": tool, "role": role, "arguments": arguments, "why": why}

    def run(self, *steps: dict[str, Any]) -> list[StepResult]:
        """Run steps with this recipe's data file and continue the numbering; returns the steps of this call."""
        new = execute_plan(list(steps), self.data, self.hints, self.folder, echo=self.echo, start_index=len(self.results) + 1, earlier=[*self.upstream, *self.results], retry=False)
        self.results.extend(new)
        return new

    def diagnose(self) -> list[StepResult]:
        """The topic's own read-only checks, with the baseline among them (if it has one) marked as the baseline."""
        steps = diagnostic_playbook(self.category, self.problem_type, self.hints)
        for s in steps:
            s["role"] = "baseline" if "baseline" in s["tool"] else "check"
        return self.run(*steps)

    # ---- reading what ran ---------------------------------------------------------------------------------------------
    def found(self, tool: str | None = None, role: str | None = None) -> list[StepResult]:
        return [r for r in self.results if r.ok and (tool is None or r.tool == tool) and (role is None or r.role == role)]

    def last(self, tool: str | None = None, role: str | None = None) -> dict[str, Any]:
        hits = self.found(tool, role)
        return hits[-1].result or {} if hits else {}

    # ---- the data -----------------------------------------------------------------------------------------------------
    @property
    def frame(self):
        """The full data table (the original file, never a derived copy)."""
        if self._frame is None:
            import pandas as pd

            suffix = Path(self.path).suffix.lower()
            self._frame = pd.read_csv(self.path, sep="\t" if suffix == ".tsv" else ",") if suffix in (".csv", ".tsv", ".txt") else pd.read_parquet(self.path)
        return self._frame

    def derive(self, name: str, frame) -> str:
        """Write a prepared copy of the data into the run's folder and return its path."""
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / name
        frame.to_csv(path, index=False)
        return str(path)

    def ids(self, keep: tuple[str, ...] = ()) -> list[str]:
        """Columns that only name a row (`customer_id`, `txn_id`): a model that sees them is memorising, not learning."""
        frame = self.frame
        return [c for c in frame.columns if c not in keep and ID_COLUMN.search(str(c)) and frame[c].nunique() > 0.05 * len(frame)]

    def note(self, **facts: Any) -> None:
        self.info.update(facts)


# --------------------------------------------------------------------------------------------------------------------------------------
# shared reading of tool results
# --------------------------------------------------------------------------------------------------------------------------------------
def leaking_columns(recipe: Recipe) -> list[str]:
    """The columns a single-feature check found nearly perfect on their own (`find_leakage_candidates_tool`)."""
    suspects = recipe.last("find_leakage_candidates_tool").get("suspects") or []
    return [str(s["column"]) for s in suspects if isinstance(s, dict) and float(s.get("score") or 0) >= LEAK_SCORE]


def winner(board: dict[str, Any], key: str, mapping: dict[str, str], default: str) -> str:
    """The tool that fits the model a `compare_*` tool ranked first; `default` when the board is empty or names something unmapped."""
    best = board.get(key)
    return mapping.get(str(best), default)


def _drop(recipe: Recipe, extra: list[str], keep: tuple[str, ...] = ()) -> list[str]:
    seen: list[str] = []
    for column in [*recipe.ids(keep), *extra]:
        if column not in seen:
            seen.append(column)
    return seen


# --------------------------------------------------------------------------------------------------------------------------------------
# supervised
# --------------------------------------------------------------------------------------------------------------------------------------
CLASSIFIERS = {
    "logistic_regression": "fit_logistic_regression_tool", "random_forest": "fit_random_forest_tool", "gradient_boosting": "fit_gradient_boosting_tool",
    "hist_gradient_boosting": "fit_hist_gradient_boosting_tool", "knn": "fit_knn_tool", "gaussian_nb": "fit_naive_bayes_tool", "extra_trees": "fit_extra_trees_tool",
    "decision_tree": "fit_decision_tree_tool", "svm": "fit_svm_tool", "mlp": "fit_mlp_tool", "adaboost": "fit_adaboost_tool", "lda": "classification_fit_lda_tool",
    "sgd": "fit_sgd_classifier_tool",
}


def classification(r: Recipe) -> None:
    r.diagnose()
    leaks = leaking_columns(r)
    drop = _drop(r, leaks)
    if leaks:
        r.note(dropped_as_leaks=leaks)
    target = r.hints.get("target")
    minority = min(r.frame[target].value_counts(normalize=True)) if target in r.frame.columns else 0.5
    binary = target in r.frame.columns and r.frame[target].nunique() == 2
    scoring = "average_precision" if binary and minority < 0.2 else ("roc_auc" if binary else "f1_macro")
    if not r.found("baseline_classifier_tool"):
        r.run(r.step("baseline_classifier_tool", "baseline", "the most frequent class: what a model has to beat"))
    board = r.run(r.step("compare_classifiers_tool", "candidate", "cross-validate the usual families on the same folds", drop=drop, scoring=scoring))[0]
    best = (board.result or {}).get("best") or "hist_gradient_boosting"
    tool = CLASSIFIERS.get(best, "fit_hist_gradient_boosting_tool")
    r.note(scoring=scoring, chosen=best)
    r.run(r.step(tool, "model", f"{best} won the cross-validation ({scoring})", drop=drop))
    r.run(r.step("evaluate_classifier_cv_tool", "check", "cross-validated score of the chosen model", drop=drop, model=best if best in CLASSIFIERS else None))


REGRESSORS = {
    "linear": "fit_linear_regression_tool", "huber": "fit_huber_regression_tool", "bayesian_ridge": "fit_bayesian_ridge_tool", "sgd": "fit_sgd_regressor_tool", "ridge": "fit_ridge_tool", "lasso": "fit_lasso_tool", "elastic_net": "fit_elastic_net_tool", "random_forest": "fit_random_forest_regressor_tool",
    "gradient_boosting": "fit_gradient_boosting_regressor_tool", "hist_gradient_boosting": "fit_hist_gradient_boosting_regressor_tool", "extra_trees": "fit_extra_trees_regressor_tool",
    "knn": "fit_knn_regressor_tool", "decision_tree": "fit_decision_tree_regressor_tool", "svr": "fit_svr_tool", "mlp": "fit_mlp_regressor_tool",
}


def earlier_clusters(r: Recipe, key: str):
    """The cluster each entity was given by a sub agent that ran before this one (its fitted model's table), as a frame of `key` and `cluster`; None when there is none."""
    import pandas as pd

    for step in reversed(r.upstream):
        if step.ok and step.role == "model":
            file = step.result.get("written_to") if step.result else None
            file = file or step.arguments.get("output_path")
            if file and Path(str(file)).is_file():
                table = pd.read_csv(file, nrows=5)
                if "cluster" in table.columns and key in table.columns:
                    table = pd.read_csv(file, usecols=[key, "cluster"])
                    return table.groupby(key, as_index=False)["cluster"].agg(lambda s: s.mode().iloc[0])
    return None


def regression(r: Recipe) -> None:
    from console import prepare

    r.diagnose()
    target = r.hints.get("target")
    group = r.hints.get("group_column")
    frame = r.frame
    keep_group = (group,) if group else ()
    if group and group in frame.columns and frame[group].nunique() < 0.5 * len(frame) and target in frame.columns:
        # Rows that follow an entity through time: how far each sensor has moved from where the entity started, and the group the entity was put in earlier.
        numeric = prepare.numeric_columns(frame, exclude=[*id_named(frame), target])
        counters = prepare.monotone_within(frame, group, numeric)
        sensors = [c for c in numeric if c not in counters and c not in prepare.constant_columns(frame, numeric)]
        order = next((c for c in counters if c != target and frame.groupby(group)[c].diff().dropna().ge(0).mean() > 0.99), None)
        frame = prepare.panel_features(frame, group, order, sensors)
        clusters = earlier_clusters(r, group)
        if clusters is not None:
            frame = frame.merge(clusters.rename(columns={"cluster": "regime"}), on=group, how="left")
        r.note(panel_features_from=sensors, ordered_by=order, regime_from_an_earlier_sub_agent=clusters is not None)
        r.data = r.derive("panel.csv", frame)
        r._frame = frame
    drop = _drop(r, [group] if group else [], keep=keep_group)
    # A table that holds many rows per entity (an engine, a customer) must be split by entity: neighbouring rows of one entity are near-copies, and a random
    # row split would test the model on rows it has effectively seen.
    train_path, valid_path = r.data, None
    if group and group in frame.columns and frame[group].nunique() < 0.5 * len(frame) and target in frame.columns:
        import numpy as np

        entities = frame[group].drop_duplicates().to_numpy()
        rng = np.random.default_rng(42)
        held = set(rng.choice(entities, size=max(1, int(0.25 * len(entities))), replace=False).tolist())
        mask = frame[group].isin(held)
        train_path, valid_path = r.derive("train_entities.csv", frame[~mask]), r.derive("valid_entities.csv", frame[mask])
        r.note(split_by=group, train_data=train_path, valid_data=valid_path, target=target, held_out_entities=len(held))
        r.data = train_path
    if not r.found("baseline_regressor_tool"):
        r.run(r.step("baseline_regressor_tool", "baseline", "predicting the mean: what a model has to beat"))
    scoring = "mae"
    board = r.run(r.step("compare_regressors_tool", "candidate", "cross-validate the usual families on the same folds", drop=drop, scoring=scoring))[0]
    best = (board.result or {}).get("best") or "hist_gradient_boosting"
    tool = REGRESSORS.get(best, "fit_hist_gradient_boosting_regressor_tool")
    r.note(chosen=best)
    fitted = r.run(r.step(tool, "model", f"{best} won the cross-validation", drop=drop))[0]
    if valid_path and fitted.ok:
        r.run(r.step("predict_with_saved_regressor_tool", "predictions", "predict the entities that were held out of the fit", data=valid_path))


def ranking(r: Recipe) -> None:
    r.diagnose()
    positions = [c for c in r.frame.columns if POSITION_COLUMN.search(str(c))]
    drop = _drop(r, positions, keep=tuple(c for c in (r.hints.get("group_column"),) if c))
    if positions:
        r.note(dropped_as_position=positions)
    approaches = {"listnet": "fit_listnet_ranker_tool", "pairwise_logistic": "fit_pairwise_logistic_ranker_tool", "pointwise_linear": "fit_pointwise_linear_ranker_tool",
                  "pointwise_classification": "fit_pointwise_classifier_ranker_tool", "pointwise_regression": "fit_pointwise_regression_ranker_tool",
                  "pointwise_random_forest": "fit_pointwise_random_forest_ranker_tool", "ranksvm": "fit_ranksvm_tool", "lambdarank": "fit_lightgbm_lambdarank_tool"}
    if not r.found("baseline_random_ranker_tool"):
        r.run(r.step("baseline_random_ranker_tool", "baseline", "a random order: what a ranker has to beat"))
    board = r.run(r.step("compare_rankers_tool", "candidate", "cross-validate the usual approaches on the same queries", drop=drop))[0]
    best = (board.result or {}).get("best") or "pointwise_regression"
    r.note(chosen=best)
    r.run(r.step(approaches.get(best, "fit_pointwise_regression_ranker_tool"), "model", f"{best} won the cross-validation (ndcg@10)", drop=drop))


MULTILABEL = {
    "binary_relevance": "fit_binary_relevance_tool", "classifier_chain": "fit_classifier_chain_tool", "label_powerset": "fit_label_powerset_tool", "rakel": "fit_rakel_tool",
    "multioutput_forest": "fit_multioutput_random_forest_tool", "mlknn": "fit_mlknn_tool", "knn": "fit_knn_multilabel_tool",
}


def multilabel(r: Recipe) -> None:
    r.diagnose()
    labels = r.hints.get("labels")
    # A column whose values name the labels (`refund|login`) is the labels written differently, and a model that reads it has the answer.
    names = [re.sub(r"^(?:label|is|has)_", "", str(label)).lower() for label in (labels or [])]
    leaky = [c for c in r.frame.columns if c not in (labels or []) and r.frame[c].dtype == object and names
             and r.frame[c].astype(str).str.lower().str.contains("|".join(re.escape(n) for n in names), regex=True).mean() > 0.2
             and r.frame[c].astype(str).str.len().mean() < 60]
    drop = _drop(r, leaky)
    if leaky:
        r.note(dropped_as_label_copies=leaky)
    if not r.found("baseline_multilabel_tool"):
        r.run(r.step("baseline_multilabel_tool", "baseline", "the most common label for every row: what a model has to beat"))
    text = r.hints.get("text_column")
    board = r.run(r.step("compare_multilabel_models_tool", "candidate", "cross-validate the usual approaches on the same folds", drop=drop, text_column=text))[0]
    best = (board.result or {}).get("best") or "binary_relevance"
    r.note(chosen=best)
    r.run(r.step(MULTILABEL.get(best, "fit_binary_relevance_tool"), "model", f"{best} won the cross-validation (micro-F1)", drop=drop, text_column=text))


ORDINAL = {
    "regression_learned_thresholds": "fit_regression_with_learned_thresholds_tool", "regression_rounding": "fit_regression_with_rounding_tool", "ordered_logit": "fit_ordered_logit_tool",
    "frank_hall_logistic": "fit_frank_hall_ordinal_tool", "frank_hall_boosting": "fit_frank_hall_ordinal_tool", "multiclass_argmax": "fit_multiclass_ignoring_order_tool",
    "multiclass_median_decode": "fit_multiclass_ignoring_order_tool",
}


def ordinal(r: Recipe) -> None:
    r.diagnose()
    drop = _drop(r, [])
    if not r.found("baseline_ordinal_tool"):
        r.run(r.step("baseline_ordinal_tool", "baseline", "the median level for every row: what a model has to beat"))
    board = r.run(r.step("compare_ordinal_models_tool", "candidate", "cross-validate the usual approaches on the same folds", drop=drop))[0]
    best = (board.result or {}).get("best") or "regression_learned_thresholds"
    extra: dict[str, Any] = {"base": "boosting"} if best == "frank_hall_boosting" else {}
    r.note(chosen=best)
    r.run(r.step(ORDINAL.get(best, "fit_regression_with_learned_thresholds_tool"), "model", f"{best} won the cross-validation (quadratic-weighted kappa)", drop=drop, **extra))


SURVIVAL = {"weibull": "fit_weibull_aft_tool", "cox_ridge": "fit_cox_regularised_tool", "cox": "fit_cox_proportional_hazards_tool", "discrete_time": "fit_discrete_time_logistic_tool",
            "lognormal": "fit_lognormal_aft_tool"}


def survival(r: Recipe) -> None:
    r.diagnose()
    drop = _drop(r, [])
    board = r.run(r.step("compare_survival_models_tool", "candidate", "cross-validate the usual survival models on the same folds", drop=drop))[0]
    best = (board.result or {}).get("best") or "cox_ridge"
    r.note(chosen=best)
    r.run(r.step(SURVIVAL.get(best, "fit_cox_regularised_tool"), "model", f"{best} won the cross-validation (concordance)", drop=drop))


# --------------------------------------------------------------------------------------------------------------------------------------
# unsupervised: no labels in the data, so the baseline is the plain run of the same family and the model is the run after the preparation the checks called for
# --------------------------------------------------------------------------------------------------------------------------------------
def id_named(frame) -> list[str]:
    return [c for c in frame.columns if ID_COLUMN.search(str(c))]


def anomaly(r: Recipe) -> None:
    from console import prepare

    r.diagnose()
    frame = r.frame
    entity = r.hints.get("group_column")
    entity = entity if entity in frame.columns and frame[entity].nunique() < 0.5 * len(frame) else None
    when = r.hints.get("date_column")
    named = id_named(frame)
    row_id = next((c for c in named if frame[c].nunique() == len(frame)), None)
    plain_drop = [c for c in [*named, when] if c]
    r.run(r.step("detect_isolation_forest_tool", "baseline", "an isolation forest on the columns as they are: every row compared with every other", drop=plain_drop, id_column=row_id))
    if entity:
        numeric = prepare.numeric_columns(frame, exclude=named)
        categories = prepare.categorical_columns(frame, exclude=[when, *named])
        counterparts = [c for c in named if c not in (row_id, entity)]
        features = prepare.contextual_features(frame, entity, numeric, categories, when, counterparts)
        if row_id:
            features.insert(0, row_id, frame[row_id])
        path = r.derive("contextual_features.csv", features)
        r.note(contextual_features=[c for c in features.columns if c != row_id], measured_against=entity)
        r.run(r.step("detect_isolation_forest_tool", "model", f"an isolation forest on each row measured against the history of its own {entity}", data=path,
                     drop=[row_id] if row_id else None, id_column=row_id))
    else:
        r.run(r.step("ensemble_detectors_tool", "model", "several detectors voting", drop=plain_drop, id_column=row_id))
    r.run(r.step("compare_detectors_tool", "check", "how far the detectors agree", drop=plain_drop))


def chosen_k(suggestion: dict[str, Any] | None, criterion: str) -> int:
    """The k the algorithm's own criterion asks for (k-means: its elbow; a Gaussian mixture: its BIC), else the criteria's consensus, else 3.

    The consensus of six criteria is what the tool reports, but it is often a compromise nobody's model would pick, and the tool itself says when it is low confidence.
    """
    votes = (suggestion or {}).get("votes") or {}
    k = votes.get(criterion) or (suggestion or {}).get("consensus_k") or 3
    return max(2, min(int(k), 10))


def clustering(r: Recipe) -> None:
    from console import prepare

    r.diagnose()
    frame = r.frame
    named = id_named(frame)
    entity = r.hints.get("group_column")
    entity = entity if entity in frame.columns and frame[entity].nunique() < 0.5 * len(frame) else None
    plain_drop = [c for c in named if c != entity] if entity else named
    raw_drop = [*plain_drop, entity] if entity else plain_drop
    ask = r.run(r.step("suggest_number_of_clusters_tool", "check", "how many clusters each criterion asks for, for the columns as they are", drop=raw_drop))[0]
    k_raw = chosen_k(ask.result, "elbow")
    r.run(r.step("cluster_kmeans_tool", "baseline", f"k-means with k={k_raw} (its own elbow criterion) on the columns as they are", n_clusters=k_raw, drop=raw_drop))

    numeric = prepare.numeric_columns(frame, exclude=named)
    if entity:
        # Rows of one entity are one unit, described by what it looks like at the start: counters (a cycle number, a time remaining) are dropped, they only say how far along it is.
        counters = prepare.monotone_within(frame, entity, numeric)
        constant = prepare.constant_columns(frame, numeric)
        columns = [c for c in numeric if c not in counters and c not in constant]
        order = next((c for c in counters if frame.groupby(entity)[c].diff().dropna().ge(0).mean() > 0.99), None)
        profile = prepare.entity_profile(frame, entity, columns, order)
        path = r.derive("entity_profile.csv", profile)
        r.note(clustered=entity, columns=columns, dropped_as_counters=counters, dropped_as_constant=constant, ordered_by=order)
        prepared, key = profile, entity
    else:
        skew = prepare.skewed_columns(frame, numeric)
        kept = prepare.shared_structure(frame, numeric)
        cats = prepare.associated_categories(frame, prepare.categorical_columns(frame, exclude=named), kept)
        prepared = prepare.log_transform(frame[[c for c in frame.columns if c in kept or c in cats or c in named]], [c for c in skew if c in kept])
        path = r.derive("prepared.csv", prepared)
        r.note(kept=kept, dropped_as_unrelated=[c for c in numeric if c not in kept], categories_kept=cats, logged=[c for c in skew if c in kept])
        key = named[0] if named else None
    drop = [key] if key else None
    ask = r.run(r.step("suggest_number_of_clusters_tool", "check", "how many clusters each criterion asks for, after preparing the columns", data=path, drop=drop))[0]
    k = chosen_k(ask.result, "gmm_bic")
    r.note(k=k, k_before_preparing=k_raw)
    r.run(r.step("cluster_gaussian_mixture_tool", "model", f"a Gaussian mixture with k={k} (its own BIC criterion) on the prepared columns", data=path, n_clusters=k, drop=drop))
    r.run(r.step("compare_clusterings_tool", "check", "does another algorithm find the same groups", data=path, n_clusters=k, drop=drop))


def association_rules(r: Recipe) -> None:
    from console import prepare

    r.diagnose()
    basket, item = r.hints.get("transaction_column"), r.hints.get("item_column")
    frame = r.frame
    fence = None
    path = r.data
    if basket in frame.columns and item in frame.columns:
        sizes = frame.groupby(basket)[item].nunique()
        fence = prepare.basket_size_fence(sizes)
        small = sizes[sizes <= fence].index
        path = r.derive("baskets_within_fence.csv", frame[frame[basket].isin(small)])
        r.note(basket_size_fence=fence, baskets_dropped=int((sizes > fence).sum()))
    # A rule has to rest on enough baskets to measure its lift: 30 of them, however small a share of all baskets that is. The sweep's own suggestion is a floor
    # only where it is lower (a long catalogue can make a count this low explode).
    baskets = int(frame[basket].nunique()) if basket in frame.columns else len(frame)
    support = min(max(30 / max(baskets, 1), 0.001), number_from(r.last("suggest_min_support_tool"), "suggested_min_support") or 0.01)
    r.note(min_support=round(support, 5), min_baskets=30)
    r.run(r.step("generate_rules_tool", "baseline", "every basket, the tool's default thresholds, the most confident rules first", sort_by="confidence"))
    r.run(r.step("generate_rules_tool", "model", "baskets without the far-out sizes, the support the sweep suggested, the strongest associations beyond chance first", data=path, min_support=support,
                 min_confidence=0.2, min_lift=1.2, sort_by="lift", max_len=3))


def number_from(result: dict[str, Any], key: str) -> float | None:
    try:
        return float(result[key])
    except (KeyError, TypeError, ValueError):
        return None


def eigengap_k(variances: list[float]) -> int | None:
    """The number of components before the steepest fall in the variance each explains: where a few strong themes end and the long tail of noise begins."""
    v = [x for x in variances if x and x > 0]
    if len(v) < 3:
        return None
    ratios = [v[i] / v[i + 1] for i in range(len(v) - 1)]
    return max(2, ratios.index(max(ratios)) + 1)


def vocabulary_settings(grid: list[dict[str, Any]]) -> tuple[float, int]:
    """`(max_df, min_df)`: the strictest document-frequency ceiling that still leaves almost no document (5%) with fewer than three terms, and a floor of five documents."""
    fine = [g for g in grid if (g.get("documents_under_3_terms") or 1.0) <= 0.05]
    return (min(g["max_df"] for g in fine), 5) if fine else (0.8, 2)


def topic_modeling(r: Recipe) -> None:
    r.diagnose()
    text = r.hints.get("text_column")
    named = id_named(r.frame)
    doc = named[0] if named else None
    structure = r.last("estimate_topic_structure_tool")
    k = eigengap_k(structure.get("singular_value_variance") or [])
    if not k:
        sweep = r.run(r.step("choose_n_topics_tool", "check", "coherence and diversity as the number of topics grows", text_column=text))[0]
        k = int((sweep.result or {}).get("suggested_k") or 8)
    max_df, min_df = vocabulary_settings(r.last("suggest_vocabulary_settings_tool").get("grid") or [])
    r.note(n_topics=k, n_topics_from="the steepest fall in the variance of the components", max_df=max_df, min_df=min_df)
    r.run(r.step("fit_nmf_topics_tool", "baseline", "NMF with the tool's defaults on the text as it is", text_column=text, id_column=doc))
    r.run(r.step("fit_nmf_topics_tool", "model", f"NMF with {k} topics, words in over {max_df:.0%} of documents set aside as boilerplate, words in under {min_df} documents ignored",
                 text_column=text, id_column=doc, n_topics=k, max_df=max_df, min_df=min_df, stop_words="english"))
    r.run(r.step("score_topic_coherence_tool", "check", "how well the words of each topic go together", text_column=text))


FORECASTERS = ("seasonal_naive", "theta", "holt_winters", "ses", "harmonic")


def forecasting(r: Recipe) -> None:
    from console import prepare

    r.diagnose()
    frame = r.frame
    date, value = r.hints.get("date_column"), r.hints.get("value_column")
    series = prepare.categorical_columns(frame, exclude=[date], max_levels=500)
    if not (date and value and series):
        return
    import pandas as pd

    frame = frame.assign(series_key=frame[series].astype(str).agg("|".join, axis=1))
    frame[date] = pd.to_datetime(frame[date])
    horizon = int(min(28, max(7, frame.groupby("series_key").size().min() // 10)))
    cut = frame[date].max() - pd.Timedelta(days=horizon)
    full, before = r.derive("series.csv", frame), r.derive("series_before_the_window.csv", frame[frame[date] <= cut])
    r.note(series_column="series_key", series_from=series, horizon=horizon, series=int(frame.series_key.nunique()))
    # The method is chosen on the window before the one it is scored on, so that choosing it does not use the answers.
    boards = [(m, r.run(r.step("forecast_many_series_tool", "candidate", f"{m} for every series, scored on the {horizon} days before the final window", data=before, date_column=date, value_column=value,
                               series_column="series_key", method=m, holdout=horizon))[0]) for m in FORECASTERS]
    scored = [(float((s.result or {}).get("median_smape", 9e9)), m) for m, s in boards if s.ok]
    best = min(scored)[1] if scored else "seasonal_naive"
    r.note(chosen=best)
    r.run(r.step("forecast_many_series_tool", "baseline", f"seasonal naive for every series, scored on the final {horizon} days", data=full, date_column=date, value_column=value,
                 series_column="series_key", method="seasonal_naive", holdout=horizon))
    r.run(r.step("forecast_many_series_tool", "model", f"{best} for every series (it had the lowest error on the window before), scored on the final {horizon} days", data=full, date_column=date,
                 value_column=value, series_column="series_key", method=best, holdout=horizon))


# --------------------------------------------------------------------------------------------------------------------------------------
# semi-supervised
# --------------------------------------------------------------------------------------------------------------------------------------
PU_METHODS = {
    "elkan_noto": "fit_pu_elkan_noto_tool", "bagging": "fit_pu_bagging_tool", "weighted_logistic": "fit_pu_weighted_logistic_tool", "two_step_spy": "fit_pu_two_step_spy_tool",
    "em": "fit_pu_expectation_maximisation_tool", "two_step_rocchio": "fit_pu_two_step_rocchio_tool", "nnpu": "fit_nnpu_tool",
}


def positive_unlabelled(r: Recipe) -> None:
    r.diagnose()
    drop = _drop(r, [])
    positive = r.hints.get("positive_label")
    if not r.found("baseline_positive_vs_unlabelled_tool"):
        r.run(r.step("baseline_positive_vs_unlabelled_tool", "baseline", "a classifier of labelled against unlabelled, taken at face value", drop=drop, positive_label=positive))
    board = r.run(r.step("compare_pu_methods_tool", "candidate", "the PU methods on the same held-out positives", drop=drop, positive_label=positive))[0]
    best = (board.result or {}).get("best") or "elkan_noto"
    r.note(chosen=best)
    r.run(r.step(PU_METHODS.get(best, "fit_pu_elkan_noto_tool"), "model", f"{best} ranked first on held-out positives", drop=drop, positive_label=positive))
    r.run(r.step("rank_unlabelled_by_positive_probability_tool", "predictions", "every unlabelled row ordered by how likely it is a hidden positive", drop=drop, positive_label=positive))


def label_propagation(r: Recipe) -> None:
    from console import prepare

    r.diagnose()
    label = r.hints.get("label_column")
    frame = r.frame
    named = id_named(frame)
    numeric = prepare.numeric_columns(frame, exclude=[*named, label])
    features = prepare.labelled_signal(frame, label, numeric) if label in frame.columns else numeric
    r.note(features=features, dropped_as_unrelated_to_the_labels=[c for c in numeric if c not in features])
    r.run(r.step("compare_propagation_methods_tool", "candidate", "graph methods against a classifier on the labels alone", features=features))
    fitted = r.run(r.step("fit_label_spreading_knn_tool", "model", "label spreading on a nearest-neighbour graph of the columns that separate the labelled rows", features=features))[0]
    if fitted.ok:
        r.run(r.step("label_propagation_predict_with_semi_model_tool", "predictions", "the label of every row, labelled or not", id_column=named[0] if named else None))


# --------------------------------------------------------------------------------------------------------------------------------------
# which recipes exist
# --------------------------------------------------------------------------------------------------------------------------------------
RECIPES: dict[tuple[str, str], Callable[[Recipe], None]] = {
    ("supervised", "classification"): classification,
    ("supervised", "regression"): regression,
    ("supervised", "ranking"): ranking,
    ("supervised", "multi_label_classification"): multilabel,
    ("supervised", "ordinal_regression"): ordinal,
    ("supervised", "survival_analysis"): survival,
    ("unsupervised", "anomaly_detection"): anomaly,
    ("unsupervised", "clustering"): clustering,
    ("semisupervised", "positive_unlabeled_learning"): positive_unlabelled,
    ("semisupervised", "label_propagation"): label_propagation,
    ("unsupervised", "association_rule_mining"): association_rules,
    ("unsupervised", "topic_modeling"): topic_modeling,
    ("supervised", "time_series_forecasting"): forecasting,
}


def has_recipe(category: str, problem_type: str) -> bool:
    return (category, problem_type) in RECIPES


def run_recipe(category: str, problem_type: str, data_path: str, hints: dict[str, Any], folder: Path, echo: Callable[..., Any] = print,
               upstream: list[StepResult] | None = None) -> tuple[list[StepResult], dict[str, Any]]:
    """Run the topic's recipe in `folder`; returns every step and what the recipe decided. Saves both as `results.json` there."""
    recipe = Recipe(category, problem_type, data_path, hints, Path(folder), echo, list(upstream or []))
    RECIPES[(category, problem_type)](recipe)
    save_results(recipe.results, recipe.folder, recipe.info)
    return recipe.results, recipe.info
