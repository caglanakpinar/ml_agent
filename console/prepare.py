"""Preparing a table for a recipe: the few transformations an analyst makes by reading the diagnostics before fitting anything.

Every function reads a pandas frame and returns columns or a new frame; none touches a file or a model. They are generic rules, not rules for one dataset:

    skewed_columns         a non-negative column with a long right tail is better clustered on its log
    shared_structure       a numeric column that moves with none of the others carries no structure that clustering could find
    associated_categories  a category that no kept numeric column differs across is noise
    contextual_features    for transactions of an entity (a card, a user): each row measured against that entity's own history
    entity_profile         one row per entity, from what the entity's rows look like at the start
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def numeric_columns(frame: pd.DataFrame, exclude: tuple[str, ...] | list[str] = ()) -> list[str]:
    return [c for c in frame.columns if c not in exclude and pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])]


def categorical_columns(frame: pd.DataFrame, exclude: tuple[str, ...] | list[str] = (), max_levels: int = 30) -> list[str]:
    return [c for c in frame.columns if c not in exclude and not pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_datetime64_any_dtype(frame[c])
            and 1 < frame[c].nunique() <= max_levels]


def skewed_columns(frame: pd.DataFrame, columns: list[str], threshold: float = 1.5) -> list[str]:
    """Columns that are never negative and lean far to the right: taking their log pulls the long tail in."""
    out = []
    for c in columns:
        s = frame[c].dropna()
        if len(s) > 20 and s.min() >= 0 and s.nunique() > 10 and float(s.skew()) > threshold:
            out.append(c)
    return out


def shared_structure(frame: pd.DataFrame, columns: list[str], threshold: float = 0.08) -> list[str]:
    """Numeric columns whose rank correlation with at least one other column reaches `threshold`.

    A column that is independent of every other column cannot help to separate groups that show up as patterns across columns; with dozens of them they only
    add distance. Spearman, so a long tail or a gross outlier does not hide a real relationship.
    """
    from scipy import stats

    if len(columns) < 2:
        return list(columns)
    sample = frame[columns].sample(min(len(frame), 5000), random_state=0) if len(frame) > 5000 else frame[columns]
    # With dozens of independent columns some pair will correlate by chance: the bar is the correlation that no pair would reach by chance (Bonferroni, 0.1%),
    # and never below `threshold`.
    pairs = len(columns) * (len(columns) - 1) / 2
    threshold = max(threshold, float(stats.norm.ppf(1 - 0.001 / (2 * pairs))) / np.sqrt(len(sample)))
    corr = sample.corr(method="spearman").abs().to_numpy(copy=True)
    np.fill_diagonal(corr, 0.0)
    keep = np.nan_to_num(corr, nan=0.0).max(axis=0) >= threshold
    return [c for c, k in zip(columns, keep) if k]


def associated_categories(frame: pd.DataFrame, categories: list[str], numeric: list[str], alpha: float = 1e-4) -> list[str]:
    """Categories across whose levels at least one of the numeric columns differs (one-way ANOVA, p below `alpha`)."""
    from scipy import stats

    keep = []
    sample = frame.sample(min(len(frame), 8000), random_state=0) if len(frame) > 8000 else frame
    for c in categories:
        for n in numeric:
            groups = [g[n].dropna().to_numpy() for _, g in sample.groupby(c) if g[n].notna().sum() > 5]
            if len(groups) > 1 and stats.f_oneway(*groups).pvalue < alpha:
                keep.append(c)
                break
    return keep


def log_transform(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for c in columns:
        out[c] = np.log1p(out[c].clip(lower=0))
    return out


def contextual_features(frame: pd.DataFrame, entity: str, numeric: list[str], categorical: list[str], time_column: str | None = None, counterparts: list[str] | None = None,
                        burst_minutes: int = 10) -> pd.DataFrame:
    """Each row measured against the history of its own entity, which a global detector cannot see.

    * numeric columns: a robust z-score within the entity (its median and spread; a skewed column on its log). A large amount is normal for a customer who always
      spends a lot, and only the second is told apart.
    * categories: how rare this row's value is among the entity's rows (a country the card has never been used in).
    * counterparts: how many rows hit the same counterparty (a merchant) within `burst_minutes` of each other, which finds a ring acting in step.
    """
    out = pd.DataFrame(index=frame.index)
    g = frame.groupby(entity)
    for c in numeric:
        x = frame[c].astype(float)
        if x.min() >= 0 and float(x.skew()) > 1.5:
            x = np.log1p(x)
        grouped = x.groupby(frame[entity])
        median = grouped.transform("median")
        spread = (x - median).abs().groupby(frame[entity]).transform("median") * 1.4826
        spread = spread.clip(lower=max(float(x.std()) * 0.05, 1e-6))
        out[f"{c}_vs_entity"] = ((x - median) / spread).clip(-30, 30)
    for c in categorical:
        share = frame.groupby([entity, c])[c].transform("size") / g[c].transform("size")
        out[f"{c}_rarity_for_entity"] = 1.0 - share
    if time_column and counterparts:
        when = pd.to_datetime(frame[time_column], errors="coerce")
        if when.notna().all():
            window = when.dt.floor(f"{burst_minutes}min")
            for c in counterparts:
                out[f"{c}_events_in_{burst_minutes}min"] = frame.groupby([c, window])[c].transform("size")
    return out


def monotone_within(frame: pd.DataFrame, entity: str, columns: list[str]) -> list[str]:
    """Columns that only ever rise or only ever fall inside each entity: counters (a cycle, a days-remaining), not facts about the entity."""
    out = []
    for c in columns:
        diffs = frame.groupby(entity)[c].diff().dropna()
        if len(diffs) and ((diffs >= 0).mean() > 0.99 or (diffs <= 0).mean() > 0.99) and diffs.abs().sum() > 0:
            out.append(c)
    return out


def entity_profile(frame: pd.DataFrame, entity: str, columns: list[str], order_column: str | None = None, first: int = 10) -> pd.DataFrame:
    """One row per entity: the mean of each column over the entity's first `first` rows (in `order_column` order when there is one)."""
    ordered = frame.sort_values([entity, order_column]) if order_column else frame
    head = ordered.groupby(entity, sort=True).head(first)
    return head.groupby(entity, sort=True)[columns].mean().reset_index()


def constant_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if frame[c].nunique(dropna=True) <= 1]


def describe_drop(reason: str, columns: list[Any]) -> str:
    return f"{reason}: {', '.join(str(c) for c in columns)}" if columns else ""


def labelled_signal(frame: pd.DataFrame, label_column: str, columns: list[str], alpha: float = 0.01) -> list[str]:
    """Numeric columns that differ across the classes among the *labelled* rows (one-way ANOVA, Bonferroni-corrected).

    With a handful of labels a distance-based method is easily drowned by columns that say nothing about the class; this keeps the ones that do.
    """
    from scipy import stats

    labelled = frame[frame[label_column].notna() & (frame[label_column].astype(str).str.strip() != "")]
    if labelled[label_column].nunique() < 2:
        return list(columns)
    threshold = alpha / max(len(columns), 1)
    keep = []
    for c in columns:
        groups = [g[c].dropna().to_numpy() for _, g in labelled.groupby(label_column) if g[c].notna().sum() > 1]
        if len(groups) > 1 and stats.f_oneway(*groups).pvalue < threshold:
            keep.append(c)
    return keep or list(columns)


def basket_size_fence(sizes: pd.Series, far: float = 3.0) -> float:
    """The basket size above which a basket is a far outlier (Tukey's far-out fence, Q3 + 3 IQR): bulk and wholesale orders that manufacture pairs no shopper chose."""
    q1, q3 = float(sizes.quantile(0.25)), float(sizes.quantile(0.75))
    return q3 + far * (q3 - q1)


def panel_features(frame: pd.DataFrame, entity: str, order: str | None, columns: list[str], window: int = 10) -> pd.DataFrame:
    """For rows that follow an entity through time: each column's change since the entity's first row, and its recent average relative to the start.

    The level of a sensor says little (units differ by entity); how far it has moved from where this entity started says how worn it is.
    """
    ordered = frame.sort_values([entity, order]) if order else frame
    out = frame.copy()
    g = ordered.groupby(entity)
    for c in columns:
        start = g[c].transform("first")
        out[f"{c}_since_start"] = (ordered[c] - start).reindex(out.index)
        out[f"{c}_recent_since_start"] = (g[c].transform(lambda s: s.rolling(window, min_periods=1).mean()) - start).reindex(out.index)
    return out
