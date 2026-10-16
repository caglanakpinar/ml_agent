"""Read-only checks on a multi-label problem and its data, before any model is fitted."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_core.tooling import tool
from supervised.multi_label_classification.tools._common import load_multilabel, multilabel_metrics


@tool
def profile_multilabel_data(data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", features: list[str] | None = None, drop: list[str] | None = None, text_column: str | None = None) -> dict:
    """Profile a multi-label dataset: rows, labels with their counts, label cardinality (average labels per row) and density, rows with no labels, distinct label combinations, and which columns are usable features.

    Args:
        labels: Columns holding one 0/1 indicator per label.
        label_column: Alternatively, one column listing each row's labels joined by `separator`.
        separator: Delimiter inside `label_column` cells.
        text_column: Column of free text to use as features (TF-IDF).
    """
    X, Y, names = load_multilabel(data, labels, label_column, separator, features, drop, text_column)
    counts = Y.sum(axis=0)
    combos = pd.Series([tuple(r) for r in Y]).value_counts()
    return {"rows": len(X), "labels": len(names), "label_counts": dict(zip(names, counts.astype(int).tolist())), "label_cardinality": round(float(Y.sum(axis=1).mean()), 3),
            "label_density": round(float(Y.mean()), 4), "rows_without_labels": int((Y.sum(axis=1) == 0).sum()), "max_labels_on_a_row": int(Y.sum(axis=1).max()),
            "distinct_label_combinations": int(len(combos)), "combinations_seen_once": int((combos == 1).sum()), "feature_columns": list(X.columns),
            "warnings": [w for w in [
                "many label combinations seen once: label powerset will struggle; prefer binary relevance, chains or RAkEL" if len(combos) > 0.5 * len(X) else None,
                "labels with under 20 positives: their per-label scores are unreliable" if (counts < 20).any() else None,
                "a label is never positive: drop it" if (counts == 0).any() else None,
                "a label is always positive: it carries no information" if (counts == len(X)).any() else None] if w]}


@tool
def label_cooccurrence(data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", k: int = 12) -> dict:
    """The label pairs that occur together most (count, Jaccard, lift over independence) — high-lift pairs mean labels depend on each other, which favours classifier chains / label powerset over independent binary models.

    Args:
        k: How many pairs to list.
    """
    _, Y, names = load_multilabel(data, labels, label_column, separator)
    n = len(Y)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            both = int((Y[:, i] & Y[:, j]).sum())
            a, b = int(Y[:, i].sum()), int(Y[:, j].sum())
            if both:
                rows.append({"a": names[i], "b": names[j], "together": both, "jaccard": round(both / (a + b - both), 4), "lift": round(both * n / (a * b), 3)})
    rows.sort(key=lambda r: -r["together"])
    strong = [r for r in rows if r["lift"] > 1.5 or r["lift"] < 0.5]
    return {"pairs": rows[:k], "dependent_pairs": len(strong), "advice": "labels clearly depend on each other: try classifier chains, label powerset or RAkEL as well as binary relevance" if len(strong) > len(names) / 2 else "labels are close to independent: binary relevance is enough"}


@tool
def check_label_imbalance(data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|") -> dict:
    """Per-label positive share and imbalance ratio (the IRLbl and MeanIR measures), and the rare labels that need class weights, oversampling or their own threshold."""
    _, Y, names = load_multilabel(data, labels, label_column, separator)
    counts = Y.sum(axis=0)
    irlbl = counts.max() / np.maximum(counts, 1)
    mean_ir = float(irlbl.mean())
    rare = [names[j] for j in range(len(names)) if irlbl[j] > mean_ir]
    return {"positive_share": {n: round(float(c / len(Y)), 4) for n, c in zip(names, counts)}, "imbalance_ratio": {n: round(float(r), 2) for n, r in zip(names, irlbl)},
            "mean_imbalance_ratio": round(mean_ir, 2), "rare_labels": rare,
            "advice": "balanced enough for default settings" if mean_ir < 2 else "imbalanced: use class-weighted base models, optimise per-label thresholds, and judge by macro F1 / average precision, not subset accuracy"}


@tool
def baseline_multilabel(data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", test_size: float = 0.2, seed: int = 42) -> dict:
    """Score two do-nothing baselines on held-out rows — predict only the labels that are positive in over half of the rows, and always predict the single most common label — so a model has a floor to beat.

    Args:
        test_size: Fraction of rows held out.
    """
    from supervised.multi_label_classification.tools._common import holdout_ml

    X, Y, names = load_multilabel(data, labels, label_column, separator)
    _, _, Y_tr, Y_te = holdout_ml(X, Y, test_size, seed)
    majority = np.tile((Y_tr.mean(axis=0) >= 0.5).astype(int), (len(Y_te), 1))
    top = np.zeros_like(Y_te)
    top[:, Y_tr.sum(axis=0).argmax()] = 1
    return {"labels": names, "majority_labels": [n for n, f in zip(names, Y_tr.mean(axis=0)) if f >= 0.5], "most_common_label": names[int(Y_tr.sum(axis=0).argmax())],
            "predict_majority_labels": multilabel_metrics(Y_te, majority), "always_most_common_label": multilabel_metrics(Y_te, top)}


@tool
def check_label_feature_signal(data: str, labels: list[str] | None = None, label_column: str | None = None, separator: str = "|", features: list[str] | None = None, drop: list[str] | None = None, k: int = 5) -> dict:
    """For each label, the numeric features most associated with it (point-biserial correlation) — a quick check that every label has some signal in the features and where it comes from.

    Args:
        k: How many features to list per label.
    """
    X, Y, names = load_multilabel(data, labels, label_column, separator, features, drop)
    num = X.select_dtypes("number")
    num = num.fillna(num.median())
    out, weak = {}, []
    for j, n in enumerate(names):
        if Y[:, j].std() == 0:
            continue
        corr = {c: float(np.corrcoef(num[c], Y[:, j])[0, 1]) for c in num.columns if num[c].std() > 0}
        top = sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:k]
        out[n] = {c: round(v, 3) for c, v in top}
        if not top or abs(top[0][1]) < 0.1:
            weak.append(n)
    return {"strongest_features_per_label": out, "labels_without_obvious_numeric_signal": weak,
            "note": "labels listed as weak may still be predictable from categorical or text columns, or from interactions"}
