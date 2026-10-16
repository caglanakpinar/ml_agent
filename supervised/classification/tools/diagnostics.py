"""Read-only checks on a classification problem and its data, before any model is fitted."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ml_core.data import as_list, load_frame
from ml_core.ml import build_preprocessor, holdout, numeric_and_categorical, sk, split_xy
from ml_core.tooling import tool
from supervised.classification.tools._common import classification_metrics


@tool
def profile_classification_data(data: str, target: str, features: list[str] | None = None) -> dict:
    """Profile a classification dataset: size, target classes and their share, column kinds, missing values, and constant or id-like columns."""
    frame = load_frame(data)
    if target not in frame.columns:
        raise KeyError(f"target {target!r} not in columns {list(frame.columns)}")
    names = as_list(features) or [c for c in frame.columns if c != target]
    X = frame[names]
    numeric, categorical = numeric_and_categorical(X, max_categories=50)
    counts = frame[target].value_counts(dropna=False)
    missing = X.isna().mean()
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "target_missing": int(frame[target].isna().sum()),
        "classes": {str(k): int(v) for k, v in counts.items()},
        "class_share": {str(k): round(float(v / len(frame)), 4) for k, v in counts.items()},
        "problem": "binary" if frame[target].nunique() == 2 else "multiclass",
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "high_cardinality_columns": [c for c in names if c not in numeric and c not in categorical],
        "constant_columns": [c for c in names if X[c].nunique(dropna=False) <= 1],
        "id_like_columns": [c for c in names if len(frame) > 20 and X[c].nunique() >= 0.98 * len(frame)],
        "columns_with_missing": {c: round(float(r), 4) for c, r in missing[missing > 0].items()},
        "duplicate_rows": int(frame.duplicated().sum()),
    }


@tool
def check_class_balance(data: str, target: str) -> dict:
    """Measure class imbalance (ratio, minority share, effective number of classes) and recommend a strategy: plain fit, class weights, resampling, or a PR-based metric."""
    y = load_frame(data)[target].dropna()
    counts = y.value_counts()
    ratio = float(counts.max() / counts.min())
    share = counts / counts.sum()
    effective = float(np.exp(-(share * np.log(share)).sum()))
    if ratio < 1.5:
        advice = "balanced enough: fit directly and report accuracy plus macro-F1."
    elif ratio < 5:
        advice = "mild imbalance: use class_weight='balanced' and judge by balanced accuracy / macro-F1."
    elif ratio < 20:
        advice = "moderate imbalance: class weights or SMOTE-style oversampling on the training fold only; judge by PR-AUC and the F1 of the minority class."
    else:
        advice = "severe imbalance: judge by PR-AUC and recall at a fixed precision; use class weights, then tune the decision threshold."
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "imbalance_ratio": round(ratio, 3),
        "minority_class": str(counts.idxmin()),
        "minority_share": round(float(share.min()), 4),
        "effective_classes": round(effective, 3),
        "recommendation": advice,
    }


@tool
def find_leakage_candidates(
    data: str, target: str, features: list[str] | None = None, threshold: float = 0.95
) -> dict:
    """Flag features that predict the target suspiciously well on their own (single-feature AUC or accuracy above `threshold`) — usually leakage, not signal.

    Args:
        threshold: Single-feature score above which a column is flagged.
    """
    sk()
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.tree import DecisionTreeClassifier

    X, y = split_xy(data, target, features)
    X_tr, X_te, y_tr, y_te = holdout(X, y, 0.3, 42, stratify=True)
    scores: dict[str, float] = {}
    for column in X.columns:
        try:
            prep = build_preprocessor(X[[column]], scale=False)
            tree = DecisionTreeClassifier(max_depth=4, random_state=0)
            tree.fit(prep.fit_transform(X_tr[[column]]), y_tr)
            Z = prep.transform(X_te[[column]])
            if y.nunique() == 2:
                col = list(tree.classes_).index(tree.classes_[-1])
                scores[column] = float(roc_auc_score(y_te == tree.classes_[-1], tree.predict_proba(Z)[:, col]))
            else:
                scores[column] = float(balanced_accuracy_score(y_te, tree.predict(Z)))
        except Exception:
            continue
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {
        "metric": "single-feature AUC" if y.nunique() == 2 else "single-feature balanced accuracy",
        "suspects": [{"column": c, "score": round(s, 4)} for c, s in ranked if s >= threshold],
        "strongest": [{"column": c, "score": round(s, 4)} for c, s in ranked[:8]],
        "note": "A suspect is not proof — check whether the column exists at prediction time before dropping or keeping it.",
    }


@tool
def baseline_classifier(
    data: str, target: str, strategy: Literal["most_frequent", "prior", "stratified"] = "most_frequent",
    test_size: float = 0.2, seed: int = 42,
) -> dict:
    """Score a do-nothing baseline (always the majority class, or class-prior guesses) so every real model has a floor to beat.

    Args:
        strategy: most_frequent, prior or stratified.
    """
    sk()
    from sklearn.dummy import DummyClassifier

    X, y = split_xy(data, target)
    X_tr, X_te, y_tr, y_te = holdout(X, y, test_size, seed, stratify=True)
    dummy = DummyClassifier(strategy=strategy, random_state=seed).fit(X_tr, y_tr)
    classes = list(dummy.classes_)
    return {
        "strategy": strategy,
        "holdout": classification_metrics(y_te, dummy.predict(X_te), dummy.predict_proba(X_te), classes),
    }


@tool
def check_feature_target_association(
    data: str, target: str, features: list[str] | None = None, k: int = 15
) -> dict:
    """Rank features by mutual information with the target — a model-free look at which columns carry signal.

    Args:
        k: How many top features to return.
    """
    sk()
    from sklearn.feature_selection import mutual_info_classif

    X, y = split_xy(data, target, features)
    numeric, categorical = numeric_and_categorical(X)
    M = pd.DataFrame(index=X.index)
    for c in numeric:
        M[c] = X[c].fillna(X[c].median())
    for c in categorical:
        M[c] = X[c].astype(str).astype("category").cat.codes
    discrete = [c in categorical for c in M.columns]
    mi = mutual_info_classif(M.to_numpy(float), y, discrete_features=discrete, random_state=0)
    ranked = sorted(zip(M.columns, mi), key=lambda item: item[1], reverse=True)
    return {"mutual_information": {c: round(float(v), 5) for c, v in ranked[:k]}, "zero_signal": [c for c, v in ranked if v <= 1e-6]}
