"""The operational loop of a labelling campaign: ingest the annotators' answers, measure what they changed, and score with the current model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.ml import build_preprocessor
from ml_core.semi import base_clf, full_proba, load_semi, predict_saved, save_model, SemiModel
from ml_core.tooling import tool


@tool
def ingest_new_labels(
    data: str, labels_file: str, label_column: str, key_column: str, new_label_column: str | None = None, unlabeled_value: str | None = None,
    base: str = "logistic_regression", output_path: str | None = None, seed: int = 42,
) -> dict:
    """Merge annotators' answers back into the dataset: rows are matched on `key_column`, their label is filled in, and the report shows how many were new, how many conflict with an existing label (never overwritten), and how the model's cross-validated accuracy changes with the extra labels.

    Args:
        labels_file: CSV/JSON from the annotators holding the key column and their label.
        key_column: Column that identifies a row in both files (an id, or `row` from the queue file).
        new_label_column: Column of the labels file holding the answer; defaults to `label_column`.
        output_path: Where to write the updated dataset.
    """
    from sklearn.model_selection import cross_val_score

    frame = load_frame(data)
    answers = load_frame(labels_file)
    col = new_label_column or label_column
    require_columns(frame, [label_column, key_column])
    require_columns(answers, [key_column, col])
    answers = answers.dropna(subset=[col]).drop_duplicates(subset=key_column, keep="last")
    unl = frame[label_column].isna()
    if unlabeled_value is not None:
        unl |= frame[label_column].astype(str) == str(unlabeled_value)
    mapping = dict(zip(answers[key_column], answers[col]))
    keys = frame[key_column]
    new = unl & keys.isin(mapping)
    conflicts = (~unl) & keys.isin(mapping) & (frame[label_column].astype(str) != keys.map(mapping).astype(str))

    def accuracy(f: pd.DataFrame) -> float | None:
        try:
            sd = load_semi(f, label_column, None, [key_column], unlabeled_value)
            Z = np.asarray(build_preprocessor(sd.X).fit_transform(sd.X), float)
            y = sd.y[sd.labeled]
            folds = max(2, min(5, int(np.bincount(y)[np.bincount(y) > 0].min())))
            return round(float(cross_val_score(base_clf(base, seed), Z[sd.labeled], y, cv=folds).mean()), 4)
        except Exception:
            return None

    before = accuracy(frame)
    updated = frame.copy()
    updated.loc[new, label_column] = keys[new].map(mapping)
    after = accuracy(updated)
    return {"answers_received": int(len(answers)), "new_labels_applied": int(new.sum()), "already_labelled_ignored": int(((~unl) & keys.isin(mapping)).sum()), "conflicts_with_existing_label": int(conflicts.sum()),
            "answers_for_unknown_keys": int((~answers[key_column].isin(keys)).sum()), "labelled_before": int((~unl).sum()), "labelled_after": int((~unl).sum() + new.sum()),
            "cv_accuracy_before": before, "cv_accuracy_after": after, "written_to": save_frame(updated, output_path, "dataset_with_new_labels.csv")}


@tool
def measure_annotator_agreement(labels_file: str, key_column: str, annotator_column: str, label_column: str) -> dict:
    """Agreement between annotators who labelled the same rows: pairwise raw agreement, Cohen's kappa, and the rows with conflicting answers — label noise caps what any model can reach, and active learning amplifies it because it asks about the hardest rows.

    Args:
        labels_file: CSV/JSON with one row per (row, annotator) answer.
        key_column: Column identifying the row that was labelled.
        annotator_column: Column identifying who answered.
        label_column: Column holding the answer.
    """
    from itertools import combinations

    from sklearn.metrics import cohen_kappa_score

    answers = load_frame(labels_file)
    require_columns(answers, [key_column, annotator_column, label_column])
    wide = answers.pivot_table(index=key_column, columns=annotator_column, values=label_column, aggfunc="first")
    pairs = []
    for a, b in combinations(wide.columns, 2):
        both = wide[[a, b]].dropna()
        if len(both) >= 5:
            pairs.append({"annotators": [str(a), str(b)], "rows_in_common": int(len(both)), "agreement": round(float((both[a] == both[b]).mean()), 4), "cohen_kappa": round(float(cohen_kappa_score(both[a], both[b])), 4)})
    multi = wide[wide.notna().sum(axis=1) > 1]
    conflict = multi[multi.nunique(axis=1) > 1]
    return {"annotators": int(wide.shape[1]), "rows_with_several_answers": int(len(multi)), "rows_with_conflicting_answers": int(len(conflict)), "pairs": pairs,
            "conflicting_examples": conflict.head(10).reset_index(),
            "reading": "kappa under ~0.6 means the labelling guidelines are ambiguous: fix them before buying more labels"}


@tool
def train_and_save_current_model(
    data: str, label_column: str, unlabeled_value: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, base: str = "logistic_regression",
    model_path: str = "active_learning_model.joblib", seed: int = 42,
) -> dict:
    """Fit the base learner on every labelled row so far and save it — the model to use between labelling rounds (and the one whose pool predictions the next query is based on). Reports cross-validated accuracy so progress can be tracked round to round.

    Args:
        base: Base learner.
        model_path: Where to save the model (joblib).
    """
    from sklearn.model_selection import cross_val_score

    sd = load_semi(data, label_column, features, drop, unlabeled_value)
    prep = build_preprocessor(sd.X, scale=True)
    Z = np.asarray(prep.fit_transform(sd.X), float)
    y = sd.y[sd.labeled]
    counts = np.bincount(y, minlength=len(sd.classes))
    cv = float(cross_val_score(base_clf(base, seed), Z[sd.labeled], y, cv=max(2, min(5, int(counts[counts > 0].min())))).mean()) if counts[counts > 0].min() >= 2 else None
    model = base_clf(base, seed).fit(Z[sd.labeled], y)

    class _Core:
        classes_ = np.arange(len(sd.classes))

        def predict_proba(self, Zn: np.ndarray) -> np.ndarray:
            return full_proba(model, Zn, len(sd.classes))

    # a module-level class is needed to pickle; wrap through Fitted instead
    from ml_core.semi import Fitted

    saved = save_model(SemiModel("active_learning", prep, Fitted(model, len(sd.classes)), sd.classes, list(sd.X.columns)), model_path)
    return {"labelled_rows": int(len(y)), "pool_rows": int(len(sd.unlabeled)), "per_class": dict(zip(sd.classes, counts.astype(int).tolist())), "cv_accuracy": round(cv, 4) if cv else None, "model_path": saved}


@tool
def predict_with_semi_model(model_path: str, data: str, id_column: str | None = None, output_path: str | None = None) -> dict:
    """Score rows with a model saved by `train_and_save_current_model` (or any semi-supervised fit tool): predicted class, confidence and every class probability.

    Args:
        model_path: Path of the saved model (joblib).
    """
    return predict_saved(model_path, data, id_column, output_path)
