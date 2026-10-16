"""The protocol every semi-supervised tool shares, so their numbers are comparable.

A semi-supervised dataset has a label column where some rows are labelled and the rest are not (blank, or
a marker such as -1). Every method is judged the same way:

    labelled rows  ->  split into  L_train  (labels the method may use)  and  L_test  (labels it never sees)
    unlabelled rows -> U            (features the method may use, labels it cannot)

The method is fitted on L_train + U and scored on L_test. The *same* base learner fitted on L_train alone is
the supervised-only baseline, and the number that matters is the difference: a semi-supervised method that
does not beat its own base learner on the labels it was denied is not helping, however many rows it labelled.

If the dataset also carries the true label of the unlabelled rows (`truth_column` — common in experiments
where labels were hidden on purpose) the pseudo-labels a method produced are audited against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ml_core.data import as_list, load_frame, require_columns
from ml_core.ml import build_preprocessor, save_model, sk

UNLABELED = -1


@dataclass
class SemiData:
    X: pd.DataFrame
    y: np.ndarray  # class codes 0..K-1, -1 where unlabelled
    classes: list[str]
    truth: np.ndarray | None  # class codes for every row where known (else -1)
    label_column: str

    @property
    def labeled(self) -> np.ndarray:
        return np.flatnonzero(self.y != UNLABELED)

    @property
    def unlabeled(self) -> np.ndarray:
        return np.flatnonzero(self.y == UNLABELED)


def load_semi(data: Any, label_column: str, features: Any = None, drop: Any = None, unlabeled_value: Any = None, truth_column: str | None = None) -> SemiData:
    """Read a partially labelled table. A row is unlabelled when its label is blank or equals `unlabeled_value`."""
    frame = load_frame(data)
    require_columns(frame, [label_column])
    raw = frame[label_column]
    unl = raw.isna()
    if unlabeled_value is not None and str(unlabeled_value) != "":
        unl |= raw.astype(str) == str(unlabeled_value)
    labelled_values = raw[~unl]
    if labelled_values.nunique() < 2:
        raise ValueError(f"only {labelled_values.nunique()} class among the labelled rows of {label_column!r}; a classifier needs at least two")
    classes = sorted(labelled_values.astype(str).unique(), key=lambda s: (len(s), s))
    index = {c: i for i, c in enumerate(classes)}
    y = np.where(unl, UNLABELED, raw.astype(str).map(index).fillna(UNLABELED)).astype(int)
    truth = None
    skip = {label_column} | set(as_list(drop) or [])
    if truth_column:
        require_columns(frame, [truth_column])
        truth = frame[truth_column].astype(str).map(index).fillna(UNLABELED).to_numpy(int)
        skip.add(truth_column)
    names = [c for c in (as_list(features) or list(frame.columns)) if c not in skip]
    require_columns(frame, names)
    return SemiData(frame[names].reset_index(drop=True), y, classes, truth, label_column)


@dataclass
class SemiSplit:
    l_train: np.ndarray
    l_test: np.ndarray
    unl: np.ndarray


def split_semi(sd: SemiData, test_size: float = 0.3, seed: int = 42) -> SemiSplit:
    sk()
    from sklearn.model_selection import train_test_split

    labeled = sd.labeled
    y = sd.y[labeled]
    strat = y if np.bincount(y).min() >= 2 else None
    tr, te = train_test_split(labeled, test_size=test_size, random_state=seed, stratify=strat)
    return SemiSplit(np.sort(tr), np.sort(te), sd.unlabeled)


@dataclass
class Context:
    """What a trainer receives: the encoded matrix of all rows, who is labelled, and the base learner's name."""

    Z: np.ndarray
    sd: SemiData
    split: SemiSplit
    base: str
    seed: int
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def K(self) -> int:
        return len(self.sd.classes)

    @property
    def Zl(self) -> np.ndarray:
        return self.Z[self.split.l_train]

    @property
    def yl(self) -> np.ndarray:
        return self.sd.y[self.split.l_train]

    @property
    def Zu(self) -> np.ndarray:
        return self.Z[self.split.unl]


def base_clf(name: str, seed: int = 0) -> Any:
    sk()
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from ml_core.ml import svc_with_probabilities

    return {"logistic_regression": lambda: LogisticRegression(max_iter=1000),
            "random_forest": lambda: RandomForestClassifier(200, min_samples_leaf=2, n_jobs=-1, random_state=seed),
            "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=seed),
            "svm": lambda: svc_with_probabilities(random_state=seed), "knn": lambda: KNeighborsClassifier(n_neighbors=7, weights="distance"),
            "naive_bayes": lambda: GaussianNB()}[name]()


def full_proba(model: Any, Z: np.ndarray, K: int) -> np.ndarray:
    """P(class) for all K classes, even when the model saw only some of them."""
    raw = np.asarray(model.predict_proba(Z))
    classes = getattr(model, "classes_", np.arange(raw.shape[1]))
    out = np.zeros((len(Z), K))
    out[:, [int(c) for c in classes]] = raw
    return out


def metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    from sklearn import metrics as m

    pred = proba.argmax(axis=1)
    out = {"accuracy": m.accuracy_score(y, pred), "balanced_accuracy": m.balanced_accuracy_score(y, pred), "f1_macro": m.f1_score(y, pred, average="macro", zero_division=0)}
    if proba.shape[1] == 2 and len(set(y)) == 2:
        out["roc_auc"] = m.roc_auc_score(y, proba[:, 1])
        out["average_precision"] = m.average_precision_score(y, proba[:, 1])
    return {k: round(float(v), 5) for k, v in out.items()}


class SemiModel:
    """Preprocessor + fitted core + class names. `core.predict_proba(Z)` must return one column per class."""

    def __init__(self, kind: str, prep: Any, core: Any, classes: list[str], columns: list[str]):
        self.kind, self.prep, self.core, self.classes, self.columns = kind, prep, core, classes, columns

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.prep.transform(X[self.columns])
        return full_proba(self.core, np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z, float), len(self.classes))

    def predict(self, X: pd.DataFrame) -> list[str]:
        return [self.classes[i] for i in self.proba(X).argmax(axis=1)]


Trainer = Callable[[Context], "tuple[Any, dict[str, Any]]"]


def make_context(data: Any, label_column: str, features: Any, drop: Any, unlabeled_value: Any, truth_column: str | None, test_size: float, seed: int, base: str, **params: Any) -> tuple[Context, Any]:
    sd = load_semi(data, label_column, features, drop, unlabeled_value, truth_column)
    split = split_semi(sd, test_size, seed)
    fit_rows = np.concatenate([split.l_train, split.unl])
    prep = build_preprocessor(sd.X, scale=True)
    prep.fit(sd.X.iloc[fit_rows])  # unlabelled rows legitimately inform scaling and encoding
    Z = prep.transform(sd.X)
    Z = np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z, float)
    return Context(Z, sd, split, base, seed, dict(params)), prep


def audit_pseudo(sd: SemiData, pseudo: dict[str, np.ndarray] | None) -> dict[str, Any] | None:
    """Accuracy of the pseudo-labels a method produced, against the hidden truth if the data carries it."""
    if not pseudo or len(pseudo.get("index", [])) == 0:
        return None
    idx, lab = np.asarray(pseudo["index"]), np.asarray(pseudo["label"])
    out: dict[str, Any] = {"pseudo_labelled": int(len(idx)), "share_of_unlabelled": round(float(len(idx) / max(len(sd.unlabeled), 1)), 4),
                           "class_counts": {sd.classes[k]: int((lab == k).sum()) for k in range(len(sd.classes))}}
    if sd.truth is not None:
        known = sd.truth[idx] != UNLABELED
        if known.any():
            out["accuracy_vs_truth"] = round(float((sd.truth[idx][known] == lab[known]).mean()), 4)
            if "confidence" in pseudo:
                conf = np.asarray(pseudo["confidence"])[known]
                correct = (sd.truth[idx][known] == lab[known])
                hi = conf >= np.median(conf)
                out["accuracy_most_confident_half"] = round(float(correct[hi].mean()), 4)
                out["accuracy_least_confident_half"] = round(float(correct[~hi].mean()), 4)
    return out


def run_semi(label: str, trainer: Trainer, data: Any, label_column: str, features: Any, drop: Any, unlabeled_value: Any, truth_column: str | None,
             test_size: float, seed: int, model_path: str | None, base: str = "logistic_regression", **params: Any) -> dict[str, Any]:
    """Fit `trainer` on L_train + U, compare it with the same base learner fitted on L_train alone, and report on L_test."""
    ctx, prep = make_context(data, label_column, features, drop, unlabeled_value, truth_column, test_size, seed, base, **params)
    sd, sp = ctx.sd, ctx.split
    if len(sp.unl) == 0:
        raise ValueError("no unlabelled rows: every row has a label, so there is nothing for a semi-supervised method to use")
    Zt, yt = ctx.Z[sp.l_test], sd.y[sp.l_test]

    supervised = base_clf(base, seed).fit(ctx.Zl, ctx.yl)
    sup = metrics(yt, full_proba(supervised, Zt, ctx.K))
    core, extra = trainer(ctx)
    semi = metrics(yt, full_proba(core, Zt, ctx.K))
    gain = {k: round(semi[k] - sup[k], 5) for k in sup}
    se = float(np.sqrt(max(sup["accuracy"] * (1 - sup["accuracy"]), 1e-4) / len(yt)))
    verdict = ("helps: gain exceeds ~2 standard errors of the held-out accuracy" if gain["accuracy"] > 2 * se else
               "hurts: the unlabelled data made the model worse" if gain["accuracy"] < -2 * se else
               "no reliable difference from the supervised-only model at this test-set size")
    pseudo = extra.pop("pseudo", None)
    result = {"model": label, "labelled_rows": int(len(sd.labeled)), "unlabelled_rows": int(len(sp.unl)), "labelled_share": round(float(len(sd.labeled) / len(sd.y)), 4),
              "labelled_train": int(len(sp.l_train)), "labelled_test": int(len(sp.l_test)), "classes": sd.classes,
              "holdout": semi, "supervised_only_baseline": sup, "gain_over_supervised": gain, "verdict": verdict,
              "test_set_standard_error": round(se, 4), **extra}
    audit = audit_pseudo(sd, pseudo)
    if audit:
        result["pseudo_labels"] = audit
    model = SemiModel(label, prep, core, sd.classes, list(sd.X.columns))
    result["model_path"] = save_model(model, model_path)
    return result


# ---- helpers every semi-supervised topic exposes as a tool --------------------------------------
def predict_saved(model_path: str, data: Any, id_column: str | None, output_path: str | None) -> dict[str, Any]:
    """Score new rows with a model saved by a semi-supervised fit tool."""
    import joblib

    from ml_core.data import save_frame

    model = joblib.load(model_path)
    frame = load_frame(data)
    require_columns(frame, model.columns)
    proba = model.proba(frame)
    out = pd.DataFrame({"prediction": [model.classes[i] for i in proba.argmax(axis=1)], "confidence": proba.max(axis=1)}, index=frame.index)
    for k, name in enumerate(model.classes):
        out[f"p_{name}"] = proba[:, k]
    if id_column and id_column in frame.columns:
        out.insert(0, id_column, frame[id_column])
    return {"rows": len(out), "prediction_counts": out["prediction"].value_counts().to_dict(), "mean_confidence": round(float(out["confidence"].mean()), 4),
            "written_to": save_frame(out, output_path, "semi_predictions.csv"), "sample": out.head(5)}


def mask_labels(data: Any, label_column: str, keep_fraction: float, seed: int, output_path: str | None, stratify: bool = True) -> dict[str, Any]:
    """Hide most labels of a fully labelled table to simulate scarcity; the true labels are kept in `<label>__truth`."""
    from ml_core.data import save_frame

    frame = load_frame(data)
    require_columns(frame, [label_column])
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(frame), bool)
    groups = frame.groupby(label_column).indices.values() if stratify else [np.arange(len(frame))]
    for idx in groups:
        n = max(2, int(round(len(idx) * keep_fraction)))
        keep[rng.choice(idx, size=min(n, len(idx)), replace=False)] = True
    truth = f"{label_column}__truth"
    out = frame.assign(**{truth: frame[label_column]})
    out.loc[~keep, label_column] = np.nan
    return {"rows": len(out), "labelled": int(keep.sum()), "unlabelled": int((~keep).sum()), "labelled_share": round(float(keep.mean()), 4), "truth_column": truth,
            "written_to": save_frame(out, output_path, "masked_labels.csv"),
            "next": f"pass label_column='{label_column}' and truth_column='{truth}' to the fit tools to audit pseudo-labels against the hidden truth"}


# ---- small shared building blocks ---------------------------------------------------------------
class Fitted:
    """A fitted learner that always answers for all K classes."""

    def __init__(self, model: Any, K: int, history: list[dict[str, Any]] | None = None):
        self.model, self.K, self.history = model, K, history or []
        self.classes_ = np.arange(K)

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        return full_proba(self.model, Z, self.K)


def fit_weighted(model: Any, Z: np.ndarray, y: np.ndarray, w: np.ndarray | None) -> Any:
    """Fit with sample weights; learners that take none (k-NN) get rows replicated in proportion to weight."""
    if w is None:
        return model.fit(Z, y)
    try:
        return model.fit(Z, y, sample_weight=w)
    except TypeError:
        reps = np.maximum(1, np.rint(w / max(w.min(), 1e-9)).astype(int))
        return model.fit(np.repeat(Z, reps, axis=0), np.repeat(y, reps))


def column_blocks(prep: Any) -> dict[str, list[int]]:
    """Which columns of the encoded matrix each original column became: one for a number, one per level for a category."""
    blocks: dict[str, list[int]] = {}
    offset = 0
    for name, transformer, cols in prep.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        cols = list(cols)
        if name == "num":
            for c in cols:
                blocks[c] = [offset]
                offset += 1
        else:  # one-hot: category count per column
            encoder = transformer.named_steps["onehot"]
            for c, cats in zip(cols, encoder.categories_):
                blocks[c] = list(range(offset, offset + len(cats)))
                offset += len(cats)
    return blocks


def view_indices(prep: Any, view: list[str]) -> np.ndarray:
    blocks = column_blocks(prep)
    missing = [c for c in view if c not in blocks]
    if missing:
        raise ValueError(f"view column(s) {missing} are not usable features; usable are {sorted(blocks)}")
    return np.asarray(sorted(i for c in view for i in blocks[c]), int)
