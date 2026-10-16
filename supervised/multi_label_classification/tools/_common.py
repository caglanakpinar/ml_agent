"""Multi-label machinery the tool modules share: reading label indicators (several 0/1 columns, or one
delimited column), metrics, per-label decision thresholds, and the problem-transformation / native models —
binary relevance, classifier chains, label powerset, RAkEL, ML-kNN and multi-output forests — all behind
one `MultiLabelModel` that returns a probability per label."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_core.data import as_list, load_frame, require_columns
from ml_core.ml import build_preprocessor, save_model, sk


# ---- reading ------------------------------------------------------------------------------------
def load_multilabel(
    data: Any, labels: Any = None, label_column: str | None = None, separator: str = "|", features: Any = None, drop: Any = None, text_column: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return the feature frame, the (n, L) 0/1 label matrix and the label names. Labels come from several 0/1 `labels` columns
    or from one `label_column` whose cells list the labels joined by `separator`."""
    frame = load_frame(data)
    names = as_list(labels)
    if names:
        require_columns(frame, names)
        Y = frame[names].fillna(0).astype(float).gt(0).astype(int).to_numpy()
        label_names, used = names, set(names)
    elif label_column:
        require_columns(frame, [label_column])
        sets = frame[label_column].fillna("").astype(str).map(lambda s: [p.strip() for p in s.split(separator) if p.strip()])
        label_names = sorted({l for s in sets for l in s})
        Y = np.array([[int(l in s) for l in label_names] for s in sets])
        used = {label_column}
    else:
        raise ValueError("pass `labels` (several 0/1 columns) or `label_column` (one delimited column)")
    if len(label_names) < 2:
        raise ValueError("multi-label needs at least two labels")
    skip = used | set(as_list(drop) or [])
    cols = as_list(features) or [c for c in frame.columns if c not in skip]
    cols = [c for c in cols if c not in skip]
    if text_column and text_column not in cols:
        cols.append(text_column)
    require_columns(frame, cols)
    return frame[cols].copy(), Y, label_names


def build_features(X: pd.DataFrame, text_column: str | None = None, max_features: int = 5000) -> Any:
    """Tabular columns through the shared preprocessor; a text column through TF-IDF; both when both exist."""
    sk()
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not text_column:
        return build_preprocessor(X, scale=True)
    others = [c for c in X.columns if c != text_column]
    parts = [("text", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True, stop_words="english"), text_column)]
    if others:
        parts.append(("tab", build_preprocessor(X[others], scale=True), others))
    return ColumnTransformer(parts, sparse_threshold=0.0)


def stack_proba(raw: Any, n_labels: int) -> np.ndarray:
    """sklearn multi-output `predict_proba` returns a list of (n, 2) arrays; make it (n, L) of P(label = 1)."""
    if isinstance(raw, list):
        cols = [(r[:, 1] if r.shape[1] > 1 else np.zeros(len(r))) for r in raw]
        return np.column_stack(cols)
    return np.asarray(raw)


# ---- metrics ------------------------------------------------------------------------------------
def multilabel_metrics(Y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None = None) -> dict[str, float]:
    sk()
    from sklearn import metrics as m

    out = {
        "subset_accuracy": m.accuracy_score(Y, pred),
        "hamming_loss": m.hamming_loss(Y, pred),
        "f1_micro": m.f1_score(Y, pred, average="micro", zero_division=0),
        "f1_macro": m.f1_score(Y, pred, average="macro", zero_division=0),
        "f1_samples": m.f1_score(Y, pred, average="samples", zero_division=0),
        "precision_micro": m.precision_score(Y, pred, average="micro", zero_division=0),
        "recall_micro": m.recall_score(Y, pred, average="micro", zero_division=0),
        "jaccard_samples": m.jaccard_score(Y, pred, average="samples", zero_division=0),
    }
    if proba is not None:
        ok = [j for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)]
        if ok:
            out["roc_auc_macro"] = float(np.mean([m.roc_auc_score(Y[:, j], proba[:, j]) for j in ok]))
            out["average_precision_macro"] = float(np.mean([m.average_precision_score(Y[:, j], proba[:, j]) for j in ok]))
        if (Y.sum(axis=1) > 0).all():
            out["label_ranking_average_precision"] = float(m.label_ranking_average_precision_score(Y, proba))
            out["ranking_loss"] = float(m.label_ranking_loss(Y, proba))
    return {k: round(float(v), 5) for k, v in out.items()}


def decide(proba: np.ndarray, thresholds: Any = 0.5, at_least_one: bool = True) -> np.ndarray:
    pred = (proba >= np.asarray(thresholds)).astype(int)
    if at_least_one:
        empty = pred.sum(axis=1) == 0
        pred[empty, proba[empty].argmax(axis=1)] = 1
    return pred


# ---- cores --------------------------------------------------------------------------------------
def base_estimator(name: str, seed: int = 0) -> Any:
    sk()
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    return {"logistic_regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
            "random_forest": lambda: RandomForestClassifier(150, min_samples_leaf=2, n_jobs=-1, random_state=seed),
            "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=seed),
            "linear_svm": lambda: CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3),
            "naive_bayes": lambda: GaussianNB()}[name]()


class BinaryRelevance:
    def __init__(self, base: str, seed: int = 0):
        self.base, self.seed, self.models = base, seed, []

    def fit(self, Z: Any, Y: np.ndarray) -> "BinaryRelevance":
        self.models = [("const", float(Y[0, j])) if Y[:, j].min() == Y[:, j].max() else ("model", base_estimator(self.base, self.seed).fit(Z, Y[:, j])) for j in range(Y.shape[1])]
        return self

    def predict_proba(self, Z: Any) -> np.ndarray:
        return np.column_stack([np.full(Z.shape[0], m[1]) if m[0] == "const" else m[1].predict_proba(Z)[:, 1] for m in self.models])


class ClassifierChains:
    """An ensemble of classifier chains with random label orders; each label's model sees the previous labels as features."""

    def __init__(self, base: str, n_chains: int = 5, seed: int = 0):
        self.base, self.n_chains, self.seed = base, n_chains, seed
        self.chains: list[Any] = []

    def fit(self, Z: Any, Y: np.ndarray) -> "ClassifierChains":
        from sklearn.multioutput import ClassifierChain

        self.chains = []
        rng = np.random.default_rng(self.seed)
        for i in range(self.n_chains):
            order = rng.permutation(Y.shape[1]) if i else np.arange(Y.shape[1])
            self.chains.append(ClassifierChain(base_estimator(self.base, self.seed + i), order=list(order), random_state=self.seed + i).fit(Z, Y))
        return self

    def predict_proba(self, Z: Any) -> np.ndarray:
        return np.mean([c.predict_proba(Z) for c in self.chains], axis=0)


class LabelPowerset:
    """Each distinct label combination is one class of a multiclass problem; rare combinations are folded into the nearest common one."""

    def __init__(self, base: str, min_count: int = 2, seed: int = 0):
        self.base, self.min_count, self.seed = base, min_count, seed

    def fit(self, Z: Any, Y: np.ndarray) -> "LabelPowerset":
        keys = [tuple(r) for r in Y]
        counts: dict[tuple, int] = {}
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
        common = [k for k, c in counts.items() if c >= self.min_count] or list(counts)
        self.sets = np.array(common)
        mapped = []
        for k in keys:
            if counts[k] >= self.min_count or k in common:
                mapped.append(common.index(k))
            else:
                mapped.append(int(np.abs(self.sets - np.array(k)).sum(axis=1).argmin()))
        self.model = base_estimator(self.base, self.seed).fit(Z, np.array(mapped))
        return self

    def predict_proba(self, Z: Any) -> np.ndarray:
        p = self.model.predict_proba(Z)
        return p @ self.sets[self.model.classes_]


class RAkEL:
    """Random k-labelsets: label-powerset models on random subsets of k labels, voting for each label."""

    def __init__(self, base: str, k: int = 3, n_models: int = 12, seed: int = 0):
        self.base, self.k, self.n_models, self.seed = base, k, n_models, seed

    def fit(self, Z: Any, Y: np.ndarray) -> "RAkEL":
        rng = np.random.default_rng(self.seed)
        L = Y.shape[1]
        k = min(self.k, L)
        self.subsets, self.models = [], []
        for _ in range(self.n_models):
            subset = np.sort(rng.choice(L, k, replace=False))
            self.subsets.append(subset)
            self.models.append(LabelPowerset(self.base, 1, self.seed).fit(Z, Y[:, subset]))
        self.L = L
        return self

    def predict_proba(self, Z: Any) -> np.ndarray:
        total, votes = np.zeros((Z.shape[0], self.L)), np.zeros(self.L)
        for subset, model in zip(self.subsets, self.models):
            total[:, subset] += model.predict_proba(Z)
            votes[subset] += 1
        return total / np.maximum(votes, 1)


class NativeMultiOutput:
    """Estimators with built-in multi-label support: random forest, extra trees, k-NN, MLP."""

    def __init__(self, kind: str, seed: int = 0, **params: Any):
        sk()
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier

        self.model = {"random_forest": lambda: RandomForestClassifier(n_estimators=params.get("n_estimators", 300), min_samples_leaf=params.get("min_samples_leaf", 2), n_jobs=-1, random_state=seed),
                      "extra_trees": lambda: ExtraTreesClassifier(n_estimators=params.get("n_estimators", 300), min_samples_leaf=params.get("min_samples_leaf", 2), n_jobs=-1, random_state=seed),
                      "knn": lambda: KNeighborsClassifier(n_neighbors=params.get("n_neighbors", 15), weights="distance"),
                      "mlp": lambda: MLPClassifier(hidden_layer_sizes=params.get("hidden", (64,)), max_iter=600, random_state=seed)}[kind]()

    def fit(self, Z: Any, Y: np.ndarray) -> "NativeMultiOutput":
        self.model.fit(Z, Y)
        self.L = Y.shape[1]
        return self

    def predict_proba(self, Z: Any) -> np.ndarray:
        return stack_proba(self.model.predict_proba(Z), self.L)


class MLkNN:
    """Multi-label k-NN (Zhang & Zhou): for each label a Bayesian posterior from how many of a row's k neighbours carry it."""

    def __init__(self, k: int = 10, smooth: float = 1.0):
        self.k, self.s = k, smooth

    def fit(self, Z: Any, Y: np.ndarray) -> "MLkNN":
        from sklearn.neighbors import NearestNeighbors

        self.Y = Y
        self.nn = NearestNeighbors(n_neighbors=self.k + 1).fit(Z)
        n, L = Y.shape
        self.prior = (self.s + Y.sum(axis=0)) / (2 * self.s + n)
        idx = self.nn.kneighbors(Z, return_distance=False)[:, 1:]
        counts = Y[idx].sum(axis=1)  # (n, L): neighbours carrying each label
        self.c1, self.c0 = np.zeros((L, self.k + 1)), np.zeros((L, self.k + 1))
        for j in range(L):
            for i in range(n):
                (self.c1 if Y[i, j] else self.c0)[j, counts[i, j]] += 1
        return self

    def predict_proba(self, Z: Any) -> np.ndarray:
        idx = self.nn.kneighbors(Z, n_neighbors=self.k, return_distance=False)
        counts = self.Y[idx].sum(axis=1)
        out = np.zeros(counts.shape)
        for j in range(counts.shape[1]):
            c1, c0 = self.c1[j], self.c0[j]
            p1 = (self.s + c1[counts[:, j]]) / (self.s * (self.k + 1) + c1.sum())
            p0 = (self.s + c0[counts[:, j]]) / (self.s * (self.k + 1) + c0.sum())
            out[:, j] = self.prior[j] * p1 / (self.prior[j] * p1 + (1 - self.prior[j]) * p0)
        return out


class MultiLabelModel:
    def __init__(self, kind: str, prep: Any, core: Any, labels: list[str], columns: list[str], text_column: str | None, thresholds: Any = 0.5):
        self.kind, self.prep, self.core, self.labels, self.columns, self.text_column, self.thresholds = kind, prep, core, labels, columns, text_column, thresholds

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.prep.transform(X[self.columns])
        return np.asarray(self.core.predict_proba(Z.toarray() if hasattr(Z, "toarray") else Z), float)

    def predict(self, X: pd.DataFrame, thresholds: Any = None, at_least_one: bool = True) -> np.ndarray:
        return decide(self.proba(X), self.thresholds if thresholds is None else thresholds, at_least_one)


def build_core(kind: str, base: str, seed: int, **params: Any) -> Any:
    if kind == "binary_relevance":
        return BinaryRelevance(base, seed)
    if kind == "classifier_chain":
        return ClassifierChains(base, params.get("n_chains", 5), seed)
    if kind == "label_powerset":
        return LabelPowerset(base, params.get("min_count", 2), seed)
    if kind == "rakel":
        return RAkEL(base, params.get("k", 3), params.get("n_models", 12), seed)
    if kind == "mlknn":
        return MLkNN(params.get("n_neighbors", 10))
    return NativeMultiOutput(kind, seed, **params)


def fit_multilabel(kind: str, X: pd.DataFrame, Y: np.ndarray, labels: list[str], text_column: str | None = None, base: str = "logistic_regression", seed: int = 0, **params: Any) -> MultiLabelModel:
    prep = build_features(X, text_column)
    Z = prep.fit_transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z, float)
    core = build_core(kind, base, seed, **params).fit(Z, Y)
    return MultiLabelModel(kind, prep, core, labels, list(X.columns), text_column)


def holdout_ml(X: pd.DataFrame, Y: np.ndarray, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    sk()
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(X))
    # stratify on the single most common-vs-rare label combination signature that keeps every stratum splittable
    key = pd.Series([tuple(r) for r in Y])
    strat = key if key.value_counts().min() >= 2 else None
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)
    return X.iloc[tr], X.iloc[te], Y[tr], Y[te]


def run_multilabel(kind: str, label: str, data: Any, labels: Any, label_column: str | None, separator: str, features: Any, drop: Any, text_column: str | None,
                   test_size: float, seed: int, model_path: str | None, base: str = "logistic_regression", threshold: float = 0.5, **params: Any) -> dict[str, Any]:
    X, Y, names = load_multilabel(data, labels, label_column, separator, features, drop, text_column)
    X_tr, X_te, Y_tr, Y_te = holdout_ml(X, Y, test_size, seed)
    model = fit_multilabel(kind, X_tr, Y_tr, names, text_column, base, seed, **params)
    model.thresholds = threshold
    proba = model.proba(X_te)
    pred = decide(proba, threshold)
    prior = np.tile((Y_tr.mean(axis=0) >= 0.5).astype(int), (len(Y_te), 1))
    from sklearn.metrics import f1_score

    per = {names[j]: round(float(f1_score(Y_te[:, j], pred[:, j], zero_division=0)), 4) for j in range(len(names))}
    out = {"model": label, "labels": names, "rows_train": len(X_tr), "rows_test": len(X_te), "label_cardinality": round(float(Y.sum(axis=1).mean()), 3),
           "holdout": multilabel_metrics(Y_te, pred, proba), "per_label_f1": per, "predicted_labels_per_row": round(float(pred.sum(axis=1).mean()), 3),
           "prior_baseline_f1_micro": multilabel_metrics(Y_te, prior)["f1_micro"]}
    out["model_path"] = save_model(model, model_path)
    return out
