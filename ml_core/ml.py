"""What the ML tools have in common: splitting a frame into features and a target, preprocessing it,
holding rows out, and keeping the model that was fitted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_core.data import as_list, load_frame, require_columns
from ml_core.tooling import MissingDependency


def sk() -> Any:
    """Import scikit-learn lazily so a missing install is reported by the tool, not at import time."""
    try:
        import sklearn  # noqa: F401
    except ImportError as error:
        raise MissingDependency(
            "scikit-learn is not installed. Run: pip install scikit-learn"
        ) from error
    return sklearn


def sklearn_version() -> tuple[int, int]:
    import sklearn

    major, minor = sklearn.__version__.split(".")[:2]
    return int(major), int("".join(c for c in minor if c.isdigit()) or 0)


def logistic_kwargs(penalty: str = "l2", C: float = 1.0, **extra: Any) -> dict[str, Any]:
    """LogisticRegression arguments for an l2 / l1 / elasticnet penalty, spelled the way the installed scikit-learn wants it: 1.8 replaced
    `penalty=` by `l1_ratio=` (0 = l2, 1 = l1) and deprecated the old form."""
    sk()
    l1_ratio = {"l2": 0.0, "l1": 1.0, "elasticnet": 0.5}[penalty]
    solver = "lbfgs" if penalty == "l2" else "saga"
    if sklearn_version() >= (1, 8):
        return {"C": C, "l1_ratio": l1_ratio, "solver": solver, **extra}
    out: dict[str, Any] = {"C": C, "penalty": penalty, "solver": solver, **extra}
    if penalty == "elasticnet":
        out["l1_ratio"] = l1_ratio
    return out


def svc_with_probabilities(**params: Any) -> Any:
    """An SVM that answers `predict_proba`. scikit-learn 1.9 deprecated `SVC(probability=True)` in favour of calibrating the margin."""
    sk()
    from sklearn.svm import SVC

    if sklearn_version() >= (1, 9):
        from sklearn.calibration import CalibratedClassifierCV

        return CalibratedClassifierCV(SVC(**params), cv=3, ensemble=False)
    return SVC(probability=True, **params)


def split_xy(
    data: Any,
    target: str,
    features: Any = None,
    drop: Any = None,
    dropna_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load `data` and split it into a feature frame and the target column.

    Rows with no target are dropped (they cannot be scored) unless `dropna_target` is off. Columns in
    `drop` (ids, leaky columns) never reach the features. With `features` unset, every other column is a
    feature.
    """
    frame = load_frame(data)
    require_columns(frame, [target])
    if dropna_target:
        frame = frame[frame[target].notna()].reset_index(drop=True)
    names = as_list(features)
    if names:
        require_columns(frame, names)
    else:
        names = [c for c in frame.columns if c != target]
    excluded = set(as_list(drop) or [])
    names = [c for c in names if c not in excluded and c != target]
    if not names:
        raise ValueError("no feature columns left after dropping the target and `drop` columns")
    return frame[names].copy(), frame[target].copy()


def numeric_and_categorical(X: pd.DataFrame, max_categories: int = 50) -> tuple[list[str], list[str]]:
    """Split columns into numeric ones and categorical ones (booleans count as numeric).

    A text-like column with more than `max_categories` distinct values is left out of both lists — one-hot
    encoding an id column is how a model memorises its training set.
    """
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]
    categorical = [
        c for c in X.columns if c not in numeric and X[c].nunique(dropna=True) <= max_categories
    ]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame, scale: bool = True, max_categories: int = 50) -> Any:
    """Median-impute and (optionally) standardise numerics; mode-impute and one-hot categoricals."""
    sk()
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric, categorical = numeric_and_categorical(X, max_categories)
    parts = []
    if numeric:
        steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scale", StandardScaler()))
        parts.append(("num", Pipeline(steps), numeric))
    if categorical:
        parts.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    if not parts:
        raise ValueError("no usable feature columns: none numeric, none categorical with few levels")
    return ColumnTransformer(parts, remainder="drop")


def make_pipeline(estimator: Any, X: pd.DataFrame, scale: bool = True) -> Any:
    """`build_preprocessor` followed by `estimator`, so imputation and scaling are fitted on training rows only."""
    sk()
    from sklearn.pipeline import Pipeline

    return Pipeline([("prep", build_preprocessor(X, scale=scale)), ("model", estimator)])


def holdout(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42, stratify: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """A random hold-out split, stratified on `y` when asked and when every class has two rows."""
    sk()
    from sklearn.model_selection import train_test_split

    strat = y if stratify and y.value_counts().min() >= 2 else None
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)


def to_matrix(
    data: Any, columns: Any = None, scale: bool = True, drop: Any = None
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Numeric design matrix for unsupervised tools: chosen (or all numeric) columns, median-imputed and
    optionally standardised. Categorical columns are one-hot encoded when named explicitly.

    Returns the matrix, its column names and the source frame (with its original index/rows).
    """
    sk()
    frame = load_frame(data)
    names = as_list(columns)
    if names:
        require_columns(frame, names)
    else:
        names = [c for c in frame.columns if c not in set(as_list(drop) or [])]
    X = frame[names]
    prep = build_preprocessor(X, scale=scale)
    matrix = prep.fit_transform(X)
    try:
        out_names = [str(n) for n in prep.get_feature_names_out()]
    except Exception:
        out_names = [f"f{i}" for i in range(matrix.shape[1])]
    return np.asarray(matrix, dtype=float), out_names, frame


def save_model(model: Any, path: str | None) -> str | None:
    """Persist a fitted model with joblib; returns the path or None when no path was asked for."""
    if not path:
        return None
    import joblib

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)
    return str(target)


def seed_everything(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


def top_items(mapping: dict[str, float], k: int = 15, by_abs: bool = True) -> dict[str, float]:
    """The `k` largest entries of a name -> score mapping, largest first."""
    key = (lambda item: abs(item[1])) if by_abs else (lambda item: item[1])
    return {name: round(float(score), 6) for name, score in sorted(mapping.items(), key=key, reverse=True)[:k]}


def feature_names(model: Any, X: pd.DataFrame) -> list[str]:
    """Output feature names of a fitted `make_pipeline` pipeline."""
    try:
        return [str(n) for n in model.named_steps["prep"].get_feature_names_out()]
    except Exception:
        return [str(c) for c in X.columns]
