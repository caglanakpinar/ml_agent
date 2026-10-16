"""What the unsupervised topics share: turning a table into a numeric matrix (keeping the fitted preprocessor so new rows can be
encoded the same way), and the optional hold-out labels some experiments carry for *external* validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml_core.data import as_list, load_frame, require_columns
from ml_core.ml import build_preprocessor, sk


@dataclass
class Prepared:
    frame: pd.DataFrame       # the rows as loaded (all columns, original order)
    X: pd.DataFrame           # the feature columns the model sees
    prep: Any                 # fitted ColumnTransformer
    Z: np.ndarray             # encoded, imputed (and standardised) matrix
    names: list[str]          # names of the encoded columns
    columns: list[str]        # names of the original feature columns
    truth: np.ndarray | None  # external labels (never used for fitting), if given

    def encode(self, data: Any) -> np.ndarray:
        frame = load_frame(data)
        require_columns(frame, self.columns)
        Z = self.prep.transform(frame[self.columns])
        return np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z, float)


def prepare(data: Any, columns: Any = None, drop: Any = None, scale: bool = True, truth_column: str | None = None, max_rows: int | None = None, seed: int = 42,
            numeric_only: bool = False) -> Prepared:
    """Load, select features (never the truth column), impute and encode. `max_rows` subsamples very large tables for tools that are quadratic in n."""
    sk()
    frame = load_frame(data)
    if max_rows and len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=seed).reset_index(drop=True)
    skip = set(as_list(drop) or [])
    truth = None
    if truth_column:
        require_columns(frame, [truth_column])
        truth = pd.factorize(frame[truth_column].astype(str))[0]
        skip.add(truth_column)
    names = as_list(columns)
    if names:
        require_columns(frame, names)
    else:
        names = [c for c in frame.columns if c not in skip]
        if numeric_only:
            names = [c for c in names if pd.api.types.is_numeric_dtype(frame[c])]
    names = [c for c in names if c not in skip]
    if not names:
        raise ValueError("no feature columns left after removing `drop` and the truth column")
    X = frame[names].copy()
    prep = build_preprocessor(X, scale=scale)
    Z = prep.fit_transform(X)
    Z = np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z, float)
    try:
        enc = [str(n) for n in prep.get_feature_names_out()]
    except Exception:
        enc = [f"f{i}" for i in range(Z.shape[1])]
    return Prepared(frame, X, prep, Z, enc, names, truth)
