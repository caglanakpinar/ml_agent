"""Judge an ordinal model: how far off the misses are, decoding rules, cut-point tuning, importance, and scoring new rows."""

from __future__ import annotations

from typing import Literal

import joblib
import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.ml import holdout, sk, top_items
from ml_core.tooling import tool
from supervised.ordinal_regression.tools._common import (
    decode,
    fit_ordinal,
    load_ordinal,
    optimise_cutpoints,
    ordinal_metrics,
)


def _split(data, target, order, features, drop, test_size, seed):
    X, y, levels, how = load_ordinal(data, target, features, drop, order)
    X_tr, X_te, y_tr, y_te = holdout(X, pd.Series(y), test_size, seed, stratify=True)
    return X_tr, X_te, y_tr.to_numpy(), y_te.to_numpy(), levels


@tool
def ordinal_confusion_report(
    data: str, target: str, order: list[str] | None = None, features: list[str] | None = None, drop: list[str] | None = None,
    model: Literal["ordered_logit", "frank_hall", "regression_thresholds", "multiclass"] = "frank_hall", test_size: float = 0.25, seed: int = 42,
) -> dict:
    """Confusion matrix plus how many predictions are off by 0, 1, 2 … levels, and the mean error per true level — shows whether mistakes are near-misses (tolerable) or far-misses (not).

    Args:
        order: The levels from lowest to highest.
        model: Ordinal approach to evaluate.
    """
    sk()
    from sklearn.metrics import confusion_matrix

    X_tr, X_te, y_tr, y_te, levels = _split(data, target, order, features, drop, test_size, seed)
    m = fit_ordinal(model, X_tr, y_tr, levels, seed=seed)
    pred = m.predict_codes(X_te)
    off = np.abs(pred - y_te)
    return {"levels": levels, "confusion_matrix": confusion_matrix(y_te, pred, labels=range(len(levels))).tolist(),
            "share_off_by": {str(k): round(float((off == k).mean()), 4) for k in range(int(off.max()) + 1)},
            "mean_signed_error_by_true_level": {levels[k]: round(float((pred[y_te == k] - k).mean()), 3) for k in range(len(levels)) if (y_te == k).any()},
            "metrics": ordinal_metrics(y_te, pred, len(levels))}


@tool
def compare_decoding_rules(
    data: str, target: str, order: list[str] | None = None, features: list[str] | None = None, drop: list[str] | None = None,
    model: Literal["ordered_logit", "frank_hall", "multiclass"] = "frank_hall", test_size: float = 0.25, seed: int = 42,
) -> dict:
    """Score the same class probabilities decoded three ways — argmax (the most likely level), median (minimises absolute error) and expected value — to pick the rule that matches the cost of being wrong by one level.

    Args:
        order: The levels from lowest to highest.
        model: Probability model to decode.
    """
    X_tr, X_te, y_tr, y_te, levels = _split(data, target, order, features, drop, test_size, seed)
    m = fit_ordinal(model, X_tr, y_tr, levels, seed=seed)
    proba = m.proba(X_te)
    rows = {how: ordinal_metrics(y_te, decode(proba, how), len(levels)) for how in ("argmax", "median", "expected")}
    best = min(rows, key=lambda h: rows[h]["mae"])
    return {"model": model, "by_rule": rows, "lowest_mae": best, "advice": "median decoding is the default for ordinal targets; argmax only wins when the cost of any miss is equal"}


@tool
def optimise_score_cutpoints(data: str, score_column: str, target: str, order: list[str] | None = None) -> dict:
    """Given any continuous score you already have (a regression output, a rating model's raw score), find the cut-points that turn it into levels with the highest quadratic weighted kappa, and compare with naive rounding.

    Args:
        score_column: Column holding the continuous score.
        order: The levels from lowest to highest.
    """
    from supervised.ordinal_regression.tools._common import encode_target

    frame = load_frame(data)
    require_columns(frame, [score_column, target])
    frame = frame.dropna(subset=[score_column, target])
    y, levels, _ = encode_target(frame[target], order)
    s = frame[score_column].to_numpy(float)
    lo, hi = s.min(), s.max()
    scaled = (s - lo) / (hi - lo + 1e-12) * (len(levels) - 1)
    cuts = optimise_cutpoints(scaled, y, len(levels))
    naive = np.clip(np.rint(scaled), 0, len(levels) - 1).astype(int)
    tuned = np.searchsorted(cuts, scaled)
    return {"levels": levels, "cutpoints_on_score_scale": [round(float(lo + c / (len(levels) - 1) * (hi - lo)), 5) for c in cuts],
            "naive_rounding": ordinal_metrics(y, naive, len(levels)), "tuned_cutpoints": ordinal_metrics(y, tuned, len(levels)),
            "note": "these cut-points were tuned on the same rows they are scored on — confirm on held-out data before trusting the gain"}


@tool
def permutation_importance_ordinal(
    data: str, target: str, order: list[str] | None = None, features: list[str] | None = None, drop: list[str] | None = None,
    model: Literal["ordered_logit", "frank_hall", "regression_thresholds", "multiclass"] = "frank_hall", n_repeats: int = 8, k: int = 15, test_size: float = 0.25, seed: int = 42,
) -> dict:
    """Permutation importance on held-out rows, measured as the loss in quadratic weighted kappa when one raw column is shuffled — which features the ordinal predictions really depend on.

    Args:
        order: The levels from lowest to highest.
        model: Ordinal approach to explain.
        n_repeats: Shuffles per column.
        k: How many features to return.
    """
    X_tr, X_te, y_tr, y_te, levels = _split(data, target, order, features, drop, test_size, seed)
    m = fit_ordinal(model, X_tr, y_tr, levels, seed=seed)
    base = ordinal_metrics(y_te, m.predict_codes(X_te), len(levels))["quadratic_weighted_kappa"]
    rng = np.random.default_rng(seed)
    drops = {}
    for c in X_te.columns:
        losses = []
        for _ in range(n_repeats):
            shuffled = X_te.copy()
            shuffled[c] = rng.permutation(shuffled[c].to_numpy())
            losses.append(base - ordinal_metrics(y_te, m.predict_codes(shuffled), len(levels))["quadratic_weighted_kappa"])
        drops[c] = float(np.mean(losses))
    return {"baseline_qwk": base, "importance_qwk_drop": top_items(drops, k), "no_effect": [c for c, v in drops.items() if v <= 0][:20]}


@tool
def predict_ordinal_levels(model_path: str, data: str, id_column: str | None = None, decode_with: Literal["median", "argmax", "expected"] | None = None, output_path: str | None = None) -> dict:
    """Score new rows with a saved ordinal model: the predicted level, the probability of every level (when the model gives them), and the expected level — one output row per input row.

    Args:
        model_path: Path of the saved ordinal model (joblib).
        decode_with: Override the decoding rule the model was saved with.
    """
    model = joblib.load(model_path)
    frame = load_frame(data)
    require_columns(frame, model.columns)
    codes = model.predict_codes(frame, decode_with)
    out = pd.DataFrame({"predicted_level": [model.levels[c] for c in codes]}, index=frame.index)
    if id_column and id_column in frame.columns:
        out.insert(0, id_column, frame[id_column])
    if not hasattr(model.core, "score"):
        proba = model.proba(frame)
        for k, lvl in enumerate(model.levels):
            out[f"p_{lvl}"] = proba[:, k]
        out["expected_level"] = proba @ np.arange(len(model.levels))
    return {"rows": len(out), "model": model.kind, "level_counts": out["predicted_level"].value_counts().to_dict(), "written_to": save_frame(out, output_path, "ordinal_predictions.csv"), "sample": out.head(5)}
