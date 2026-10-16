"""Combine, compare and operationalise detectors: ensembles, labelled evaluation, stability, thresholds, explanations, a review queue, and scoring new rows."""

from __future__ import annotations

from typing import Literal

import joblib
import numpy as np
import pandas as pd

from ml_core.data import load_frame, require_columns, save_frame
from ml_core.tooling import tool
from ml_core.unsup import prepare
from unsupervised.anomaly_detection.tools._common import (
    SCORERS,
    evaluate_scores,
    explain_rows,
    threshold_for,
)

DEFAULT = ["isolation_forest", "robust_zscore", "mahalanobis", "ecod", "knn_distance", "pca_reconstruction", "lof"]


def _truth(P):
    if P.truth is None:
        return None
    t = P.truth
    t = (t == t.max()).astype(int) if set(np.unique(t)) != {0, 1} else t
    return t if 0 < t.sum() < len(t) else None


def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).to_numpy()


def _scores(P, kinds, seed, params=None):
    out = {}
    for k in kinds:
        try:
            out[k] = SCORERS[k](P.Z, P.Z, (params or {}).get(k, {}), seed)[0]
        except Exception as error:
            out[k] = error
    return out


@tool
def ensemble_detectors(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, detectors: list[str] | None = None, contamination: float = 0.05,
                       combine: Literal["rank_average", "max_rank", "vote"] = "rank_average", id_column: str | None = None, top: int = 15, max_rows: int = 8000, seed: int = 42, output_path: str | None = None) -> dict:
    """Combine several detectors: each one's scores become percentile ranks, which are averaged (or maxed, or counted as votes above the contamination cut). Detectors fail in different ways, so an ensemble is rarely the worst and often the best; the agreement count is also a confidence for each flagged row.

    Args:
        truth_column: Column of known anomalies, used only to score each detector and the ensemble.
        detectors: Any of zscore, robust_zscore, iqr, mahalanobis, ecod, hbos, gmm, isolation_forest, lof, one_class_svm, elliptic_envelope, knn_distance, pca_reconstruction, autoencoder, kmeans_distance.
        combine: rank_average, max_rank or vote.
        id_column: Column to show beside the flagged rows.
        top: Rows listed.
    """
    P = prepare(data, columns, drop, True, truth_column, max_rows, seed)
    raw = _scores(P, detectors or DEFAULT, seed)
    ok = {k: v for k, v in raw.items() if not isinstance(v, Exception)}
    ranks = np.column_stack([_rank(v) for v in ok.values()])
    votes = np.column_stack([v >= threshold_for(v, contamination) for v in ok.values()]).sum(axis=1)
    score = ranks.mean(axis=1) if combine == "rank_average" else ranks.max(axis=1) if combine == "max_rank" else votes / len(ok)
    thr = threshold_for(score, contamination)
    order = np.argsort(-score)[:top]
    worst = P.frame.iloc[order].copy()
    worst.insert(0, "row", order)
    worst["ensemble_score"], worst["detectors_flagging"], worst["why"] = score[order], votes[order], explain_rows(P, order)
    if id_column and id_column in worst.columns:
        worst = worst[["row", id_column] + [c for c in worst.columns if c not in ("row", id_column)]]
    out = {"detectors_used": list(ok), "detectors_failed": {k: str(v)[:80] for k, v in raw.items() if isinstance(v, Exception)}, "combine": combine, "flagged": int((score >= thr).sum()), "top_anomalies": worst}
    t = _truth(P)
    if t is not None:
        out["evaluation_against_labels"] = {"ensemble": evaluate_scores(t, score, contamination), **{k: evaluate_scores(t, v, contamination)["roc_auc"] for k, v in ok.items()}}
    pairs = pd.DataFrame(ranks, columns=list(ok)).corr(method="spearman")
    out["mean_agreement_between_detectors"] = round(float((pairs.to_numpy().sum() - len(pairs)) / max(len(pairs) * (len(pairs) - 1), 1)), 3)
    out["written_to"] = save_frame(P.frame.assign(ensemble_score=score, detectors_flagging=votes, is_flagged=(score >= thr).astype(int)), output_path, "anomalies_ensemble.csv") if output_path else None
    return out


@tool
def compare_detectors(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, detectors: list[str] | None = None, contamination: float = 0.05, max_rows: int = 6000, seed: int = 42) -> dict:
    """Run many detectors on the same data and compare them — with a truth column by ROC-AUC, average precision and precision at the true anomaly count; without one by how much their top-k lists overlap (Jaccard) and how strongly their rankings correlate. With labels this is the benchmark; without, it shows which detectors agree and which are idiosyncratic.

    Args:
        truth_column: Column of known anomalies, used only to score the detectors.
        detectors: Detectors to compare; defaults to eight.
    """
    P = prepare(data, columns, drop, True, truth_column, max_rows, seed)
    raw = _scores(P, detectors or DEFAULT + ["hbos"], seed)
    ok = {k: v for k, v in raw.items() if not isinstance(v, Exception)}
    t = _truth(P)
    rows = []
    cut = {k: set(np.argsort(-v)[: max(1, int(contamination * len(v)))]) for k, v in ok.items()}
    for k, v in ok.items():
        row = {"detector": k}
        if t is not None:
            row.update(evaluate_scores(t, v, contamination))
        others = [len(cut[k] & cut[o]) / len(cut[k] | cut[o]) for o in ok if o != k]
        row["mean_topk_overlap_with_others"] = round(float(np.mean(others)), 3) if others else None
        rows.append(row)
    rows.sort(key=lambda r: -(r.get("roc_auc") if t is not None else (r["mean_topk_overlap_with_others"] or 0)))
    return {"ranked_by": "roc_auc (labels)" if t is not None else "agreement with the other detectors (no labels: a high value means typical, not necessarily correct)", "detectors": rows, "failed": {k: str(v)[:80] for k, v in raw.items() if isinstance(v, Exception)}}


@tool
def evaluate_score_column(data: str, score_column: str, truth_column: str, contamination: float | None = None, top_fractions: list[float] | None = None) -> dict:
    """Score any anomaly scores you already have (a column of scores, higher = more anomalous) against known anomalies: ROC-AUC, average precision, precision at the true count, and the recall found in the top 1%, 2%, 5% and 10% of rows — the review-budget view.

    Args:
        score_column: Column holding the anomaly score.
        truth_column: Column marking known anomalies.
        contamination: If given, precision / recall / F1 at that flagged share.
        top_fractions: Review budgets to report recall for.
    """
    frame = load_frame(data)
    require_columns(frame, [score_column, truth_column])
    t = frame[truth_column]
    t = (pd.to_numeric(t, errors="coerce").fillna(0) > 0).astype(int).to_numpy() if pd.api.types.is_numeric_dtype(t) else t.astype(str).isin(["1", "true", "True", "anomaly", "yes"]).astype(int).to_numpy()
    s = frame[score_column].to_numpy(float)
    order = np.argsort(-s)
    out = evaluate_scores(t, s, contamination)
    out["recall_in_top_fraction"] = {f"{f:.0%}": round(float(t[order[: max(1, int(f * len(s)))]].sum() / max(t.sum(), 1)), 4) for f in (top_fractions or [0.01, 0.02, 0.05, 0.1])}
    out["anomalies"] = int(t.sum())
    return out


@tool
def detector_stability(data: str, detector: str = "isolation_forest", columns: list[str] | None = None, drop: list[str] | None = None, contamination: float = 0.05, n_runs: int = 10, sample_share: float = 0.8, max_rows: int = 6000, seed: int = 42) -> dict:
    """How reproducible is a detector's top list? Re-fit on random subsamples with different seeds and measure the overlap (Jaccard) of the flagged sets and the rank correlation of the scores. Randomised detectors (isolation forest) and density detectors (LOF) can flip borderline rows between runs; stable rows are the ones to trust.

    Args:
        detector: Detector name (isolation_forest, lof, knn_distance, one_class_svm, autoencoder, ...).
        n_runs: Re-fits.
        sample_share: Share of rows used to fit each run (scoring is always on all rows).
    """
    from scipy.stats import spearmanr

    P = prepare(data, columns, drop, True, None, max_rows, seed)
    rng = np.random.default_rng(seed)
    runs = []
    for i in range(n_runs):
        idx = rng.choice(len(P.Z), int(len(P.Z) * sample_share), replace=False)
        runs.append(SCORERS[detector](P.Z[idx], P.Z, {}, int(rng.integers(1 << 30)))[0])
    flags = [set(np.flatnonzero(r >= threshold_for(r, contamination))) for r in runs]
    jac = [len(flags[i] & flags[j]) / max(len(flags[i] | flags[j]), 1) for i in range(n_runs) for j in range(i + 1, n_runs)]
    corr = [spearmanr(runs[i], runs[j])[0] for i in range(n_runs) for j in range(i + 1, n_runs)]
    freq = np.zeros(len(P.Z))
    for f in flags:
        freq[list(f)] += 1
    return {"detector": detector, "mean_flagged_set_overlap": round(float(np.mean(jac)), 4), "mean_score_rank_correlation": round(float(np.mean(corr)), 4), "rows_flagged_in_every_run": int((freq == n_runs).sum()),
            "rows_flagged_in_some_runs": int(((freq > 0) & (freq < n_runs)).sum()), "verdict": "stable" if np.mean(jac) > 0.7 else "borderline rows flip between runs: report only rows flagged in most runs"}


@tool
def choose_threshold(data: str, detector: str = "isolation_forest", columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, review_budget: int | None = None, max_rows: int = 8000, seed: int = 42) -> dict:
    """Three ways to turn scores into a flag: a fixed review budget (the top k rows), the biggest gap in the sorted scores, and an extreme-value threshold (a generalised Pareto fitted to the upper tail, so the cut is set where scores stop looking like the bulk). With labels, also the F1-optimal threshold.

    Args:
        detector: Detector whose scores are thresholded.
        truth_column: Column of known anomalies; adds the F1-optimal cut.
        review_budget: Rows someone can review; the top-k cut is reported for it.
    """
    from scipy.stats import genpareto

    P = prepare(data, columns, drop, True, truth_column, max_rows, seed)
    s = SCORERS[detector](P.Z, P.Z, {}, seed)[0]
    srt = np.sort(s)[::-1]
    head = srt[: max(10, len(s) // 10)]
    gap_k = int(np.argmax(-np.diff(head))) + 1
    u = np.quantile(s, 0.9)
    tail = s[s > u] - u
    c, loc, scale = genpareto.fit(tail, floc=0) if len(tail) > 10 else (0, 0, tail.std() or 1)
    evt = float(u + genpareto.ppf(1 - 0.01 / 0.1, c, loc=0, scale=scale)) if len(tail) > 10 else float(srt[0])
    out = {"detector": detector, "largest_gap": {"rows_above": gap_k, "threshold": round(float(srt[gap_k - 1]), 4)}, "extreme_value": {"threshold": round(evt, 4), "rows_above": int((s >= evt).sum())}}
    if review_budget:
        out["review_budget"] = {"threshold": round(float(srt[min(review_budget, len(s)) - 1]), 4), "rows": review_budget}
    t = _truth(P)
    if t is not None:
        from sklearn.metrics import precision_recall_curve

        p, r, th = precision_recall_curve(t, s)
        f1 = 2 * p * r / np.maximum(p + r, 1e-9)
        i = int(np.argmax(f1[:-1]))
        out["f1_optimal"] = {"threshold": round(float(th[i]), 4), "f1": round(float(f1[i]), 4), "precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4), "rows_above": int((s >= th[i]).sum())}
    return out


@tool
def explain_anomalies(data: str, rows: list[int], columns: list[str] | None = None, drop: list[str] | None = None, detector: str = "isolation_forest", k: int = 5, seed: int = 42) -> dict:
    """Explain why chosen rows look anomalous: for each, the columns where it deviates most from the typical value (robust z-scores, with the typical value for comparison) and — for rows flagged jointly rather than by one column — how much a one-column-at-a-time replacement by the median lowers the detector's score (the column whose repair helps most is the culprit).

    Args:
        rows: Row numbers (0-based positions in the data) to explain.
        detector: Detector used for the replacement test.
        k: Columns listed per row.
    """
    P = prepare(data, columns, drop, True, None, None, seed)
    base, _ = SCORERS[detector](P.Z, P.Z[rows], {}, seed) if detector in ("isolation_forest", "lof", "knn_distance", "pca_reconstruction") else SCORERS[detector](P.Z, P.Z[rows], {}, seed)
    num = P.X.select_dtypes("number")
    med = num.median()
    out = []
    for n, r in enumerate(rows):
        row = {"row": int(r), "score": round(float(base[n]), 4), "deviations": explain_rows(P, np.array([r]), k)[0]}
        drops = {}
        for j, c in enumerate(P.X.columns):
            if c in num.columns:
                alt = P.X.iloc[[r]].copy()
                alt[c] = med[c]
                Z2 = np.asarray(P.prep.transform(alt))
                new = SCORERS[detector](P.Z, Z2, {}, seed)[0][0]
                drops[c] = float(base[n] - new)
        top = sorted(drops.items(), key=lambda kv: -kv[1])[:3]
        row["repairing_this_column_lowers_score_most"] = {c: round(v, 4) for c, v in top}
        out.append(row)
    return {"detector": detector, "explanations": out}


@tool
def build_review_queue(data: str, columns: list[str] | None = None, drop: list[str] | None = None, detectors: list[str] | None = None, review_budget: int = 50, id_column: str | None = None, max_rows: int = 20000, seed: int = 42, output_path: str | None = None) -> dict:
    """The operational deliverable: the `review_budget` most anomalous rows by ensemble score, each with its id, how many detectors flagged it, the columns that explain it and the agreement-based confidence — written as a file for whoever investigates.

    Args:
        detectors: Detectors in the ensemble; defaults to seven.
        review_budget: Rows to put in the queue.
        id_column: Column that identifies a row for the reviewer.
    """
    P = prepare(data, columns, drop, True, None, max_rows, seed)
    raw = _scores(P, detectors or DEFAULT, seed)
    ok = {k: v for k, v in raw.items() if not isinstance(v, Exception)}
    ranks = np.column_stack([_rank(v) for v in ok.values()])
    score = ranks.mean(axis=1)
    k = min(review_budget, len(score))
    order = np.argsort(-score)[:k]
    agree = (ranks[order] >= 0.95).sum(axis=1)
    q = P.frame.iloc[order].copy()
    q.insert(0, "row", order)
    q["priority"], q["ensemble_score"], q["detectors_in_top5pct"], q["confidence"], q["why"] = np.arange(1, k + 1), score[order], agree, np.where(agree >= 0.7 * len(ok), "high", np.where(agree >= 0.4 * len(ok), "medium", "low")), explain_rows(P, order)
    if id_column and id_column in q.columns:
        q = q[["priority", "row", id_column] + [c for c in q.columns if c not in ("priority", "row", id_column)]]
    return {"queue_rows": k, "detectors": list(ok), "confidence_counts": pd.Series(q["confidence"]).value_counts().to_dict(), "top": q.head(10), "written_to": save_frame(q, output_path, "anomaly_review_queue.csv")}


@tool
def score_new_rows(model_path: str, data: str, id_column: str | None = None, output_path: str | None = None) -> dict:
    """Score new rows with a saved detector and flag them at the threshold set when it was fitted — for monitoring: the same preprocessing, the same reference data, the same cut.

    Args:
        model_path: Path of the saved detector (joblib).
        id_column: Column to carry through to the output.
    """
    model = joblib.load(model_path)
    frame = load_frame(data)
    require_columns(frame, model.columns)
    s = model.scores(frame)
    out = pd.DataFrame({"anomaly_score": s, "is_flagged": (s >= model.threshold).astype(int)}, index=frame.index)
    if id_column and id_column in frame.columns:
        out.insert(0, id_column, frame[id_column])
    return {"detector": model.kind, "rows": len(out), "flagged": int(out["is_flagged"].sum()), "share": round(float(out["is_flagged"].mean()), 4), "training_contamination": model.contamination, "threshold": round(float(model.threshold), 4),
            "written_to": save_frame(out, output_path, "anomaly_scores_new_rows.csv"), "sample": out.head(5)}
