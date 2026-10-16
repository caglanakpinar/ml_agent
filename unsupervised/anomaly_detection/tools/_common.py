"""Anomaly-detection machinery the tool modules share. Every detector reduces to one function: rows in, an anomaly score out
(higher = more anomalous). That is what lets ten different algorithms be thresholded, compared, ensembled, explained and saved the same way."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_core.data import save_frame
from ml_core.ml import save_model
from ml_core.unsup import Prepared, prepare


# ---- scoring functions: (Z_fit, Z_score, params, seed) -> scores, fitted object ------------------
def s_zscore(Zf: np.ndarray, Zs: np.ndarray, p: dict, seed: int) -> tuple[np.ndarray, Any]:
    mu, sd = Zf.mean(axis=0), np.where(Zf.std(axis=0) > 0, Zf.std(axis=0), 1)
    return np.abs((Zs - mu) / sd).max(axis=1), (mu, sd)


def s_robust_zscore(Zf: np.ndarray, Zs: np.ndarray, p: dict, seed: int) -> tuple[np.ndarray, Any]:
    med = np.median(Zf, axis=0)
    mad = np.median(np.abs(Zf - med), axis=0)
    mad = np.where(mad > 0, mad, np.where(Zf.std(axis=0) > 0, Zf.std(axis=0) / 1.4826, 1))
    return np.abs(0.6745 * (Zs - med) / mad).max(axis=1), (med, mad)


def s_iqr(Zf: np.ndarray, Zs: np.ndarray, p: dict, seed: int) -> tuple[np.ndarray, Any]:
    q1, q3 = np.quantile(Zf, 0.25, axis=0), np.quantile(Zf, 0.75, axis=0)
    iqr = np.where(q3 - q1 > 0, q3 - q1, 1)
    over = np.maximum((q1 - Zs) / iqr, (Zs - q3) / iqr)  # distance beyond the box, in IQRs
    return np.maximum(over, 0).max(axis=1) + 1e-6 * np.abs(Zs).max(axis=1), (q1, q3, iqr)


def s_mahalanobis(Zf: np.ndarray, Zs: np.ndarray, p: dict, seed: int) -> tuple[np.ndarray, Any]:
    if p.get("robust", True):
        from sklearn.covariance import MinCovDet

        import warnings

        with warnings.catch_warnings():  # MCD warns when its determinant stalls on tied data; the estimate is still usable
            warnings.simplefilter("ignore", RuntimeWarning)
            est = MinCovDet(random_state=seed, support_fraction=p.get("support_fraction")).fit(Zf)
        return np.sqrt(est.mahalanobis(Zs)), est
    from sklearn.covariance import EmpiricalCovariance

    est = EmpiricalCovariance().fit(Zf)
    return np.sqrt(est.mahalanobis(Zs)), est


def s_ecod(Zf: np.ndarray, Zs: np.ndarray, p: dict, seed: int) -> tuple[np.ndarray, Any]:
    """ECOD (Li et al.): per-column empirical tail probabilities, summed on the log scale — parameter-free, fast, and handles skew."""
    n = len(Zf)
    srt = np.sort(Zf, axis=0)
    left = np.stack([np.searchsorted(srt[:, j], Zs[:, j], side="right") / n for j in range(Zf.shape[1])], axis=1)
    right = np.stack([1 - np.searchsorted(srt[:, j], Zs[:, j], side="left") / n for j in range(Zf.shape[1])], axis=1)
    skew = np.sign(((Zf - Zf.mean(axis=0)) ** 3).mean(axis=0))
    tail = np.where(skew > 0, right, np.where(skew < 0, left, np.minimum(left, right)))
    o = -np.log(np.clip(np.minimum(left, right), 1.0 / (n + 1), 1)).sum(axis=1)
    o_skew = -np.log(np.clip(tail, 1.0 / (n + 1), 1)).sum(axis=1)
    return np.maximum(o, o_skew), srt


def s_hbos(Zf: np.ndarray, Zs: np.ndarray, p: dict, seed: int) -> tuple[np.ndarray, Any]:
    """Histogram-based outlier score: a histogram per column, score = sum of −log(bin density). Very fast; assumes independent columns."""
    bins = p.get("bins") or int(np.sqrt(len(Zf)))
    total = np.zeros(len(Zs))
    hists = []
    for j in range(Zf.shape[1]):
        h, edges = np.histogram(Zf[:, j], bins=bins)
        dens = (h + 1e-3) / (h.sum() + 1e-3 * len(h))
        idx = np.clip(np.searchsorted(edges, Zs[:, j], side="right") - 1, 0, len(h) - 1)
        outside = (Zs[:, j] < edges[0]) | (Zs[:, j] > edges[-1])
        total += -np.log(np.where(outside, 1e-4, dens[idx]))
        hists.append((edges, dens))
    return total, hists


def s_gmm(Zf: np.ndarray, Zs: np.ndarray, p: dict, seed: int) -> tuple[np.ndarray, Any]:
    from sklearn.mixture import GaussianMixture

    gm = GaussianMixture(p.get("n_components", 3), covariance_type=p.get("covariance_type", "full"), n_init=2, random_state=seed).fit(Zf)
    return -gm.score_samples(Zs), gm


def s_iforest(Zf: np.ndarray, Zs: np.ndarray, p: dict, seed: int) -> tuple[np.ndarray, Any]:
    from sklearn.ensemble import IsolationForest

    m = IsolationForest(n_estimators=p.get("n_estimators", 300), max_samples=p.get("max_samples", "auto"), contamination="auto", random_state=seed, n_jobs=-1).fit(Zf)
    return -m.score_samples(Zs), m


def s_lof(Zf: np.ndarray, Zs: np.ndarray, p: dict, seed: int) -> tuple[np.ndarray, Any]:
    from sklearn.neighbors import LocalOutlierFactor

    m = LocalOutlierFactor(n_neighbors=min(p.get("n_neighbors", 20), len(Zf) - 1), novelty=True).fit(Zf)
    return -m.score_samples(Zs), m


def s_ocsvm(Zf: np.ndarray, Zs: np.ndarray, p: dict, seed: int) -> tuple[np.ndarray, Any]:
    from sklearn.svm import OneClassSVM

    m = OneClassSVM(kernel="rbf", gamma=p.get("gamma", "scale"), nu=p.get("nu", 0.05)).fit(Zf)
    return -m.score_samples(Zs), m


def s_elliptic(Zf: np.ndarray, Zs: np.ndarray, p: dict, seed: int) -> tuple[np.ndarray, Any]:
    from sklearn.covariance import EllipticEnvelope

    m = EllipticEnvelope(contamination=min(max(p.get("contamination", 0.05), 0.001), 0.5), random_state=seed).fit(Zf)
    return -m.score_samples(Zs), m


def s_knn(Zf: np.ndarray, Zs: np.ndarray, p: dict, seed: int) -> tuple[np.ndarray, Any]:
    from sklearn.neighbors import NearestNeighbors

    k = min(p.get("n_neighbors", 10), len(Zf) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Zf)
    d = nn.kneighbors(Zs, n_neighbors=k + 1)[0]
    same = np.allclose(d[:, 0], 0)  # scoring the training rows themselves: skip the self-distance
    d = d[:, 1:] if (same and len(Zs) == len(Zf)) else d[:, :k]
    return (d.mean(axis=1) if p.get("aggregate", "mean") == "mean" else d[:, -1]), nn


def s_pca(Zf: np.ndarray, Zs: np.ndarray, p: dict, seed: int) -> tuple[np.ndarray, Any]:
    from sklearn.decomposition import PCA

    n = p.get("n_components") or max(1, min(Zf.shape[1] - 1, int(np.searchsorted(np.cumsum(PCA().fit(Zf).explained_variance_ratio_), p.get("variance", 0.9)) + 1)))
    m = PCA(n_components=n, random_state=seed).fit(Zf)
    return np.linalg.norm(Zs - m.inverse_transform(m.transform(Zs)), axis=1), m


def s_autoencoder(Zf: np.ndarray, Zs: np.ndarray, p: dict, seed: int) -> tuple[np.ndarray, Any]:
    from sklearn.neural_network import MLPRegressor

    hidden = p.get("hidden") or (max(2, Zf.shape[1] // 2), max(1, Zf.shape[1] // 4), max(2, Zf.shape[1] // 2))
    m = MLPRegressor(hidden_layer_sizes=tuple(hidden), activation="tanh", max_iter=p.get("max_iter", 600), random_state=seed, early_stopping=False).fit(Zf, Zf)
    return np.linalg.norm(Zs - m.predict(Zs), axis=1), m


def s_kmeans(Zf: np.ndarray, Zs: np.ndarray, p: dict, seed: int) -> tuple[np.ndarray, Any]:
    from sklearn.cluster import KMeans

    km = KMeans(p.get("n_clusters", 5), n_init=5, random_state=seed).fit(Zf)
    d_fit = np.linalg.norm(Zf - km.cluster_centers_[km.labels_], axis=1)
    scale = np.array([d_fit[km.labels_ == c].std() or 1 for c in range(km.n_clusters)])
    lab = km.predict(Zs)
    return np.linalg.norm(Zs - km.cluster_centers_[lab], axis=1) / scale[lab], km


SCORERS = {"zscore": s_zscore, "robust_zscore": s_robust_zscore, "iqr": s_iqr, "mahalanobis": s_mahalanobis, "ecod": s_ecod, "hbos": s_hbos, "gmm": s_gmm, "isolation_forest": s_iforest,
           "lof": s_lof, "one_class_svm": s_ocsvm, "elliptic_envelope": s_elliptic, "knn_distance": s_knn, "pca_reconstruction": s_pca, "autoencoder": s_autoencoder, "kmeans_distance": s_kmeans}


def threshold_for(scores: np.ndarray, contamination: float) -> float:
    return float(np.quantile(scores, 1 - min(max(contamination, 1e-4), 0.5)))


def evaluate_scores(truth: np.ndarray, scores: np.ndarray, contamination: float | None = None) -> dict[str, float]:
    from sklearn import metrics as m

    out = {"roc_auc": round(float(m.roc_auc_score(truth, scores)), 4), "average_precision": round(float(m.average_precision_score(truth, scores)), 4)}
    k = int(truth.sum())
    top = np.argsort(-scores)[:k]
    out["precision_at_k_true_anomalies"] = round(float(truth[top].mean()), 4)
    if contamination:
        pred = scores >= threshold_for(scores, contamination)
        out.update({"precision_at_contamination": round(float(m.precision_score(truth, pred, zero_division=0)), 4), "recall_at_contamination": round(float(m.recall_score(truth, pred, zero_division=0)), 4), "f1_at_contamination": round(float(m.f1_score(truth, pred, zero_division=0)), 4)})
    return out


def explain_rows(P: Prepared, idx: np.ndarray, k: int = 3) -> list[str]:
    """For each row, the numeric columns it deviates from the typical value most (robust z-scores) — a first answer to 'why was this flagged?'."""
    num = P.X.select_dtypes("number")
    med, mad = num.median(), (num - num.median()).abs().median().replace(0, np.nan)
    mad = mad.fillna(num.std().replace(0, 1))
    out = []
    for i in idx:
        z = (0.6745 * (num.iloc[i] - med) / mad).dropna()
        top = z.reindex(z.abs().sort_values(ascending=False).index).head(k)
        out.append("; ".join(f"{c} = {num.iloc[i][c]:.3g} ({v:+.1f}σ)" for c, v in top.items()))
    return out


class AnomalyModel:
    """Preprocessor + a way to score new rows. Detectors with no out-of-sample scoring refit on the training matrix."""

    def __init__(self, kind: str, prep: Any, columns: list[str], Zfit: np.ndarray, params: dict, seed: int, threshold: float, contamination: float):
        self.kind, self.prep, self.columns, self.Zfit, self.params, self.seed, self.threshold, self.contamination = kind, prep, columns, Zfit, params, seed, threshold, contamination

    def scores(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.prep.transform(X[self.columns])
        Z = np.asarray(Z.toarray() if hasattr(Z, "toarray") else Z, float)
        return SCORERS[self.kind](self.Zfit, Z, self.params, self.seed)[0]


def run_detector(kind: str, label: str, data: Any, columns: Any, drop: Any, scale: bool, truth_column: str | None, contamination: float, params: dict, seed: int, output_path: str | None,
                 model_path: str | None, id_column: str | None = None, top: int = 15, max_rows: int | None = None) -> dict[str, Any]:
    """Fit on all rows (unsupervised), score every row, flag the top `contamination` share, explain the worst, and score against labels when they exist."""
    P = prepare(data, columns, drop, scale, truth_column, max_rows, seed)
    scores, _ = SCORERS[kind](P.Z, P.Z, params, seed)
    thr = threshold_for(scores, contamination)
    flagged = scores >= thr
    order = np.argsort(-scores)[:top]
    worst = P.frame.iloc[order].copy()
    worst.insert(0, "row", order)
    worst["anomaly_score"] = scores[order]
    worst["why"] = explain_rows(P, order)
    if id_column and id_column in worst.columns:
        worst = worst[["row", id_column] + [c for c in worst.columns if c not in ("row", id_column)]]
    out: dict[str, Any] = {"detector": label, "rows": int(len(scores)), "contamination_assumed": contamination, "threshold": round(thr, 4), "flagged": int(flagged.sum()),
                           "score_quantiles": {q: round(float(np.quantile(scores, q)), 4) for q in (0.5, 0.9, 0.99)}, "top_anomalies": worst}
    if P.truth is not None:
        t = (P.truth == (pd.Series(P.truth).max())).astype(int) if set(np.unique(P.truth)) != {0, 1} else P.truth
        if 0 < t.sum() < len(t):
            out["evaluation_against_labels"] = evaluate_scores(t, scores, contamination)
    table = P.frame.assign(anomaly_score=scores, is_flagged=flagged.astype(int))
    out["written_to"] = save_frame(table, output_path, f"anomalies_{kind}.csv") if output_path else None
    out["model_path"] = save_model(AnomalyModel(kind, P.prep, P.columns, P.Z, params, seed, thr, contamination), model_path)
    return out
