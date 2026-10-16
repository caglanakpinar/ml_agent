"""Read-only checks before anomaly detection: the data, a first estimate of how contaminated it is, which detector families suit its shape, and what labels (if any) exist."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_core.data import load_frame
from ml_core.ml import numeric_and_categorical
from ml_core.tooling import tool
from ml_core.unsup import prepare
from unsupervised.anomaly_detection.tools._common import SCORERS


@tool
def profile_anomaly_data(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None) -> dict:
    """Profile a table for anomaly detection: rows, numeric vs categorical columns, missing values, duplicate rows, scale and skew of each column, the share of IQR outliers per column, and whether a column of known anomalies exists (and its rate)."""
    frame = load_frame(data)
    skip = set(drop or []) | ({truth_column} if truth_column else set())
    names = [c for c in (columns or frame.columns) if c not in skip]
    X = frame[names]
    num, cat = numeric_and_categorical(X)
    q1, q3 = X[num].quantile(.25), X[num].quantile(.75)
    iqr = (q3 - q1).replace(0, np.nan)
    outlier_share = (((X[num] < q1 - 1.5 * iqr) | (X[num] > q3 + 1.5 * iqr)).mean()).round(4)
    out = {"rows": len(frame), "numeric": num, "categorical": cat, "columns_with_missing": {c: round(float(X[c].isna().mean()), 4) for c in names if X[c].isna().any()}, "duplicate_rows": int(frame.duplicated().sum()),
           "skewed_columns": [c for c in num if abs(X[c].skew()) > 2], "iqr_outlier_share_by_column": {c: float(v) for c, v in outlier_share.sort_values(ascending=False).head(8).items()}}
    if truth_column:
        t = frame[truth_column]
        out["labelled_anomaly_rate"] = round(float((t.astype(str).isin(["1", "True", "true", "anomaly", "yes"]) if not pd.api.types.is_numeric_dtype(t) else t > 0).mean()), 4)
    return out


@tool
def estimate_contamination(data: str, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, seed: int = 42) -> dict:
    """Estimate what share of the rows are anomalous, three independent ways: the share beyond the IQR fences of the robust Mahalanobis distance, the point where the isolation-forest score distribution separates (largest gap in the sorted tail), and a two-component mixture fitted to the log scores. The contamination is the one setting every detector needs; agreement between the estimates is the evidence.

    Args:
        truth_column: Column of known anomalies, only used to compare the estimates with.
    """
    from sklearn.mixture import GaussianMixture

    P = prepare(data, columns, drop, True, truth_column, 20000, seed)
    d, _ = SCORERS["mahalanobis"](P.Z, P.Z, {"robust": True}, seed)
    q1, q3 = np.quantile(d, [0.25, 0.75])
    est_iqr = float((d > q3 + 1.5 * (q3 - q1)).mean())
    iso, _ = SCORERS["isolation_forest"](P.Z, P.Z, {}, seed)
    srt = np.sort(iso)[::-1]
    head = srt[: max(5, len(srt) // 5)]
    gap = int(np.argmax(-np.diff(head))) + 1
    est_gap = gap / len(srt)
    ls = np.log(np.maximum(iso - iso.min() + 1e-6, 1e-6)).reshape(-1, 1)
    gm = GaussianMixture(2, random_state=seed).fit(ls)
    hi = int(np.argmax(gm.means_.ravel()))
    est_gmm = float(gm.weights_[hi])
    out = {"iqr_on_robust_mahalanobis": round(est_iqr, 4), "isolation_score_gap": round(est_gap, 4), "two_component_mixture": round(est_gmm, 4),
           "suggested_contamination": round(float(np.median([est_iqr, est_gap, min(est_gmm, 0.3)])), 4),
           "reading": "if the estimates differ by more than a factor of 3 the data has no clean anomaly/normal split: choose a review budget (top-k) instead of a contamination share"}
    if P.truth is not None:
        t = P.truth
        out["labelled_rate"] = round(float((t == t.max()).mean() if set(np.unique(t)) != {0, 1} else t.mean()), 4)
    return out


@tool
def recommend_detector_family(data: str, columns: list[str] | None = None, drop: list[str] | None = None, seed: int = 42) -> dict:
    """Look at the shape of the data and say which detector families fit it: correlated numeric columns (Mahalanobis, PCA reconstruction), many columns (isolation forest), several dense groups of different density (LOF, k-means distance), heavy tails and skew (ECOD, robust z-score, HBOS), mixed types (isolation forest after encoding) — with the evidence for each recommendation.

    """
    from sklearn.decomposition import PCA

    P = prepare(data, columns, drop, True, None, 5000, seed)
    num = P.X.select_dtypes("number")
    corr = num.corr().abs().to_numpy()
    mean_corr = float((corr.sum() - len(corr)) / max(len(corr) * (len(corr) - 1), 1)) if len(corr) > 1 else 0.0
    ev = PCA().fit(P.Z).explained_variance_ratio_
    top2 = float(ev[:2].sum())
    skew = float(num.skew().abs().mean()) if len(num.columns) else 0.0
    recs = []
    if mean_corr > 0.3 or top2 > 0.7:
        recs.append({"family": "mahalanobis / pca_reconstruction / elliptic_envelope", "because": f"columns are correlated (mean |r| = {mean_corr:.2f}); anomalies often break the correlation, which univariate rules cannot see"})
    if P.Z.shape[1] > 10:
        recs.append({"family": "isolation_forest / ecod / hbos", "because": f"{P.Z.shape[1]} encoded columns: distance-based methods degrade, tree isolation and per-column methods do not"})
    if skew > 1:
        recs.append({"family": "ecod / robust_zscore / hbos", "because": f"heavy skew (mean |skew| = {skew:.1f}): Gaussian assumptions over-flag the long tail"})
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    sil = float(silhouette_score(P.Z, KMeans(3, n_init=3, random_state=seed).fit_predict(P.Z), sample_size=min(2000, len(P.Z)), random_state=seed))
    if sil > 0.3:
        recs.append({"family": "lof / kmeans_distance / knn_distance", "because": f"the data forms groups (silhouette {sil:.2f}): anomalies are rows far from *their own* group, which a global method misses"})
    recs.append({"family": "isolation_forest", "because": "the safe general-purpose default — no distributional assumption, scales, handles mixed scales"})
    return {"mean_abs_correlation": round(mean_corr, 3), "variance_in_first_two_components": round(top2, 3), "mean_abs_skew": round(skew, 2), "cluster_silhouette": round(sil, 3), "recommendations": recs}


@tool
def check_label_availability(data: str, truth_column: str, columns: list[str] | None = None, drop: list[str] | None = None, seed: int = 42) -> dict:
    """If some rows are known anomalies, are there enough to evaluate with, and can a supervised model already separate them? Counts of labelled anomalies, and a cross-validated AUC of a random forest — a high AUC means a supervised classifier (see `classification`) may beat any unsupervised detector.

    Args:
        truth_column: Column marking known anomalies (1 / true / label).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score

    P = prepare(data, columns, drop, True, truth_column, 20000, seed)
    t = P.truth
    t = (t == t.max()).astype(int) if set(np.unique(t)) != {0, 1} else t
    n = int(t.sum())
    out = {"labelled_anomalies": n, "rows": int(len(t)), "rate": round(float(t.mean()), 4)}
    if n < 10:
        out["verdict"] = "fewer than 10 labelled anomalies: too few to evaluate or train on — rely on unsupervised detectors and review the top rows by hand"
        return out
    p = cross_val_predict(RandomForestClassifier(200, class_weight="balanced", min_samples_leaf=2, random_state=seed, n_jobs=-1), P.Z, t, cv=StratifiedKFold(min(5, n), shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    out["supervised_cv_auc"] = round(float(roc_auc_score(t, p)), 4)
    out["verdict"] = "labels are informative and numerous: train a classifier (supervised/classification) and use the unsupervised detectors as extra features" if out["supervised_cv_auc"] > 0.9 and n >= 50 else "usable for evaluating unsupervised detectors; too few for a dependable supervised model"
    return out
