"""Pseudo-label generators and the retraining step. A generator looks at the labelled rows and the unlabelled pool and
returns which pool rows to label, with what, and how confidently; `pseudo_trainer` then (optionally) filters them,
retrains on labelled + pseudo-labelled rows and writes the relabelled dataset."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ml_core.data import save_frame
from ml_core.semi import Context, Fitted, base_clf, fit_weighted, full_proba, sk


def teacher_for(ctx: Context, calibrated: bool = False) -> Any:
    model = base_clf(ctx.base, ctx.seed)
    if calibrated and np.bincount(ctx.yl, minlength=ctx.K).min() >= 3:
        from sklearn.calibration import CalibratedClassifierCV

        model = CalibratedClassifierCV(model, cv=3, method="sigmoid")
    return model.fit(ctx.Zl, ctx.yl)


def result(pos: np.ndarray, proba: np.ndarray) -> dict[str, np.ndarray]:
    return {"pos": np.asarray(pos, int), "label": proba[pos].argmax(axis=1), "conf": proba[pos].max(axis=1)}


def gen_threshold(ctx: Context, threshold: float = 0.9, calibrated: bool = False) -> dict[str, np.ndarray]:
    proba = full_proba(teacher_for(ctx, calibrated), ctx.Zu, ctx.K)
    return result(np.flatnonzero(proba.max(axis=1) >= threshold), proba)


def gen_top_per_class(ctx: Context, fraction: float = 0.3, min_conf: float = 0.5) -> dict[str, np.ndarray]:
    proba = full_proba(teacher_for(ctx), ctx.Zu, ctx.K)
    pred, conf = proba.argmax(axis=1), proba.max(axis=1)
    pos = []
    for k in range(ctx.K):
        rows = np.flatnonzero((pred == k) & (conf >= min_conf))
        pos += list(rows[np.argsort(-conf[rows])[: max(1, int(len(rows) * fraction))]]) if len(rows) else []
    return result(np.asarray(pos, int), proba)


def gen_ensemble(ctx: Context, threshold: float = 0.85, bases: list[str] | None = None) -> dict[str, np.ndarray]:
    names = bases or list(dict.fromkeys([ctx.base, "random_forest", "logistic_regression", "hist_gradient_boosting"]))
    probas = np.stack([full_proba(base_clf(n, ctx.seed).fit(ctx.Zl, ctx.yl), ctx.Zu, ctx.K) for n in names])
    votes = probas.argmax(axis=2)
    agree = (votes == votes[0]).all(axis=0)
    mean = probas.mean(axis=0)
    return result(np.flatnonzero(agree & (mean.max(axis=1) >= threshold)), mean)


def gen_cluster_consistent(ctx: Context, threshold: float = 0.7, n_clusters: int | None = None, purity: float = 0.8) -> dict[str, np.ndarray]:
    """Cluster assumption: label a pool row only if the classifier and the labelled majority of its cluster agree."""
    sk()
    from sklearn.cluster import KMeans

    k = n_clusters or max(2 * ctx.K, min(30, len(ctx.yl) // 5))
    allZ = np.vstack([ctx.Zl, ctx.Zu])
    km = KMeans(k, n_init=5, random_state=ctx.seed).fit(allZ)
    cl_l, cl_u = km.labels_[: len(ctx.yl)], km.labels_[len(ctx.yl):]
    majority, pur = {}, {}
    for c in range(k):
        ys = ctx.yl[cl_l == c]
        if len(ys) >= 2:
            counts = np.bincount(ys, minlength=ctx.K)
            majority[c], pur[c] = int(counts.argmax()), counts.max() / counts.sum()
    proba = full_proba(teacher_for(ctx), ctx.Zu, ctx.K)
    keep = [i for i in range(len(ctx.Zu)) if cl_u[i] in majority and pur[cl_u[i]] >= purity and proba[i].argmax() == majority[cl_u[i]] and proba[i].max() >= threshold]
    return result(np.asarray(keep, int), proba)


def gen_soft(ctx: Context, min_prob: float = 0.2) -> dict[str, np.ndarray]:
    proba = full_proba(teacher_for(ctx), ctx.Zu, ctx.K)
    return {"pos": np.arange(len(ctx.Zu)), "label": proba.argmax(axis=1), "conf": proba.max(axis=1), "soft": proba, "min_prob": np.asarray(min_prob)}


def loss_filter(ctx: Context, gen: dict[str, np.ndarray], drop_fraction: float) -> dict[str, np.ndarray]:
    """Small-loss trick: cross-validate on labelled + pseudo-labelled rows and drop the pseudo-labels the model finds most surprising."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    pos, lab = gen["pos"], gen["label"]
    if len(pos) < 10 or drop_fraction <= 0:
        return gen
    Z = np.vstack([ctx.Zl, ctx.Zu[pos]])
    y = np.concatenate([ctx.yl, lab])
    folds = StratifiedKFold(max(2, min(5, int(np.bincount(y, minlength=ctx.K)[np.bincount(y, minlength=ctx.K) > 0].min()))), shuffle=True, random_state=ctx.seed)
    oof = cross_val_predict(base_clf(ctx.base, ctx.seed), Z, y, cv=folds, method="predict_proba")
    classes = np.unique(y)
    p_given = oof[np.arange(len(y)), np.searchsorted(classes, y)][len(ctx.yl):]
    keep = np.argsort(p_given)[int(len(pos) * drop_fraction):]
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(pos) else v) for k, v in gen.items()}


def retrain(ctx: Context, gen: dict[str, np.ndarray], weight_by_confidence: bool = False, base: str | None = None) -> Fitted:
    Zl, yl, w = ctx.Zl, ctx.yl, np.ones(len(ctx.yl))
    pos = gen["pos"]
    if "soft" in gen:  # one replicated row per class the teacher gives enough probability
        soft, min_p = gen["soft"], float(gen["min_prob"])
        rows, labs, ws = [], [], []
        for k in range(ctx.K):
            take = np.flatnonzero(soft[:, k] >= min_p)
            rows.append(ctx.Zu[take]); labs.append(np.full(len(take), k)); ws.append(soft[take, k])
        Zl, yl, w = np.vstack([Zl] + rows), np.concatenate([yl] + labs), np.concatenate([w] + ws)
    elif len(pos):
        Zl = np.vstack([Zl, ctx.Zu[pos]])
        yl = np.concatenate([yl, gen["label"]])
        w = np.concatenate([w, gen["conf"] if weight_by_confidence else np.ones(len(pos))])
    return Fitted(fit_weighted(base_clf(base or ctx.base, ctx.seed), Zl, yl, w), ctx.K)


def write_dataset(ctx: Context, gen: dict[str, np.ndarray], path: str | None, default: str) -> str | None:
    """Every row with its final label, where the label came from, and the pseudo-label's confidence."""
    if path is None:
        return None
    sd = ctx.sd
    frame = sd.X.copy()
    label = np.array([sd.classes[c] if c >= 0 else None for c in sd.y], dtype=object)
    source = np.where(sd.y >= 0, "human", "unlabelled").astype(object)
    conf = np.full(len(sd.y), np.nan)
    pos = ctx.split.unl[gen["pos"]]
    label[pos] = [sd.classes[c] for c in gen["label"]]
    source[pos] = "pseudo"
    conf[pos] = gen["conf"]
    frame[sd.label_column] = label
    frame["label_source"] = source
    frame["pseudo_confidence"] = conf
    return save_frame(frame, path, default)


def pseudo_trainer(generator: Callable[..., dict[str, np.ndarray]], gen_kwargs: dict[str, Any], drop_fraction: float = 0.0, weight_by_confidence: bool = False,
                   output_path: str | None = None, final_base: str | None = None) -> Callable[[Context], tuple[Fitted, dict[str, Any]]]:
    def trainer(ctx: Context) -> tuple[Fitted, dict[str, Any]]:
        gen = generator(ctx, **gen_kwargs)
        before = len(gen["pos"])
        if drop_fraction:
            gen = loss_filter(ctx, gen, drop_fraction)
        model = retrain(ctx, gen, weight_by_confidence, final_base)
        extra: dict[str, Any] = {"candidates_before_filter": before, "kept_after_filter": int(len(gen["pos"])),
                                 "pseudo": {"index": ctx.split.unl[gen["pos"]], "label": gen["label"][: len(gen["pos"])], "confidence": gen["conf"][: len(gen["pos"])]}}
        written = write_dataset(ctx, gen, output_path, "pseudo_labelled.csv")
        if written:
            extra["relabelled_dataset"] = written
        return model, extra

    return trainer


def noisy_student(ctx: Context, threshold: float = 0.8, rounds: int = 3, noise: float = 0.3, student_base: str = "random_forest", output_path: str | None = None) -> tuple[Fitted, dict[str, Any]]:
    """Teacher labels the pool; a (usually larger) student trains on labelled + pseudo rows *with noise*; the student becomes the next teacher."""
    rng = np.random.default_rng(ctx.seed)
    teacher = teacher_for(ctx)
    history = []
    gen = {"pos": np.array([], int), "label": np.array([], int), "conf": np.array([])}
    student = teacher
    for r in range(rounds):
        proba = full_proba(teacher, ctx.Zu, ctx.K)
        gen = result(np.flatnonzero(proba.max(axis=1) >= threshold), proba)
        Zp, yp = ctx.Zu[gen["pos"]], gen["label"]
        Zn = np.vstack([ctx.Zl, Zp])
        yn = np.concatenate([ctx.yl, yp])
        jitter = Zn + rng.normal(scale=noise, size=Zn.shape)
        dropout = Zn * (rng.random(Zn.shape) > 0.1)
        student = base_clf(student_base, ctx.seed).fit(np.vstack([Zn, jitter, dropout]), np.concatenate([yn, yn, yn]))
        history.append({"round": r, "pseudo_labelled": int(len(gen["pos"])), "student": student_base})
        teacher = student
    extra = {"rounds": history, "pseudo": {"index": ctx.split.unl[gen["pos"]], "label": gen["label"], "confidence": gen["conf"]}}
    written = write_dataset(ctx, gen, output_path, "noisy_student.csv")
    if written:
        extra["relabelled_dataset"] = written
    return Fitted(student, ctx.K), extra
