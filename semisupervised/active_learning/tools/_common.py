"""Active-learning machinery: an environment (labelled seed set, unlabelled pool, optional oracle), the query strategies
that rank pool rows by how much a label would teach the model, batch selection, and the simulation loop that
replays 'ask for labels, retrain, repeat' against known answers so strategies can be compared before any money is spent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ml_core.ml import build_preprocessor
from ml_core.semi import base_clf, full_proba, load_semi, metrics


@dataclass
class Env:
    Z: np.ndarray
    y: np.ndarray            # oracle answers (codes) where known, -1 where nobody knows
    classes: list[str]
    labelled: np.ndarray     # row indices with a label the learner may use
    pool: np.ndarray         # row indices a query may pick from
    test: np.ndarray         # held-out rows with known labels (empty when there are none)
    names: list[str]
    seed: int
    simulated: bool

    @property
    def K(self) -> int:
        return len(self.classes)


def build_env(data: Any, label_column: str, features: Any, drop: Any, unlabeled_value: Any, truth_column: str | None, seed: int, test_size: float = 0.3,
              initial_size: int | None = None, simulate: bool = False) -> Env:
    """Real mode: the labelled rows are the seed set, unlabelled rows are the pool, and (if present) `truth_column` is the oracle.
    Simulation mode (`simulate`, or no unlabelled rows): hide labels of a fully labelled dataset — a random class-covering seed of
    `initial_size` rows stays labelled, the rest become the pool and the real labels become the oracle."""
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    sd = load_semi(data, label_column, features, drop, unlabeled_value, truth_column)
    prep = build_preprocessor(sd.X, scale=True)
    Z = np.asarray(prep.fit_transform(sd.X), float)
    y_known = sd.y.copy()
    if sd.truth is not None:
        y_known = np.where(sd.truth >= 0, sd.truth, y_known)
    if simulate or len(sd.unlabeled) == 0:
        full = np.flatnonzero(y_known >= 0)
        strat = y_known[full] if np.bincount(y_known[full]).min() >= 2 else None
        rest, test = train_test_split(full, test_size=test_size, random_state=seed, stratify=strat)
        n0 = initial_size or max(2 * len(sd.classes), 10)
        seed_rows = []
        for k in range(len(sd.classes)):
            ks = rest[y_known[rest] == k]
            seed_rows += list(rng.choice(ks, size=min(max(2, n0 // len(sd.classes)), len(ks)), replace=False))
        labelled = np.asarray(seed_rows, int)
        pool = np.setdiff1d(rest, labelled)
        return Env(Z, y_known, sd.classes, labelled, pool, np.sort(test), list(sd.X.columns), seed, True)
    labelled = sd.labeled
    test = np.array([], int)
    if sd.truth is not None:
        pool_known = sd.unlabeled[sd.truth[sd.unlabeled] >= 0]
        test = pool_known  # truth for pool rows doubles as an evaluation set (they are not trained on until queried)
    return Env(Z, y_known, sd.classes, labelled, sd.unlabeled, np.sort(test), list(sd.X.columns), seed, False)


# ---- uncertainty and strategies -----------------------------------------------------------------
def proba_of(model: Any, Z: np.ndarray, K: int) -> np.ndarray:
    return full_proba(model, Z, K)


def uncertainty(proba: np.ndarray, kind: str) -> np.ndarray:
    srt = np.sort(proba, axis=1)
    if kind == "least_confident":
        return 1 - srt[:, -1]
    if kind == "margin":
        return 1 - (srt[:, -1] - srt[:, -2])
    return -(proba * np.log(proba + 1e-12)).sum(axis=1)  # entropy


def committee(env: Env, base: str, n_members: int, rng: np.random.Generator, lab: np.ndarray | None = None) -> list[Any]:
    """Bagged committee: each member is fitted on a class-stratified bootstrap of the labelled rows."""
    lab = env.labelled if lab is None else lab
    members = []
    for i in range(n_members):
        idx = np.concatenate([rng.choice(lab[env.y[lab] == k], size=int((env.y[lab] == k).sum()), replace=True) for k in range(env.K) if (env.y[lab] == k).any()])
        members.append(base_clf(base, int(rng.integers(1 << 30))).fit(env.Z[idx], env.y[idx]))
    return members


def committee_scores(env: Env, members: list[Any], Zp: np.ndarray, kind: str) -> np.ndarray:
    P = np.stack([proba_of(m, Zp, env.K) for m in members])  # (members, n, K)
    mean = P.mean(axis=0)
    if kind == "vote_entropy":
        votes = np.eye(env.K)[P.argmax(axis=2)].mean(axis=0)
        return -(votes * np.log(votes + 1e-12)).sum(axis=1)
    if kind == "kl":
        return (P * (np.log(P + 1e-12) - np.log(mean + 1e-12))).sum(axis=2).mean(axis=0)
    # BALD: entropy of the mean prediction minus the mean entropy of each member
    return -(mean * np.log(mean + 1e-12)).sum(axis=1) - (-(P * np.log(P + 1e-12)).sum(axis=2)).mean(axis=0)


def density(Zp: np.ndarray, sample: int = 600, seed: int = 0) -> np.ndarray:
    """Mean cosine similarity of each pool row to a sample of the pool — how representative (central) a row is."""
    rng = np.random.default_rng(seed)
    ref = Zp[rng.choice(len(Zp), size=min(sample, len(Zp)), replace=False)]
    a = Zp / np.maximum(np.linalg.norm(Zp, axis=1, keepdims=True), 1e-9)
    b = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), 1e-9)
    return (a @ b.T).mean(axis=1)


def k_center_greedy(Zp: np.ndarray, Zl: np.ndarray, k: int) -> list[int]:
    """Core-set selection: repeatedly pick the pool row farthest from everything already labelled or picked."""
    from sklearn.metrics import pairwise_distances

    d = pairwise_distances(Zp, Zl).min(axis=1) if len(Zl) else np.full(len(Zp), np.inf)
    chosen = []
    for _ in range(min(k, len(Zp))):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(Zp - Zp[i], axis=1))
    return chosen


def diverse_top(scores: np.ndarray, Zp: np.ndarray, k: int, oversample: int = 5, seed: int = 0) -> list[int]:
    """Batch of k: take the top `oversample*k` by score, cluster them, and pick the highest-scoring row of each cluster."""
    from sklearn.cluster import KMeans

    cand = np.argsort(-scores)[: max(k, oversample * k)]
    if len(cand) <= k:
        return list(cand)
    labels = KMeans(k, n_init=3, random_state=seed).fit_predict(Zp[cand])
    return [int(cand[labels == c][np.argmax(scores[cand][labels == c])]) for c in range(k) if (labels == c).any()]


def expected_error_reduction(env: Env, base: str, Zp: np.ndarray, k: int, n_candidates: int, seed: int) -> tuple[list[int], np.ndarray]:
    """For a sample of uncertain candidates: for each possible label, retrain with it and measure the expected log-loss left on the pool."""
    rng = np.random.default_rng(seed)
    model = base_clf(base, seed).fit(env.Z[env.labelled], env.y[env.labelled])
    p0 = proba_of(model, Zp, env.K)
    cand = np.argsort(-uncertainty(p0, "entropy"))[:n_candidates]
    ref = rng.choice(len(Zp), size=min(300, len(Zp)), replace=False)
    scores = np.full(len(Zp), np.inf)
    for i in cand:
        risk = 0.0
        for c in range(env.K):
            m = base_clf(base, seed).fit(np.vstack([env.Z[env.labelled], Zp[i : i + 1]]), np.concatenate([env.y[env.labelled], [c]]))
            pr = proba_of(m, Zp[ref], env.K)
            risk += p0[i, c] * float(-(pr * np.log(pr + 1e-12)).sum(axis=1).mean())
        scores[i] = risk
    return list(np.argsort(scores)[:k]), scores


def select(strategy: str, env: Env, base: str, batch: int, lab: np.ndarray, pool: np.ndarray, rng: np.random.Generator, *, beta: float = 1.0, n_members: int = 5, n_candidates: int = 25) -> tuple[list[int], np.ndarray]:
    """Positions within `pool` to query, and the per-row scores that ranked them (for the queue file)."""
    Zp = env.Z[pool]
    saved, env.labelled = env.labelled, lab
    try:
        if strategy == "random":
            scores = rng.random(len(pool))
            return list(np.argsort(-scores)[:batch]), scores
        if strategy == "core_set":
            picks = k_center_greedy(Zp, env.Z[lab], batch)
            return picks, np.zeros(len(pool))
        if strategy == "expected_error_reduction":
            picks, sc = expected_error_reduction(env, base, Zp, batch, n_candidates, int(rng.integers(1 << 30)))
            return picks, -np.where(np.isfinite(sc), sc, sc[np.isfinite(sc)].max() + 1)
        model = base_clf(base, int(rng.integers(1 << 30))).fit(env.Z[lab], env.y[lab])
        proba = proba_of(model, Zp, env.K)
        if strategy in ("least_confident", "margin", "entropy"):
            scores = uncertainty(proba, strategy)
        elif strategy in ("vote_entropy", "kl", "bald"):
            scores = committee_scores(env, committee(env, base, n_members, rng, lab), Zp, strategy)
        elif strategy == "density_weighted":
            scores = uncertainty(proba, "entropy") * density(Zp, seed=int(rng.integers(1 << 30))) ** beta
        elif strategy == "uncertainty_diversity":
            scores = uncertainty(proba, "entropy")
            return diverse_top(scores, Zp, batch, seed=int(rng.integers(1 << 30))), scores
        else:
            raise ValueError(f"unknown strategy {strategy!r}")
        return list(np.argsort(-scores)[:batch]), scores
    finally:
        env.labelled = saved


STRATEGIES = ("random", "least_confident", "margin", "entropy", "vote_entropy", "kl", "bald", "density_weighted", "core_set", "uncertainty_diversity")


def run_simulation(env: Env, strategy: str, base: str, batch: int, rounds: int, rng: np.random.Generator, **kw: Any) -> list[dict[str, Any]]:
    """Replay active learning: query a batch, reveal the oracle's labels, retrain, record accuracy on the held-out rows."""
    lab, pool = env.labelled.copy(), env.pool.copy()
    curve = []
    for r in range(rounds + 1):
        model = base_clf(base, env.seed).fit(env.Z[lab], env.y[lab])
        m = metrics(env.y[env.test], proba_of(model, env.Z[env.test], env.K))
        curve.append({"round": r, "labels": int(len(lab)), "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})
        if r == rounds or len(pool) == 0:
            break
        picks, _ = select(strategy, env, base, min(batch, len(pool)), lab, pool, rng, **kw)
        lab = np.concatenate([lab, pool[picks]])
        pool = np.delete(pool, picks)
    return curve


def curve_area(curve: list[dict[str, Any]]) -> float:
    """Area under the learning curve (accuracy over labels), normalised by the label range — one number for 'good early'."""
    x = np.array([c["labels"] for c in curve], float)
    y = np.array([c["accuracy"] for c in curve], float)
    return float(np.trapezoid(y, x) / max(x[-1] - x[0], 1)) if len(x) > 1 else float(y[0])
