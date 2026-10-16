"""Fit a PU model: Elkan–Noto rescaling, bagging, two-step methods (spy and Rocchio reliable negatives), Lee–Liu weighting and a non-negative PU risk estimator."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ml_core.tooling import tool
from semisupervised.positive_unlabeled_learning.tools._common import (
    ScoreModel,
    clf,
    elkan_noto_c,
    pos_proba,
    run_pu,
)

Base = Literal["logistic_regression", "random_forest", "hist_gradient_boosting", "naive_bayes"]


class Bagged:
    """Average of the positive-vs-sample classifiers of every bag."""

    def __init__(self, models):
        self.models = models

    def scores(self, Z):
        return np.mean([pos_proba(m, Z) for m in self.models], axis=0)


def _run(label, trainer, data, label_column, positive_label, features, drop, truth_column, base, prior, test_size, seed, model_path, truth_positive=None):
    return run_pu(label, trainer, data, label_column, positive_label, features, drop, truth_column, base, prior, test_size, seed, model_path, truth_positive)


@tool
def fit_pu_elkan_noto(
    data: str, label_column: str, positive_label: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "random_forest", prior: float | None = None, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Elkan–Noto: train labelled-vs-unlabelled, then divide the output by c = P(labelled | positive) to turn 'probability of being labelled' into 'probability of being positive'. Needs SCAR and a calibrated classifier; the simplest correct PU method.

    Args:
        base: Classifier for labelled-vs-unlabelled.
        prior: Class prior if known (otherwise estimated).
    """
    def trainer(ctx):
        c = elkan_noto_c(ctx, base)["e1_mean"]
        model = clf(base, seed).fit(np.vstack([ctx.Zp, ctx.Zu]), np.concatenate([np.ones(len(ctx.Zp), int), np.zeros(len(ctx.Zu), int)]))
        return ScoreModel(model, 1.0 / c, "elkan_noto"), {"c_label_frequency": round(c, 4)}

    return _run(f"elkan_noto({base})", trainer, data, label_column, positive_label, features, drop, truth_column, base, prior, test_size, seed, model_path)


@tool
def fit_pu_bagging(
    data: str, label_column: str, positive_label: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "random_forest", n_bags: int = 30, negatives_per_positive: float = 1.0, prior: float | None = None, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """PU bagging (Mordelet & Vert): repeatedly draw a small random sample of the unlabelled rows, treat it as negatives against all labelled positives, and train; the score of an unlabelled row is the average over the bags that did NOT train on it. Because most unlabelled rows are sampled rarely, hidden positives are seldom held against the model.

    Args:
        n_bags: Number of resamples.
        negatives_per_positive: Unlabelled rows sampled per labelled positive in each bag.
    """
    def trainer(ctx):
        rng = np.random.default_rng(seed)
        models, oob = [], np.zeros(len(ctx.Z))
        counts = np.zeros(len(ctx.Z))
        n_neg = max(5, int(len(ctx.p_train) * negatives_per_positive))
        for _ in range(n_bags):
            neg = rng.choice(ctx.unl, size=min(n_neg, len(ctx.unl)), replace=False)
            Z = np.vstack([ctx.Zp, ctx.Z[neg]])
            y = np.concatenate([np.ones(len(ctx.Zp), int), np.zeros(len(neg), int)])
            m = clf(base, seed).fit(Z, y)
            models.append(m)
            held = np.setdiff1d(np.arange(len(ctx.Z)), np.concatenate([ctx.p_train, neg]))
            oob[held] += pos_proba(m, ctx.Z[held])
            counts[held] += 1
        return ScoreModel(Bagged(models), 1.0, "bagging"), {"bags": n_bags, "negatives_per_bag": n_neg, "rows_never_out_of_bag": int((counts == 0).sum())}

    return _run(f"pu_bagging({base}x{n_bags})", trainer, data, label_column, positive_label, features, drop, truth_column, base, prior, test_size, seed, model_path)


@tool
def fit_pu_two_step_spy(
    data: str, label_column: str, positive_label: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", spy_share: float = 0.15, noise_level: float = 0.05, prior: float | None = None, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Two-step PU with the spy technique (Liu et al.): hide a share of the labelled positives inside the unlabelled set as 'spies', train positive-vs-unlabelled, and use how the spies score to set a cut below which unlabelled rows are *reliable negatives*. Step two trains positives against those reliable negatives only.

    Args:
        spy_share: Share of labelled positives planted as spies.
        noise_level: Share of spies allowed below the cut (lower is stricter about what counts as negative).
    """
    def trainer(ctx):
        rng = np.random.default_rng(seed)
        spies = rng.choice(len(ctx.Zp), size=max(3, int(len(ctx.Zp) * spy_share)), replace=False)
        keep = np.setdiff1d(np.arange(len(ctx.Zp)), spies)
        Z1 = np.vstack([ctx.Zp[keep], ctx.Zp[spies], ctx.Zu])
        y1 = np.concatenate([np.ones(len(keep), int), np.zeros(len(spies) + len(ctx.Zu), int)])
        m1 = clf(base, seed).fit(Z1, y1)
        cut = float(np.quantile(pos_proba(m1, ctx.Zp[spies]), noise_level))
        reliable = np.flatnonzero(pos_proba(m1, ctx.Zu) < cut)
        if len(reliable) < 5:
            reliable = np.argsort(pos_proba(m1, ctx.Zu))[: max(5, len(ctx.Zu) // 10)]
        m2 = clf(base, seed).fit(np.vstack([ctx.Zp, ctx.Zu[reliable]]), np.concatenate([np.ones(len(ctx.Zp), int), np.zeros(len(reliable), int)]))
        return ScoreModel(m2, 1.0, "spy"), {"spy_cut": round(cut, 4), "reliable_negatives": int(len(reliable)), "share_of_unlabelled_marked_negative": round(float(len(reliable) / len(ctx.Zu)), 4)}

    return _run(f"two_step_spy({base})", trainer, data, label_column, positive_label, features, drop, truth_column, base, prior, test_size, seed, model_path)


@tool
def fit_pu_two_step_rocchio(
    data: str, label_column: str, positive_label: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", negative_share: float = 0.3, prior: float | None = None, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Two-step PU with Rocchio reliable negatives: the unlabelled rows farthest from the labelled-positive centroid (and closest to the unlabelled centroid) are taken as negatives; step two trains positives against them. Cheap, needs no first-stage classifier, and works well when positives form a compact cluster.

    Args:
        negative_share: Share of the unlabelled rows to take as reliable negatives.
    """
    def trainer(ctx):
        cp, cu = ctx.Zp.mean(axis=0), ctx.Zu.mean(axis=0)
        score = np.linalg.norm(ctx.Zu - cp, axis=1) - np.linalg.norm(ctx.Zu - cu, axis=1)  # large = far from positives, near the unlabelled bulk
        reliable = np.argsort(-score)[: max(5, int(len(ctx.Zu) * negative_share))]
        m = clf(base, seed).fit(np.vstack([ctx.Zp, ctx.Zu[reliable]]), np.concatenate([np.ones(len(ctx.Zp), int), np.zeros(len(reliable), int)]))
        return ScoreModel(m, 1.0, "rocchio"), {"reliable_negatives": int(len(reliable)), "share_of_unlabelled_marked_negative": round(float(len(reliable) / len(ctx.Zu)), 4)}

    return _run(f"two_step_rocchio({base})", trainer, data, label_column, positive_label, features, drop, truth_column, base, prior, test_size, seed, model_path)


@tool
def fit_pu_weighted_logistic(
    data: str, label_column: str, positive_label: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    weights: list[float] | None = None, prior: float | None = None, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Biased (Lee–Liu) weighting: a logistic regression of positives vs unlabelled where each labelled positive is weighted up, with the weight chosen by the Lee–Liu criterion r²/Pr(f=1) — high recall on held-out positives while predicting few unlabelled rows positive — evaluated on labelled positives held out from the weight selection.

    Args:
        weights: Positive-class weights to try; defaults to 1, 2, 5, 10, 20, 50.
    """
    def trainer(ctx):
        rng = np.random.default_rng(seed)
        val = rng.choice(len(ctx.Zp), size=max(3, len(ctx.Zp) // 4), replace=False)
        fit = np.setdiff1d(np.arange(len(ctx.Zp)), val)
        best, rows = None, []
        for w in weights or [1, 2, 5, 10, 20, 50]:
            m = clf("logistic_regression", seed, class_weight={1: float(w), 0: 1.0}).fit(np.vstack([ctx.Zp[fit], ctx.Zu]), np.concatenate([np.ones(len(fit), int), np.zeros(len(ctx.Zu), int)]))
            pu = pos_proba(m, ctx.Zu) >= 0.5
            r = float((pos_proba(m, ctx.Zp[val]) >= 0.5).mean())
            crit = r * r / max(pu.mean(), 1e-6)
            rows.append({"weight": w, "recall_on_validation_positives": round(r, 4), "share_unlabelled_predicted_positive": round(float(pu.mean()), 4), "criterion": round(crit, 4)})
            if best is None or crit > best[0]:
                best = (crit, w)
        final = clf("logistic_regression", seed, class_weight={1: float(best[1]), 0: 1.0}).fit(np.vstack([ctx.Zp, ctx.Zu]), np.concatenate([np.ones(len(ctx.Zp), int), np.zeros(len(ctx.Zu), int)]))
        return ScoreModel(final, 1.0, "weighted"), {"weight_search": rows, "chosen_weight": best[1]}

    return _run("pu_weighted_logistic(lee_liu)", trainer, data, label_column, positive_label, features, drop, truth_column, "logistic_regression", prior, test_size, seed, model_path)


class NNPUNet:
    """Random-feature linear scorer trained with the non-negative PU risk (Kiryo et al.)."""

    def __init__(self, dim: int, hidden: int = 256, gamma: float = 0.3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.R, self.b = rng.normal(scale=np.sqrt(2 * gamma), size=(dim, hidden)), rng.uniform(0, 2 * np.pi, hidden)
        self.w, self.c, self.hidden = np.zeros(hidden), 0.0, hidden

    def phi(self, Z: np.ndarray) -> np.ndarray:
        return np.sqrt(2.0 / self.hidden) * np.cos(Z @ self.R + self.b)

    def scores(self, Z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(self.phi(Z) @ self.w + self.c, -30, 30)))


@tool
def fit_nnpu(
    data: str, label_column: str, positive_label: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    prior: float | None = None, epochs: int = 400, lr: float = 0.05, beta: float = 0.0, gamma_shrink: float = 0.5, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """Non-negative PU learning (nnPU): minimise the unbiased PU risk π·E_P[ℓ(f,+1)] + max(0, E_U[ℓ(f,−1)] − π·E_P[ℓ(f,−1)]), where the clamp stops the flexible model from driving the estimated negative risk below zero (overfitting the positives). Needs the class prior; a numpy random-feature network, no deep-learning framework required.

    Args:
        prior: Class prior (share of all rows that are positive); estimated by Elkan–Noto when omitted — a wrong prior biases every score.
        epochs: Gradient steps.
        beta: Tolerance below zero before the correction step kicks in.
        gamma_shrink: Step shrinkage applied when the negative risk goes below −beta.
    """
    def trainer(ctx):
        net = NNPUNet(ctx.Z.shape[1], seed=seed)
        Fp, Fu = net.phi(ctx.Zp), net.phi(ctx.Zu)
        pi = float(ctx.prior)
        m1, v1 = np.zeros(net.hidden + 1), np.zeros(net.hidden + 1)
        theta = np.zeros(net.hidden + 1)
        sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))  # noqa: E731
        log_trace = []
        for step in range(1, epochs + 1):
            fp, fu = Fp @ theta[:-1] + theta[-1], Fu @ theta[:-1] + theta[-1]
            # logistic loss: l(f, +1) = log(1+e^-f), l(f, -1) = log(1+e^f); d/df = -sig(-f), sig(f)
            risk_p_pos = np.mean(np.logaddexp(0, -fp))
            risk_p_neg = np.mean(np.logaddexp(0, fp))
            risk_u_neg = np.mean(np.logaddexp(0, fu))
            neg = risk_u_neg - pi * risk_p_neg
            gp_pos = np.append(Fp.T @ (-sig(-fp)) / len(fp), (-sig(-fp)).mean())
            gp_neg = np.append(Fp.T @ sig(fp) / len(fp), sig(fp).mean())
            gu_neg = np.append(Fu.T @ sig(fu) / len(fu), sig(fu).mean())
            grad_neg = gu_neg - pi * gp_neg
            if neg >= -beta:
                grad = pi * gp_pos + grad_neg
            else:  # Kiryo et al.: ascend on the negative part so the empirical negative risk is pushed back to >= 0
                grad = -gamma_shrink * grad_neg
            grad = grad + 1e-3 * np.append(theta[:-1], 0)
            m1 = 0.9 * m1 + 0.1 * grad
            v1 = 0.999 * v1 + 0.001 * grad**2
            theta -= lr * (m1 / (1 - 0.9**step)) / (np.sqrt(v1 / (1 - 0.999**step)) + 1e-8)
            if step % 100 == 0 or step == epochs:
                log_trace.append({"step": step, "pu_risk": round(float(pi * risk_p_pos + max(neg, 0.0)), 4), "negative_part": round(float(neg), 4)})
        net.w, net.c = theta[:-1], float(theta[-1])
        return ScoreModel(net, 1.0, "nnpu"), {"training_trace": log_trace, "prior_used": round(pi, 4)}

    return _run("nnpu(random_features)", trainer, data, label_column, positive_label, features, drop, truth_column, "logistic_regression", prior, test_size, seed, model_path)


@tool
def fit_pu_expectation_maximisation(
    data: str, label_column: str, positive_label: str | None = None, features: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None,
    base: Base = "logistic_regression", rounds: int = 10, prior: float | None = None, test_size: float = 0.3, seed: int = 42, model_path: str | None = None,
) -> dict:
    """EM-style soft relabelling: start from the Elkan–Noto probabilities for the unlabelled rows, then repeat — fit a classifier on labelled positives (weight 1) plus every unlabelled row twice (as positive with weight p and negative with weight 1−p), and update p from the new model. The class prior is held fixed by rescaling p each round, so the estimate cannot drift to all-negative.

    Args:
        rounds: EM iterations.
    """
    def trainer(ctx):
        c = elkan_noto_c(ctx, base)["e1_mean"]
        first = clf(base, seed).fit(np.vstack([ctx.Zp, ctx.Zu]), np.concatenate([np.ones(len(ctx.Zp), int), np.zeros(len(ctx.Zu), int)]))
        p = np.clip(pos_proba(first, ctx.Zu) / c, 0.01, 0.99)
        target = max(float(ctx.prior) * len(ctx.Z) - len(ctx.Zp), 1.0)
        model, trace = first, []
        for r in range(rounds):
            p = np.clip(p * (target / max(p.sum(), 1e-6)), 0.01, 0.99)  # keep the expected number of hidden positives at the prior
            Z = np.vstack([ctx.Zp, ctx.Zu, ctx.Zu])
            y = np.concatenate([np.ones(len(ctx.Zp), int), np.ones(len(ctx.Zu), int), np.zeros(len(ctx.Zu), int)])
            w = np.concatenate([np.ones(len(ctx.Zp)), p, 1 - p])
            model = clf(base, seed)
            try:
                model.fit(Z, y, sample_weight=w)
            except TypeError:
                model.fit(Z, y)
            new_p = np.clip(pos_proba(model, ctx.Zu), 0.01, 0.99)
            trace.append({"round": r, "mean_p_unlabelled": round(float(new_p.mean()), 4), "change": round(float(np.abs(new_p - p).mean()), 5)})
            p = new_p
        return ScoreModel(model, 1.0, "em"), {"em_trace": trace}

    return _run(f"pu_em({base})", trainer, data, label_column, positive_label, features, drop, truth_column, base, prior, test_size, seed, model_path)
