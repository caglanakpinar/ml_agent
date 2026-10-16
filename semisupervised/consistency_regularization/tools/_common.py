"""Consistency-regularisation machinery for tabular data, in plain numpy so it needs no deep-learning framework.

The idea: a good classifier should give (nearly) the same answer for a row and for a slightly perturbed copy of
it. That holds for unlabelled rows too, so it is a training signal that needs no label:

    supervised loss   cross-entropy on the labelled rows
  + consistency loss  the model's prediction on one perturbed copy of an unlabelled row should match its
                      prediction on another copy (Pi-model), on an exponential-moving-average copy of itself
                      (Mean Teacher), or a confident hard label from a weakly perturbed copy (FixMatch)

The model is a fixed random Fourier-feature map followed by a softmax layer trained with Adam — non-linear
enough to separate real data, small enough to differentiate by hand."""

from __future__ import annotations

from typing import Any

import numpy as np

from ml_core.semi import Context, full_proba, metrics


def augment(Z: np.ndarray, kind: str, strength: float, rng: np.random.Generator) -> np.ndarray:
    """A perturbed copy of the (standardised) rows."""
    n, d = Z.shape
    if kind == "gaussian":
        return Z + rng.normal(scale=strength, size=Z.shape)
    if kind == "dropout":
        return Z * (rng.random(Z.shape) > min(strength, 0.9))
    if kind == "swap_noise":  # replace a share of each column's values with values from other rows
        mask = rng.random(Z.shape) < min(strength, 0.9)
        shuffled = np.take_along_axis(Z, rng.permuted(np.tile(np.arange(n)[:, None], (1, d)), axis=0), axis=0)
        return np.where(mask, shuffled, Z)
    if kind == "mixup":  # pull each row towards a random other row; `strength` scales how far (0 = no change, 1 = full mixup)
        lam = rng.beta(0.4, 0.4, size=(n, 1))
        lam = np.maximum(lam, 1 - lam)
        return Z + min(strength, 1.0) * (1 - lam) * (Z[rng.permutation(n)] - Z)
    if kind == "mixed":
        return augment(augment(Z, "gaussian", strength, rng), "dropout", strength / 2, rng)
    raise ValueError(f"unknown augmentation {kind!r}")


def softmax(a: np.ndarray) -> np.ndarray:
    e = np.exp(a - a.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class RFFNet:
    """Random Fourier features + softmax. Trainable parameters: W (features x classes) and c (classes)."""

    def __init__(self, dim: int, K: int, hidden: int = 256, gamma: float = 0.5, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.R = rng.normal(scale=np.sqrt(2 * gamma), size=(dim, hidden))
        self.b = rng.uniform(0, 2 * np.pi, hidden)
        self.W, self.c = np.zeros((hidden, K)), np.zeros(K)
        self.K, self.hidden = K, hidden
        self.classes_ = np.arange(K)

    def phi(self, Z: np.ndarray) -> np.ndarray:
        return np.sqrt(2.0 / self.hidden) * np.cos(Z @ self.R + self.b)

    def predict_proba(self, Z: np.ndarray, W: np.ndarray | None = None, c: np.ndarray | None = None) -> np.ndarray:
        return softmax(self.phi(Z) @ (self.W if W is None else W) + (self.c if c is None else c))


def train_net(
    ctx: Context, mode: str = "pi", augmentation: str = "gaussian", strength: float = 0.3, epochs: int = 300, lr: float = 0.05, consistency_weight: float = 5.0,
    ema: float = 0.95, threshold: float = 0.9, hidden: int = 256, gamma: float = 0.5, l2: float = 1e-3, monitor_every: int = 25,
) -> tuple[RFFNet, dict[str, Any]]:
    """Train on labelled rows + a consistency term over the unlabelled rows.

    mode:  supervised      cross-entropy on augmented labelled rows only (the augmentation-only control)
           pi              consistency between two perturbed passes of the same network
           mean_teacher    consistency with an exponential-moving-average copy of the network
           fixmatch        a confident prediction on a weak copy is the hard label for a strong copy
    """
    rng = np.random.default_rng(ctx.seed)
    net = RFFNet(ctx.Z.shape[1], ctx.K, hidden, gamma, ctx.seed)
    Zl, yl, Zu = ctx.Zl, ctx.yl, ctx.Zu
    Y = np.eye(ctx.K)[yl]
    teacher_W, teacher_c = net.W.copy(), net.c.copy()
    state = {"mW": np.zeros_like(net.W), "vW": np.zeros_like(net.W), "mc": np.zeros_like(net.c), "vc": np.zeros_like(net.c)}
    Zt, yt = ctx.Z[ctx.split.l_test], ctx.sd.y[ctx.split.l_test]
    history, confident_used = [], 0

    for step in range(1, epochs + 1):
        ramp = np.exp(-5 * (1 - min(step / (0.5 * epochs), 1.0)) ** 2)  # slow ramp-up of the unlabelled term
        Zl_in = augment(Zl, augmentation, strength, rng)
        F = net.phi(Zl_in)
        P = softmax(F @ net.W + net.c)
        gW = F.T @ (P - Y) / len(yl) + l2 * net.W
        gc = (P - Y).mean(axis=0)
        cons = 0.0
        if mode != "supervised" and len(Zu):
            if mode == "fixmatch":
                weak = augment(Zu, "gaussian", strength / 3, rng)
                q = net.predict_proba(weak)
                keep = q.max(axis=1) >= threshold
                confident_used = int(keep.sum())
                if keep.any():
                    strong = augment(Zu[keep], "mixed" if augmentation == "gaussian" else augmentation, strength * 1.5, rng)
                    Fs = net.phi(strong)
                    Ps = softmax(Fs @ net.W + net.c)
                    T = np.eye(ctx.K)[q[keep].argmax(axis=1)]
                    w = consistency_weight * ramp * keep.sum() / len(Zu)
                    gW += w * Fs.T @ (Ps - T) / keep.sum()
                    gc += w * (Ps - T).mean(axis=0)
                    cons = float(-(T * np.log(Ps + 1e-9)).sum(axis=1).mean())
            else:
                a, b = augment(Zu, augmentation, strength, rng), augment(Zu, augmentation, strength, rng)
                Fa = net.phi(a)
                p = softmax(Fa @ net.W + net.c)
                q = softmax(net.phi(b) @ (teacher_W if mode == "mean_teacher" else net.W) + (teacher_c if mode == "mean_teacher" else net.c))
                g = p - q
                dlogits = p * (g - (g * p).sum(axis=1, keepdims=True))  # gradient of 0.5*||p - q||^2 through the softmax
                w = consistency_weight * ramp
                gW += w * Fa.T @ dlogits / len(Zu)
                gc += w * dlogits.mean(axis=0)
                cons = float(0.5 * (g ** 2).sum(axis=1).mean())
        for name, g_, p_ in (("W", gW, net.W), ("c", gc, net.c)):
            state["m" + name] = 0.9 * state["m" + name] + 0.1 * g_
            state["v" + name] = 0.999 * state["v" + name] + 0.001 * g_ ** 2
            p_ -= lr * (state["m" + name] / (1 - 0.9 ** step)) / (np.sqrt(state["v" + name] / (1 - 0.999 ** step)) + 1e-8)
        if mode == "mean_teacher":
            teacher_W, teacher_c = ema * teacher_W + (1 - ema) * net.W, ema * teacher_c + (1 - ema) * net.c
        if step % monitor_every == 0 or step == epochs:
            e = {"step": step, "consistency_loss": round(cons, 5), "holdout_accuracy": metrics(yt, net.predict_proba(Zt))["accuracy"]}
            if mode == "fixmatch":
                e["confident_unlabelled_rows"] = confident_used
            history.append(e)
    if mode == "mean_teacher":
        net.W, net.c = teacher_W, teacher_c  # the averaged weights are the better model
    q = net.predict_proba(Zu) if len(Zu) else np.zeros((0, ctx.K))
    sure = np.flatnonzero(q.max(axis=1) >= threshold)
    return net, {"mode": mode, "augmentation": augmentation, "strength": strength, "trajectory": history,
                 "pseudo": {"index": ctx.split.unl[sure], "label": q[sure].argmax(axis=1), "confidence": q[sure].max(axis=1)},
                 "note": "the pseudo block lists unlabelled rows the final network is confident about; it is an audit view, the network was trained on the consistency loss, not on these labels"}


def prediction_stability(ctx: Context, model: Any, kind: str, strength: float, n_aug: int, rows: str = "unlabelled") -> dict[str, Any]:
    """How often a trained model's prediction survives perturbation: per row, the share of `n_aug` perturbed copies that keep the original predicted class."""
    rng = np.random.default_rng(ctx.seed)
    Z = ctx.Zu if rows == "unlabelled" else ctx.Zl
    base = full_proba(model, Z, ctx.K).argmax(axis=1)
    keep = np.mean([full_proba(model, augment(Z, kind, strength, rng), ctx.K).argmax(axis=1) == base for _ in range(n_aug)], axis=0)
    return {"per_row": keep, "mean_stability": float(keep.mean()), "fully_stable_share": float((keep == 1).mean()), "base_prediction": base}
