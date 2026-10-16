"""Learned representations of table rows: autoencoders, contrastive learning, random features, sparse coding, RBMs, k-means and tree embeddings, and (with labels) metric learning."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ml_core.tooling import tool
from ml_core.unsup import prepare
from unsupervised.representation_learning.tools._common import finish


class LayerEncoder:
    """The encoder half of a trained scikit-learn MLP autoencoder (its first layers' weights), usable on new rows."""

    def __init__(self, coefs: list[np.ndarray], intercepts: list[np.ndarray], activation: str):
        self.coefs, self.intercepts, self.act = coefs, intercepts, activation

    def transform(self, Z: np.ndarray) -> np.ndarray:
        h = Z
        for W, b in zip(self.coefs, self.intercepts):
            h = h @ W + b
            h = np.tanh(h) if self.act == "tanh" else np.maximum(h, 0) if self.act == "relu" else h
        return h


def _prep(data, columns, drop, scale, truth_column, max_rows, seed):
    return prepare(data, columns, drop, scale, truth_column, max_rows, seed)


@tool
def embed_autoencoder(data: str, n_components: int = 8, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, hidden_layers: str | None = None, noise: float = 0.0,
                      activation: Literal["tanh", "relu"] = "tanh", max_iter: int = 400, keep_columns: list[str] | None = None, max_rows: int = 30000, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Autoencoder: a network trained to reproduce its input through a narrow bottleneck; the bottleneck activations are the learned representation — a non-linear, trainable generalisation of PCA. With `noise` > 0 it becomes a *denoising* autoencoder, trained to recover clean rows from corrupted ones, which forces it to learn structure rather than copy.

    Args:
        n_components: Width of the bottleneck (the embedding size).
        truth_column: Column of known labels, kept aside to *probe* the embedding (never used for fitting).
        hidden_layers: Comma-separated widths of the layers before the bottleneck, e.g. "32,16"; chosen from the data when omitted.
        noise: Standard deviation of Gaussian corruption added to the input during training (denoising autoencoder).
        activation: tanh or relu.
        max_iter: Training epochs.
        keep_columns: Original columns copied next to the embedding.
        max_rows: Rows sampled.
    """
    from sklearn.neural_network import MLPRegressor

    P = _prep(data, columns, drop, scale, truth_column, max_rows, seed)
    d = P.Z.shape[1]
    pre = tuple(int(p) for p in hidden_layers.replace(" ", "").split(",") if p) if hidden_layers else (max(n_components * 2, min(d, 32)),)
    sizes = pre + (n_components,) + pre[::-1]
    rng = np.random.default_rng(seed)
    ae = MLPRegressor(hidden_layer_sizes=sizes, activation=activation, max_iter=max_iter, random_state=seed, early_stopping=False)
    if noise > 0:  # corrupt the input each pass: train on several noisy copies against the clean target
        X = np.vstack([P.Z + rng.normal(scale=noise, size=P.Z.shape) for _ in range(3)])
        ae.fit(X, np.vstack([P.Z] * 3))
    else:
        ae.fit(P.Z, P.Z)
    enc = LayerEncoder(ae.coefs_[: len(pre) + 1], ae.intercepts_[: len(pre) + 1], activation)
    E = enc.transform(P.Z)
    rec = ae.predict(P.Z)
    return finish("autoencoder", f"{'denoising_' if noise else ''}autoencoder({'-'.join(map(str, sizes))})", P, E, enc, truth_column, keep_columns, "classification", seed, output_path, model_path,
                  {"reconstruction_r2": round(float(1 - ((P.Z - rec) ** 2).sum() / ((P.Z - P.Z.mean(axis=0)) ** 2).sum()), 4), "training_loss": round(float(ae.loss_), 5), "epochs": int(ae.n_iter_)})


class ContrastiveEncoder:
    def __init__(self, R: np.ndarray, b: np.ndarray, W: np.ndarray, hidden: int):
        self.R, self.b, self.W, self.h = R, b, W, hidden

    def transform(self, Z: np.ndarray) -> np.ndarray:
        H = np.sqrt(2.0 / self.h) * np.cos(Z @ self.R + self.b) @ self.W
        return H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-9)


@tool
def embed_contrastive(data: str, n_components: int = 8, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, augmentation: Literal["gaussian", "dropout", "mixed"] = "mixed",
                      strength: float = 0.3, temperature: float = 0.2, epochs: int = 150, batch_size: int = 256, hidden_features: int = 256, lr: float = 0.05, keep_columns: list[str] | None = None, max_rows: int = 20000, seed: int = 42,
                      output_path: str | None = None, model_path: str | None = None) -> dict:
    """Contrastive learning (SimCLR-style, in numpy): two perturbed views of the same row are pulled together in the embedding and pushed away from every other row in the batch (InfoNCE loss), so the representation keeps what survives perturbation and discards noise — no labels. A random-feature map followed by a trained linear projection; reports alignment and uniformity, the two quantities the loss trades off.

    Args:
        n_components: Embedding size.
        truth_column: Column of known labels, kept aside to probe the embedding.
        augmentation: gaussian noise, feature dropout or both.
        strength: Perturbation size.
        temperature: InfoNCE temperature; lower sharpens the contrast.
        epochs: Passes over the data.
        batch_size: Rows per batch (also the number of negatives).
        hidden_features: Width of the random Fourier feature map.
        lr: Adam step size.
        keep_columns: Original columns copied next to the embedding.
        max_rows: Rows sampled.
    """
    P = _prep(data, columns, drop, scale, truth_column, max_rows, seed)
    rng = np.random.default_rng(seed)
    R, b = rng.normal(scale=1.0, size=(P.Z.shape[1], hidden_features)), rng.uniform(0, 2 * np.pi, hidden_features)
    phi = lambda Z: np.sqrt(2.0 / hidden_features) * np.cos(Z @ R + b)  # noqa: E731
    W = rng.normal(scale=0.1, size=(hidden_features, n_components))
    m, v = np.zeros_like(W), np.zeros_like(W)

    def aug(Z: np.ndarray) -> np.ndarray:
        out = Z
        if augmentation in ("gaussian", "mixed"):
            out = out + rng.normal(scale=strength, size=Z.shape)
        if augmentation in ("dropout", "mixed"):
            out = out * (rng.random(Z.shape) > strength / 2)
        return out

    def norm_rows(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-9)
        return H / n, n

    step, losses = 0, []
    for ep in range(epochs):
        order = rng.permutation(len(P.Z))
        for s in range(0, len(order) - batch_size + 1, batch_size):
            Zb = P.Z[order[s : s + batch_size]]
            F1, F2 = phi(aug(Zb)), phi(aug(Zb))
            H1, H2 = F1 @ W, F2 @ W
            Z1, n1 = norm_rows(H1)
            Z2, n2 = norm_rows(H2)
            S = Z1 @ Z2.T / temperature
            Pm = np.exp(S - S.max(axis=1, keepdims=True))
            Pm /= Pm.sum(axis=1, keepdims=True)
            Pt = np.exp(S.T - S.T.max(axis=1, keepdims=True))
            Pt /= Pt.sum(axis=1, keepdims=True)
            B = len(Zb)
            I = np.eye(B)
            loss = -0.5 * (np.log(np.diag(Pm) + 1e-12).mean() + np.log(np.diag(Pt) + 1e-12).mean())
            G = ((Pm - I) + (Pt.T - I)) / (2 * temperature * B)  # d loss / d (z1_i . z2_j) for the symmetric InfoNCE loss
            dZ1, dZ2 = G @ Z2, G.T @ Z1
            dH1 = (dZ1 - (dZ1 * Z1).sum(axis=1, keepdims=True) * Z1) / n1  # back through the row normalisation
            dH2 = (dZ2 - (dZ2 * Z2).sum(axis=1, keepdims=True) * Z2) / n2
            g = F1.T @ dH1 + F2.T @ dH2
            step += 1
            m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g**2
            W -= lr * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
        losses.append(float(loss))

    enc = ContrastiveEncoder(R, b, W, hidden_features)
    E = enc.transform(P.Z)
    v1, v2 = enc.transform(aug(P.Z[:1000])), enc.transform(aug(P.Z[:1000]))
    alignment = float(((v1 - v2) ** 2).sum(axis=1).mean())
    m = min(1000, len(E))
    sq = ((E[:m, None, :] - E[None, :m, :]) ** 2).sum(axis=2)
    uniformity = float(np.log(np.exp(-2 * sq[np.triu_indices(m, 1)]).mean()))
    return finish("contrastive", f"contrastive({augmentation}={strength},tau={temperature})", P, E, enc, truth_column, keep_columns, "classification", seed, output_path, model_path,
                  {"final_loss": round(losses[-1], 4), "first_loss": round(losses[0], 4), "alignment_lower_is_better": round(alignment, 4), "uniformity_lower_is_better": round(uniformity, 4)})


@tool
def embed_random_fourier_features(data: str, n_components: int = 64, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, gamma: float | None = None, keep_columns: list[str] | None = None,
                                  max_rows: int = 30000, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Random Fourier features: an explicit, finite-dimensional approximation of the RBF kernel — a linear model on these features behaves like a kernel machine, without the quadratic cost. No training; the representation 'lifts' the data so non-linear structure becomes linearly separable. A baseline every learned embedding should beat.

    Args:
        n_components: Number of random features.
        truth_column: Column of known labels, kept aside to probe the embedding.
        gamma: RBF width; 1 / number of features when omitted.
        keep_columns: Original columns copied next to the embedding.
        max_rows: Rows sampled.
    """
    from sklearn.kernel_approximation import RBFSampler

    P = _prep(data, columns, drop, scale, truth_column, max_rows, seed)
    m = RBFSampler(gamma=gamma or 1.0 / P.Z.shape[1], n_components=n_components, random_state=seed).fit(P.Z)
    return finish("random_fourier", f"random_fourier_features({n_components})", P, m.transform(P.Z), m, truth_column, keep_columns, "classification", seed, output_path, model_path)


@tool
def embed_sparse_coding(data: str, n_components: int = 16, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, alpha: float = 1.0, keep_columns: list[str] | None = None,
                        max_rows: int = 10000, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Dictionary learning (sparse coding): learn a set of 'atoms' so each row is a combination of only a few of them; the (mostly zero) coefficients are the representation. Over-complete (more atoms than columns) and sparse — parts-like features that often separate better than dense ones.

    Args:
        n_components: Number of dictionary atoms.
        truth_column: Column of known labels, kept aside to probe the embedding.
        alpha: Sparsity penalty; larger gives sparser codes.
        keep_columns: Original columns copied next to the embedding.
        max_rows: Rows sampled.
    """
    from sklearn.decomposition import MiniBatchDictionaryLearning

    P = _prep(data, columns, drop, scale, truth_column, max_rows, seed)
    m = MiniBatchDictionaryLearning(n_components=n_components, alpha=alpha, transform_algorithm="lasso_lars", transform_alpha=alpha, random_state=seed, max_iter=50).fit(P.Z)
    E = m.transform(P.Z)
    return finish("sparse_coding", f"sparse_coding({n_components},alpha={alpha})", P, E, m, truth_column, keep_columns, "classification", seed, output_path, model_path, {"mean_active_atoms_per_row": round(float((np.abs(E) > 1e-6).sum(axis=1).mean()), 2)})


@tool
def embed_rbm(data: str, n_components: int = 16, columns: list[str] | None = None, drop: list[str] | None = None, truth_column: str | None = None, learning_rate: float = 0.05, n_iter: int = 30, keep_columns: list[str] | None = None, max_rows: int = 20000,
              seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Restricted Boltzmann machine: an energy-based model whose hidden units learn to represent patterns of co-activation among the (min-max scaled, 0–1) columns — the classic unsupervised pre-training layer; works best on binary or bounded data.

    Args:
        n_components: Hidden units.
        truth_column: Column of known labels, kept aside to probe the embedding.
        learning_rate: Step size.
        n_iter: Passes over the data.
        keep_columns: Original columns copied next to the embedding.
        max_rows: Rows sampled.
    """
    from sklearn.neural_network import BernoulliRBM
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler

    P = _prep(data, columns, drop, False, truth_column, max_rows, seed)
    pipe = make_pipeline(MinMaxScaler(), BernoulliRBM(n_components=n_components, learning_rate=learning_rate, n_iter=n_iter, random_state=seed)).fit(P.Z)
    return finish("rbm", f"rbm({n_components})", P, pipe.transform(P.Z), pipe, truth_column, keep_columns, "classification", seed, output_path, model_path)


@tool
def embed_kmeans_features(data: str, n_components: int = 32, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, truth_column: str | None = None, encoding: Literal["triangle", "distance", "hard"] = "triangle",
                          keep_columns: list[str] | None = None, max_rows: int = 30000, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """K-means feature learning (Coates & Ng): learn k centroids, then describe each row by its relation to all of them — the triangle encoding max(0, mean distance − distance to centroid) is sparse and surprisingly strong, rivalling deeper models on small data. Very cheap and a robust baseline.

    Args:
        n_components: Number of centroids (the embedding size).
        truth_column: Column of known labels, kept aside to probe the embedding.
        encoding: triangle, distance (to every centroid) or hard (one-hot nearest).
        keep_columns: Original columns copied next to the embedding.
        max_rows: Rows sampled.
    """
    from sklearn.cluster import MiniBatchKMeans

    P = _prep(data, columns, drop, scale, truth_column, max_rows, seed)
    km = MiniBatchKMeans(n_clusters=n_components, n_init=3, random_state=seed).fit(P.Z)
    enc = KMeansEncoder(km, encoding)
    return finish("kmeans_features", f"kmeans_features({n_components},{encoding})", P, enc.transform(P.Z), enc, truth_column, keep_columns, "classification", seed, output_path, model_path)


class KMeansEncoder:
    def __init__(self, km, encoding: str):
        self.km, self.encoding = km, encoding

    def transform(self, Z: np.ndarray) -> np.ndarray:
        d = self.km.transform(Z)
        if self.encoding == "distance":
            return d
        if self.encoding == "hard":
            return np.eye(d.shape[1])[d.argmin(axis=1)]
        return np.maximum(0, d.mean(axis=1, keepdims=True) - d)


@tool
def embed_tree_ensemble(data: str, n_components: int = 16, columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = False, truth_column: str | None = None, n_trees: int = 100, max_depth: int = 5, keep_columns: list[str] | None = None,
                        max_rows: int = 30000, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Totally random trees embedding: every row is described by the leaf it reaches in each of many random trees (a very high-dimensional, sparse, unsupervised code), then compressed with SVD. Rows that often share leaves are close — an embedding built from the data's own partition structure, robust to scale and to irrelevant columns.

    Args:
        n_components: Final SVD dimensions.
        truth_column: Column of known labels, kept aside to probe the embedding.
        n_trees: Random trees.
        max_depth: Depth of each tree.
        keep_columns: Original columns copied next to the embedding.
        max_rows: Rows sampled.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import RandomTreesEmbedding
    from sklearn.pipeline import make_pipeline

    P = _prep(data, columns, drop, scale, truth_column, max_rows, seed)
    pipe = make_pipeline(RandomTreesEmbedding(n_estimators=n_trees, max_depth=max_depth, random_state=seed, n_jobs=-1), TruncatedSVD(n_components, random_state=seed)).fit(P.Z)
    return finish("tree_ensemble", f"random_trees({n_trees}x{max_depth})+svd({n_components})", P, pipe.transform(P.Z), pipe, truth_column, keep_columns, "classification", seed, output_path, model_path)


@tool
def embed_metric_learning(data: str, label_column: str, n_components: int = 8, method: Literal["nca", "lda"] = "nca", columns: list[str] | None = None, drop: list[str] | None = None, scale: bool = True, keep_columns: list[str] | None = None,
                          max_rows: int = 5000, seed: int = 42, output_path: str | None = None, model_path: str | None = None) -> dict:
    """Supervised representation (metric learning): use class labels to learn a linear map in which rows of the same class are close and different classes far apart — neighbourhood components analysis (directly optimises k-NN accuracy) or linear discriminant analysis. The probe is computed on rows the map was *not* fitted on (cross-fitting), so the gain over unsupervised embeddings is honest.

    Args:
        label_column: Column of class labels used for fitting.
        n_components: Output dimensions (LDA allows at most classes − 1).
        method: nca or lda.
        keep_columns: Original columns copied next to the embedding.
        max_rows: Rows sampled.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import KFold
    from sklearn.neighbors import NeighborhoodComponentsAnalysis

    P = _prep(data, columns, list(drop or []) + [label_column], scale, label_column, max_rows, seed)
    y = P.truth
    k = min(n_components, P.Z.shape[1], (len(set(y)) - 1) if method == "lda" else n_components)
    make = (lambda: NeighborhoodComponentsAnalysis(n_components=k, random_state=seed, max_iter=100)) if method == "nca" else (lambda: LinearDiscriminantAnalysis(n_components=k))
    E = np.zeros((len(P.Z), k))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(P.Z):
        E[te] = make().fit(P.Z[tr], y[tr]).transform(P.Z[te])  # embeddings of held-out rows come from a map that never saw their labels
    model = make().fit(P.Z, y)
    return finish("metric_learning", f"{method}({k})", P, E, model, label_column, keep_columns, "classification", seed, output_path, model_path, {"note": "embeddings were produced cross-fitted (5 folds), so the probe is not inflated by label leakage; the saved model is fitted on all rows"})
