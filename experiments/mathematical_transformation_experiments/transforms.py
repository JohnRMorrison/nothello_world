"""
Pluggable transformation classes for generating boolean labels from game sequences.

Each transform class implements:
    generate(games, rng, n=1) -> (params_dict, scalars of shape (N, n))
    apply(games, params)       -> scalars of shape (N, n)

where games is shape (N, 60), normalized to [0,1].

Registry: TRANSFORMS maps name -> instance.
"""

import numpy as np
from abc import ABC, abstractmethod


def _random_vec_excluding_zero(shape, rng: np.random.Generator):
    """Generate a random array in [-1,1] excluding (-0.1, 0.1)."""
    raw = rng.uniform(0.1, 1.0, size=shape)
    signs = rng.choice([-1, 1], size=shape)
    return raw * signs


class Transform(ABC):
    @abstractmethod
    def generate(self, games: np.ndarray, rng: np.random.Generator, n: int = 1):
        """Generate random params and compute scalars.

        Returns:
            params: dict of numpy arrays (transform-specific)
            scalars: np.ndarray of shape (N, n)
        """

    @abstractmethod
    def apply(self, games: np.ndarray, params: dict) -> np.ndarray:
        """Apply saved params to games.

        Returns:
            scalars: np.ndarray of shape (N, n)
        """


class DotProduct(Transform):
    """Linear: games @ W. Easiest."""

    def generate(self, games, rng, n=1):
        W = _random_vec_excluding_zero((60, n), rng)
        scalars = games @ W
        return {"vector": W}, scalars

    def apply(self, games, params):
        return games @ params["vector"]


class MaxProjection(Transform):
    """Piecewise linear: max over k=4 linear projections."""

    k = 4

    def generate(self, games, rng, n=1):
        # W shape: (60, k*n) — k projections per transform
        W = _random_vec_excluding_zero((60, self.k * n), rng)
        all_proj = games @ W  # (N, k*n)
        # Reshape to (N, k, n) and take max over k
        scalars = all_proj.reshape(len(games), self.k, n).max(axis=1)  # (N, n)
        return {"W": W}, scalars

    def apply(self, games, params):
        W = params["W"]
        n = W.shape[1] // self.k
        all_proj = games @ W  # (N, k*n)
        return all_proj.reshape(len(games), self.k, n).max(axis=1)


class ReluFeatures(Transform):
    """One hidden layer with ReLU: ReLU(games @ W + b) @ v. Moderate difficulty."""

    h = 32  # hidden units per transform

    def generate(self, games, rng, n=1):
        W = _random_vec_excluding_zero((60, self.h * n), rng)    # (60, h*n)
        b = rng.standard_normal(self.h * n).astype(np.float64)   # (h*n,)
        v = _random_vec_excluding_zero((self.h, n), rng)         # (h, n)

        hidden = games @ W  # (N, h*n)
        hidden += b
        np.maximum(hidden, 0, out=hidden)  # ReLU in-place
        # Reshape to (N, h, n), multiply by v (h, n), sum over h
        hidden = hidden.reshape(len(games), self.h, n)
        scalars = (hidden * v).sum(axis=1)  # (N, n)

        return {"W": W, "b": b, "v": v}, scalars

    def apply(self, games, params):
        W, b, v = params["W"], params["b"], params["v"]
        n = v.shape[1] if v.ndim == 2 else 1
        hidden = games @ W + b
        np.maximum(hidden, 0, out=hidden)
        hidden = hidden.reshape(len(games), self.h, n)
        return (hidden * v).sum(axis=1)


class Quadratic(Transform):
    """Sum of squared projections: sum_j (games @ W_j)^2. Moderate-hard."""

    k = 8  # number of squared terms per transform

    def generate(self, games, rng, n=1):
        W = _random_vec_excluding_zero((60, self.k * n), rng)
        proj = games @ W  # (N, k*n)
        proj_sq = proj ** 2
        # Reshape to (N, k, n), sum over k
        scalars = proj_sq.reshape(len(games), self.k, n).sum(axis=1)  # (N, n)
        return {"W": W}, scalars

    def apply(self, games, params):
        W = params["W"]
        n = W.shape[1] // self.k
        proj = games @ W
        return (proj ** 2).reshape(len(games), self.k, n).sum(axis=1)


class Periodic(Transform):
    """Sinusoidal: sin(games @ W + phi). Hard (non-monotonic)."""

    def generate(self, games, rng, n=1):
        W = _random_vec_excluding_zero((60, n), rng)
        phi = rng.uniform(0, 2 * np.pi, size=n)
        scalars = np.sin(games @ W + phi)  # (N, n)
        return {"W": W, "phi": phi}, scalars

    def apply(self, games, params):
        return np.sin(games @ params["W"] + params["phi"])


class SparseParity(Transform):
    """XOR of k=5 thresholded features. Hardest (parity is notoriously hard)."""

    k = 5  # number of features to XOR

    def generate(self, games, rng, n=1):
        # Pick k random feature indices per transform
        indices = np.stack([rng.choice(60, size=self.k, replace=False) for _ in range(n)],
                          axis=0)  # (n, k)
        # Thresholds per feature: use median of each selected feature
        thresholds = np.zeros((n, self.k))
        for i in range(n):
            for j in range(self.k):
                thresholds[i, j] = np.median(games[:, indices[i, j]])

        # Compute: threshold each feature to 0/1, XOR (= parity = sum mod 2)
        scalars = self._compute(games, indices, thresholds)
        return {"indices": indices, "thresholds": thresholds}, scalars

    def apply(self, games, params):
        return self._compute(games, params["indices"], params["thresholds"])

    def _compute(self, games, indices, thresholds):
        """Returns float scalars (N, n) — values are 0.0 or 1.0."""
        n = indices.shape[0]
        N = len(games)
        bits = np.zeros((N, n), dtype=np.int8)
        for i in range(n):
            for j in range(self.k):
                feat = games[:, indices[i, j]]  # (N,)
                bits[:, i] ^= (feat > thresholds[i, j]).astype(np.int8)
        return bits.astype(np.float64)


TRANSFORMS = {
    "dot_product": DotProduct(),
    "max_projection": MaxProjection(),
    "relu_features": ReluFeatures(),
    "quadratic": Quadratic(),
    "periodic": Periodic(),
    "sparse_parity": SparseParity(),
}
