"""
Pluggable transformation functions for generating boolean labels from game sequences.

Each transform function takes:
    - games: np.ndarray of shape (N, 60), normalized to [0,1]
    - rng: np.random.Generator for reproducible random vector generation
And returns:
    - transform_vec: np.ndarray of shape (60,) — the fixed vector used
    - scalars: np.ndarray of shape (N,) — one scalar per game

To add a new transformation, define a function with the same signature
and pass it to generate_labels.py via --transform.
"""

import numpy as np


def dot_product(games: np.ndarray, rng: np.random.Generator):
    """
    Default transform: dot product with a random vector.
    Vector values drawn uniformly from [-1, -0.1] ∪ [0.1, 1].
    """
    vec = _random_vec_excluding_zero(60, rng)
    scalars = games @ vec
    return vec, scalars


def _random_vec_excluding_zero(length: int, rng: np.random.Generator):
    """Generate a random vector in [-1,1] excluding (-0.1, 0.1)."""
    raw = rng.uniform(0.1, 1.0, size=length)
    signs = rng.choice([-1, 1], size=length)
    return raw * signs


TRANSFORMS = {
    "dot_product": dot_product,
}