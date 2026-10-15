"""Change detection from image embeddings.

Planet's team is expanding into "land-cover change detection using embeddings".
The idea: encode each image patch (or time step) into a vector with a pretrained
model, then measure how far apart two times are in embedding space. Large
distances flag places where the land changed (construction, deforestation,
harvest) without hand-labelling every transition.

This module is the model-agnostic part: given embeddings, compute distances and
threshold them. Producing the embeddings themselves is the ``model`` scaffold.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def cosine_distance(a: ArrayLike, b: ArrayLike, eps: float = 1e-10) -> NDArray:
    """Row-wise cosine distance in [0, 2] between two sets of embeddings.

    0 means identical direction (no change); larger means more different.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("embedding arrays must have the same shape")
    a2 = np.atleast_2d(a)
    b2 = np.atleast_2d(b)
    num = np.sum(a2 * b2, axis=1)
    den = np.linalg.norm(a2, axis=1) * np.linalg.norm(b2, axis=1) + eps
    return 1.0 - num / den


def change_scores(before: ArrayLike, after: ArrayLike) -> NDArray:
    """Per-location change score = cosine distance between before/after embeddings."""
    return cosine_distance(before, after)


def flag_changes(scores: ArrayLike, threshold: float) -> NDArray:
    """Boolean mask of locations whose change score exceeds ``threshold``."""
    return np.asarray(scores, dtype=float) > threshold
