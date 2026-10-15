"""Stratified train/validation/test splits.

Land-cover classes are imbalanced, so a random split can leave a rare class
under-represented (or absent) in the test set, giving noisy or misleading scores.
A stratified split keeps each class's proportions roughly equal across the splits.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def stratified_split(
    labels: ArrayLike,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) with per-class proportions preserved.

    ``fractions`` must sum to 1. Within each class the items are shuffled, then
    split by the cumulative fractions, so every class contributes to every split.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    labels = np.asarray(labels)
    rng = np.random.default_rng(random_state)
    f_train, f_val, _ = fractions

    train, val, test = [], [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = round(f_train * n)
        n_val = round(f_val * n)
        train.extend(idx[:n_train].tolist())
        val.extend(idx[n_train : n_train + n_val].tolist())
        test.extend(idx[n_train + n_val :].tolist())

    return np.array(sorted(train)), np.array(sorted(val)), np.array(sorted(test))
