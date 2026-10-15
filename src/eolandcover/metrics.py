"""Classification metrics for land-cover models.

Overall accuracy hides per-class failures — a model can look good while missing a
rare but important class entirely. So we report per-class accuracy and the full
confusion matrix alongside the headline number, the same habit as reporting a
result with its breakdown rather than a single figure.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Overall fraction correct."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike, n_classes: int | None = None) -> NDArray:
    """Confusion matrix with rows = true class, columns = predicted class."""
    labels = list(range(n_classes)) if n_classes is not None else None
    return _sk_confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=labels)


def per_class_accuracy(y_true: ArrayLike, y_pred: ArrayLike, n_classes: int | None = None) -> NDArray:
    """Recall per class: of each true class, the fraction predicted correctly.

    Classes absent from ``y_true`` yield ``nan`` rather than a misleading 0.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    totals = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = np.diag(cm) / totals
    acc[totals == 0] = np.nan
    return acc
