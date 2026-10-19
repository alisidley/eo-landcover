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


def plot_confusion_matrix(
    cm: ArrayLike,
    class_names: list[str] | None = None,
    normalise: bool = True,
    path: str = "confusion_matrix.png",
    title: str = "Confusion matrix",
) -> str:
    """Save a confusion-matrix heatmap and return the path.

    Rows are the true class, columns the predicted class. With ``normalise`` the
    rows are divided by their totals, so the diagonal reads as per-class recall
    (easier to compare across imbalanced classes). Requires the ``plots`` extra
    (matplotlib); imported lazily so the core stays dependency-light.
    """
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    if normalise:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = cm / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    n = cm.shape[0]
    labels = class_names if class_names is not None else [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * n, 1.0 + 0.6 * n))
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0 if normalise else None)
    ax.set_xticks(range(n), labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = (cm.max() if cm.size else 1.0) / 2.0
    fmt = "{:.2f}" if normalise else "{:.0f}"
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, fmt.format(cm[i, j]),
                ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
