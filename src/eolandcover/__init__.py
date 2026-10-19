"""eo-landcover — land-cover classification and change detection from satellite imagery.

A learning project built around Sentinel-2 imagery (EuroSAT). The reusable core
here is domain logic that needs no heavy dependencies and is fully tested:

* ``indices``  — spectral indices (NDVI, normalised differences) from bands;
* ``metrics``  — classification metrics (accuracy, per-class, confusion matrix);
* ``change``   — change detection from image embeddings (distance + thresholding);
* ``split``    — stratified train/validation/test splits.

The satellite-imagery modelling (``data``, ``model``, ``train``) lives behind the
optional ``torch`` extra and is a guided scaffold — see the README roadmap.
"""
from __future__ import annotations

from .change import change_scores, cosine_distance, flag_changes
from .indices import ndvi, normalised_difference
from .metrics import accuracy, confusion_matrix, per_class_accuracy, plot_confusion_matrix
from .split import stratified_split

__all__ = [
    "accuracy",
    "change_scores",
    "confusion_matrix",
    "cosine_distance",
    "flag_changes",
    "ndvi",
    "normalised_difference",
    "per_class_accuracy",
    "plot_confusion_matrix",
    "stratified_split",
]
__version__ = "0.1.0"
