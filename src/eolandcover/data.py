"""EuroSAT data loading (optional ``torch`` extra).

EuroSAT is the standard starter dataset for satellite imagery: 27,000 Sentinel-2
patches labelled with 10 land-cover classes (forest, river, residential, crops,
pasture, ...). TorchGeo can download and load it for you.

This is a guided scaffold — the TODOs are the learning. Install the extra first:
    pip install -e '.[torch]'
"""
from __future__ import annotations

from pathlib import Path


def load_eurosat(root: str | Path = "data", download: bool = True):
    """Return a TorchGeo EuroSAT dataset.

    TODO (your turn):
      1. `from torchgeo.datasets import EuroSAT`
      2. Return `EuroSAT(root=str(root), download=download)`.
      3. Inspect one sample: it's a dict with an image tensor and a label. Print
         its shape and dtype, and note how many bands Sentinel-2 has.

    Kept as a scaffold so the chemistry of loading real imagery is something you
    do hands-on (and can therefore talk about in an interview).
    """
    raise NotImplementedError(
        "Implement load_eurosat using torchgeo.datasets.EuroSAT — see the TODO."
    )


def to_features(dataset) -> tuple:
    """Turn an image dataset into a plain (X, y) feature table for a quick baseline.

    TODO (your turn): a strong first baseline needs no deep learning —
      1. For each sample, compute a few per-band summary statistics (mean, std)
         and the mean NDVI (use `eolandcover.indices.ndvi` on the NIR/red bands).
      2. Stack these into X (n_samples, n_features) and collect labels y.
      3. Hand X, y to a scikit-learn classifier (see examples/02_train_baseline.py).
    This "simple features first" step is a good sanity check before embeddings.
    """
    raise NotImplementedError("Implement to_features — see the TODO.")
