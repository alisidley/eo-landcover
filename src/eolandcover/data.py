"""EuroSAT data loading (optional ``torch`` extra).

EuroSAT is the standard starter dataset for satellite imagery: 27,000 Sentinel-2
patches labelled with 10 land-cover classes (forest, river, residential, crops,
pasture, ...). TorchGeo can download and load it for you.

Requires the optional ``torch`` extra:
    pip install -e '.[torch]'
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .indices import ndvi


def load_eurosat(root: str | Path = "data", download: bool = True):
    """Load the EuroSAT dataset (Sentinel-2, 10 land-cover classes) via TorchGeo.

    Downloads to ``root`` on first use. Requires the ``torch`` extra.
    """
    from torchgeo.datasets import EuroSAT
    return EuroSAT(root=str(root), download=download)


def to_features(dataset) -> tuple:
    """Reduce each patch to simple per-band features for a quick baseline.

    Per-band mean and std over the patch's pixels plus mean NDVI, giving an
    (N, 2*C + 1) feature matrix X and integer labels y for a scikit-learn model.
    """
    B4, B8 = 3, 7   # red/NIR channel indices

    feats, labels = [], []
    for sample in dataset:
        img = sample["image"].float()          # (C, H, W) torch tensor
        mean = img.mean(dim=(1, 2))            # length C  (mean over the H*W pixels, per band)
        std  = img.std(dim=(1, 2))             # length C
        ndvi_mean = ndvi(img[B8], img[B4]).mean()   # scalar, mean NDVI over the patch
        vec = np.concatenate([mean.numpy(), std.numpy(), [float(ndvi_mean)]])
        feats.append(vec)
        labels.append(int(sample["label"]))
    X = np.stack(feats)        # (N, 2C+1)
    y = np.array(labels)
    return X, y



