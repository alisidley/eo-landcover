"""Spectral indices from satellite bands.

Vegetation and land cover show up strongly in simple band ratios. NDVI
(Normalised Difference Vegetation Index) is the classic one: healthy vegetation
reflects near-infrared and absorbs red, so (NIR - Red) / (NIR + Red) is high for
vegetation and low for bare ground, water, or built-up areas. These indices are a
cheap, interpretable feature set before (or alongside) learned representations.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalised_difference(a: ArrayLike, b: ArrayLike, eps: float = 1e-10) -> NDArray:
    """(a - b) / (a + b), guarded against divide-by-zero. Result in [-1, 1]."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return (a - b) / (a + b + eps)


def ndvi(nir: ArrayLike, red: ArrayLike) -> NDArray:
    """Normalised Difference Vegetation Index from the NIR and red bands.

    For Sentinel-2 these are bands B8 (NIR) and B4 (red). High values indicate
    dense vegetation; near-zero/negative values indicate soil, water, or built-up.
    """
    return normalised_difference(nir, red)
