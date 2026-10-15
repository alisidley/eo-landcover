"""Explore EuroSAT — your first hands-on step (needs the ``torch`` extra).

Goal: build intuition for the data before modelling. Run after
`pip install -e '.[torch]'`.

This is a scaffold with TODOs — filling them in is the exercise.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eolandcover.data import load_eurosat
from eolandcover.indices import ndvi


def main() -> None:
    dataset = load_eurosat(root="data", download=True)  # implement in eolandcover.data

    # TODO (your turn):
    #   1. Print len(dataset) and the list of class names.
    #   2. Take one sample; print the image tensor's shape and dtype.
    #      Sentinel-2 EuroSAT has 13 bands — which indices are red (B4) and NIR (B8)?
    #   3. Compute the mean NDVI of a 'Forest' patch and a 'Highway' patch with
    #      `ndvi(nir_band, red_band)` and confirm forest is higher.
    #   4. If matplotlib is installed, show an RGB composite (bands B4, B3, B2).
    print(f"dataset loaded: {len(dataset)} samples")
    _ = ndvi  # used once you implement step 3


if __name__ == "__main__":
    main()
