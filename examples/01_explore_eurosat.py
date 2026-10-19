"""Explore EuroSAT — build intuition for the data before modelling.

Prints the dataset size and class names, inspects one sample's structure, and
checks that NDVI behaves as expected (vegetation > built-up). Needs the ``torch``
extra:  pip install -e '.[torch]'
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eolandcover.data import load_eurosat
from eolandcover.indices import ndvi

# Sentinel-2 band order in EuroSAT: B01, B02, B03, B04(red), ..., B08(NIR), ...
RED, NIR = 3, 7


def first_of_class(dataset, class_index: int):
    """Return the first sample whose label is ``class_index``."""
    for sample in dataset:
        if int(sample["label"]) == class_index:
            return sample
    raise ValueError(f"no sample found for class index {class_index}")


def main() -> None:
    dataset = load_eurosat(root="data", download=True)
    classes = list(dataset.classes)

    print(f"samples: {len(dataset)}")
    print(f"classes ({len(classes)}): {classes}")

    sample = dataset[0]
    img = sample["image"]
    print(f"sample keys: {list(sample.keys())}")
    print(f"image shape: {tuple(img.shape)}  dtype: {img.dtype}")  # (bands, H, W)
    print(f"label: {int(sample['label'])} -> {classes[int(sample['label'])]}")

    # NDVI sanity check: vegetation should score higher than built-up land.
    forest = first_of_class(dataset, classes.index("Forest"))["image"].float()
    highway = first_of_class(dataset, classes.index("Highway"))["image"].float()
    forest_ndvi = float(ndvi(forest[NIR], forest[RED]).mean())
    highway_ndvi = float(ndvi(highway[NIR], highway[RED]).mean())
    print(f"mean NDVI — Forest: {forest_ndvi:.3f}  Highway: {highway_ndvi:.3f}")
    print("NDVI higher for Forest (as expected)" if forest_ndvi > highway_ndvi
          else "unexpected: Highway NDVI >= Forest — check the band indices")


if __name__ == "__main__":
    main()
