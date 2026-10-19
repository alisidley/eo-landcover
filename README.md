# eo-landcover

**Land-cover classification and change detection from Sentinel-2 satellite imagery.**

An Earth-observation project built around **EuroSAT** (Sentinel-2 patches, 10
land-cover classes). It mirrors the shape of real operational work: extract
validated markers from satellite imagery, starting from simple interpretable
features and moving toward learned embeddings and change detection.

## What's here

- **`data`** — load EuroSAT via TorchGeo and reduce each patch to per-band summary
  features (mean, std) plus NDVI for a fast baseline.
- **`indices`** — spectral indices (NDVI and normalised differences) from bands.
- **`metrics`** — accuracy, per-class accuracy, confusion matrix, and a confusion-matrix
  plot (per-class matters: land-cover classes are imbalanced).
- **`change`** — change detection from image embeddings (cosine distance + thresholding).
- **`split`** — stratified train/val/test splits that preserve class proportions.
- **`examples/`** — explore the data, run the baseline, and a change-detection demo.
- Tests + GitHub Actions CI (lint + test on Python 3.10 / 3.12). The core modules
  need no heavy dependencies; the imagery work sits behind the optional `torch` extra.

## Quickstart

Core + the runnable change-detection demo (no satellite download needed):

```bash
pip install -e '.[dev,plots]'
pytest -q                               # the implemented core is tested
python examples/03_change_detection.py  # change detection on simulated embeddings
```

For the imagery work (downloads EuroSAT the first time):

```bash
pip install -e '.[torch]'
python examples/01_explore_eurosat.py   # look at a few patches + their NDVI
python examples/02_train_baseline.py    # train + evaluate the baseline
```

## Results — baseline

Each 64×64 patch is reduced to simple, interpretable features — per-band mean and
standard deviation plus mean NDVI — and classified with a RandomForest on a
stratified 70/15/15 split. No deep learning.

**Test accuracy: 88%** across the 10 classes (chance is 10%); validation 88.5% and
test 88.0% track closely, so the model isn't overfitting.

![EuroSAT confusion matrix — baseline](assets/confusion_matrix.png)

The confusion matrix (row-normalised, so the diagonal is per-class recall) tells a
clear story. Spectrally uniform classes are easy — **Forest 0.99, SeaLake 0.98**.
The one weak class is **Highway (0.56)**, which scatters into AnnualCrop,
Residential, Industrial and others.

That failure is informative rather than disappointing: a highway is a **thin, linear**
feature, and collapsing a patch to per-band *statistics* discards *where* things
are — so a mostly-crop patch with a road through it looks spectrally like crops.
The baseline's main weakness is precisely the class defined by spatial structure,
which is the motivation for the next step: a representation that **keeps** spatial
layout (a CNN, or embeddings from one).

## Roadmap

- [x] **Explore** — load EuroSAT; inspect bands, patch shape, and NDVI.
- [x] **Baseline** — per-band features + NDVI → RandomForest, with per-class accuracy
  and confusion matrix (**88%** test — see Results).
- [ ] **Embeddings** — use a pretrained (satellite) CNN as a feature extractor and
  classify from embeddings; compare against the baseline, especially on Highway.
- [ ] **Change detection on real imagery** — embed two Sentinel-2 acquisitions of the
  same area and flag change by embedding distance (`change.py` already does the
  distance/thresholding; this wires in real pairs).

## Why this project

Earth observation turns satellite data into decisions about agriculture, land, and
climate. The hard part is rarely a single model — it's validated, reproducible markers
and honest evaluation on messy real-world imagery. This repo is built around that:
metrics and splits that don't fool you, simple interpretable features before fancy ones,
and change detection framed as a distance in a learned space.

## License

MIT.
