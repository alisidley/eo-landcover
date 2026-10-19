"""Baseline: classify EuroSAT land cover from simple per-band features.

This is the MVP — no deep learning yet. It proves the end-to-end loop
(load -> features -> stratified split -> train -> validate) and gives a number to
beat with embeddings later.

Run (after `pip install -e '.[torch]'`; downloads EuroSAT on first use):
    python examples/02_train_baseline.py

Reaches ~88% test accuracy and saves a confusion matrix to assets/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.ensemble import RandomForestClassifier

from eolandcover import (
    accuracy,
    confusion_matrix,
    per_class_accuracy,
    plot_confusion_matrix,
    stratified_split,
)
from eolandcover.data import load_eurosat, to_features


def main() -> None:
    dataset = load_eurosat(root="data", download=True)
    X, y = to_features(dataset)

    train_idx, val_idx, test_idx = stratified_split(y, random_state=0)

    clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    clf.fit(X[train_idx], y[train_idx])

    for name, idx in (("val", val_idx), ("test", test_idx)):
        pred = clf.predict(X[idx])
        print(f"{name}: accuracy = {accuracy(y[idx], pred):.3f}")
        if name == "test":
            print("per-class accuracy:", per_class_accuracy(y[idx], pred))
            cm = confusion_matrix(y[idx], pred)
            print("confusion matrix:\n", cm)
            class_names = getattr(dataset, "classes", None)
            path = plot_confusion_matrix(cm, class_names, path="assets/confusion_matrix.png")
            print(f"saved {path}")


if __name__ == "__main__":
    main()
