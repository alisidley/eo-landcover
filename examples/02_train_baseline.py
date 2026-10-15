"""Baseline: classify EuroSAT land cover from simple per-band features.

This is the MVP — no deep learning yet. It proves the end-to-end loop
(load -> features -> stratified split -> train -> validate) and gives a number to
beat with embeddings later.

Run (after `pip install -e '.[torch]'` and letting EuroSAT download once):
    python examples/02_train_baseline.py

It is written to fail with a clear message until you implement the TODOs in
`eolandcover.data` — that implementation is the point of the exercise.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.ensemble import RandomForestClassifier

from eolandcover import accuracy, confusion_matrix, per_class_accuracy, stratified_split
from eolandcover.data import load_eurosat, to_features


def main() -> None:
    dataset = load_eurosat(root="data", download=True)  # implement in eolandcover.data
    X, y = to_features(dataset)                          # implement in eolandcover.data

    train_idx, val_idx, test_idx = stratified_split(y, random_state=0)

    clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    clf.fit(X[train_idx], y[train_idx])

    for name, idx in (("val", val_idx), ("test", test_idx)):
        pred = clf.predict(X[idx])
        print(f"{name}: accuracy = {accuracy(y[idx], pred):.3f}")
        if name == "test":
            print("per-class accuracy:", per_class_accuracy(y[idx], pred))
            print("confusion matrix:\n", confusion_matrix(y[idx], pred))


if __name__ == "__main__":
    main()
