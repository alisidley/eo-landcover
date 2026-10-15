"""Change detection from embeddings — runnable now, no satellite data needed.

Simulates "before" and "after" embeddings for a grid of locations where a few
have genuinely changed, then shows that thresholding the embedding distance
recovers them. Once you implement `eolandcover.model.embed`, swap the synthetic
embeddings for real ones from two Sentinel-2 acquisitions.

Run:  python examples/03_change_detection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from eolandcover import change_scores, flag_changes


def main() -> None:
    rng = np.random.default_rng(0)
    n_locations, dim = 200, 32

    before = rng.standard_normal((n_locations, dim))
    after = before.copy()

    # 15 locations genuinely change: perturb their embeddings substantially.
    changed = rng.choice(n_locations, size=15, replace=False)
    after[changed] += 2.5 * rng.standard_normal((len(changed), dim))
    # everywhere else: small acquisition noise (clouds, illumination, ...).
    after += 0.1 * rng.standard_normal((n_locations, dim))

    scores = change_scores(before, after)
    flagged = np.where(flag_changes(scores, threshold=0.4))[0]

    truth = set(changed.tolist())
    found = set(flagged.tolist())
    tp = len(truth & found)
    print(f"true changes: {len(truth)}  flagged: {len(found)}  correct: {tp}")
    print(f"precision: {tp / max(len(found), 1):.2f}  recall: {tp / len(truth):.2f}")
    print("Tune the threshold to trade precision against recall.")


if __name__ == "__main__":
    main()
