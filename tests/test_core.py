import numpy as np
import pytest

from eolandcover import (
    accuracy,
    change_scores,
    confusion_matrix,
    cosine_distance,
    flag_changes,
    ndvi,
    normalised_difference,
    per_class_accuracy,
    plot_confusion_matrix,
    stratified_split,
)


# --- indices ---
def test_ndvi_vegetation_vs_bare():
    # high NIR, low red -> vegetation -> NDVI near 1; equal bands -> ~0
    assert ndvi(0.8, 0.1) > 0.5
    assert abs(ndvi(0.3, 0.3)) < 1e-6


def test_normalised_difference_bounds():
    nd = normalised_difference([0.0, 1.0, 0.5], [1.0, 0.0, 0.5])
    assert np.all(nd >= -1.0) and np.all(nd <= 1.0)


# --- metrics ---
def test_accuracy_and_shape_check():
    assert accuracy([0, 1, 2, 2], [0, 1, 2, 1]) == 0.75
    with pytest.raises(ValueError):
        accuracy([0, 1], [0, 1, 2])


def test_per_class_accuracy_handles_absent_class():
    # class 2 never appears in y_true -> nan, not 0
    acc = per_class_accuracy([0, 0, 1], [0, 1, 1], n_classes=3)
    assert acc[0] == 0.5 and acc[1] == 1.0 and np.isnan(acc[2])


def test_confusion_matrix_shape():
    cm = confusion_matrix([0, 1, 2], [0, 1, 2], n_classes=3)
    assert cm.shape == (3, 3)
    assert np.trace(cm) == 3


# --- change detection ---
def test_cosine_distance_identical_is_zero():
    v = np.array([[1.0, 2.0, 3.0]])
    assert cosine_distance(v, v)[0] == pytest.approx(0.0, abs=1e-9)


def test_change_scores_and_flagging():
    before = np.array([[1.0, 0.0], [1.0, 0.0]])
    after = np.array([[1.0, 0.0], [0.0, 1.0]])  # second row flipped direction
    scores = change_scores(before, after)
    assert scores[0] < scores[1]
    flags = flag_changes(scores, threshold=0.5)
    assert not flags[0] and flags[1]


# --- split ---
def test_stratified_split_preserves_classes_and_is_disjoint():
    labels = np.array([0] * 50 + [1] * 30 + [2] * 20)
    tr, va, te = stratified_split(labels, (0.7, 0.15, 0.15), random_state=1)
    # disjoint and complete
    assert len(set(tr) | set(va) | set(te)) == len(labels)
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    # every class present in train
    assert set(labels[tr]) == {0, 1, 2}


def test_stratified_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        stratified_split([0, 1, 2], (0.5, 0.3, 0.3))


# --- plotting (skips when matplotlib isn't installed, e.g. in CI) ---
def test_plot_confusion_matrix_writes_file(tmp_path):
    pytest.importorskip("matplotlib")
    cm = np.array([[5, 1], [2, 4]])
    out = tmp_path / "cm.png"
    path = plot_confusion_matrix(cm, ["a", "b"], path=str(out))
    assert out.exists() and path == str(out)
