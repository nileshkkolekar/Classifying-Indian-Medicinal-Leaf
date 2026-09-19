"""Metric computation."""

from __future__ import annotations

import numpy as np
import pytest

from medicinal_leaf.evaluation.metrics import compute_metrics, top_k_accuracy

CLASSES = ["Aloevera", "Amla", "Mint"]


def test_perfect_predictions_score_one():
    truth = np.array([0, 1, 2, 0, 1, 2])
    metrics = compute_metrics(truth, truth, CLASSES)

    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.balanced_accuracy == 1.0


def test_confusion_matrix_shape_and_trace():
    truth = np.array([0, 1, 2, 0])
    pred = np.array([0, 1, 2, 1])
    metrics = compute_metrics(truth, pred, CLASSES)

    assert metrics.confusion.shape == (3, 3)
    assert metrics.confusion.trace() == 3
    assert metrics.confusion.sum() == 4


def test_per_class_support_matches_the_labels():
    truth = np.array([0, 0, 0, 1, 2])
    pred = np.array([0, 0, 1, 1, 2])
    metrics = compute_metrics(truth, pred, CLASSES)

    assert metrics.per_class["Aloevera"]["support"] == 3
    assert metrics.per_class["Amla"]["support"] == 1


def test_macro_f1_punishes_an_ignored_class():
    # Class 2 is never predicted, so accuracy stays high but macro-F1 drops.
    truth = np.array([0] * 8 + [1] * 8 + [2] * 2)
    pred = np.array([0] * 8 + [1] * 8 + [0] * 2)
    metrics = compute_metrics(truth, pred, CLASSES)

    assert metrics.accuracy > 0.88
    assert metrics.macro_f1 < 0.70


def test_metrics_frame_has_expected_columns():
    truth = np.array([0, 1, 2])
    metrics = compute_metrics(truth, truth, CLASSES)
    frame = metrics.to_frame()

    assert list(frame.columns) == ["precision", "recall", "f1", "support"]
    assert list(frame.index) == CLASSES


def test_to_dict_contains_only_scalars():
    truth = np.array([0, 1, 2])
    payload = compute_metrics(truth, truth, CLASSES, loss=0.25).to_dict()

    assert payload["loss"] == 0.25
    assert all(isinstance(v, float) for v in payload.values())


def test_confusion_frame_normalises_by_row():
    truth = np.array([0, 0, 1, 2])
    pred = np.array([0, 1, 1, 2])
    frame = compute_metrics(truth, pred, CLASSES).confusion_frame(normalize=True)

    assert frame.loc["Aloevera"].sum() == pytest.approx(1.0)
    assert frame.loc["Aloevera", "Aloevera"] == pytest.approx(0.5)


def test_top_k_accuracy_is_monotonic():
    probabilities = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.1, 0.2, 0.7],
            [0.4, 0.35, 0.25],
        ]
    )
    truth = np.array([0, 2, 2, 1])

    assert top_k_accuracy(probabilities, truth, 1) <= top_k_accuracy(probabilities, truth, 2)
    assert top_k_accuracy(probabilities, truth, 3) == 1.0


def test_top_k_rejects_one_dimensional_input():
    with pytest.raises(ValueError, match="2-D probabilities"):
        top_k_accuracy(np.array([0.1, 0.9]), np.array([1]))


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_metrics(np.array([0, 1]), np.array([0]), CLASSES)


def test_empty_predictions_are_rejected():
    with pytest.raises(ValueError, match="zero predictions"):
        compute_metrics(np.array([]), np.array([]), CLASSES)
