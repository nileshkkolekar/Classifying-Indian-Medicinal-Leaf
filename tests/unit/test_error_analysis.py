"""Failure-mode summaries."""

from __future__ import annotations

import pandas as pd
import pytest

from medicinal_leaf.evaluation.error_analysis import (
    least_confident_correct,
    most_confident_errors,
    per_class_error_rate,
    save_report,
    summarize_errors,
    top_confusions,
)


@pytest.fixture
def predictions() -> pd.DataFrame:
    rows = [
        # Mint is repeatedly mistaken for Tulsi, and confidently so.
        ("a.jpg", "Mint", "Tulsi", 0.95),
        ("b.jpg", "Mint", "Tulsi", 0.91),
        ("c.jpg", "Mint", "Tulsi", 0.88),
        ("d.jpg", "Mint", "Mint", 0.52),
        ("e.jpg", "Neem", "Amla", 0.60),
        ("f.jpg", "Neem", "Neem", 0.99),
        ("g.jpg", "Amla", "Amla", 0.97),
        ("h.jpg", "Amla", "Amla", 0.80),
    ]
    frame = pd.DataFrame(rows, columns=["file_path", "true_label", "predicted_label", "confidence"])
    frame["correct"] = frame["true_label"] == frame["predicted_label"]
    frame["true_class_prob"] = frame["confidence"].where(frame["correct"], 1 - frame["confidence"])
    return frame


def test_top_confusions_ranks_by_frequency(predictions):
    confusions = top_confusions(predictions)
    top = confusions.iloc[0]

    assert top["true_label"] == "Mint"
    assert top["predicted_label"] == "Tulsi"
    assert top["count"] == 3


def test_top_confusions_on_a_perfect_model():
    perfect = pd.DataFrame(
        {
            "true_label": ["Mint"],
            "predicted_label": ["Mint"],
            "confidence": [0.9],
            "correct": [True],
        }
    )
    assert top_confusions(perfect).empty


def test_per_class_error_rate(predictions):
    rates = per_class_error_rate(predictions)

    assert rates.loc["Mint", "errors"] == 3
    assert rates.loc["Mint", "error_rate"] == pytest.approx(0.75)
    assert rates.loc["Amla", "error_rate"] == 0.0
    # Sorted worst-first.
    assert rates.index[0] == "Mint"


def test_most_confident_errors_are_sorted(predictions):
    worst = most_confident_errors(predictions, n=2)
    assert len(worst) == 2
    assert worst.iloc[0]["confidence"] == 0.95
    assert not worst["correct"].any()


def test_least_confident_correct(predictions):
    shaky = least_confident_correct(predictions, n=1)
    assert shaky.iloc[0]["file_path"] == "d.jpg"
    assert shaky["correct"].all()


def test_summary_mentions_the_error_rate(predictions):
    summary = summarize_errors(predictions)
    assert "4/8" in summary
    assert "Mint" in summary


def test_save_report_writes_three_files(predictions, tmp_path):
    written = save_report(predictions, tmp_path / "eval")

    assert set(written) == {"predictions", "confusions", "summary"}
    for path in written.values():
        assert path.is_file()
        assert path.stat().st_size > 0
