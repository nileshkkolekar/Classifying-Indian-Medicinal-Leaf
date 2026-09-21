"""The UI's CSV-to-table conversion.

A queued job returns CSV while the immediate path returns JSON; both feed the
same renderer, so this is where the two shapes have to agree.
"""

from __future__ import annotations

import pandas as pd
import pytest

from medicinal_leaf.ui.streamlit_app import results_frame, results_frame_from_csv

COLUMNS = ["File", "Predicted species", "Confidence", "Verdict", "Needs review", "Note"]

CSV = (
    "filename,verdict,label,confidence,needs_review,note\n"
    "a.jpg,classified,Tulsi,0.964000,False,\n"
    "b.jpg,needs_review,Mint,0.551000,True,Confidence below threshold\n"
    "c.jpg,unable_to_classify,,0.180000,True,Below the floor\n"
    "d.jpg,error,,,True,Could not read image\n"
)


def test_csv_maps_onto_the_display_columns():
    frame = results_frame_from_csv(CSV)
    assert list(frame.columns) == COLUMNS
    assert len(frame) == 4


def test_confidence_becomes_numeric():
    frame = results_frame_from_csv(CSV)
    assert frame["Confidence"].iloc[0] == pytest.approx(0.964)
    # A row with no confidence must be NaN, not the string "".
    assert pd.isna(frame["Confidence"].iloc[3])


def test_missing_label_shows_a_dash():
    """FR-14: a declined image names no species."""
    frame = results_frame_from_csv(CSV)
    assert frame["Predicted species"].iloc[2] == "—"
    assert frame["Predicted species"].iloc[0] == "Tulsi"


def test_needs_review_becomes_boolean():
    frame = results_frame_from_csv(CSV)
    assert frame["Needs review"].tolist() == [False, True, True, True]


def test_empty_csv_yields_an_empty_frame():
    for text in ("", "   ", "filename,verdict,label,confidence,needs_review,note\n"):
        frame = results_frame_from_csv(text)
        assert list(frame.columns) == COLUMNS
        assert frame.empty


def test_both_paths_produce_the_same_columns():
    """The JSON and CSV shapes must converge, or the shared renderer breaks."""
    json_frame = results_frame(
        [
            {
                "filename": "a.jpg",
                "verdict": "classified",
                "label": "Tulsi",
                "confidence": 0.964,
                "needs_review": False,
                "note": None,
            }
        ]
    )
    assert list(json_frame.columns) == list(results_frame_from_csv(CSV).columns)
