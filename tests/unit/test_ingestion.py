"""Indexing a folder-per-class image tree."""

from __future__ import annotations

import pytest

from medicinal_leaf.data.ingestion import (
    INDEX_COLUMNS,
    build_index,
    discover_classes,
    iter_image_paths,
    read_image_record,
)
from tests.conftest import CLASS_SPECS


def test_discover_classes_is_sorted(image_root):
    assert discover_classes(image_root) == sorted(CLASS_SPECS)


def test_discover_classes_rejects_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_classes(tmp_path / "nope")


def test_discover_classes_rejects_flat_directory(tmp_path):
    (tmp_path / "loose.jpg").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="No class subdirectories"):
        discover_classes(tmp_path)


def test_index_has_one_row_per_image(index_frame):
    assert len(index_frame) == sum(CLASS_SPECS.values())
    assert list(index_frame.columns) == list(INDEX_COLUMNS)


def test_index_counts_match_the_tree(index_frame):
    counts = index_frame["class_name"].value_counts().to_dict()
    assert counts == CLASS_SPECS


def test_aspect_ratio_is_width_over_height(index_frame):
    row = index_frame.iloc[0]
    assert row["aspect_ratio"] == pytest.approx(row["width"] / row["height"])


def test_unreadable_files_are_skipped_not_raised(image_root):
    (image_root / "Mint" / "truncated.jpg").write_bytes(b"\xff\xd8\xff not really a jpeg")
    frame = build_index(image_root)
    # The corrupt file is dropped; everything else survives.
    assert len(frame) == sum(CLASS_SPECS.values())


def test_extension_filter_is_honoured(image_root):
    (image_root / "Mint" / "notes.txt").write_text("ignore me")
    paths = list(iter_image_paths(image_root / "Mint"))
    assert all(p.suffix == ".jpg" for p in paths)
    assert len(paths) == CLASS_SPECS["Mint"]


def test_read_image_record_returns_none_for_garbage(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"definitely not an image")
    assert read_image_record(bad, "Mint") is None


def test_index_restricted_to_named_classes(image_root):
    frame = build_index(image_root, classes=["Mint", "Neem"])
    assert set(frame["class_name"]) == {"Mint", "Neem"}
