"""Dataset integrity checks."""

from __future__ import annotations

import shutil

import pytest

from medicinal_leaf.data.ingestion import build_index
from medicinal_leaf.data.validation import (
    content_digest,
    find_duplicates,
    raise_for_errors,
    validate_index,
)


def test_clean_dataset_passes(index_frame):
    report = validate_index(index_frame, min_images_per_class=5)
    assert report.ok
    assert report.n_classes == 5
    raise_for_errors(report)  # must not raise


def test_empty_index_is_an_error():
    import pandas as pd

    report = validate_index(pd.DataFrame(columns=["file_path", "class_name"]))
    assert not report.ok
    assert report.errors[0].code == "empty_index"


def test_duplicate_within_class_is_flagged(image_root):
    source = next((image_root / "Mint").glob("*.jpg"))
    shutil.copy(source, image_root / "Mint" / "copy.jpg")

    frame = build_index(image_root)
    report = validate_index(frame, min_images_per_class=5)

    codes = {issue.code for issue in report.issues}
    assert "duplicate_image" in codes
    assert not report.ok


def test_duplicate_across_classes_is_flagged(image_root):
    source = next((image_root / "Mint").glob("*.jpg"))
    shutil.copy(source, image_root / "Neem" / "leaked.jpg")

    frame = build_index(image_root)
    report = validate_index(frame, min_images_per_class=5)

    codes = {issue.code for issue in report.issues}
    assert "duplicate_across_classes" in codes


def test_small_class_is_an_error(index_frame):
    report = validate_index(index_frame, min_images_per_class=100, check_duplicates=False)
    assert not report.ok
    assert all(i.code == "class_too_small" for i in report.errors)
    with pytest.raises(ValueError, match="validation failed"):
        raise_for_errors(report)


def test_tiny_images_warn_but_do_not_block(index_frame):
    report = validate_index(
        index_frame, min_side=1000, min_images_per_class=5, check_duplicates=False
    )
    assert report.ok  # warnings only
    assert {i.code for i in report.warnings} == {"small_image"}


def test_imbalance_warning(index_frame):
    trimmed = index_frame.drop(index_frame[index_frame["class_name"] == "Amla"].index[2:])
    report = validate_index(
        trimmed, min_images_per_class=2, max_imbalance_ratio=1.5, check_duplicates=False
    )
    assert "class_imbalance" in {i.code for i in report.warnings}


def test_missing_file_is_an_error(index_frame, tmp_path):
    frame = index_frame.copy()
    frame.loc[0, "file_path"] = str(tmp_path / "vanished.jpg")
    report = validate_index(frame, min_images_per_class=5, check_duplicates=False)
    assert "missing_file" in {i.code for i in report.errors}


def test_content_digest_is_stable_and_distinguishing(image_root):
    paths = sorted((image_root / "Mint").glob("*.jpg"))[:2]
    assert content_digest(paths[0]) == content_digest(paths[0])
    assert content_digest(paths[0]) != content_digest(paths[1])


def test_find_duplicates_groups_identical_bytes(image_root):
    source = next((image_root / "Tulsi").glob("*.jpg"))
    shutil.copy(source, image_root / "Tulsi" / "again.jpg")

    frame = build_index(image_root)
    duplicates = find_duplicates(frame)

    assert len(duplicates) == 1
    assert len(next(iter(duplicates.values()))) == 2


def test_report_frame_and_summary(index_frame):
    report = validate_index(index_frame, min_side=1000, min_images_per_class=5)
    frame = report.to_frame()
    assert list(frame.columns) == ["severity", "code", "message", "file_path"]
    assert "warning(s)" in report.summary()
