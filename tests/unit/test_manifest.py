"""Manifest round-tripping and the label mapping contract."""

from __future__ import annotations

import pytest

from medicinal_leaf.data.manifest import (
    class_mapping,
    manifest_fingerprint,
    read_manifest,
    sidecar_path,
    split_frame,
    write_manifest,
)


def test_write_then_read_round_trips(split_frame_fixture, tmp_path):
    path = tmp_path / "manifest.csv"
    meta = write_manifest(split_frame_fixture, path)

    frame, loaded_meta = read_manifest(path)

    assert len(frame) == len(split_frame_fixture)
    assert loaded_meta.class_to_idx == meta.class_to_idx
    assert loaded_meta.fingerprint == meta.fingerprint
    assert loaded_meta.n_images == len(frame)


def test_sidecar_is_written(split_frame_fixture, tmp_path):
    path = tmp_path / "manifest.csv"
    write_manifest(split_frame_fixture, path)
    assert sidecar_path(path).is_file()


def test_label_idx_matches_mapping(split_frame_fixture, tmp_path):
    path = tmp_path / "manifest.csv"
    meta = write_manifest(split_frame_fixture, path)
    frame, _ = read_manifest(path)

    for _, row in frame.iterrows():
        assert row["label_idx"] == meta.class_to_idx[row["class_name"]]


def test_class_mapping_is_alphabetical(split_frame_fixture):
    mapping = class_mapping(split_frame_fixture)
    assert list(mapping) == sorted(mapping)
    assert list(mapping.values()) == list(range(len(mapping)))


def test_class_names_are_ordered_by_index(split_frame_fixture, tmp_path):
    path = tmp_path / "manifest.csv"
    meta = write_manifest(split_frame_fixture, path)
    assert meta.class_names == sorted(meta.class_to_idx)
    assert meta.num_classes == 5


def test_fingerprint_is_order_independent(split_frame_fixture):
    shuffled = split_frame_fixture.sample(frac=1.0, random_state=0)
    assert manifest_fingerprint(split_frame_fixture) == manifest_fingerprint(shuffled)


def test_fingerprint_changes_with_the_data(split_frame_fixture):
    original = manifest_fingerprint(split_frame_fixture)
    altered = split_frame_fixture.copy()
    altered.loc[altered.index[0], "split"] = "test"
    assert manifest_fingerprint(altered) != original


def test_missing_sidecar_is_rebuilt(split_frame_fixture, tmp_path):
    path = tmp_path / "manifest.csv"
    meta = write_manifest(split_frame_fixture, path)
    sidecar_path(path).unlink()

    _, rebuilt = read_manifest(path)
    assert rebuilt.class_to_idx == meta.class_to_idx
    assert rebuilt.fingerprint == meta.fingerprint


def test_reading_a_missing_manifest_explains_itself(tmp_path):
    with pytest.raises(FileNotFoundError, match="leaf-train prepare"):
        read_manifest(tmp_path / "absent.csv")


def test_unsplit_frame_is_rejected(index_frame, tmp_path):
    with pytest.raises(KeyError, match="stratified_split"):
        write_manifest(index_frame, tmp_path / "manifest.csv")


def test_empty_frame_is_rejected(split_frame_fixture, tmp_path):
    with pytest.raises(ValueError, match="empty manifest"):
        write_manifest(split_frame_fixture.head(0), tmp_path / "manifest.csv")


def test_split_frame_selects_rows(split_frame_fixture):
    train = split_frame(split_frame_fixture, "train")
    assert (train["split"] == "train").all()
    assert len(train) < len(split_frame_fixture)


def test_split_frame_reports_available_splits(split_frame_fixture):
    with pytest.raises(ValueError, match="Available"):
        split_frame(split_frame_fixture, "holdout")
