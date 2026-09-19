"""Stratified splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from medicinal_leaf.data.splitting import (
    assert_no_leakage,
    split_proportions,
    split_summary,
    stratified_split,
)
from tests.conftest import CLASS_SPECS


def test_every_row_gets_exactly_one_split(split_frame_fixture):
    assert split_frame_fixture["split"].notna().all()
    assert set(split_frame_fixture["split"].unique()) == {"train", "val", "test"}
    assert len(split_frame_fixture) == sum(CLASS_SPECS.values())


def test_split_sizes_are_close_to_requested(index_frame):
    frame = stratified_split(index_frame, train_size=0.7, val_size=0.15, test_size=0.15, seed=3)
    shares = frame["split"].value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.05)
    assert shares["val"] == pytest.approx(0.15, abs=0.05)
    assert shares["test"] == pytest.approx(0.15, abs=0.05)


def test_every_class_appears_in_every_split(split_frame_fixture):
    table = pd.crosstab(split_frame_fixture["class_name"], split_frame_fixture["split"])
    assert (table > 0).all().all()


def test_stratification_holds_per_class(split_frame_fixture):
    proportions = split_proportions(split_frame_fixture)
    # Each class should devote a similar share to train.
    assert proportions["train"].max() - proportions["train"].min() < 0.2


def test_no_file_appears_in_two_splits(split_frame_fixture):
    assert_no_leakage(split_frame_fixture)


def test_leakage_is_detected():
    frame = pd.DataFrame(
        {
            "file_path": ["a.jpg", "a.jpg", "b.jpg"],
            "class_name": ["Mint", "Mint", "Neem"],
            "split": ["train", "test", "train"],
        }
    )
    with pytest.raises(ValueError, match="multiple splits"):
        assert_no_leakage(frame)


def test_split_is_reproducible(index_frame):
    first = stratified_split(index_frame, seed=11)
    second = stratified_split(index_frame, seed=11)
    pd.testing.assert_series_equal(first["split"], second["split"])


def test_different_seeds_give_different_splits(index_frame):
    first = stratified_split(index_frame, seed=1)
    second = stratified_split(index_frame, seed=2)
    assert not first["split"].equals(second["split"])


def test_fractions_must_sum_to_one(index_frame):
    with pytest.raises(ValueError, match="sum to 1.0"):
        stratified_split(index_frame, train_size=0.5, val_size=0.2, test_size=0.2)


def test_class_with_too_few_images_is_rejected(index_frame):
    crippled = index_frame.drop(index_frame[index_frame["class_name"] == "Neem"].index[1:])
    with pytest.raises(ValueError, match="at least 3 images"):
        stratified_split(crippled)


def test_empty_frame_is_rejected():
    with pytest.raises(ValueError, match="empty frame"):
        stratified_split(pd.DataFrame(columns=["class_name"]))


def test_summary_has_totals(split_frame_fixture):
    table = split_summary(split_frame_fixture)
    assert table.loc["total", "total"] == len(split_frame_fixture)
    assert list(table.columns) == ["train", "val", "test", "total"]
