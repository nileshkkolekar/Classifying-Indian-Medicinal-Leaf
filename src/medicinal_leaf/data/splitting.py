"""Stratified, reproducible train/validation/test splits.

Stratification matters here: with five species and an uneven collection, a
random split can leave a class almost absent from validation, which makes
macro-F1 jump around for reasons that have nothing to do with the model.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")


def stratified_split(
    frame: pd.DataFrame,
    *,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    label_col: str = "class_name",
    split_col: str = "split",
) -> pd.DataFrame:
    """Return a copy of ``frame`` with a ``split`` column added.

    Implemented as two successive stratified splits: first carve off the test
    set, then divide the remainder into train and validation.
    """
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total:.4f}")
    if frame.empty:
        raise ValueError("Cannot split an empty frame.")

    counts = frame[label_col].value_counts()
    # Two successive splits need at least one sample per class on each side.
    too_small = counts[counts < 3]
    if not too_small.empty:
        raise ValueError(
            "Every class needs at least 3 images for a three-way stratified split; "
            f"too small: {too_small.to_dict()}"
        )

    working = frame.reset_index(drop=True)

    remainder, test = train_test_split(
        working,
        test_size=test_size,
        stratify=working[label_col],
        random_state=seed,
        shuffle=True,
    )

    # Rescale: val_size is a fraction of the whole, not of the remainder.
    val_fraction_of_remainder = val_size / (train_size + val_size)
    train, val = train_test_split(
        remainder,
        test_size=val_fraction_of_remainder,
        stratify=remainder[label_col],
        random_state=seed,
        shuffle=True,
    )

    result = working.copy()
    result[split_col] = pd.Series(dtype="object")
    for name, part in zip(SPLIT_NAMES, (train, val, test), strict=True):
        result.loc[part.index, split_col] = name

    unassigned = int(result[split_col].isna().sum())
    if unassigned:  # pragma: no cover - defensive
        raise AssertionError(f"{unassigned} rows were not assigned to a split.")

    logger.info(
        "Split %d images -> %s",
        len(result),
        result[split_col].value_counts().to_dict(),
    )
    return result


def split_summary(
    frame: pd.DataFrame,
    label_col: str = "class_name",
    split_col: str = "split",
) -> pd.DataFrame:
    """Class-by-split counts, with row and column totals."""
    table = pd.crosstab(frame[label_col], frame[split_col])
    ordered = [name for name in SPLIT_NAMES if name in table.columns]
    table = table[ordered]
    table["total"] = table.sum(axis=1)
    table.loc["total"] = table.sum(axis=0)
    return table


def split_proportions(
    frame: pd.DataFrame,
    label_col: str = "class_name",
    split_col: str = "split",
) -> pd.DataFrame:
    """Per-class share of each split — should be near-identical across rows."""
    table = pd.crosstab(frame[label_col], frame[split_col], normalize="index")
    ordered = [name for name in SPLIT_NAMES if name in table.columns]
    return table[ordered].round(4)


def assert_no_leakage(
    frame: pd.DataFrame,
    key_col: str = "file_path",
    split_col: str = "split",
) -> None:
    """Raise if any key appears in more than one split."""
    per_key = frame.groupby(key_col)[split_col].nunique()
    offenders = per_key[per_key > 1]
    if not offenders.empty:
        sample = list(offenders.index[:10])
        raise ValueError(f"{len(offenders)} item(s) appear in multiple splits, e.g. {sample}")
