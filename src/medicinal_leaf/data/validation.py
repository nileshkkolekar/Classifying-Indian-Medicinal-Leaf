"""Integrity checks that run between indexing and splitting.

The point is to fail loudly *before* a GPU-hour is spent: duplicate images
leaking across splits, a class with eight examples, a batch of CMYK scans, or
a folder of 40×40 thumbnails all produce metrics that look fine and
generalise terribly.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

Severity = Literal["error", "warning", "info"]

#: Modes that survive `.convert("RGB")` without surprises.
EXPECTED_MODES = frozenset({"RGB", "L", "RGBA", "P"})


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    file_path: str | None = None

    def __str__(self) -> str:
        where = f" [{self.file_path}]" if self.file_path else ""
        return f"{self.severity.upper()}: {self.message}{where}"


@dataclass
class ValidationReport:
    """Outcome of :func:`validate_index`."""

    n_images: int
    n_classes: int
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True when nothing blocking was found."""
        return not self.errors

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "severity": i.severity,
                    "code": i.code,
                    "message": i.message,
                    "file_path": i.file_path,
                }
                for i in self.issues
            ],
            columns=["severity", "code", "message", "file_path"],
        )

    def summary(self) -> str:
        counts: dict[str, int] = defaultdict(int)
        for issue in self.issues:
            counts[issue.code] += 1
        head = (
            f"{self.n_images} images / {self.n_classes} classes — "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )
        if not counts:
            return f"{head}\n  clean"
        lines = [f"  {code}: {n}" for code, n in sorted(counts.items())]
        return "\n".join([head, *lines])

    def __str__(self) -> str:
        return self.summary()


def content_digest(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of the file's bytes.

    Byte-level hashing catches exact re-saves and copy-paste duplicates, which
    is the common failure in scraped plant datasets. It will *not* catch
    re-encoded or resized near-duplicates — for those, compare perceptual
    hashes or embedding distances.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def find_duplicates(
    frame: pd.DataFrame,
    path_col: str = "file_path",
) -> dict[str, list[str]]:
    """Map digest -> file paths, keeping only digests seen more than once."""
    by_digest: dict[str, list[str]] = defaultdict(list)
    for path in frame[path_col]:
        try:
            by_digest[content_digest(path)].append(path)
        except OSError as exc:
            logger.warning("Could not hash %s: %s", path, exc)
    return {digest: paths for digest, paths in by_digest.items() if len(paths) > 1}


def _check_decodable(frame: pd.DataFrame, path_col: str) -> list[ValidationIssue]:
    """Fully decode every image — slow, so it is opt-in."""
    issues: list[ValidationIssue] = []
    for path in frame[path_col]:
        try:
            with Image.open(path) as image:
                image.load()
        except Exception as exc:  # noqa: BLE001 - report whatever the decoder raised
            issues.append(
                ValidationIssue("error", "undecodable", f"Image failed to decode: {exc}", path)
            )
    return issues


def validate_index(
    frame: pd.DataFrame,
    *,
    min_side: int = 32,
    max_aspect_ratio: float = 4.0,
    min_images_per_class: int = 10,
    max_imbalance_ratio: float = 5.0,
    check_duplicates: bool = True,
    check_decodable: bool = False,
    label_col: str = "class_name",
    path_col: str = "file_path",
) -> ValidationReport:
    """Run every check and collect the findings.

    Errors are conditions that make training results untrustworthy (empty
    index, duplicates, undecodable files, a class too small to stratify).
    Warnings are worth a look but not blocking.
    """
    issues: list[ValidationIssue] = []

    if frame.empty:
        return ValidationReport(
            n_images=0,
            n_classes=0,
            issues=[ValidationIssue("error", "empty_index", "No images were indexed.")],
        )

    n_classes = int(frame[label_col].nunique())

    # ── Missing files ────────────────────────────────────────────────────
    missing = [p for p in frame[path_col] if not Path(p).is_file()]
    issues.extend(
        ValidationIssue("error", "missing_file", "Indexed file no longer exists.", p)
        for p in missing
    )

    # ── Geometry ─────────────────────────────────────────────────────────
    too_small = frame[(frame["width"] < min_side) | (frame["height"] < min_side)]
    issues.extend(
        ValidationIssue(
            "warning",
            "small_image",
            f"Image is smaller than {min_side}px on a side "
            f"({row.width}x{row.height}); upscaling will blur detail.",
            row.file_path,
        )
        for row in too_small.itertuples()
    )

    ratios = frame["aspect_ratio"]
    extreme = frame[(ratios > max_aspect_ratio) | (ratios < 1 / max_aspect_ratio)]
    issues.extend(
        ValidationIssue(
            "warning",
            "extreme_aspect_ratio",
            f"Aspect ratio {row.aspect_ratio:.2f} exceeds {max_aspect_ratio}; "
            "letterboxing will waste most of the frame.",
            row.file_path,
        )
        for row in extreme.itertuples()
    )

    # ── Colour mode ──────────────────────────────────────────────────────
    unexpected_modes = frame[~frame["mode"].isin(EXPECTED_MODES)]
    issues.extend(
        ValidationIssue(
            "warning",
            "unexpected_mode",
            f"Unusual colour mode {row.mode!r}; conversion to RGB may shift colours.",
            row.file_path,
        )
        for row in unexpected_modes.itertuples()
    )

    # ── Class balance ────────────────────────────────────────────────────
    counts = frame[label_col].value_counts()
    for class_name, count in counts.items():
        if count < min_images_per_class:
            issues.append(
                ValidationIssue(
                    "error",
                    "class_too_small",
                    f"Class {class_name!r} has {count} images, "
                    f"below the minimum of {min_images_per_class}.",
                )
            )

    if len(counts) > 1:
        imbalance = float(counts.max() / counts.min())
        if imbalance > max_imbalance_ratio:
            issues.append(
                ValidationIssue(
                    "warning",
                    "class_imbalance",
                    f"Largest class is {imbalance:.1f}x the smallest "
                    f"({counts.idxmax()}={counts.max()}, {counts.idxmin()}={counts.min()}); "
                    "enable training.class_weighting or resample.",
                )
            )

    # ── Duplicates ───────────────────────────────────────────────────────
    if check_duplicates:
        for digest, paths in find_duplicates(frame, path_col).items():
            labels = {
                frame.loc[frame[path_col] == p, label_col].iloc[0]
                for p in paths
                if (frame[path_col] == p).any()
            }
            cross_class = len(labels) > 1
            issues.append(
                ValidationIssue(
                    "error",
                    "duplicate_across_classes" if cross_class else "duplicate_image",
                    (
                        f"{len(paths)} byte-identical copies"
                        + (f" spanning classes {sorted(labels)}" if cross_class else "")
                        + f" (sha256 {digest[:12]}): {', '.join(paths)}"
                    ),
                )
            )

    if check_decodable:
        issues.extend(_check_decodable(frame, path_col))

    report = ValidationReport(n_images=len(frame), n_classes=n_classes, issues=issues)
    logger.info("Validation: %s", report.summary().replace("\n", "; "))
    return report


def raise_for_errors(report: ValidationReport) -> None:
    """Raise if the report contains anything blocking."""
    if report.ok:
        return
    details = "\n".join(f"  - {issue}" for issue in report.errors[:20])
    more = "" if len(report.errors) <= 20 else f"\n  ... and {len(report.errors) - 20} more"
    raise ValueError(
        f"Dataset validation failed with {len(report.errors)} error(s):\n{details}{more}"
    )
