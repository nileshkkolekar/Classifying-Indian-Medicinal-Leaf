"""Turn a folder-per-class image tree into a tabular index.

The expected layout is one directory per species::

    Data/
        Aloevera/ img001.jpg ...
        Amla/     img001.jpg ...
        ...

Reading only the image *header* (``Image.open`` is lazy) keeps a full scan of
a few thousand files to well under a second; pixels are decoded later, in the
Dataset, and only for the images a batch actually needs.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from medicinal_leaf.config.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """One row of the index — metadata only, no pixels."""

    file_path: str
    class_name: str
    width: int
    height: int
    aspect_ratio: float
    mode: str
    size_bytes: int


#: Canonical column order of the index frame. The manifest extends this.
INDEX_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(ImageRecord))


def discover_classes(root: Path) -> list[str]:
    """Return the sorted class names, one per immediate subdirectory."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    classes = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
    if not classes:
        raise ValueError(
            f"No class subdirectories found under {root}. Expected one folder per species."
        )
    return classes


def iter_image_paths(
    class_dir: Path,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
) -> Iterator[Path]:
    """Yield image files in ``class_dir``, sorted for reproducibility."""
    allowed = {ext.lower() for ext in extensions}
    for path in sorted(class_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in allowed:
            yield path


def read_image_record(path: Path, class_name: str) -> ImageRecord | None:
    """Build a record from an image's header, or ``None`` if it cannot be read."""
    try:
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
    except Exception as exc:  # noqa: BLE001 - any decoder failure means "skip it"
        logger.warning("Could not read %s: %s", path, exc)
        return None

    if height == 0:
        logger.warning("Skipping zero-height image: %s", path)
        return None

    return ImageRecord(
        file_path=str(path),
        class_name=class_name,
        width=width,
        height=height,
        aspect_ratio=width / height,
        mode=mode,
        size_bytes=path.stat().st_size,
    )


def build_index(
    root: Path,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    classes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Scan ``root`` and return one row per readable image.

    Unreadable files are logged and dropped rather than raising — a single
    truncated JPEG should not abort a scan of several thousand images. Use
    :func:`medicinal_leaf.data.validation.validate_index` to turn the
    survivors into a pass/fail judgement.
    """
    root = Path(root)
    class_names = list(classes) if classes is not None else discover_classes(root)

    records: list[ImageRecord] = []
    skipped = 0

    for class_name in class_names:
        class_dir = root / class_name
        if not class_dir.is_dir():
            logger.warning("Declared class has no directory, skipping: %s", class_dir)
            continue

        for image_path in iter_image_paths(class_dir, extensions):
            record = read_image_record(image_path, class_name)
            if record is None:
                skipped += 1
            else:
                records.append(record)

    frame = pd.DataFrame(records, columns=list(INDEX_COLUMNS))
    logger.info(
        "Indexed %d images across %d classes (%d skipped) from %s",
        len(frame),
        frame["class_name"].nunique() if not frame.empty else 0,
        skipped,
        root,
    )
    return frame


def build_index_from_settings(settings: Settings) -> pd.DataFrame:
    """Convenience wrapper reading paths and extensions from configuration."""
    return build_index(settings.data.raw_dir, settings.data.image_extensions)


class LeafDataset(Dataset):
    """Serve ``(image_tensor, label)`` pairs from an index frame.

    The frame is expected to carry at least ``file_path`` and ``class_name``.
    Passing a pre-filtered view (one split) keeps the dataset honest about
    which rows it may touch::

        train = LeafDataset(frame[frame.split == "train"], mapping, transform)
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        class_to_idx: dict[str, int],
        transform: Any | None = None,
        *,
        label_col: str = "class_name",
        path_col: str = "file_path",
        return_path: bool = False,
    ) -> None:
        missing = {label_col, path_col} - set(frame.columns)
        if missing:
            raise KeyError(f"Frame is missing required column(s): {sorted(missing)}")

        unknown = set(frame[label_col].unique()) - set(class_to_idx)
        if unknown:
            raise KeyError(f"Labels absent from class_to_idx: {sorted(unknown)}")

        self.frame = frame.reset_index(drop=True)
        self.class_to_idx = dict(class_to_idx)
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.transform = transform
        self.label_col = label_col
        self.path_col = path_col
        self.return_path = return_path

        self._paths: list[str] = self.frame[path_col].tolist()
        self._targets: list[int] = [self.class_to_idx[name] for name in self.frame[label_col]]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        path = self._paths[index]
        target = self._targets[index]

        with Image.open(path) as image:
            # Grayscale and palette images appear in field-collected data;
            # normalise everything to three channels up front.
            sample: Any = image.convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        if self.return_path:
            return sample, target, path
        return sample, target

    @property
    def targets(self) -> list[int]:
        """Integer labels in row order — for samplers and class weights."""
        return list(self._targets)

    @property
    def classes(self) -> list[str]:
        return [self.idx_to_class[i] for i in sorted(self.idx_to_class)]

    def class_counts(self) -> dict[str, int]:
        counts = self.frame[self.label_col].value_counts()
        return {name: int(counts.get(name, 0)) for name in self.classes}

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights, normalised to mean 1.0."""
        counts = torch.tensor(
            [max(self.class_counts()[name], 1) for name in self.classes],
            dtype=torch.float32,
        )
        weights = counts.sum() / (len(counts) * counts)
        return weights / weights.mean()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n={len(self)}, classes={len(self.class_to_idx)}, "
            f"transform={type(self.transform).__name__ if self.transform else None})"
        )
