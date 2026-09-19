"""Persist the split index so every later stage sees identical data.

The manifest is the contract between EDA, training, evaluation and inference:
a CSV of rows plus a JSON sidecar holding the label mapping and a fingerprint.
Checkpoints record that fingerprint, so a model can always be traced back to
the exact split it was trained on.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from medicinal_leaf.data.ingestion import INDEX_COLUMNS

logger = logging.getLogger(__name__)

#: Index columns plus the split assignment and integer label.
MANIFEST_COLUMNS: tuple[str, ...] = (*INDEX_COLUMNS, "split", "label_idx")

SIDECAR_SUFFIX = ".meta.json"


@dataclass(slots=True)
class ManifestMeta:
    """Everything needed to interpret a manifest CSV."""

    class_to_idx: dict[str, int]
    n_images: int
    fingerprint: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    split_counts: dict[str, int] = field(default_factory=dict)
    version: int = 1

    @property
    def class_names(self) -> list[str]:
        """Class names ordered by their integer label."""
        return [name for name, _ in sorted(self.class_to_idx.items(), key=lambda kv: kv[1])]

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)


def sidecar_path(manifest_path: str | Path) -> Path:
    """Location of the JSON sidecar belonging to ``manifest_path``."""
    path = Path(manifest_path)
    return path.with_suffix(path.suffix + SIDECAR_SUFFIX)


def class_mapping(frame: pd.DataFrame, label_col: str = "class_name") -> dict[str, int]:
    """Stable name -> index mapping.

    Sorted alphabetically so the mapping is identical across machines and
    runs; never derive it from row order.
    """
    return {name: idx for idx, name in enumerate(sorted(frame[label_col].unique()))}


def manifest_fingerprint(frame: pd.DataFrame) -> str:
    """Short digest over (path, class, split) — changes whenever the data does."""
    digest = hashlib.sha256()
    columns = [c for c in ("file_path", "class_name", "split") if c in frame.columns]
    for row in frame.sort_values("file_path")[columns].itertuples(index=False):
        digest.update("\x1f".join(str(value) for value in row).encode("utf-8"))
    return digest.hexdigest()[:16]


def write_manifest(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    class_to_idx: dict[str, int] | None = None,
    label_col: str = "class_name",
) -> ManifestMeta:
    """Write the manifest CSV and its sidecar, returning the metadata.

    A ``label_idx`` column is derived from ``class_to_idx`` if absent, so the
    CSV alone is enough to reconstruct targets.
    """
    if frame.empty:
        raise ValueError("Refusing to write an empty manifest.")
    if "split" not in frame.columns:
        raise KeyError("Frame has no 'split' column — run stratified_split first.")

    mapping = class_to_idx or class_mapping(frame, label_col)
    out = frame.copy()
    out["label_idx"] = out[label_col].map(mapping).astype(int)

    ordered = [c for c in MANIFEST_COLUMNS if c in out.columns]
    extras = [c for c in out.columns if c not in ordered]
    out = out[[*ordered, *extras]]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    meta = ManifestMeta(
        class_to_idx=mapping,
        n_images=len(out),
        fingerprint=manifest_fingerprint(out),
        split_counts={str(k): int(v) for k, v in out["split"].value_counts().items()},
    )
    sidecar_path(path).write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    logger.info("Wrote manifest: %s (%d rows, fingerprint %s)", path, len(out), meta.fingerprint)
    return meta


def read_manifest(path: str | Path) -> tuple[pd.DataFrame, ManifestMeta]:
    """Load a manifest and its sidecar.

    If the sidecar is missing the metadata is rebuilt from the CSV, so a
    hand-edited manifest still loads.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No manifest at {path}. Build one with `leaf-train prepare`.")

    frame = pd.read_csv(path)

    sidecar = sidecar_path(path)
    if sidecar.is_file():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        meta = ManifestMeta(**payload)
    else:
        logger.warning("Sidecar %s missing; rebuilding metadata from the CSV.", sidecar)
        mapping = class_mapping(frame)
        meta = ManifestMeta(
            class_to_idx=mapping,
            n_images=len(frame),
            fingerprint=manifest_fingerprint(frame),
            split_counts={str(k): int(v) for k, v in frame["split"].value_counts().items()},
        )

    return frame, meta


def load_class_names(path: str | Path) -> list[str]:
    """Class names in label order, without loading the full manifest frame."""
    _, meta = read_manifest(path)
    return meta.class_names


def split_frame(frame: pd.DataFrame, split: str, split_col: str = "split") -> pd.DataFrame:
    """Rows belonging to one split, with a clear error when it is empty."""
    subset = frame[frame[split_col] == split]
    if subset.empty:
        available = sorted(frame[split_col].dropna().unique())
        raise ValueError(f"Split {split!r} is empty. Available: {available}")
    return subset.reset_index(drop=True)
