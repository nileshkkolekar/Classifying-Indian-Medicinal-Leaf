"""Dataset discovery, integrity checks, splitting and manifest IO."""

from medicinal_leaf.data.ingestion import (
    INDEX_COLUMNS,
    ImageRecord,
    LeafDataset,
    build_index,
    build_index_from_settings,
    discover_classes,
)
from medicinal_leaf.data.manifest import (
    ManifestMeta,
    class_mapping,
    read_manifest,
    write_manifest,
)
from medicinal_leaf.data.splitting import (
    assert_no_leakage,
    split_summary,
    stratified_split,
)
from medicinal_leaf.data.validation import (
    ValidationIssue,
    ValidationReport,
    raise_for_errors,
    validate_index,
)

__all__ = [
    "INDEX_COLUMNS",
    "ImageRecord",
    "LeafDataset",
    "ManifestMeta",
    "ValidationIssue",
    "ValidationReport",
    "assert_no_leakage",
    "build_index",
    "build_index_from_settings",
    "class_mapping",
    "discover_classes",
    "raise_for_errors",
    "read_manifest",
    "split_summary",
    "stratified_split",
    "validate_index",
    "write_manifest",
]
