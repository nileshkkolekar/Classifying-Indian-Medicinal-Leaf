"""Read the dataset and model artifacts from Amazon S3 (FR-1).

Optional throughout: with no ``aws.*`` configuration the pipeline runs
entirely from local disk. boto3 lives in the ``aws`` extra and is imported
lazily, so a training-only or test environment need not install it.

No credentials appear here or anywhere in configuration. boto3 resolves them
from the environment, an instance profile, or — in the deployed case — an ECS
task role, which is what keeps them out of git (NFR-4) and scopeable to one
bucket (NFR-7).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from medicinal_leaf.config.settings import Settings

logger = logging.getLogger(__name__)

S3_SCHEME = "s3"

DEFAULT_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")


class S3ConfigurationError(RuntimeError):
    """S3 was asked for but is unavailable or misconfigured."""


@dataclass(frozen=True, slots=True)
class S3Location:
    """A bucket and key, parsed from an ``s3://`` URI."""

    bucket: str
    key: str

    @classmethod
    def from_uri(cls, uri: str) -> S3Location:
        parsed = urlparse(uri)
        if parsed.scheme != S3_SCHEME:
            raise ValueError(f"Expected an s3:// URI, got {uri!r}")
        if not parsed.netloc:
            raise ValueError(f"URI has no bucket: {uri!r}")
        return cls(bucket=parsed.netloc, key=parsed.path.lstrip("/"))

    @property
    def uri(self) -> str:
        return f"{S3_SCHEME}://{self.bucket}/{self.key}"

    @property
    def prefix(self) -> str:
        """The key as a listing prefix, with exactly one trailing slash."""
        return f"{self.key.rstrip('/')}/" if self.key else ""

    def __str__(self) -> str:
        return self.uri


def is_s3_uri(value: str | Path | None) -> bool:
    return isinstance(value, str) and value.startswith(f"{S3_SCHEME}://")


def build_client(region: str | None = None, endpoint_url: str | None = None) -> Any:
    """Create an S3 client, with a useful error when the extra is missing."""
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise S3ConfigurationError(
            "boto3 is not installed. Install the aws extra: pip install -e '.[aws]'"
        ) from exc

    return boto3.client("s3", region_name=region, endpoint_url=endpoint_url)


def client_from_settings(settings: Settings) -> Any:
    return build_client(settings.aws.region, settings.aws.endpoint_url)


def safe_destination(root: Path, relative_key: str) -> Path:
    """Resolve ``relative_key`` under ``root``, refusing to escape it.

    Object keys are attacker-influenced in the general case, and a key like
    ``../../.ssh/authorized_keys`` would otherwise write outside the download
    directory — the same class of bug as zip slip.

    Anomalous keys are rejected rather than normalised. Silently rewriting
    ``a/../../b.jpg`` to ``b.jpg`` would risk clobbering a legitimate
    ``b.jpg``, and a traversal segment in our own dataset bucket is worth
    surfacing instead of quietly repairing. Callers skip what this rejects.
    """
    pure = PurePosixPath(relative_key)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Object key escapes the destination directory: {relative_key!r}")

    parts = [p for p in pure.parts if p != "."]
    if not parts:
        raise ValueError(f"Object key resolves to no path: {relative_key!r}")

    candidate = root.joinpath(*parts).resolve()
    # Defence in depth: catches anything the segment check cannot anticipate,
    # such as a Windows drive letter arriving inside a key.
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Object key escapes the destination directory: {relative_key!r}")
    return candidate


def iter_object_keys(location: S3Location, *, client: Any) -> Iterator[str]:
    """Every key under a prefix, following pagination."""
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=location.bucket, Prefix=location.prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            # Console-created "folders" are zero-byte keys ending in a slash.
            if not key.endswith("/"):
                yield key


def download_file(location: S3Location, destination: Path, *, client: Any) -> Path:
    """Download one object, creating parent directories."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", location.uri, destination)
    client.download_file(location.bucket, location.key, str(destination))
    return destination


def download_prefix(
    location: S3Location,
    destination: Path,
    *,
    client: Any,
    extensions: Sequence[str] | None = DEFAULT_EXTENSIONS,
    overwrite: bool = False,
) -> list[Path]:
    """Mirror a prefix into ``destination``, preserving relative structure.

    The structure matters: ``s3://bucket/data/Neem/img.jpg`` must land as
    ``Data/Neem/img.jpg``, because the folder name is the class label.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    allowed = {e.lower() for e in extensions} if extensions else None

    downloaded: list[Path] = []
    skipped = 0

    for key in iter_object_keys(location, client=client):
        relative = key[len(location.prefix) :] if location.prefix else key
        if allowed is not None and PurePosixPath(relative).suffix.lower() not in allowed:
            continue

        try:
            target = safe_destination(destination, relative)
        except ValueError as exc:
            logger.warning("Skipping unsafe object key: %s", exc)
            skipped += 1
            continue

        if target.exists() and not overwrite:
            skipped += 1
            continue

        download_file(S3Location(location.bucket, key), target, client=client)
        downloaded.append(target)

    logger.info("Synced %d object(s) from %s (%d skipped)", len(downloaded), location.uri, skipped)
    return downloaded


def upload_file(path: Path, location: S3Location, *, client: Any) -> S3Location:
    """Upload one local file, for publishing a trained checkpoint."""
    logger.info("Uploading %s -> %s", path, location.uri)
    client.upload_file(str(path), location.bucket, location.key)
    return location


# ── Settings-driven helpers ──────────────────────────────────────────────


def sync_dataset(settings: Settings, *, client: Any | None = None) -> list[Path]:
    """Mirror ``aws.dataset_uri`` into ``data.raw_dir`` (FR-1)."""
    if not settings.aws.dataset_uri:
        raise S3ConfigurationError(
            "No dataset URI configured. Set aws.dataset_uri or MLC_AWS__DATASET_URI."
        )

    location = S3Location.from_uri(settings.aws.dataset_uri)
    return download_prefix(
        location,
        settings.data.raw_dir,
        client=client or client_from_settings(settings),
        extensions=settings.data.image_extensions,
    )


def download_checkpoint(settings: Settings, *, client: Any | None = None) -> Path:
    """Fetch ``aws.checkpoint_uri`` to the configured local checkpoint path."""
    if not settings.aws.checkpoint_uri:
        raise S3ConfigurationError(
            "No checkpoint URI configured. Set aws.checkpoint_uri or MLC_AWS__CHECKPOINT_URI."
        )

    location = S3Location.from_uri(settings.aws.checkpoint_uri)
    destination = settings.serving.checkpoint_path
    download_file(location, destination, client=client or client_from_settings(settings))

    # The sidecar is written alongside checkpoints and is useful but optional.
    sidecar = S3Location(location.bucket, f"{location.key.rsplit('.', 1)[0]}.meta.json")
    try:
        download_file(
            sidecar,
            destination.with_suffix(".meta.json"),
            client=client or client_from_settings(settings),
        )
    except Exception as exc:  # noqa: BLE001 - absence is normal, not an error
        logger.debug("No checkpoint sidecar at %s (%s)", sidecar.uri, exc)

    return destination
