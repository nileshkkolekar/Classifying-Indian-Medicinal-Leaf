"""S3 URI handling, key safety and prefix mirroring.

A stub client stands in for boto3 throughout: these tests are about our
logic — structure preservation, extension filtering, refusing to write
outside the destination — not about AWS.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medicinal_leaf.data.s3 import (
    S3ConfigurationError,
    S3Location,
    download_prefix,
    is_s3_uri,
    safe_destination,
    sync_dataset,
)


class StubS3Client:
    """Minimal boto3 S3 stand-in: lists keys and writes their payloads."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.downloaded: list[str] = []

    def get_paginator(self, _operation: str):
        objects = self.objects

        class _Paginator:
            def paginate(self, Bucket: str, Prefix: str = ""):  # noqa: N803 - boto3 casing
                contents = [
                    {"Key": key, "Size": len(payload)}
                    for key, payload in objects.items()
                    if key.startswith(Prefix)
                ]
                # Exercise the pagination path rather than one big page.
                for start in range(0, max(len(contents), 1), 2):
                    yield {"Contents": contents[start : start + 2]}

        return _Paginator()

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:  # noqa: N803
        self.downloaded.append(Key)
        Path(Filename).write_bytes(self.objects[Key])

    def upload_file(self, Filename: str, Bucket: str, Key: str) -> None:  # noqa: N803
        self.objects[Key] = Path(Filename).read_bytes()


# ── URI parsing ──────────────────────────────────────────────────────────


def test_parses_bucket_and_key():
    location = S3Location.from_uri("s3://leaf-data/datasets/v1/Neem/img.jpg")
    assert location.bucket == "leaf-data"
    assert location.key == "datasets/v1/Neem/img.jpg"


def test_round_trips_to_uri():
    uri = "s3://leaf-data/datasets/v1"
    assert S3Location.from_uri(uri).uri == uri


def test_prefix_gets_exactly_one_trailing_slash():
    assert S3Location.from_uri("s3://b/data").prefix == "data/"
    assert S3Location.from_uri("s3://b/data/").prefix == "data/"
    assert S3Location.from_uri("s3://b").prefix == ""


@pytest.mark.parametrize("uri", ["https://example.com/x", "/local/path", "leaf-data/key"])
def test_rejects_non_s3_uris(uri):
    with pytest.raises(ValueError, match="Expected an s3"):
        S3Location.from_uri(uri)


def test_rejects_uri_without_bucket():
    with pytest.raises(ValueError, match="no bucket"):
        S3Location.from_uri("s3:///just-a-key")


def test_is_s3_uri():
    assert is_s3_uri("s3://bucket/key")
    assert not is_s3_uri("/local/path")
    assert not is_s3_uri(Path("/local/path"))
    assert not is_s3_uri(None)


# ── Key safety ───────────────────────────────────────────────────────────


def test_safe_destination_joins_under_root(tmp_path):
    assert safe_destination(tmp_path, "Neem/img.jpg") == (tmp_path / "Neem" / "img.jpg").resolve()


def test_safe_destination_rejects_traversal(tmp_path):
    """A key must never write outside the download directory."""
    with pytest.raises(ValueError, match="escapes"):
        safe_destination(tmp_path, "../../etc/passwd")


def test_safe_destination_rejects_absolute_keys(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        safe_destination(tmp_path, "/etc/passwd")


def test_safe_destination_allows_a_dot_segment(tmp_path):
    """`./` is noise, not an escape attempt."""
    assert safe_destination(tmp_path, "./Neem/img.jpg").name == "img.jpg"


def test_safe_destination_rejects_empty_key(tmp_path):
    with pytest.raises(ValueError, match="no path"):
        safe_destination(tmp_path, "")


# ── Prefix mirroring ─────────────────────────────────────────────────────


@pytest.fixture
def dataset_client() -> StubS3Client:
    return StubS3Client(
        {
            "datasets/v1/Neem/a.jpg": b"neem-a",
            "datasets/v1/Neem/b.jpg": b"neem-b",
            "datasets/v1/Tulsi/c.png": b"tulsi-c",
            "datasets/v1/README.txt": b"not an image",
            "datasets/v1/Tulsi/": b"",
        }
    )


def test_mirror_preserves_class_folders(tmp_path, dataset_client):
    """The folder name is the label, so structure has to survive the copy."""
    location = S3Location.from_uri("s3://leaf-data/datasets/v1")
    downloaded = download_prefix(location, tmp_path, client=dataset_client)

    assert len(downloaded) == 3
    assert (tmp_path / "Neem" / "a.jpg").read_bytes() == b"neem-a"
    assert (tmp_path / "Tulsi" / "c.png").read_bytes() == b"tulsi-c"


def test_mirror_filters_by_extension(tmp_path, dataset_client):
    location = S3Location.from_uri("s3://leaf-data/datasets/v1")
    download_prefix(location, tmp_path, client=dataset_client)
    assert not (tmp_path / "README.txt").exists()


def test_mirror_ignores_folder_placeholder_keys(tmp_path, dataset_client):
    location = S3Location.from_uri("s3://leaf-data/datasets/v1")
    download_prefix(location, tmp_path, client=dataset_client)
    assert "datasets/v1/Tulsi/" not in dataset_client.downloaded


def test_mirror_skips_existing_files_by_default(tmp_path, dataset_client):
    location = S3Location.from_uri("s3://leaf-data/datasets/v1")
    download_prefix(location, tmp_path, client=dataset_client)
    dataset_client.downloaded.clear()

    again = download_prefix(location, tmp_path, client=dataset_client)
    assert again == []
    assert dataset_client.downloaded == []


def test_mirror_can_overwrite(tmp_path, dataset_client):
    location = S3Location.from_uri("s3://leaf-data/datasets/v1")
    download_prefix(location, tmp_path, client=dataset_client)

    again = download_prefix(location, tmp_path, client=dataset_client, overwrite=True)
    assert len(again) == 3


def test_unsafe_key_is_skipped_not_fatal(tmp_path):
    client = StubS3Client({"data/../../escape.jpg": b"x", "data/ok.jpg": b"y"})
    location = S3Location.from_uri("s3://bucket/data")

    downloaded = download_prefix(location, tmp_path, client=client)

    assert [p.name for p in downloaded] == ["ok.jpg"]


# ── Settings integration ─────────────────────────────────────────────────


def test_sync_dataset_requires_configuration(tmp_path):
    from medicinal_leaf.config.settings import Settings

    settings = Settings()
    settings.aws.dataset_uri = None

    with pytest.raises(S3ConfigurationError, match="No dataset URI"):
        sync_dataset(settings)


def test_sync_dataset_uses_configured_locations(tmp_path, dataset_client):
    from medicinal_leaf.config.settings import Settings

    settings = Settings()
    settings.aws.dataset_uri = "s3://leaf-data/datasets/v1"
    settings.data.raw_dir = tmp_path / "Data"

    downloaded = sync_dataset(settings, client=dataset_client)

    assert len(downloaded) == 3
    assert (tmp_path / "Data" / "Neem" / "a.jpg").is_file()
