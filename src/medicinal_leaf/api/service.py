"""Upload handling and the classify / flag / decline decision.

Kept free of FastAPI so the policy is testable without HTTP, and so the same
rules could be reused by a batch job or a Lambda handler.

Two things here are security-relevant rather than incidental:

* **Nothing touches disk.** Uploads are decoded from memory and dropped when
  the request ends, which is what NFR-8 asks for.
* **Archives are bounded in every dimension.** Entry count, per-file size,
  total uncompressed size and compression ratio are all checked *before*
  decompressing, because a ZIP that expands to gigabytes is trivial to build.
"""

from __future__ import annotations

import io
import logging
import zipfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, TYPE_CHECKING, Protocol

from PIL import Image, ImageOps, UnidentifiedImageError

from medicinal_leaf.api.schemas import BatchSummary, PredictionResult, Verdict

if TYPE_CHECKING:
    from medicinal_leaf.inference.predictor import Prediction

logger = logging.getLogger(__name__)


class SupportsPrediction(Protocol):
    """The slice of ``LeafPredictor`` this module needs.

    A Protocol rather than the concrete class so tests can substitute a stub
    without loading a checkpoint.
    """

    class_names: list[str]

    def predict_batch(
        self, sources: list[Image.Image], batch_size: int = ...
    ) -> list[Prediction]: ...


class UploadRejectedError(Exception):
    """An upload violated a configured limit.

    ``status_code`` distinguishes "malformed" (400) from "too large" (413) so
    the HTTP layer does not have to re-derive it from the message.
    """

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(slots=True)
class ZipLimits:
    """Bounds applied to an uploaded archive."""

    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    max_entries: int = 200
    max_uncompressed_bytes: int = 500 * 1024 * 1024
    max_file_bytes: int = 15 * 1024 * 1024
    max_compression_ratio: float = 100.0


def decide_verdict(
    confidence: float,
    *,
    review_threshold: float,
    unknown_threshold: float,
) -> tuple[Verdict, str | None]:
    """Map a top-class probability onto a verdict and an explanation.

    Below ``unknown_threshold`` the service declines to name a species at all
    (FR-14) — a forced label on an unrecognisable photo is worse than an
    honest refusal, because it looks like an answer.
    """
    if confidence < unknown_threshold:
        return (
            Verdict.UNABLE_TO_CLASSIFY,
            f"Top score {confidence:.1%} is below the {unknown_threshold:.0%} floor; "
            "no leaf was identified with usable certainty.",
        )
    if confidence < review_threshold:
        return (
            Verdict.NEEDS_REVIEW,
            f"Confidence {confidence:.1%} is below the {review_threshold:.0%} "
            "review threshold; confirm before relying on this.",
        )
    return Verdict.CLASSIFIED, None


def display_name(raw: str) -> str:
    """Make an archive member name safe to show and log.

    Nothing here is ever used as a filesystem path — entries are read straight
    from the archive object — so this is about not echoing ``../../`` noise
    back to a user, not about preventing traversal.
    """
    cleaned = raw.replace("\\", "/").lstrip("/")
    parts = [p for p in PurePosixPath(cleaned).parts if p not in ("..", ".")]
    return "/".join(parts) or "unnamed"


def decode_image(data: bytes) -> Image.Image:
    """Decode bytes to an RGB image, honouring EXIF rotation.

    Raises :class:`ValueError` for anything unreadable, including Pillow's
    decompression-bomb guard, so callers can report per-file rather than
    failing the whole batch.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        raise ValueError(str(exc) or type(exc).__name__) from exc


#: Anything ``zipfile`` can open: raw bytes, a path, or an open file object.
ZipSource = bytes | str | Path | IO[bytes]


def _open_archive(source: ZipSource) -> zipfile.ZipFile:
    """Open a ZIP from bytes, a path, or a file handle.

    Accepting a handle is what lets a multi-gigabyte upload be read from its
    spooled temp file instead of being copied into memory first.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    try:
        return zipfile.ZipFile(source)
    except zipfile.BadZipFile as exc:
        raise UploadRejectedError(f"Not a readable ZIP archive: {exc}") from exc


def _vet_candidates(
    archive: zipfile.ZipFile, limits: ZipLimits
) -> list[tuple[zipfile.ZipInfo, str]]:
    """Select the image members and check every limit that can be checked early.

    All of this reads the central directory only — nothing is decompressed, so
    an oversized or bomb-shaped archive is rejected before it costs anything.
    """
    candidates: list[tuple[zipfile.ZipInfo, str]] = []
    for info in archive.infolist():
        if info.is_dir() or "__MACOSX" in info.filename:
            continue
        name = display_name(info.filename)
        if PurePosixPath(name).name.startswith("."):
            continue
        if PurePosixPath(name).suffix.lower() not in limits.allowed_extensions:
            continue
        candidates.append((info, name))

    if not candidates:
        allowed = ", ".join(limits.allowed_extensions)
        raise UploadRejectedError(f"The archive contains no images ({allowed}).")

    if len(candidates) > limits.max_entries:
        raise UploadRejectedError(
            f"Archive holds {len(candidates)} images, over the "
            f"{limits.max_entries} per-upload limit.",
            status_code=413,
        )

    declared_total = sum(info.file_size for info, _ in candidates)
    if declared_total > limits.max_uncompressed_bytes:
        raise UploadRejectedError(
            f"Archive expands to {declared_total / 1048576:.0f} MB, over the "
            f"{limits.max_uncompressed_bytes / 1048576:.0f} MB limit.",
            status_code=413,
        )

    for info, name in candidates:
        if info.file_size > limits.max_file_bytes:
            raise UploadRejectedError(
                f"{name} is {info.file_size / 1048576:.1f} MB, over the "
                f"{limits.max_file_bytes / 1048576:.0f} MB per-image limit.",
                status_code=413,
            )
        ratio = info.file_size / max(info.compress_size, 1)
        if ratio > limits.max_compression_ratio:
            raise UploadRejectedError(
                f"{name} expands {ratio:.0f}x, above the "
                f"{limits.max_compression_ratio:.0f}x limit; refusing as a possible zip bomb.",
                status_code=413,
            )

    return candidates


def count_zip_images(source: ZipSource, limits: ZipLimits) -> int:
    """How many images the archive holds, without decompressing any of them.

    Lets a queued job report a total to count progress against before the
    first image is touched.
    """
    with _open_archive(source) as archive:
        return len(_vet_candidates(archive, limits))


def iter_zip_entries(source: ZipSource, limits: ZipLimits) -> Iterator[tuple[str, bytes]]:
    """Yield ``(name, bytes)`` one image at a time.

    Limits are validated *eagerly* — this function raises before returning the
    generator, so a bad archive is rejected at request time rather than
    halfway through processing. Only the decompression is lazy, which keeps
    peak memory at one image regardless of archive size.
    """
    archive = _open_archive(source)
    try:
        candidates = _vet_candidates(archive, limits)
    except Exception:
        archive.close()
        raise

    def _generate() -> Iterator[tuple[str, bytes]]:
        read_total = 0
        with archive:
            for info, name in candidates:
                # Read one byte past the cap so a lying header is caught too.
                with archive.open(info) as handle:
                    payload = handle.read(limits.max_file_bytes + 1)
                if len(payload) > limits.max_file_bytes:
                    raise UploadRejectedError(
                        f"{name} exceeds the {limits.max_file_bytes / 1048576:.0f} MB "
                        "per-image limit once decompressed.",
                        status_code=413,
                    )

                read_total += len(payload)
                if read_total > limits.max_uncompressed_bytes:
                    raise UploadRejectedError(
                        "Archive exceeds the total uncompressed size limit.",
                        status_code=413,
                    )
                yield name, payload

        logger.info("Streamed %d image(s) from archive (%d bytes)", len(candidates), read_total)

    return _generate()


def safe_zip_entries(source: ZipSource, limits: ZipLimits) -> list[tuple[str, bytes]]:
    """Every acceptable image in the archive, materialised.

    Convenient for small synchronous uploads. For anything large, iterate
    :func:`iter_zip_entries` instead — this holds the whole archive at once.
    """
    return list(iter_zip_entries(source, limits))


def to_result(
    filename: str,
    prediction: Prediction,
    *,
    review_threshold: float,
    unknown_threshold: float,
) -> PredictionResult:
    """Turn a raw prediction into the flagged, policy-applied wire result."""
    verdict, note = decide_verdict(
        prediction.confidence,
        review_threshold=review_threshold,
        unknown_threshold=unknown_threshold,
    )
    return PredictionResult(
        filename=filename,
        verdict=verdict,
        # Withhold the label entirely when declining (FR-14).
        label=None if verdict is Verdict.UNABLE_TO_CLASSIFY else prediction.label,
        confidence=round(prediction.confidence, 6),
        needs_review=verdict is not Verdict.CLASSIFIED,
        probabilities={k: round(v, 6) for k, v in prediction.probabilities.items()},
        note=note,
    )


#: Decoded RGB is roughly 10x its JPEG size, so a chunk is budgeted by pixels
#: rather than by file count — 64 thumbnails and 64 DSLR frames differ by two
#: orders of magnitude in memory.
DEFAULT_CHUNK_SIZE = 32
DEFAULT_CHUNK_MEGAPIXELS = 256.0


def classify_stream(
    predictor: SupportsPrediction,
    entries: Iterable[tuple[str, bytes]],
    *,
    review_threshold: float,
    unknown_threshold: float,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_chunk_megapixels: float = DEFAULT_CHUNK_MEGAPIXELS,
    on_progress: Callable[[int], None] | None = None,
) -> Iterator[PredictionResult]:
    """Classify an arbitrarily long stream of images in bounded memory.

    Images are decoded, predicted and released a chunk at a time, so peak
    memory tracks the chunk rather than the archive: a 5 GB upload costs the
    same as a 50 MB one. A chunk closes when it reaches ``chunk_size`` images
    *or* ``max_chunk_megapixels`` of decoded pixels, whichever comes first —
    the pixel budget is the one that actually bounds memory.

    Results are emitted in input order. A file that will not decode becomes an
    ``ERROR`` row held in its own slot, so a single bad image neither sinks the
    batch nor jumps ahead of its neighbours in the output.
    """
    slots: list[PredictionResult | None] = []
    names: list[str] = []
    images: list[Image.Image] = []
    image_slots: list[int] = []
    megapixels = 0.0
    processed = 0

    def flush() -> Iterator[PredictionResult]:
        nonlocal processed
        if images:
            predictions = predictor.predict_batch(images, len(images))
            for slot, prediction in zip(image_slots, predictions, strict=True):
                slots[slot] = to_result(
                    names[slot],
                    prediction,
                    review_threshold=review_threshold,
                    unknown_threshold=unknown_threshold,
                )
        for result in slots:
            if result is not None:
                processed += 1
                yield result
        if on_progress is not None:
            on_progress(processed)

    for name, payload in entries:
        try:
            image = decode_image(payload)
        except ValueError as exc:
            logger.warning("Could not decode %s: %s", name, exc)
            names.append(name)
            slots.append(
                PredictionResult(
                    filename=name,
                    verdict=Verdict.ERROR,
                    needs_review=True,
                    note=f"Could not read image: {exc}",
                )
            )
        else:
            names.append(name)
            slots.append(None)
            image_slots.append(len(slots) - 1)
            images.append(image)
            megapixels += (image.width * image.height) / 1_000_000

        if len(slots) >= chunk_size or megapixels >= max_chunk_megapixels:
            yield from flush()
            # Dropping the references here is what actually frees the pixels.
            slots, names, images, image_slots = [], [], [], []
            megapixels = 0.0

    if slots:
        yield from flush()


def classify_entries(
    predictor: SupportsPrediction,
    entries: Iterable[tuple[str, bytes]],
    *,
    review_threshold: float,
    unknown_threshold: float,
    batch_size: int = DEFAULT_CHUNK_SIZE,
) -> list[PredictionResult]:
    """Classify every entry and return the results as a list.

    A thin wrapper over :func:`classify_stream` for callers that want the
    whole answer at once.
    """
    return list(
        classify_stream(
            predictor,
            entries,
            review_threshold=review_threshold,
            unknown_threshold=unknown_threshold,
            chunk_size=batch_size,
        )
    )


def summarize(results: list[PredictionResult]) -> BatchSummary:
    """Count verdicts for the results header."""
    summary = BatchSummary(total=len(results))
    for result in results:
        match result.verdict:
            case Verdict.CLASSIFIED:
                summary.classified += 1
            case Verdict.NEEDS_REVIEW:
                summary.needs_review += 1
            case Verdict.UNABLE_TO_CLASSIFY:
                summary.unable_to_classify += 1
            case Verdict.ERROR:
                summary.errors += 1
    return summary
