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
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Protocol

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


def safe_zip_entries(data: bytes, limits: ZipLimits) -> list[tuple[str, bytes]]:
    """Return ``(name, bytes)`` for every acceptable image in the archive.

    Every limit is checked against the central directory *before* any entry is
    decompressed, then the read itself is capped in case the header lied.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise UploadRejectedError(f"Not a readable ZIP archive: {exc}") from exc

    with archive:
        candidates = []
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

        entries: list[tuple[str, bytes]] = []
        read_total = 0
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
            entries.append((name, payload))

    logger.info("Accepted %d image(s) from archive (%d bytes)", len(entries), read_total)
    return entries


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


def classify_entries(
    predictor: SupportsPrediction,
    entries: list[tuple[str, bytes]],
    *,
    review_threshold: float,
    unknown_threshold: float,
    batch_size: int = 32,
) -> list[PredictionResult]:
    """Classify every entry, keeping per-file failures local.

    Decoding happens first so the readable images can go through the model in
    batches; an undecodable file becomes an ``ERROR`` row instead of taking
    the whole upload down with it.
    """
    results: list[PredictionResult | None] = [None] * len(entries)
    images: list[Image.Image] = []
    slots: list[int] = []

    for index, (name, payload) in enumerate(entries):
        try:
            images.append(decode_image(payload))
            slots.append(index)
        except ValueError as exc:
            logger.warning("Could not decode %s: %s", name, exc)
            results[index] = PredictionResult(
                filename=name,
                verdict=Verdict.ERROR,
                needs_review=True,
                note=f"Could not read image: {exc}",
            )

    if images:
        predictions = predictor.predict_batch(images, batch_size)
        for slot, prediction in zip(slots, predictions, strict=True):
            results[slot] = to_result(
                entries[slot][0],
                prediction,
                review_threshold=review_threshold,
                unknown_threshold=unknown_threshold,
            )

    return [r for r in results if r is not None]


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
