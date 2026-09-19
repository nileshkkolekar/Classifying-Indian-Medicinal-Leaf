"""The FastAPI application.

    uvicorn medicinal_leaf.api.app:app --reload
    leaf-api                                    # same thing, host/port from config

Endpoints are thin: they enforce upload limits, hand bytes to
:mod:`medicinal_leaf.api.service`, and shape the response. All policy lives in
the service module.

The model is loaded once at startup. A missing checkpoint is *not* fatal —
the app still serves ``/health`` so a deployment can report why it is not
ready, instead of crash-looping with no explanation.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from medicinal_leaf import __version__
from medicinal_leaf.api import service
from medicinal_leaf.api.schemas import (
    BatchResult,
    HealthResponse,
    SingleResult,
    Thresholds,
)

# Imported at runtime rather than under TYPE_CHECKING: FastAPI resolves
# endpoint type hints when it registers routes, so a forward reference to a
# type-checking-only name would be a NameError at import.
from medicinal_leaf.api.service import SupportsPrediction, UploadRejectedError, ZipLimits
from medicinal_leaf.config.settings import Settings, load_settings
from medicinal_leaf.inference.predictor import LeafPredictor

logger = logging.getLogger(__name__)

_UPLOAD_CHUNK = 1 << 20


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load configuration and the checkpoint once, before serving."""
    settings = load_settings()
    app.state.settings = settings
    app.state.predictor = None

    try:
        app.state.predictor = LeafPredictor.from_checkpoint(settings.serving.checkpoint_path)
        logger.info("Model ready: %s", settings.serving.checkpoint_path)
    except FileNotFoundError:
        # Serve anyway so /health can explain the problem.
        logger.warning(
            "No checkpoint at %s - prediction endpoints will return 503 until one exists.",
            settings.serving.checkpoint_path,
        )
    except Exception:
        logger.exception("Failed to load the checkpoint; prediction endpoints will return 503.")

    yield

    app.state.predictor = None


app = FastAPI(
    title="Medicinal Leaf Classification API",
    description=(
        "Identify Indian medicinal leaf species from a photograph. "
        "Educational and botanical use only - not medical guidance."
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.exception_handler(UploadRejectedError)
async def _upload_rejected_handler(_: Request, exc: UploadRejectedError) -> JSONResponse:
    """Render a limit violation as its own status code rather than a 500."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


# ── Dependencies ─────────────────────────────────────────────────────────


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_predictor(request: Request) -> SupportsPrediction:
    """The loaded model, or a 503 explaining that there isn't one.

    Overridden in tests to inject a stub, which is why it is a dependency
    rather than a direct attribute read inside each endpoint.
    """
    predictor = request.app.state.predictor
    if predictor is None:
        raise UploadRejectedError(
            "No trained model is loaded. Train one with `leaf-train fit`, or point "
            "MLC_SERVING__CHECKPOINT_PATH at an existing checkpoint.",
            status_code=503,
        )
    return predictor


SettingsDep = Annotated[Settings, Depends(get_settings)]
PredictorDep = Annotated[SupportsPrediction, Depends(get_predictor)]


def resolve_thresholds(
    settings: Settings,
    review_threshold: float | None,
    unknown_threshold: float | None,
) -> Thresholds:
    """Per-request overrides on top of configured defaults (FR-13)."""
    review = settings.serving.review_threshold if review_threshold is None else review_threshold
    unknown = settings.serving.unknown_threshold if unknown_threshold is None else unknown_threshold
    if unknown > review:
        raise UploadRejectedError(
            f"unknown_threshold ({unknown}) must not exceed review_threshold ({review}).",
            status_code=422,
        )
    return Thresholds(review_threshold=review, unknown_threshold=unknown)


async def read_capped(upload: UploadFile, limit: int, what: str) -> bytes:
    """Read an upload, aborting past ``limit``.

    Streaming with a running total rather than ``await upload.read()``: the
    latter buys whatever the client chose to send before any check runs.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(_UPLOAD_CHUNK):
        total += len(chunk)
        if total > limit:
            raise UploadRejectedError(
                f"{what} exceeds the {limit / 1048576:.0f} MB limit.", status_code=413
            )
        chunks.append(chunk)

    if total == 0:
        raise UploadRejectedError(f"{what} is empty.")
    return b"".join(chunks)


# ── Endpoints ────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(settings: SettingsDep, request: Request) -> HealthResponse:
    """Liveness, plus whether a model is actually loaded."""
    predictor = request.app.state.predictor
    return HealthResponse(
        status="ok" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        backbone=getattr(getattr(predictor, "meta", None), "backbone", None),
        classes=list(getattr(predictor, "class_names", [])),
        thresholds=Thresholds(
            review_threshold=settings.serving.review_threshold,
            unknown_threshold=settings.serving.unknown_threshold,
        ),
        version=__version__,
    )


@app.post("/predict", response_model=SingleResult, tags=["predict"])
async def predict(
    settings: SettingsDep,
    predictor: PredictorDep,
    file: Annotated[UploadFile, File(description="A single leaf image (JPG or PNG).")],
    review_threshold: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    unknown_threshold: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
) -> SingleResult:
    """Classify one uploaded image (FR-8, FR-10).

    Returns a confidence score always, a species only when the model clears
    the floor, and a ``needs_review`` flag when it clears the floor but not
    the review threshold.
    """
    thresholds = resolve_thresholds(settings, review_threshold, unknown_threshold)
    payload = await read_capped(file, settings.serving.max_image_bytes, "Image")

    name = service.display_name(file.filename or "upload")
    results = service.classify_entries(
        predictor,
        [(name, payload)],
        review_threshold=thresholds.review_threshold,
        unknown_threshold=thresholds.unknown_threshold,
    )
    return SingleResult(result=results[0], thresholds=thresholds)


@app.post("/predict/batch", response_model=BatchResult, tags=["predict"])
async def predict_batch(
    settings: SettingsDep,
    predictor: PredictorDep,
    file: Annotated[UploadFile, File(description="A ZIP archive of leaf images.")],
    review_threshold: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    unknown_threshold: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
) -> BatchResult:
    """Classify every image inside an uploaded ZIP (FR-9, FR-11).

    One unreadable file does not fail the upload; it comes back as an
    ``error`` row so the rest of the batch is still usable.
    """
    thresholds = resolve_thresholds(settings, review_threshold, unknown_threshold)
    payload = await read_capped(file, settings.serving.max_archive_bytes, "Archive")

    entries = service.safe_zip_entries(
        payload,
        ZipLimits(
            allowed_extensions=settings.serving.allowed_extensions,
            max_entries=settings.serving.max_zip_entries,
            max_uncompressed_bytes=settings.serving.max_zip_uncompressed_bytes,
            max_file_bytes=settings.serving.max_image_bytes,
            max_compression_ratio=settings.serving.max_compression_ratio,
        ),
    )

    results = service.classify_entries(
        predictor,
        entries,
        review_threshold=thresholds.review_threshold,
        unknown_threshold=thresholds.unknown_threshold,
    )
    return BatchResult(
        results=results,
        summary=service.summarize(results),
        thresholds=thresholds,
    )


def main() -> None:
    """Console-script entry point: serve with host/port from configuration."""
    import uvicorn

    settings = load_settings()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "medicinal_leaf.api.app:app",
        host=settings.serving.host,
        port=settings.serving.port,
    )


if __name__ == "__main__":
    main()
