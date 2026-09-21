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
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles

from medicinal_leaf import __version__
from medicinal_leaf.api import auth, service
from medicinal_leaf.api.auth import AuthenticationError, Principal
from medicinal_leaf.api.jobs import JobQueue, JobStore, JobWorker
from medicinal_leaf.api.schemas import (
    BatchResult,
    HealthResponse,
    JobAccepted,
    JobList,
    JobState,
    JobStatus,
    SingleResult,
    Thresholds,
    Token,
    UserInfo,
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

    # Say so loudly if this deployment is open, or locked out of itself.
    auth.warn_if_unprotected(settings)

    # FR-1: in a deployed container the checkpoint usually lives in S3 rather
    # than in the image, so fetch it before trying to load from disk.
    if settings.aws.checkpoint_uri:
        try:
            from medicinal_leaf.data import s3

            s3.download_checkpoint(settings)
        except Exception:
            logger.exception(
                "Could not fetch the checkpoint from %s; falling back to local disk.",
                settings.aws.checkpoint_uri,
            )

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

    # Bulk jobs outlive a request, so the store is on disk and survives a
    # restart. Anything caught mid-flight by the last shutdown is failed
    # rather than left "running" forever for a client that is still polling.
    store = JobStore(settings.queue.job_dir)
    store.fail_interrupted()
    store.purge_expired(settings.queue.retention_hours)

    app.state.jobs = JobQueue(store, settings.queue.max_queued_jobs)
    app.state.worker = JobWorker(app.state.jobs, settings, lambda: app.state.predictor)
    if settings.queue.enabled:
        app.state.worker.start()

    yield

    await app.state.worker.stop()
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


@app.exception_handler(AuthenticationError)
async def _authentication_handler(_: Request, exc: AuthenticationError) -> JSONResponse:
    """401 with the challenge header clients expect."""
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": exc.message},
        headers={"WWW-Authenticate": "Bearer"},
    )


# ``auto_error=False`` so a missing header reaches our own handler, which can
# also consider an API key, instead of FastAPI short-circuiting to 401.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


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


def require_principal(
    settings: SettingsDep,
    token: Annotated[str | None, Depends(oauth2_scheme)] = None,
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> Principal:
    """Identify the caller, or reject the request.

    Accepts either a bearer token (people, through the UI) or an API key
    (scripts). With auth disabled every caller resolves to anonymous, which
    keeps local development and the test suite straightforward.
    """
    return auth.principal_from_credentials(settings, bearer_token=token, api_key=api_key)


PrincipalDep = Annotated[Principal, Depends(require_principal)]


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


async def spool_upload(upload: UploadFile, destination: Path, limit: int, what: str) -> int:
    """Stream an upload straight to disk, aborting past ``limit``.

    The queued path never materialises the archive in memory — a 5 GB ZIP
    would not fit. It is written once, read back entry by entry, and deleted
    when the job finishes.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with destination.open("wb") as handle:
            while chunk := await upload.read(_UPLOAD_CHUNK):
                total += len(chunk)
                if total > limit:
                    raise UploadRejectedError(
                        f"{what} exceeds the {limit / 1073741824:.1f} GB limit.",
                        status_code=413,
                    )
                handle.write(chunk)
        if total == 0:
            raise UploadRejectedError(f"{what} is empty.")
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return total


def get_jobs(request: Request) -> JobQueue:
    queue: JobQueue | None = getattr(request.app.state, "jobs", None)
    if queue is None:
        raise UploadRejectedError("The job queue is not available.", status_code=503)
    return queue


JobsDep = Annotated[JobQueue, Depends(get_jobs)]


# ── Authentication ───────────────────────────────────────────────────────


@app.post("/auth/token", response_model=Token, tags=["auth"])
def issue_token(
    settings: SettingsDep,
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    """Exchange a username and password for a short-lived access token."""
    if not settings.auth.enabled:
        raise UploadRejectedError(
            "Authentication is disabled on this deployment; no token is needed.",
            status_code=404,
        )

    principal = auth.authenticate(form.username, form.password, settings)
    token, expires_in = auth.create_access_token(principal.name, settings)
    logger.info("Issued a token for %s", principal.name)
    return Token(access_token=token, expires_in=expires_in, username=principal.name)


@app.get("/auth/me", response_model=UserInfo, tags=["auth"])
def whoami(settings: SettingsDep, principal: PrincipalDep) -> UserInfo:
    """Who the supplied credentials identify. Useful for a UI session check."""
    return UserInfo(
        username=principal.name,
        kind=principal.kind,
        auth_enabled=settings.auth.enabled,
    )


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
    _principal: PrincipalDep,
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
    _principal: PrincipalDep,
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


# ── Bulk jobs ────────────────────────────────────────────────────────────


@app.post(
    "/jobs",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["jobs"],
)
async def submit_job(
    settings: SettingsDep,
    jobs: JobsDep,
    predictor: PredictorDep,  # noqa: ARG001 - fail fast if no model is loaded
    principal: PrincipalDep,
    file: Annotated[UploadFile, File(description="A ZIP archive of leaf images.")],
    review_threshold: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    unknown_threshold: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
) -> JobAccepted:
    """Queue a large archive for background classification.

    Returns immediately with a job id. Use this instead of ``/predict/batch``
    when the archive is big enough that a synchronous response would time
    out — thousands of images take minutes of CPU, not seconds.
    """
    thresholds = resolve_thresholds(settings, review_threshold, unknown_threshold)
    record = jobs.store.create(service.display_name(file.filename or "upload.zip"), thresholds)

    try:
        await spool_upload(
            file,
            jobs.store.archive_path(record.job_id),
            settings.queue.max_archive_bytes,
            "Archive",
        )
        # Counting from the central directory validates the archive and gives
        # the client a denominator to show progress against, without
        # decompressing anything.
        record.total = service.count_zip_images(
            jobs.store.archive_path(record.job_id),
            ZipLimits(
                allowed_extensions=settings.serving.allowed_extensions,
                max_entries=settings.queue.max_zip_entries,
                max_uncompressed_bytes=settings.queue.max_zip_uncompressed_bytes,
                max_file_bytes=settings.serving.max_image_bytes,
                max_compression_ratio=settings.serving.max_compression_ratio,
            ),
        )
        jobs.store.save(record)
        jobs.submit(record)
    except Exception:
        # Never leave an orphaned directory or upload behind a failed submit.
        jobs.store.delete(record.job_id)
        raise

    logger.info("Queued job %s for %s (%d images)", record.job_id, principal.name, record.total)
    return JobAccepted(
        job_id=record.job_id,
        state=record.state,
        total=record.total,
        status_url=f"/jobs/{record.job_id}",
        results_url=f"/jobs/{record.job_id}/results",
    )


@app.get("/jobs", response_model=JobList, tags=["jobs"])
def list_jobs(
    jobs: JobsDep,
    _principal: PrincipalDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 25,
) -> JobList:
    """Recent jobs, newest first."""
    return JobList(jobs=[record.to_status() for record in jobs.store.list(limit)])


@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["jobs"])
def job_status(jobs: JobsDep, _principal: PrincipalDep, job_id: str) -> JobStatus:
    """Progress and outcome for one job. Poll this while it runs."""
    record = jobs.store.get(job_id)
    if record is None:
        raise UploadRejectedError(f"No job {job_id}.", status_code=404)
    return record.to_status()


@app.get("/jobs/{job_id}/results", tags=["jobs"])
def job_results(jobs: JobsDep, _principal: PrincipalDep, job_id: str) -> FileResponse:
    """Download the results CSV: one row per image (FR-11)."""
    record = jobs.store.get(job_id)
    if record is None:
        raise UploadRejectedError(f"No job {job_id}.", status_code=404)

    path = jobs.store.results_path(job_id)
    if not path.is_file():
        raise UploadRejectedError(
            f"Job {job_id} is {record.state.value}; no results yet.",
            status_code=409,
        )
    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"{Path(record.filename).stem or 'results'}_{job_id[:8]}.csv",
    )


@app.delete("/jobs/{job_id}", tags=["jobs"])
def delete_job(jobs: JobsDep, _principal: PrincipalDep, job_id: str) -> dict[str, str]:
    """Cancel a job if it is still running, and remove everything it wrote."""
    record = jobs.store.get(job_id)
    if record is None:
        raise UploadRejectedError(f"No job {job_id}.", status_code=404)

    if not record.state.is_terminal:
        # The worker checks this between chunks; it cannot be interrupted
        # mid-batch, so cancellation takes effect within one chunk.
        jobs.cancel(job_id)
        return {"job_id": job_id, "state": JobState.CANCELLED.value, "detail": "Cancelling."}

    jobs.store.delete(job_id)
    return {"job_id": job_id, "state": record.state.value, "detail": "Deleted."}


# ── Static frontend ──────────────────────────────────────────────────────
#
# Mounted last, so every API route above is matched first and only unclaimed
# paths fall through to the bundle. Serving the app from the API's own origin
# means the browser never makes a cross-origin request, so there is no CORS
# policy to write, loosen under deadline pressure, and get wrong.
#
# Absent in a checkout that has not run `npm run build`; the API then serves
# JSON only, which is exactly what CI and the test suite exercise.
def mount_frontend(application: FastAPI, directory: Path) -> bool:
    if not directory.is_dir():
        logger.info("No built frontend at %s; serving the API only.", directory)
        return False

    application.mount("/", StaticFiles(directory=directory, html=True), name="frontend")
    logger.info("Serving the React bundle from %s", directory)
    return True


try:
    mount_frontend(app, load_settings().serving.frontend_dir)
except Exception:  # pragma: no cover - configuration is validated elsewhere
    # Never let a UI problem stop the API from importing; the JSON surface is
    # what matters and /health has to keep answering.
    logger.exception("Could not mount the frontend; serving the API only.")


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
