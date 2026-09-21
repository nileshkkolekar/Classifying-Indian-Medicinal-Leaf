"""Asynchronous bulk classification: a queue, a store, and one worker.

A gigabyte archive cannot be classified inside an HTTP request — at roughly
25 images/second on CPU, a few thousand images outlives any sane timeout. So
bulk work is submitted, processed in the background, and collected later.

Three deliberate constraints:

* **One worker, jobs run sequentially.** Torch already saturates the available
  cores on a single batch; running two jobs at once would halve each without
  finishing either sooner.
* **Results stream to CSV as they are produced.** Memory stays bounded by the
  chunk, and a crash leaves the rows completed so far rather than nothing.
* **The uploaded archive is deleted the moment the job ends.** It has to touch
  disk to be processed later at all, but it is held no longer than the work
  requires (NFR-8).

The job store is a directory per job, which means jobs survive a restart. It
is *not* shared between processes: with more than one API replica a job
submitted to one instance is invisible to the others. That is the point at
which this wants a real broker — SQS or Redis — behind the same interface.
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import json
import logging
import shutil
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from medicinal_leaf.api import service
from medicinal_leaf.api.schemas import BatchSummary, JobState, JobStatus, Thresholds
from medicinal_leaf.api.service import UploadRejectedError, ZipLimits

if TYPE_CHECKING:
    from medicinal_leaf.api.service import SupportsPrediction
    from medicinal_leaf.config.settings import Settings

logger = logging.getLogger(__name__)

RECORD_NAME = "job.json"
RESULTS_NAME = "results.csv"
ARCHIVE_NAME = "input.zip"

CSV_COLUMNS = (
    "filename",
    "verdict",
    "label",
    "confidence",
    "needs_review",
    "note",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class JobRecord:
    """Everything known about one bulk job. Serialised to ``job.json``."""

    job_id: str
    filename: str
    state: JobState = JobState.QUEUED
    total: int = 0
    processed: int = 0
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    review_threshold: float = 0.70
    unknown_threshold: float = 0.30
    classified: int = 0
    needs_review: int = 0
    unable_to_classify: int = 0
    errors: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        """Restore ``state`` to an enum after a round trip through JSON.

        ``JobState`` serialises as a bare string, and a dataclass does not
        coerce field types on construction — so a record loaded from disk
        would otherwise carry a plain ``str`` here, and every ``.is_terminal``
        check against it would raise.
        """
        if not isinstance(self.state, JobState):
            self.state = JobState(self.state)

    def to_status(self) -> JobStatus:
        return JobStatus(
            job_id=self.job_id,
            state=self.state,
            filename=self.filename,
            total=self.total,
            processed=self.processed,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            summary=BatchSummary(
                total=self.processed,
                classified=self.classified,
                needs_review=self.needs_review,
                unable_to_classify=self.unable_to_classify,
                errors=self.errors,
            ),
            thresholds=Thresholds(
                review_threshold=self.review_threshold,
                unknown_threshold=self.unknown_threshold,
            ),
            error=self.error,
        )


class JobStore:
    """A directory per job: the record, the spooled upload, the results CSV."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Paths ────────────────────────────────────────────────────────────

    def job_dir(self, job_id: str) -> Path:
        # Job ids are generated here, never supplied by a caller, but keep the
        # directory flat regardless so a crafted id cannot walk the tree.
        return self.root / Path(job_id).name

    def record_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / RECORD_NAME

    def results_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / RESULTS_NAME

    def archive_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / ARCHIVE_NAME

    # ── Records ──────────────────────────────────────────────────────────

    def create(self, filename: str, thresholds: Thresholds) -> JobRecord:
        record = JobRecord(
            job_id=uuid.uuid4().hex,
            filename=filename,
            review_threshold=thresholds.review_threshold,
            unknown_threshold=thresholds.unknown_threshold,
        )
        self.job_dir(record.job_id).mkdir(parents=True, exist_ok=True)
        self.save(record)
        return record

    def save(self, record: JobRecord) -> None:
        path = self.record_path(record.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a reader polling mid-write never sees half a file.
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        temporary.replace(path)

    def get(self, job_id: str) -> JobRecord | None:
        path = self.record_path(job_id)
        if not path.is_file():
            return None
        try:
            return JobRecord(**json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, TypeError) as exc:
            logger.warning("Unreadable job record %s: %s", job_id, exc)
            return None

    def list(self, limit: int = 50) -> list[JobRecord]:
        records = [r for d in self.root.iterdir() if d.is_dir() and (r := self.get(d.name))]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit]

    def delete(self, job_id: str) -> bool:
        directory = self.job_dir(job_id)
        if not directory.is_dir():
            return False
        shutil.rmtree(directory, ignore_errors=True)
        return True

    def discard_archive(self, job_id: str) -> None:
        """Drop the uploaded ZIP as soon as it is no longer needed (NFR-8)."""
        self.archive_path(job_id).unlink(missing_ok=True)

    def purge_expired(self, retention_hours: float) -> int:
        """Remove jobs that finished longer ago than the retention window."""
        cutoff = datetime.now(UTC) - timedelta(hours=retention_hours)
        removed = 0
        for record in self.list(limit=10_000):
            if not record.state.is_terminal or not record.finished_at:
                continue
            try:
                finished = datetime.fromisoformat(record.finished_at)
            except ValueError:
                continue
            if finished < cutoff and self.delete(record.job_id):
                removed += 1
        if removed:
            logger.info("Purged %d expired job(s)", removed)
        return removed

    # ── Restart recovery ─────────────────────────────────────────────────

    def fail_interrupted(self) -> int:
        """Mark jobs that were mid-flight when the process died.

        Without this a job stuck in ``running`` from a previous life would be
        polled forever by a client waiting on a worker that no longer exists.
        """
        affected = 0
        for record in self.list(limit=10_000):
            if record.state in {JobState.RUNNING, JobState.QUEUED}:
                record.state = JobState.FAILED
                record.error = "The service restarted while this job was in progress."
                record.finished_at = _now()
                self.save(record)
                self.discard_archive(record.job_id)
                affected += 1
        if affected:
            logger.warning("Marked %d interrupted job(s) as failed", affected)
        return affected


class JobQueue:
    """The submit side: bounded queue plus the store backing it."""

    def __init__(self, store: JobStore, max_queued: int) -> None:
        self.store = store
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queued)
        self._cancelled: set[str] = set()

    def submit(self, record: JobRecord) -> None:
        """Enqueue a job, refusing rather than blocking when full."""
        try:
            self.queue.put_nowait(record.job_id)
        except asyncio.QueueFull as exc:
            raise UploadRejectedError(
                f"The queue is full ({self.queue.maxsize} jobs waiting). Retry shortly.",
                status_code=429,
            ) from exc

    def cancel(self, job_id: str) -> None:
        """Ask the worker to stop between chunks."""
        self._cancelled.add(job_id)

    def is_cancelled(self, job_id: str) -> bool:
        return job_id in self._cancelled


class JobWorker:
    """Consumes the queue, one job at a time."""

    def __init__(
        self,
        jobs: JobQueue,
        settings: Settings,
        predictor_provider: Callable[[], SupportsPrediction | None],
    ) -> None:
        self.jobs = jobs
        self.store = jobs.store
        self.settings = settings
        self.predictor_provider = predictor_provider
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="leaf-job-worker")
            logger.info("Job worker started")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
            logger.info("Job worker stopped")

    async def _run(self) -> None:
        while True:
            job_id = await self.jobs.queue.get()
            try:
                # Inference is blocking CPU work; off the event loop it goes,
                # or /health and /jobs would stall for the whole job.
                await asyncio.to_thread(self._process, job_id)
            except Exception:
                logger.exception("Job %s crashed", job_id)
            finally:
                self.jobs.queue.task_done()

    # ── The actual work ──────────────────────────────────────────────────

    def _process(self, job_id: str) -> None:
        record = self.store.get(job_id)
        if record is None:
            logger.warning("Job %s vanished before it ran", job_id)
            return

        if self.jobs.is_cancelled(job_id):
            self._finish(record, JobState.CANCELLED)
            return

        predictor = self.predictor_provider()
        if predictor is None:
            record.error = "No model is loaded."
            self._finish(record, JobState.FAILED)
            return

        record.state = JobState.RUNNING
        record.started_at = _now()
        self.store.save(record)

        queue_cfg = self.settings.queue
        limits = ZipLimits(
            allowed_extensions=self.settings.serving.allowed_extensions,
            max_entries=queue_cfg.max_zip_entries,
            max_uncompressed_bytes=queue_cfg.max_zip_uncompressed_bytes,
            max_file_bytes=self.settings.serving.max_image_bytes,
            max_compression_ratio=self.settings.serving.max_compression_ratio,
        )
        archive = self.store.archive_path(job_id)

        try:
            entries = service.iter_zip_entries(archive, limits)
            results_path = self.store.results_path(job_id)

            with results_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
                writer.writeheader()

                for result in service.classify_stream(
                    predictor,
                    entries,
                    review_threshold=record.review_threshold,
                    unknown_threshold=record.unknown_threshold,
                    chunk_size=queue_cfg.chunk_size,
                    max_chunk_megapixels=queue_cfg.max_chunk_megapixels,
                ):
                    writer.writerow(
                        {
                            "filename": result.filename,
                            "verdict": result.verdict.value,
                            "label": result.label or "",
                            "confidence": (
                                f"{result.confidence:.6f}" if result.confidence is not None else ""
                            ),
                            "needs_review": result.needs_review,
                            "note": result.note or "",
                        }
                    )
                    self._tally(record, result)

                    # Flush so a polling client sees rows appear, and so a
                    # crash leaves completed work on disk.
                    if record.processed % queue_cfg.chunk_size == 0:
                        handle.flush()
                        self.store.save(record)

                    if self.jobs.is_cancelled(job_id):
                        handle.flush()
                        self._finish(record, JobState.CANCELLED)
                        return

            self._finish(record, JobState.SUCCEEDED)

        except UploadRejectedError as exc:
            record.error = exc.message
            self._finish(record, JobState.FAILED)
        except Exception as exc:  # noqa: BLE001 - any failure must reach the client
            logger.exception("Job %s failed", job_id)
            record.error = f"{type(exc).__name__}: {exc}"
            self._finish(record, JobState.FAILED)

    def _tally(self, record: JobRecord, result: object) -> None:
        record.processed += 1
        verdict = getattr(result, "verdict", None)
        if verdict == "classified":
            record.classified += 1
        elif verdict == "needs_review":
            record.needs_review += 1
        elif verdict == "unable_to_classify":
            record.unable_to_classify += 1
        elif verdict == "error":
            record.errors += 1

    def _finish(self, record: JobRecord, state: JobState) -> None:
        record.state = state
        record.finished_at = _now()
        self.store.save(record)
        # The upload has served its purpose either way.
        self.store.discard_archive(record.job_id)
        logger.info(
            "Job %s %s (%d/%d images)", record.job_id, state.value, record.processed, record.total
        )
