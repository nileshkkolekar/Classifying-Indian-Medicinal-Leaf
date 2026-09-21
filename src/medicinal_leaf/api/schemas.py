"""Request and response models for the prediction API.

These are the wire contract: the Streamlit UI, the bulk CSV export and any
future client all read these field names, so treat renames as breaking.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Verdict(StrEnum):
    """What the service concluded about one image.

    Distinguishing ``UNABLE_TO_CLASSIFY`` from ``ERROR`` matters: the first is
    a readable photo the model cannot confidently place (FR-14), the second is
    a file that could not be decoded at all. They need different follow-up.
    """

    CLASSIFIED = "classified"
    NEEDS_REVIEW = "needs_review"
    UNABLE_TO_CLASSIFY = "unable_to_classify"
    ERROR = "error"


class Thresholds(BaseModel):
    """The thresholds a given response was evaluated against.

    Echoed back on every response so a result is interpretable on its own —
    a "needs review" flag means nothing without the bar it failed.
    """

    review_threshold: float = Field(ge=0.0, le=1.0)
    unknown_threshold: float = Field(ge=0.0, le=1.0)


class PredictionResult(BaseModel):
    """One image's outcome."""

    filename: str
    verdict: Verdict
    #: ``None`` when the service declined to name a species.
    label: str | None = None
    #: Top-class probability; ``None`` when the image could not be read.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    #: True whenever a human should look — low confidence, or a failure.
    needs_review: bool = False
    #: Full distribution, so a caller can apply its own policy.
    probabilities: dict[str, float] = Field(default_factory=dict)
    #: Human-readable explanation of a non-``classified`` verdict.
    note: str | None = None


class BatchSummary(BaseModel):
    """Counts across a bulk upload, for the results header."""

    total: int = 0
    classified: int = 0
    needs_review: int = 0
    unable_to_classify: int = 0
    errors: int = 0

    @property
    def flagged(self) -> int:
        """Everything a reviewer needs to touch."""
        return self.needs_review + self.unable_to_classify + self.errors


class BatchResult(BaseModel):
    """Response for a ZIP upload (FR-9, FR-11)."""

    results: list[PredictionResult] = Field(default_factory=list)
    summary: BatchSummary = Field(default_factory=BatchSummary)
    thresholds: Thresholds


class SingleResult(BaseModel):
    """Response for a single image upload (FR-8)."""

    result: PredictionResult
    thresholds: Thresholds


class HealthResponse(BaseModel):
    """Liveness plus enough detail to debug a misconfigured deployment."""

    status: str
    model_loaded: bool
    backbone: str | None = None
    classes: list[str] = Field(default_factory=list)
    thresholds: Thresholds
    version: str


class ErrorResponse(BaseModel):
    """Body returned for a rejected upload."""

    detail: str


class JobState(StrEnum):
    """Where a queued bulk job has got to."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}


class JobStatus(BaseModel):
    """A bulk job's progress and outcome."""

    job_id: str
    state: JobState
    filename: str
    #: Images the archive holds, counted from the central directory up front.
    total: int = 0
    processed: int = 0
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    summary: BatchSummary = Field(default_factory=BatchSummary)
    thresholds: Thresholds
    #: Populated only when ``state`` is ``failed``.
    error: str | None = None

    @property
    def percent(self) -> float:
        return 100.0 * self.processed / self.total if self.total else 0.0


class JobAccepted(BaseModel):
    """Returned by the submit endpoint, before any work has been done."""

    job_id: str
    state: JobState
    total: int
    status_url: str
    results_url: str


class JobList(BaseModel):
    """Recent jobs, newest first."""

    jobs: list[JobStatus] = Field(default_factory=list)
