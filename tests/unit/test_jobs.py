"""The job store and queue.

No HTTP and no model here — this covers persistence, restart recovery,
retention and back-pressure, which are the parts that decide whether a
client polling a job id gets a sensible answer.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from medicinal_leaf.api.jobs import JobQueue, JobRecord, JobStore
from medicinal_leaf.api.schemas import JobState, Thresholds
from medicinal_leaf.api.service import UploadRejectedError

THRESHOLDS = Thresholds(review_threshold=0.7, unknown_threshold=0.3)


@pytest.fixture
def store(tmp_path) -> JobStore:
    return JobStore(tmp_path / "jobs")


# ── Records ──────────────────────────────────────────────────────────────


def test_create_assigns_an_id_and_persists(store):
    record = store.create("batch.zip", THRESHOLDS)

    assert record.job_id
    assert record.state is JobState.QUEUED
    assert store.record_path(record.job_id).is_file()


def test_record_round_trips(store):
    created = store.create("batch.zip", THRESHOLDS)
    created.total = 120
    created.processed = 7
    store.save(created)

    loaded = store.get(created.job_id)
    assert loaded is not None
    assert loaded.total == 120
    assert loaded.processed == 7
    assert loaded.review_threshold == pytest.approx(0.7)


def test_missing_job_is_none(store):
    assert store.get("does-not-exist") is None


def test_unreadable_record_is_none_not_a_crash(store):
    record = store.create("batch.zip", THRESHOLDS)
    store.record_path(record.job_id).write_text("{ not json", encoding="utf-8")
    assert store.get(record.job_id) is None


def test_state_survives_the_json_round_trip(store):
    """JSON gives back a bare string; the record must restore the enum."""
    record = store.create("batch.zip", THRESHOLDS)
    record.state = JobState.SUCCEEDED
    store.save(record)

    loaded = store.get(record.job_id)
    assert isinstance(loaded.state, JobState)
    assert loaded.state.is_terminal is True


def test_list_is_newest_first(store):
    """Timestamps are set explicitly — the Windows clock is coarse enough
    that two create() calls can land on the same microsecond."""
    first = store.create("a.zip", THRESHOLDS)
    first.created_at = "2026-01-01T00:00:00+00:00"
    store.save(first)

    second = store.create("b.zip", THRESHOLDS)
    second.created_at = "2026-01-02T00:00:00+00:00"
    store.save(second)

    assert [r.job_id for r in store.list()] == [second.job_id, first.job_id]


def test_delete_removes_everything(store):
    record = store.create("batch.zip", THRESHOLDS)
    store.archive_path(record.job_id).write_bytes(b"zip bytes")

    assert store.delete(record.job_id) is True
    assert store.get(record.job_id) is None
    assert not store.job_dir(record.job_id).exists()


def test_deleting_an_absent_job_is_false(store):
    assert store.delete("nope") is False


def test_job_id_cannot_escape_the_store_root(store):
    """Ids are generated internally, but the path must be flat regardless."""
    escaped = store.job_dir("../../etc/passwd")
    assert escaped.parent == store.root


# ── The uploaded archive ─────────────────────────────────────────────────


def test_archive_is_discardable(store):
    """NFR-8: the upload is kept only as long as the work needs it."""
    record = store.create("batch.zip", THRESHOLDS)
    store.archive_path(record.job_id).write_bytes(b"zip bytes")

    store.discard_archive(record.job_id)

    assert not store.archive_path(record.job_id).exists()
    # The record itself survives, so status remains pollable.
    assert store.get(record.job_id) is not None


def test_discarding_a_missing_archive_is_harmless(store):
    record = store.create("batch.zip", THRESHOLDS)
    store.discard_archive(record.job_id)
    store.discard_archive(record.job_id)


# ── Restart recovery ─────────────────────────────────────────────────────


def test_interrupted_jobs_are_failed_on_startup(store):
    """Otherwise a client polls a 'running' job whose worker no longer exists."""
    running = store.create("a.zip", THRESHOLDS)
    running.state = JobState.RUNNING
    store.save(running)
    queued = store.create("b.zip", THRESHOLDS)

    assert store.fail_interrupted() == 2

    for job_id in (running.job_id, queued.job_id):
        record = store.get(job_id)
        assert record.state is JobState.FAILED
        assert "restarted" in record.error


def test_finished_jobs_survive_startup(store):
    done = store.create("a.zip", THRESHOLDS)
    done.state = JobState.SUCCEEDED
    done.finished_at = datetime.now(UTC).isoformat()
    store.save(done)

    store.fail_interrupted()

    assert store.get(done.job_id).state is JobState.SUCCEEDED


# ── Retention ────────────────────────────────────────────────────────────


def test_expired_jobs_are_purged(store):
    old = store.create("old.zip", THRESHOLDS)
    old.state = JobState.SUCCEEDED
    old.finished_at = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
    store.save(old)

    fresh = store.create("fresh.zip", THRESHOLDS)
    fresh.state = JobState.SUCCEEDED
    fresh.finished_at = datetime.now(UTC).isoformat()
    store.save(fresh)

    assert store.purge_expired(retention_hours=24.0) == 1
    assert store.get(old.job_id) is None
    assert store.get(fresh.job_id) is not None


def test_unfinished_jobs_are_never_purged(store):
    running = store.create("a.zip", THRESHOLDS)
    running.state = JobState.RUNNING
    store.save(running)

    assert store.purge_expired(retention_hours=0.0001) == 0
    assert store.get(running.job_id) is not None


# ── Queue back-pressure ──────────────────────────────────────────────────


def test_queue_accepts_up_to_its_limit(store):
    async def scenario():
        jobs = JobQueue(store, max_queued=2)
        jobs.submit(store.create("a.zip", THRESHOLDS))
        jobs.submit(store.create("b.zip", THRESHOLDS))
        return jobs

    jobs = asyncio.run(scenario())
    assert jobs.queue.qsize() == 2


def test_full_queue_refuses_rather_than_blocking(store):
    """A submit that blocked would hold the request open indefinitely."""

    async def scenario():
        jobs = JobQueue(store, max_queued=1)
        jobs.submit(store.create("a.zip", THRESHOLDS))
        with pytest.raises(UploadRejectedError, match="queue is full") as excinfo:
            jobs.submit(store.create("b.zip", THRESHOLDS))
        assert excinfo.value.status_code == 429

    asyncio.run(scenario())


def test_cancellation_is_recorded(store):
    async def scenario():
        jobs = JobQueue(store, max_queued=4)
        record = store.create("a.zip", THRESHOLDS)
        assert jobs.is_cancelled(record.job_id) is False
        jobs.cancel(record.job_id)
        assert jobs.is_cancelled(record.job_id) is True

    asyncio.run(scenario())


# ── Status projection ────────────────────────────────────────────────────


def test_status_carries_counts_and_thresholds():
    record = JobRecord(
        job_id="abc",
        filename="batch.zip",
        total=10,
        processed=4,
        classified=2,
        needs_review=1,
        errors=1,
    )
    status = record.to_status()

    assert status.summary.classified == 2
    assert status.summary.errors == 1
    assert status.summary.total == 4
    assert status.percent == pytest.approx(40.0)


def test_percent_is_zero_before_the_total_is_known():
    assert JobRecord(job_id="a", filename="b.zip").to_status().percent == 0.0


@pytest.mark.parametrize(
    ("state", "terminal"),
    [
        (JobState.QUEUED, False),
        (JobState.RUNNING, False),
        (JobState.SUCCEEDED, True),
        (JobState.FAILED, True),
        (JobState.CANCELLED, True),
    ],
)
def test_terminal_states(state, terminal):
    assert state.is_terminal is terminal
