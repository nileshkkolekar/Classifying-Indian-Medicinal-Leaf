"""The queued bulk endpoints, end to end through a real ASGI client.

A stub model stands in for the real one, but the worker, the queue, the
spooled upload and the CSV are all genuine — these tests exercise the
submit / poll / download cycle a client actually performs.
"""

from __future__ import annotations

import csv
import io
import time

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

TERMINAL = {"succeeded", "failed", "cancelled"}


@pytest.fixture
def job_client(tmp_path, monkeypatch, stub_predictor):
    """A client whose job store is scoped to tmp_path, with a model loaded.

    The store directory is set through the environment because the store is
    built during lifespan, before any fixture could reach in and patch it.
    """
    monkeypatch.setenv("MLC_QUEUE__JOB_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("MLC_QUEUE__CHUNK_SIZE", "4")
    monkeypatch.setenv("MLC_AUTH__ENABLED", "false")

    from medicinal_leaf.api.app import app, get_predictor

    app.dependency_overrides[get_predictor] = lambda: stub_predictor
    with TestClient(app) as client:
        # The worker reads the predictor off app.state, not the dependency.
        client.app.state.predictor = stub_predictor
        yield client
    app.dependency_overrides.clear()


def wait_for_terminal(client, job_id: str, timeout: float = 60.0) -> dict:
    """Poll until the job reaches a terminal state, as a real client would."""
    deadline = time.time() + timeout
    body: dict = {}
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["state"] in TERMINAL:
            return body
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} never finished; last state {body.get('state')!r}")


def zip_upload(payload: bytes, name: str = "bulk.zip"):
    return {"file": (name, payload, "application/zip")}


# ── Submit ───────────────────────────────────────────────────────────────


def test_submit_returns_immediately_with_a_job_id(job_client, make_zip, jpeg_bytes):
    archive = make_zip({f"leaf_{i}.jpg": jpeg_bytes for i in range(6)})
    response = job_client.post("/jobs", files=zip_upload(archive))

    assert response.status_code == 202
    body = response.json()
    assert body["job_id"]
    assert body["state"] == "queued"
    # Counted from the central directory, before any image is decompressed.
    assert body["total"] == 6
    assert body["status_url"] == f"/jobs/{body['job_id']}"


def test_submit_rejects_a_non_zip(job_client, jpeg_bytes):
    response = job_client.post("/jobs", files=zip_upload(jpeg_bytes))
    assert response.status_code == 400
    assert "ZIP" in response.json()["detail"]


def test_submit_rejects_an_empty_upload(job_client):
    response = job_client.post("/jobs", files=zip_upload(b""))
    assert response.status_code == 400


def test_failed_submit_leaves_no_orphan_job(job_client, jpeg_bytes):
    job_client.post("/jobs", files=zip_upload(jpeg_bytes))
    assert job_client.get("/jobs").json()["jobs"] == []


def test_oversize_archive_is_rejected(job_client, make_zip, jpeg_bytes):
    job_client.app.state.settings.queue.max_archive_bytes = 64
    archive = make_zip({f"leaf_{i}.jpg": jpeg_bytes for i in range(3)})

    response = job_client.post("/jobs", files=zip_upload(archive))
    assert response.status_code == 413


# ── Run to completion ────────────────────────────────────────────────────


def test_job_runs_and_reports_progress(job_client, make_zip, jpeg_bytes):
    archive = make_zip({f"leaf_{i}.jpg": jpeg_bytes for i in range(9)})
    job_id = job_client.post("/jobs", files=zip_upload(archive)).json()["job_id"]

    body = wait_for_terminal(job_client, job_id)

    assert body["state"] == "succeeded"
    assert body["processed"] == 9
    assert body["summary"]["classified"] == 9
    assert body["finished_at"]


def test_results_csv_has_one_row_per_image(job_client, make_zip, jpeg_bytes):
    archive = make_zip({f"leaf_{i}.jpg": jpeg_bytes for i in range(5)})
    job_id = job_client.post("/jobs", files=zip_upload(archive)).json()["job_id"]
    wait_for_terminal(job_client, job_id)

    response = job_client.get(f"/jobs/{job_id}/results")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == 5
    assert set(rows[0]) == {"filename", "verdict", "label", "confidence", "needs_review", "note"}
    assert rows[0]["label"] == "Tulsi"


def test_a_corrupt_member_does_not_fail_the_job(job_client, make_zip, jpeg_bytes):
    archive = make_zip({"ok.jpg": jpeg_bytes, "broken.jpg": b"not an image"})
    job_id = job_client.post("/jobs", files=zip_upload(archive)).json()["job_id"]

    body = wait_for_terminal(job_client, job_id)

    assert body["state"] == "succeeded"
    assert body["summary"]["classified"] == 1
    assert body["summary"]["errors"] == 1


def test_uploaded_archive_is_deleted_once_the_job_ends(job_client, make_zip, jpeg_bytes):
    """NFR-8: held only as long as the work requires."""
    archive = make_zip({f"leaf_{i}.jpg": jpeg_bytes for i in range(4)})
    job_id = job_client.post("/jobs", files=zip_upload(archive)).json()["job_id"]
    wait_for_terminal(job_client, job_id)

    assert not job_client.app.state.jobs.store.archive_path(job_id).exists()
    # The record survives so the client can still collect results.
    assert job_client.get(f"/jobs/{job_id}").status_code == 200


def test_chunking_does_not_reorder_results(job_client, make_zip, jpeg_bytes):
    """Chunk size is 4 here, so 10 images span three chunks."""
    names = [f"leaf_{i:02d}.jpg" for i in range(10)]
    archive = make_zip(dict.fromkeys(names, jpeg_bytes))
    job_id = job_client.post("/jobs", files=zip_upload(archive)).json()["job_id"]
    wait_for_terminal(job_client, job_id)

    rows = list(csv.DictReader(io.StringIO(job_client.get(f"/jobs/{job_id}/results").text)))
    assert [r["filename"] for r in rows] == names


# ── Listing, lookup, deletion ────────────────────────────────────────────


def test_unknown_job_is_404(job_client):
    assert job_client.get("/jobs/nope").status_code == 404
    assert job_client.get("/jobs/nope/results").status_code == 404
    assert job_client.delete("/jobs/nope").status_code == 404


def test_jobs_are_listed(job_client, make_zip, jpeg_bytes):
    archive = make_zip({"leaf.jpg": jpeg_bytes})
    job_id = job_client.post("/jobs", files=zip_upload(archive)).json()["job_id"]
    wait_for_terminal(job_client, job_id)

    listed = job_client.get("/jobs").json()["jobs"]
    assert [j["job_id"] for j in listed] == [job_id]


def test_finished_job_can_be_deleted(job_client, make_zip, jpeg_bytes):
    archive = make_zip({"leaf.jpg": jpeg_bytes})
    job_id = job_client.post("/jobs", files=zip_upload(archive)).json()["job_id"]
    wait_for_terminal(job_client, job_id)

    assert job_client.delete(f"/jobs/{job_id}").json()["detail"] == "Deleted."
    assert job_client.get(f"/jobs/{job_id}").status_code == 404


def test_thresholds_are_honoured_per_job(job_client, make_zip, jpeg_bytes):
    archive = make_zip({f"leaf_{i}.jpg": jpeg_bytes for i in range(3)})
    job_id = job_client.post(
        "/jobs", files=zip_upload(archive), params={"review_threshold": 0.99}
    ).json()["job_id"]

    body = wait_for_terminal(job_client, job_id)

    assert body["thresholds"]["review_threshold"] == pytest.approx(0.99)
    assert body["summary"]["needs_review"] == 3
    assert body["summary"]["classified"] == 0
