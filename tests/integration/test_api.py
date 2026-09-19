"""HTTP surface of the prediction API.

Exercised through a real ASGI client with a stub model injected, so the tests
cover routing, multipart parsing, status codes and upload limits without
needing a trained checkpoint.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from medicinal_leaf.api.app import app, get_predictor
from medicinal_leaf.api.schemas import Verdict

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    """A client with lifespan run — no model loaded unless one is injected."""
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def ready(client, stub_predictor):
    """A client with the stub model wired in."""
    app.dependency_overrides[get_predictor] = lambda: stub_predictor
    yield client
    app.dependency_overrides.clear()


def image_upload(payload: bytes, name: str = "leaf.jpg"):
    return {"file": (name, payload, "image/jpeg")}


def zip_upload(payload: bytes, name: str = "batch.zip"):
    return {"file": (name, payload, "application/zip")}


# ── Health ───────────────────────────────────────────────────────────────


def test_health_reports_missing_model_rather_than_failing(client):
    """A deployment with no checkpoint must explain itself, not crash-loop."""
    body = client.get("/health").json()

    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["thresholds"]["review_threshold"] > 0


def test_health_exposes_thresholds(client):
    thresholds = client.get("/health").json()["thresholds"]
    assert 0.0 <= thresholds["unknown_threshold"] <= thresholds["review_threshold"] <= 1.0


# ── Guard rails ──────────────────────────────────────────────────────────


def test_prediction_without_a_model_is_503(client, jpeg_bytes):
    response = client.post("/predict", files=image_upload(jpeg_bytes))
    assert response.status_code == 503
    assert "leaf-train fit" in response.json()["detail"]


def test_empty_upload_is_rejected(ready):
    response = ready.post("/predict", files=image_upload(b""))
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_oversize_upload_is_rejected(ready, jpeg_bytes):
    ready.app.state.settings.serving.max_image_bytes = 32
    response = ready.post("/predict", files=image_upload(jpeg_bytes))

    assert response.status_code == 413
    assert "limit" in response.json()["detail"]


def test_thresholds_must_be_ordered(ready, jpeg_bytes):
    response = ready.post(
        "/predict",
        files=image_upload(jpeg_bytes),
        params={"review_threshold": 0.2, "unknown_threshold": 0.9},
    )
    assert response.status_code == 422


def test_threshold_outside_zero_to_one_is_rejected(ready, jpeg_bytes):
    response = ready.post(
        "/predict", files=image_upload(jpeg_bytes), params={"review_threshold": 1.5}
    )
    assert response.status_code == 422


# ── Single image (FR-8, FR-10) ───────────────────────────────────────────


def test_single_image_is_classified(ready, jpeg_bytes):
    body = ready.post("/predict", files=image_upload(jpeg_bytes)).json()
    result = body["result"]

    assert result["verdict"] == Verdict.CLASSIFIED
    assert result["label"] == "Tulsi"
    assert result["confidence"] == pytest.approx(0.95)
    assert result["needs_review"] is False
    assert result["filename"] == "leaf.jpg"


def test_confidence_accompanies_every_prediction(ready, jpeg_bytes):
    """FR-10: a score is always returned, whatever the verdict."""
    result = ready.post("/predict", files=image_upload(jpeg_bytes)).json()["result"]
    assert result["confidence"] is not None
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_low_confidence_is_flagged_for_review(ready, jpeg_bytes):
    body = ready.post(
        "/predict", files=image_upload(jpeg_bytes), params={"review_threshold": 0.99}
    ).json()

    assert body["result"]["verdict"] == Verdict.NEEDS_REVIEW
    assert body["result"]["needs_review"] is True
    assert body["result"]["note"]
    # The policy applied is echoed back with the result.
    assert body["thresholds"]["review_threshold"] == pytest.approx(0.99)


def test_very_low_confidence_returns_no_species(ready, jpeg_bytes):
    """FR-14: an honest refusal instead of a forced prediction."""
    result = ready.post(
        "/predict",
        files=image_upload(jpeg_bytes),
        params={"review_threshold": 0.99, "unknown_threshold": 0.99},
    ).json()["result"]

    assert result["verdict"] == Verdict.UNABLE_TO_CLASSIFY
    assert result["label"] is None
    assert result["confidence"] is not None


def test_undecodable_image_reports_an_error(ready):
    result = ready.post("/predict", files=image_upload(b"nonsense bytes")).json()["result"]
    assert result["verdict"] == Verdict.ERROR
    assert result["needs_review"] is True


# ── Bulk ZIP (FR-9, FR-11) ───────────────────────────────────────────────


def test_zip_classifies_every_image(ready, make_zip, jpeg_bytes):
    archive = make_zip({f"leaf_{i}.jpg": jpeg_bytes for i in range(4)})
    body = ready.post("/predict/batch", files=zip_upload(archive)).json()

    assert body["summary"]["total"] == 4
    assert body["summary"]["classified"] == 4
    assert len(body["results"]) == 4


def test_zip_results_carry_the_columns_the_table_needs(ready, make_zip, jpeg_bytes):
    """FR-11: filename, species and confidence for each file."""
    archive = make_zip({"folder/leaf.jpg": jpeg_bytes})
    result = ready.post("/predict/batch", files=zip_upload(archive)).json()["results"][0]

    assert result["filename"] == "folder/leaf.jpg"
    assert result["label"] == "Tulsi"
    assert result["confidence"] is not None
    assert "verdict" in result


def test_zip_reports_per_file_failures(ready, make_zip, jpeg_bytes):
    archive = make_zip({"ok.jpg": jpeg_bytes, "broken.jpg": b"not an image"})
    body = ready.post("/predict/batch", files=zip_upload(archive)).json()

    assert body["summary"]["total"] == 2
    assert body["summary"]["classified"] == 1
    assert body["summary"]["errors"] == 1


def test_zip_flagging_respects_the_threshold(ready, make_zip, jpeg_bytes):
    archive = make_zip({f"leaf_{i}.jpg": jpeg_bytes for i in range(3)})
    body = ready.post(
        "/predict/batch", files=zip_upload(archive), params={"review_threshold": 0.99}
    ).json()

    assert body["summary"]["needs_review"] == 3
    assert body["summary"]["classified"] == 0


def test_non_zip_upload_is_rejected(ready, jpeg_bytes):
    response = ready.post("/predict/batch", files=zip_upload(jpeg_bytes))
    assert response.status_code == 400
    assert "ZIP" in response.json()["detail"]


def test_zip_without_images_is_rejected(ready, make_zip):
    archive = make_zip({"readme.txt": b"nothing to classify"})
    response = ready.post("/predict/batch", files=zip_upload(archive))

    assert response.status_code == 400
    assert "no images" in response.json()["detail"]


def test_zip_entry_limit_is_enforced(ready, make_zip, jpeg_bytes):
    ready.app.state.settings.serving.max_zip_entries = 2
    archive = make_zip({f"leaf_{i}.jpg": jpeg_bytes for i in range(5)})
    response = ready.post("/predict/batch", files=zip_upload(archive))

    assert response.status_code == 413


def test_oversize_archive_is_rejected(ready, make_zip, jpeg_bytes):
    ready.app.state.settings.serving.max_archive_bytes = 64
    archive = make_zip({f"leaf_{i}.jpg": jpeg_bytes for i in range(3)})
    response = ready.post("/predict/batch", files=zip_upload(archive))

    assert response.status_code == 413
