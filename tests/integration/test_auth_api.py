"""The HTTP surface of authentication, with it switched ON.

Every other API test module disables auth so it can focus on prediction
behaviour. This one is the opposite: it exists to prove the endpoints are
actually closed, and that the ways in work.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from medicinal_leaf.api.auth import generate_api_key, hash_api_key, hash_password

pytestmark = pytest.mark.integration

PASSWORD = "correct-horse-battery-staple"
USERNAME = "botanist"
API_KEY = generate_api_key()

PROTECTED = ["/predict", "/predict/batch", "/jobs"]


@pytest.fixture
def secured(tmp_path, monkeypatch, stub_predictor):
    """A client with authentication enabled and one known user."""
    monkeypatch.setenv("MLC_AUTH__ENABLED", "true")
    monkeypatch.setenv("MLC_AUTH__SECRET_KEY", "test-signing-key-not-used-anywhere-real")
    monkeypatch.setenv("MLC_AUTH__USERS", json.dumps({USERNAME: hash_password(PASSWORD)}))
    monkeypatch.setenv("MLC_AUTH__API_KEY_SHA256", json.dumps([hash_api_key(API_KEY)]))
    monkeypatch.setenv("MLC_QUEUE__JOB_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("MLC_SERVING__CHECKPOINT_PATH", str(tmp_path / "absent.pt"))
    monkeypatch.delenv("MLC_AWS__CHECKPOINT_URI", raising=False)

    from medicinal_leaf.api.app import app, get_predictor

    app.dependency_overrides[get_predictor] = lambda: stub_predictor
    with TestClient(app) as client:
        client.app.state.predictor = stub_predictor
        yield client
    app.dependency_overrides.clear()


def login(client) -> str:
    response = client.post("/auth/token", data={"username": USERNAME, "password": PASSWORD})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def image(payload: bytes):
    return {"file": ("leaf.jpg", payload, "image/jpeg")}


# ── The door is shut ─────────────────────────────────────────────────────


@pytest.mark.parametrize("path", PROTECTED)
def test_protected_endpoints_reject_anonymous_callers(secured, jpeg_bytes, path):
    response = secured.post(path, files=image(jpeg_bytes))
    assert response.status_code == 401


def test_rejection_carries_the_challenge_header(secured, jpeg_bytes):
    response = secured.post("/predict", files=image(jpeg_bytes))
    assert response.headers["www-authenticate"] == "Bearer"


def test_job_listing_is_protected(secured):
    assert secured.get("/jobs").status_code == 401


def test_a_forged_token_is_rejected(secured, jpeg_bytes):
    response = secured.post(
        "/predict",
        files=image(jpeg_bytes),
        headers={"Authorization": "Bearer not.a.real.token"},
    )
    assert response.status_code == 401


@pytest.fixture
def secured_without_a_model(tmp_path, monkeypatch):
    """Authentication on, and deliberately no model loaded."""
    monkeypatch.setenv("MLC_AUTH__ENABLED", "true")
    monkeypatch.setenv("MLC_AUTH__SECRET_KEY", "test-signing-key-not-used-anywhere-real")
    monkeypatch.setenv("MLC_AUTH__USERS", json.dumps({USERNAME: hash_password(PASSWORD)}))
    monkeypatch.setenv("MLC_QUEUE__JOB_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("MLC_SERVING__CHECKPOINT_PATH", str(tmp_path / "absent.pt"))
    monkeypatch.delenv("MLC_AWS__CHECKPOINT_URI", raising=False)

    from medicinal_leaf.api.app import app

    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.mark.parametrize("path", PROTECTED)
def test_credentials_are_checked_before_the_model(secured_without_a_model, jpeg_bytes, path):
    """A stranger must not learn whether this deployment has a model.

    FastAPI resolves dependencies in signature order. With the model check
    declared first, an anonymous request got 503 instead of 401 — which
    quietly discloses deployment state to someone who has not authenticated.
    Caught by running the container, not by the unit tests.
    """
    response = secured_without_a_model.post(path, files=image(jpeg_bytes))

    assert response.status_code == 401
    assert "model" not in response.json()["detail"].lower()


def test_health_stays_public(secured):
    """ECS and load-balancer probes cannot present credentials."""
    assert secured.get("/health").status_code == 200


def test_openapi_stays_public(secured):
    assert secured.get("/openapi.json").status_code == 200


# ── Logging in ───────────────────────────────────────────────────────────


def test_token_is_issued_for_valid_credentials(secured):
    body = secured.post("/auth/token", data={"username": USERNAME, "password": PASSWORD}).json()

    assert body["token_type"] == "bearer"
    assert body["username"] == USERNAME
    assert body["expires_in"] > 0
    assert body["access_token"]


def test_wrong_password_is_rejected(secured):
    response = secured.post("/auth/token", data={"username": USERNAME, "password": "wrong"})
    assert response.status_code == 401


def test_unknown_user_is_rejected_identically(secured):
    """The response must not reveal whether the account exists."""
    wrong_password = secured.post("/auth/token", data={"username": USERNAME, "password": "wrong"})
    unknown_user = secured.post("/auth/token", data={"username": "nobody", "password": PASSWORD})
    assert wrong_password.status_code == unknown_user.status_code == 401
    assert wrong_password.json() == unknown_user.json()


def test_the_password_never_appears_in_the_response(secured):
    response = secured.post("/auth/token", data={"username": USERNAME, "password": PASSWORD})
    assert PASSWORD not in response.text


# ── Getting in ───────────────────────────────────────────────────────────


def test_a_token_opens_the_prediction_endpoint(secured, jpeg_bytes):
    token = login(secured)
    response = secured.post(
        "/predict",
        files=image(jpeg_bytes),
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json()["result"]["label"] == "Tulsi"


def test_an_api_key_opens_it_too(secured, jpeg_bytes):
    """Scripts should not have to perform a password exchange."""
    response = secured.post("/predict", files=image(jpeg_bytes), headers={"X-API-Key": API_KEY})
    assert response.status_code == 200


def test_a_wrong_api_key_does_not(secured, jpeg_bytes):
    response = secured.post(
        "/predict", files=image(jpeg_bytes), headers={"X-API-Key": generate_api_key()}
    )
    assert response.status_code == 401


def test_a_token_opens_the_queue(secured, make_zip, jpeg_bytes):
    token = login(secured)
    archive = make_zip({"leaf.jpg": jpeg_bytes})

    response = secured.post(
        "/jobs",
        files={"file": ("bulk.zip", archive, "application/zip")},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 202


# ── Who am I ─────────────────────────────────────────────────────────────


def test_me_identifies_the_token_holder(secured):
    token = login(secured)
    body = secured.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).json()

    assert body["username"] == USERNAME
    assert body["kind"] == "user"
    assert body["auth_enabled"] is True


def test_me_identifies_an_api_key(secured):
    body = secured.get("/auth/me", headers={"X-API-Key": API_KEY}).json()
    assert body["kind"] == "api_key"


def test_me_requires_credentials(secured):
    assert secured.get("/auth/me").status_code == 401
