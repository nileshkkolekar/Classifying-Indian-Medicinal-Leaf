"""Serving the React bundle from the API.

The mount sits at "/" and therefore catches everything unclaimed. If it were
ever registered before the API routes it would swallow them and the whole
service would answer with HTML — silently, and only in a built checkout.
That ordering is what these tests pin down.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from medicinal_leaf.api.app import mount_frontend


def build_bundle(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "index.html").write_text("<!doctype html><title>Leaf</title>", encoding="utf-8")
    return directory


def test_absent_bundle_is_not_mounted(tmp_path):
    """An un-built checkout must still serve the API."""
    application = FastAPI()
    assert mount_frontend(application, tmp_path / "never-built") is False


def test_present_bundle_is_mounted(tmp_path):
    application = FastAPI()
    assert mount_frontend(application, build_bundle(tmp_path / "dist")) is True


def test_index_is_served_at_the_root(tmp_path):
    application = FastAPI()
    mount_frontend(application, build_bundle(tmp_path / "dist"))

    with TestClient(application) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Leaf" in response.text


def test_api_routes_are_not_shadowed_by_the_mount(tmp_path):
    """The reason the mount is registered last."""
    application = FastAPI()

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    mount_frontend(application, build_bundle(tmp_path / "dist"))

    with TestClient(application) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_the_real_app_still_answers_json(tmp_path):
    """Guards the live app rather than a hand-built stand-in."""
    from medicinal_leaf.api.app import app

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert "model_loaded" in response.json()
