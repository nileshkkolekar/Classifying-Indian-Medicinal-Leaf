"""Configuration layering and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from medicinal_leaf.config.settings import (
    PROJECT_ROOT,
    DataConfig,
    Settings,
    load_settings,
)


def test_defaults_are_usable():
    settings = Settings()
    assert (
        settings.data.train_size + settings.data.val_size + settings.data.test_size
        == pytest.approx(1.0)
    )
    assert settings.preprocessing.image_size > 0
    assert settings.model.backbone


def test_relative_paths_resolve_against_project_root():
    config = DataConfig(raw_dir="Data")
    assert config.raw_dir.is_absolute()
    assert config.raw_dir == (PROJECT_ROOT / "Data").resolve()


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValidationError, match="must equal 1.0"):
        DataConfig(train_size=0.8, val_size=0.3, test_size=0.3)


def test_extensions_are_normalised():
    config = DataConfig(image_extensions=("JPG", ".PNG"))
    assert config.image_extensions == (".jpg", ".png")


def test_environment_overrides_yaml(monkeypatch):
    monkeypatch.setenv("MLC_TRAINING__EPOCHS", "77")
    monkeypatch.setenv("MLC_SEED", "1234")
    settings = load_settings("development")
    assert settings.training.epochs == 77
    assert settings.seed == 1234


def test_keyword_overrides_beat_everything(monkeypatch):
    monkeypatch.setenv("MLC_SEED", "1234")
    settings = load_settings("development", seed=9)
    assert settings.seed == 9


def test_development_yaml_is_loaded(monkeypatch):
    monkeypatch.delenv("MLC_MODEL__BACKBONE", raising=False)
    settings = load_settings("development")
    # development.yaml deliberately picks a small backbone.
    assert settings.model.backbone == "resnet18"
    assert settings.env == "development"


def test_summary_is_flat_and_serialisable():
    summary = Settings().summary()
    assert set(summary) >= {"env", "seed", "backbone", "epochs", "device"}
    assert all(isinstance(v, (str, int, float, bool)) for v in summary.values())


def test_ensure_directories_creates_outputs(tmp_path, monkeypatch):
    settings = Settings()
    settings.data.processed_dir = tmp_path / "processed"
    settings.data.manifest_path = tmp_path / "m" / "manifest.csv"
    settings.training.checkpoint_dir = tmp_path / "ckpt"
    settings.training.run_dir = tmp_path / "runs"

    settings.ensure_directories()

    assert settings.data.processed_dir.is_dir()
    assert settings.data.manifest_path.parent.is_dir()
    assert settings.training.checkpoint_dir.is_dir()
