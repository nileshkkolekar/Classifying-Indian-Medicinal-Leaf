"""Typed configuration assembled from YAML, ``.env`` and the environment.

Resolution order, first match wins::

    1. keyword arguments passed to ``Settings(...)``
    2. environment variables       (``MLC_TRAINING__EPOCHS=40``)
    3. the ``.env`` file at the repository root
    4. ``configs/<MLC_ENV>.yaml``
    5. the defaults declared in this module

Nested keys are addressed with a double underscore, so ``training.epochs``
becomes ``MLC_TRAINING__EPOCHS``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# src/medicinal_leaf/config/settings.py -> repository root.
#
# Deriving this from __file__ only holds for a source checkout or an editable
# install. Installed normally — in a container, say — the package lives under
# site-packages and this would point somewhere with no configs/ in it, so the
# location is overridable. Read at import time, before any Settings exist.
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.environ.get("MLC_PROJECT_ROOT", _DEFAULT_PROJECT_ROOT)).resolve()
CONFIG_DIR = PROJECT_ROOT / "configs"

DEFAULT_ENV = "development"
ENV_VAR = "MLC_ENV"


def _absolutize(value: str | Path) -> Path:
    """Resolve a possibly-relative path against the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def config_path_for(env: str) -> Path:
    return CONFIG_DIR / f"{env}.yaml"


class DataConfig(BaseModel):
    """Where the images live and how the index/splits are produced."""

    raw_dir: Path = Path("Data")
    processed_dir: Path = Path("artifacts/processed")
    manifest_path: Path = Path("artifacts/manifest.csv")

    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")

    # Validation thresholds — see data.validation.
    min_side: int = Field(default=32, ge=1)
    max_aspect_ratio: float = Field(default=4.0, gt=1.0)
    min_images_per_class: int = Field(default=10, ge=1)
    max_imbalance_ratio: float = Field(default=5.0, gt=1.0)

    # Split fractions; must sum to 1.
    train_size: float = Field(default=0.70, gt=0.0, lt=1.0)
    val_size: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_size: float = Field(default=0.15, gt=0.0, lt=1.0)

    @field_validator("raw_dir", "processed_dir", "manifest_path")
    @classmethod
    def _resolve(cls, value: Path) -> Path:
        return _absolutize(value)

    @field_validator("image_extensions")
    @classmethod
    def _normalise_extensions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            ext if ext.startswith(".") else f".{ext}" for ext in (e.lower() for e in value)
        )

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> DataConfig:
        total = self.train_size + self.val_size + self.test_size
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train_size + val_size + test_size must equal 1.0, got {total:.4f}")
        return self


class PreprocessingConfig(BaseModel):
    """Deterministic image preparation applied to every split."""

    image_size: int = Field(default=224, ge=32)
    # "letterbox" preserves aspect ratio by padding; "resize" squashes.
    resize_strategy: Literal["letterbox", "resize", "center_crop"] = "letterbox"
    pad_color: tuple[int, int, int] = (0, 0, 0)

    # ImageNet statistics — correct whenever `model.pretrained` is true.
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Background removal before resizing (see preprocessing.segmentation).
    segment_leaf: bool = False
    segment_pad: int = Field(default=8, ge=0)


class AugmentationConfig(BaseModel):
    """Stochastic transforms applied to the training split only."""

    enabled: bool = True
    horizontal_flip: float = Field(default=0.5, ge=0.0, le=1.0)
    vertical_flip: float = Field(default=0.2, ge=0.0, le=1.0)
    rotation_degrees: float = Field(default=20.0, ge=0.0, le=180.0)
    random_resized_crop_scale: tuple[float, float] = (0.7, 1.0)
    color_jitter_brightness: float = Field(default=0.2, ge=0.0)
    color_jitter_contrast: float = Field(default=0.2, ge=0.0)
    color_jitter_saturation: float = Field(default=0.2, ge=0.0)
    color_jitter_hue: float = Field(default=0.02, ge=0.0, le=0.5)
    random_erasing: float = Field(default=0.0, ge=0.0, le=1.0)


class ModelConfig(BaseModel):
    """Backbone selection and classification head shape."""

    backbone: str = "resnet50"
    pretrained: bool = True
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    # Freeze everything but the head — useful for a fast first baseline.
    freeze_backbone: bool = False
    # Unfreeze the backbone after this many epochs; None disables the schedule.
    unfreeze_after_epoch: int | None = None


class TrainingConfig(BaseModel):
    """Optimisation loop, hardware and checkpointing."""

    epochs: int = Field(default=25, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)  # SGD only
    scheduler: Literal["cosine", "plateau", "step", "none"] = "cosine"
    warmup_epochs: int = Field(default=0, ge=0)
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)
    # Weight the loss by inverse class frequency.
    class_weighting: bool = False
    gradient_clip_norm: float | None = None

    early_stopping_patience: int | None = 5
    early_stopping_metric: Literal["val_loss", "val_accuracy", "val_macro_f1"] = "val_macro_f1"

    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = False
    mixed_precision: bool = False
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"

    checkpoint_dir: Path = Path("artifacts/checkpoints")
    run_dir: Path = Path("artifacts/runs")

    @field_validator("checkpoint_dir", "run_dir")
    @classmethod
    def _resolve(cls, value: Path) -> Path:
        return _absolutize(value)

    def resolved_device(self) -> str:
        """Turn ``"auto"`` into a concrete device string.

        Imported lazily so that configuration stays usable (in the EDA
        notebook, say) without torch installed.
        """
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is a hard dep in practice
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


class AWSConfig(BaseModel):
    """Amazon S3 locations for the dataset and model artifacts (FR-1).

    Everything is optional: with nothing set the pipeline works entirely from
    local disk, which is what the tests and a laptop run do. Credentials are
    never configured here — boto3 resolves them from the environment, an
    instance profile or an ECS task role, so none can be committed (NFR-4).
    """

    #: ``None`` defers to AWS_REGION or the active profile.
    region: str | None = None
    #: Override for LocalStack or MinIO in testing; unused in production.
    endpoint_url: str | None = None

    #: ``s3://bucket/prefix`` holding one folder per species.
    dataset_uri: str | None = None
    #: ``s3://bucket/key`` of a trained checkpoint.
    checkpoint_uri: str | None = None

    @field_validator("dataset_uri", "checkpoint_uri")
    @classmethod
    def _must_be_s3_uri(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("s3://"):
            raise ValueError(f"Expected an s3:// URI, got {value!r}")
        return value


class ServingConfig(BaseModel):
    """The prediction API and the Streamlit UI in front of it.

    The two thresholds implement the BRD's review policy and are config, not
    constants (FR-13): ``review_threshold`` flags a prediction as needing a
    human look (FR-12), and below ``unknown_threshold`` the service declines
    to name a species at all rather than forcing one (FR-14).
    """

    checkpoint_path: Path = Path("artifacts/checkpoints/best.pt")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    review_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    unknown_threshold: float = Field(default=0.30, ge=0.0, le=1.0)

    # Upload limits. An endpoint that unpacks archives is the obvious denial
    # of service target, so every dimension is bounded.
    max_image_bytes: int = Field(default=15 * 1024 * 1024, ge=1)
    max_archive_bytes: int = Field(default=100 * 1024 * 1024, ge=1)
    max_zip_entries: int = Field(default=200, ge=1)
    max_zip_uncompressed_bytes: int = Field(default=500 * 1024 * 1024, ge=1)
    # A zip bomb's whole trick is an absurd uncompressed:compressed ratio.
    max_compression_ratio: float = Field(default=100.0, gt=1.0)

    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")

    # Where the Streamlit UI looks for the API.
    api_base_url: str = "http://127.0.0.1:8000"
    request_timeout_seconds: float = Field(default=120.0, gt=0.0)

    @field_validator("checkpoint_path")
    @classmethod
    def _resolve(cls, value: Path) -> Path:
        return _absolutize(value)

    @field_validator("allowed_extensions")
    @classmethod
    def _normalise_extensions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            ext if ext.startswith(".") else f".{ext}" for ext in (e.lower() for e in value)
        )

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> ServingConfig:
        if self.unknown_threshold > self.review_threshold:
            raise ValueError(
                "unknown_threshold must not exceed review_threshold "
                f"({self.unknown_threshold} > {self.review_threshold})"
            )
        return self


class Settings(BaseSettings):
    """Root configuration object handed to every stage of the pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="MLC_",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_*` config keys would otherwise collide with pydantic internals.
        protected_namespaces=(),
    )

    env: str = DEFAULT_ENV
    seed: int = 42

    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    aws: AWSConfig = Field(default_factory=AWSConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_path = config_path_for(os.getenv(ENV_VAR, DEFAULT_ENV))
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        if yaml_path.is_file():
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path))
        sources.append(file_secret_settings)
        return tuple(sources)

    def ensure_directories(self) -> None:
        """Create every output directory the pipeline writes to."""
        for path in (
            self.data.processed_dir,
            self.data.manifest_path.parent,
            self.training.checkpoint_dir,
            self.training.run_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def summary(self) -> dict[str, Any]:
        """Flat, log-friendly view of the knobs that usually matter."""
        return {
            "env": self.env,
            "seed": self.seed,
            "raw_dir": str(self.data.raw_dir),
            "image_size": self.preprocessing.image_size,
            "backbone": self.model.backbone,
            "pretrained": self.model.pretrained,
            "epochs": self.training.epochs,
            "batch_size": self.training.batch_size,
            "learning_rate": self.training.learning_rate,
            "device": self.training.resolved_device(),
        }


def load_settings(env: str | None = None, **overrides: Any) -> Settings:
    """Build a :class:`Settings` for ``env`` (default: ``$MLC_ENV``).

    ``overrides`` win over every other source, which makes this the seam to
    poke from tests and notebooks::

        settings = load_settings("development", seed=0)
    """
    if env is not None:
        os.environ[ENV_VAR] = env
    resolved_env = os.getenv(ENV_VAR, DEFAULT_ENV)
    overrides.setdefault("env", resolved_env)
    return Settings(**overrides)
