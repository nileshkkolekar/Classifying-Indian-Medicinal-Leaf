"""Assemble models, optimisers, schedulers and criteria from configuration.

Centralising construction keeps the training loop free of ``if
cfg.optimizer == ...`` branches, and gives checkpoints one place to record
exactly how a model was built.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)

from medicinal_leaf.config.settings import ModelConfig, PreprocessingConfig, TrainingConfig
from medicinal_leaf.models.model import LeafClassifier

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CheckpointMeta:
    """Everything needed to rebuild a model and reproduce its preprocessing."""

    backbone: str
    num_classes: int
    class_names: list[str]
    image_size: int
    resize_strategy: str
    normalize_mean: list[float]
    normalize_std: list[float]
    segment_leaf: bool
    dropout: float = 0.2
    epoch: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    manifest_fingerprint: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def preprocessing_config(self) -> PreprocessingConfig:
        """Rebuild the exact preprocessing the model was trained with."""
        return PreprocessingConfig(
            image_size=self.image_size,
            resize_strategy=self.resize_strategy,  # type: ignore[arg-type]
            normalize_mean=tuple(self.normalize_mean),  # type: ignore[arg-type]
            normalize_std=tuple(self.normalize_std),  # type: ignore[arg-type]
            segment_leaf=self.segment_leaf,
        )


def build_model(config: ModelConfig, num_classes: int) -> LeafClassifier:
    """Instantiate the classifier described by ``config``."""
    return LeafClassifier(
        backbone=config.backbone,
        num_classes=num_classes,
        pretrained=config.pretrained,
        dropout=config.dropout,
        freeze_backbone=config.freeze_backbone,
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """``(total, trainable)`` parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    """Optimiser over the parameters that are currently trainable.

    Filtering on ``requires_grad`` matters when the backbone is frozen —
    otherwise the optimiser holds state for parameters that never move.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Model has no trainable parameters — is the backbone frozen?")

    if config.optimizer == "adamw":
        return AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "adam":
        return Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return SGD(
            params,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")


def build_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
) -> LRScheduler | ReduceLROnPlateau | None:
    """Per-epoch learning-rate schedule, optionally preceded by a warmup."""
    if config.scheduler == "none":
        return None

    if config.scheduler == "plateau":
        # Stepped with the monitored metric rather than blindly each epoch.
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    main: LRScheduler
    if config.scheduler == "cosine":
        main = CosineAnnealingLR(optimizer, T_max=max(1, config.epochs - config.warmup_epochs))
    elif config.scheduler == "step":
        main = StepLR(optimizer, step_size=max(1, config.epochs // 3), gamma=0.1)
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler!r}")

    if config.warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=config.warmup_epochs)
        return SequentialLR(
            optimizer,
            schedulers=[warmup, main],
            milestones=[config.warmup_epochs],
        )
    return main


def build_criterion(
    config: TrainingConfig,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    """Cross-entropy, optionally smoothed and class-weighted."""
    weight = class_weights if config.class_weighting else None
    if config.class_weighting and class_weights is None:
        logger.warning("class_weighting is enabled but no weights were supplied.")
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=config.label_smoothing)


def save_checkpoint(
    path: str | Path,
    model: LeafClassifier,
    meta: CheckpointMeta,
    *,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
) -> Path:
    """Write weights plus metadata; optimiser state is optional (resume only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "meta": asdict(meta),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(payload, path)
    # A human-readable twin, so a checkpoint's provenance is greppable.
    path.with_suffix(".meta.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    logger.info("Saved checkpoint: %s (epoch %d)", path, meta.epoch)
    return path


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    *,
    strict: bool = True,
) -> tuple[LeafClassifier, CheckpointMeta]:
    """Rebuild the model described by a checkpoint and load its weights.

    ``pretrained=False`` because the stored weights are about to overwrite
    everything — downloading ImageNet weights first would be wasted work.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No checkpoint at {path}")

    payload = torch.load(path, map_location=device, weights_only=True)
    meta = CheckpointMeta(**payload["meta"])

    model = LeafClassifier(
        backbone=meta.backbone,
        num_classes=meta.num_classes,
        pretrained=False,
        dropout=meta.dropout,
    )
    model.load_state_dict(payload["state_dict"], strict=strict)
    model.to(device)
    model.eval()

    logger.info("Loaded %s from %s (epoch %d)", meta.backbone, path, meta.epoch)
    return model, meta
