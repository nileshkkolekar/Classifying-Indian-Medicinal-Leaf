"""The fit loop: epochs, validation, early stopping, checkpointing.

Kept deliberately free of configuration branching — everything that varies
(optimiser, schedule, loss) arrives pre-built from
:mod:`medicinal_leaf.models.factory`.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from medicinal_leaf.evaluation.metrics import ClassificationMetrics, compute_metrics
from medicinal_leaf.models.factory import CheckpointMeta, save_checkpoint

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    from medicinal_leaf.config.settings import TrainingConfig
    from medicinal_leaf.models.model import LeafClassifier

logger = logging.getLogger(__name__)

#: Metrics where a lower value is better.
MINIMIZE = frozenset({"val_loss", "loss"})


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch.

    ``deterministic`` additionally pins cuDNN's algorithm choice, which makes
    runs bit-reproducible at a noticeable cost in throughput — worth it when
    chasing a regression, not for routine training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Seeded everything with %d (deterministic=%s)", seed, deterministic)


@dataclass(slots=True)
class EpochResult:
    """One row of the training history."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    learning_rate: float
    seconds: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"epoch {self.epoch:>3} | "
            f"train loss {self.train_loss:.4f} acc {self.train_accuracy:.4f} | "
            f"val loss {self.val_loss:.4f} acc {self.val_accuracy:.4f} "
            f"f1 {self.val_macro_f1:.4f} | lr {self.learning_rate:.2e} | {self.seconds:.1f}s"
        )


class EarlyStopping:
    """Stop when the monitored metric has not improved for ``patience`` epochs."""

    def __init__(self, patience: int, *, mode: str = "max", min_delta: float = 1e-4) -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.epochs_without_improvement = 0

    def improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float) -> bool:
        """Record ``value``; return True if it is a new best."""
        if self.improved(value):
            self.best = value
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience


class Trainer:
    """Drive a model through its epochs and keep the best checkpoint."""

    def __init__(
        self,
        model: LeafClassifier,
        optimizer: Optimizer,
        criterion: nn.Module,
        config: TrainingConfig,
        *,
        class_names: list[str],
        scheduler: Any | None = None,
        device: str | torch.device | None = None,
        checkpoint_meta: CheckpointMeta | None = None,
        checkpoint_name: str = "best.pt",
        unfreeze_after_epoch: int | None = None,
    ) -> None:
        self.device = torch.device(device or config.resolved_device())
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)
        self.scheduler = scheduler
        self.config = config
        self.class_names = list(class_names)
        self.checkpoint_meta = checkpoint_meta
        self.checkpoint_path = Path(config.checkpoint_dir) / checkpoint_name
        # Lives on ModelConfig, so the caller has to hand it over.
        self.unfreeze_after_epoch = unfreeze_after_epoch

        # AMP only pays off on CUDA; elsewhere it is a no-op wrapper.
        self._amp_enabled = config.mixed_precision and self.device.type == "cuda"
        self._scaler = torch.amp.GradScaler(self.device.type, enabled=self._amp_enabled)

        self.history: list[EpochResult] = []
        self.best_metric: float | None = None
        self.best_epoch: int | None = None

        mode = "min" if config.early_stopping_metric in MINIMIZE else "max"
        self._stopper = (
            EarlyStopping(config.early_stopping_patience, mode=mode)
            if config.early_stopping_patience
            else None
        )

        logger.info(
            "Trainer ready on %s (amp=%s, epochs=%d, monitoring %s)",
            self.device,
            self._amp_enabled,
            config.epochs,
            config.early_stopping_metric,
        )

    # ── Single epoch ─────────────────────────────────────────────────────

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        seen = 0

        progress = tqdm(loader, desc=f"train {epoch}", leave=False)
        for batch in progress:
            images, targets = batch[0], batch[1]
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(self.device.type, enabled=self._amp_enabled):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            self._scaler.scale(loss).backward()

            if self.config.gradient_clip_norm:
                # Unscale first, or the clip threshold applies to scaled grads.
                self._scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)

            self._scaler.step(self.optimizer)
            self._scaler.update()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            correct += int((logits.argmax(dim=1) == targets).sum())
            seen += batch_size
            progress.set_postfix(loss=running_loss / seen, acc=correct / seen)

        return running_loss / max(seen, 1), correct / max(seen, 1)

    @torch.inference_mode()
    def evaluate(self, loader: DataLoader, desc: str = "val") -> ClassificationMetrics:
        """Full metric sweep over a loader."""
        self.model.eval()
        running_loss = 0.0
        seen = 0
        all_targets: list[int] = []
        all_preds: list[int] = []
        all_probs: list[np.ndarray] = []

        for batch in tqdm(loader, desc=desc, leave=False):
            images, targets = batch[0], batch[1]
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.autocast(self.device.type, enabled=self._amp_enabled):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            probabilities = torch.softmax(logits.float(), dim=1)
            running_loss += loss.item() * targets.size(0)
            seen += targets.size(0)
            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(probabilities.argmax(dim=1).cpu().tolist())
            all_probs.append(probabilities.cpu().numpy())

        return compute_metrics(
            np.array(all_targets),
            np.array(all_preds),
            self.class_names,
            loss=running_loss / max(seen, 1),
            probabilities=np.concatenate(all_probs) if all_probs else None,
        )

    # ── Full run ─────────────────────────────────────────────────────────

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> list[EpochResult]:
        """Train for the configured number of epochs, saving the best model."""
        unfreeze_at = self.unfreeze_after_epoch

        for epoch in range(1, self.config.epochs + 1):
            started = time.perf_counter()

            if unfreeze_at and epoch == unfreeze_at + 1 and self.model.backbone_frozen:
                self.model.unfreeze_backbone()

            train_loss, train_accuracy = self.train_one_epoch(train_loader, epoch)
            metrics = self.evaluate(val_loader)

            result = EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=metrics.loss or float("nan"),
                val_accuracy=metrics.accuracy,
                val_macro_f1=metrics.macro_f1,
                learning_rate=self.optimizer.param_groups[0]["lr"],
                seconds=time.perf_counter() - started,
            )
            self.history.append(result)
            logger.info("%s", result)

            self._step_scheduler(result)

            monitored = self._monitored_value(result)
            if self._stopper is None or self._stopper.step(monitored):
                self.best_metric = monitored
                self.best_epoch = epoch
                self._save_best(epoch, metrics)

            if self._stopper is not None and self._stopper.should_stop:
                logger.info(
                    "Early stopping at epoch %d; best %s=%.4f at epoch %d",
                    epoch,
                    self.config.early_stopping_metric,
                    self.best_metric or float("nan"),
                    self.best_epoch,
                )
                break

        return self.history

    def _step_scheduler(self, result: EpochResult) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(result.val_loss)
        else:
            self.scheduler.step()

    def _monitored_value(self, result: EpochResult) -> float:
        return {
            "val_loss": result.val_loss,
            "val_accuracy": result.val_accuracy,
            "val_macro_f1": result.val_macro_f1,
        }[self.config.early_stopping_metric]

    def _save_best(self, epoch: int, metrics: ClassificationMetrics) -> None:
        if self.checkpoint_meta is None:
            logger.debug("No checkpoint metadata supplied; skipping save.")
            return
        self.checkpoint_meta.epoch = epoch
        self.checkpoint_meta.metrics = metrics.to_dict()
        save_checkpoint(self.checkpoint_path, self.model, self.checkpoint_meta)

    def history_frame(self) -> Any:
        """History as a DataFrame — handy for plotting learning curves."""
        import pandas as pd

        return pd.DataFrame([r.to_dict() for r in self.history])
