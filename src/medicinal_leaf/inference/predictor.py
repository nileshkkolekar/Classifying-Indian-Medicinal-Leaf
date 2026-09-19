"""Serve predictions from a trained checkpoint.

The preprocessing is rebuilt from the checkpoint's own metadata rather than
from the current config file. That is the whole point of storing it: a model
trained at 320px with segmentation enabled keeps behaving that way even if
someone later edits ``production.yaml``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from medicinal_leaf.models.factory import CheckpointMeta, load_checkpoint
from medicinal_leaf.models.model import LeafClassifier
from medicinal_leaf.preprocessing.augmentation import build_eval_transform
from medicinal_leaf.preprocessing.image import load_image

logger = logging.getLogger(__name__)

ImageSource = str | Path | Image.Image


@dataclass(slots=True)
class Prediction:
    """A single verdict with its full probability distribution."""

    label: str
    confidence: float
    probabilities: dict[str, float]
    source: str | None = None

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        ranked = sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:k]

    def is_confident(self, threshold: float = 0.5) -> bool:
        """Whether the verdict clears a confidence bar.

        Worth checking before showing a species name to a user — an
        out-of-distribution photo still produces a top-1 label.
        """
        return self.confidence >= threshold

    def __str__(self) -> str:
        return f"{self.label} ({self.confidence:.1%})"


class LeafPredictor:
    """Load once, predict many times."""

    def __init__(
        self,
        model: LeafClassifier,
        meta: CheckpointMeta,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.meta = meta
        self.class_names = list(meta.class_names)
        self.transform = build_eval_transform(meta.preprocessing_config())

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = "auto",
    ) -> LeafPredictor:
        """Build a predictor from a saved checkpoint."""
        resolved = _resolve_device(device)
        model, meta = load_checkpoint(path, resolved)
        logger.info("Predictor ready: %s on %s", meta.backbone, resolved)
        return cls(model, meta, resolved)

    @torch.inference_mode()
    def predict(self, source: ImageSource) -> Prediction:
        """Classify one image."""
        batch = self._to_batch([source])
        probabilities = torch.softmax(self.model(batch), dim=1)[0].cpu()
        return self._to_prediction(probabilities, source)

    @torch.inference_mode()
    def predict_batch(
        self,
        sources: Sequence[ImageSource],
        batch_size: int = 32,
    ) -> list[Prediction]:
        """Classify many images, chunked to bound memory use."""
        if not sources:
            return []

        predictions: list[Prediction] = []
        for start in range(0, len(sources), batch_size):
            chunk = list(sources[start : start + batch_size])
            batch = self._to_batch(chunk)
            probabilities = torch.softmax(self.model(batch), dim=1).cpu()
            predictions.extend(
                self._to_prediction(row, source)
                for row, source in zip(probabilities, chunk, strict=True)
            )
        return predictions

    @torch.inference_mode()
    def embed(self, sources: Sequence[ImageSource]) -> torch.Tensor:
        """Pooled backbone features — for similarity search or clustering."""
        return self.model.forward_features(self._to_batch(list(sources))).cpu()

    def __call__(self, source: ImageSource) -> Prediction:
        return self.predict(source)

    # ── Internals ────────────────────────────────────────────────────────

    def _to_batch(self, sources: Iterable[ImageSource]) -> torch.Tensor:
        tensors = []
        for source in sources:
            image = source if isinstance(source, Image.Image) else load_image(source)
            tensors.append(self.transform(image))
        return torch.stack(tensors).to(self.device)

    def _to_prediction(self, probabilities: torch.Tensor, source: ImageSource) -> Prediction:
        scores = {name: float(p) for name, p in zip(self.class_names, probabilities, strict=True)}
        best = max(scores, key=scores.__getitem__)
        return Prediction(
            label=best,
            confidence=scores[best],
            probabilities=scores,
            source=None if isinstance(source, Image.Image) else str(source),
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(backbone={self.meta.backbone}, "
            f"classes={len(self.class_names)}, device={self.device})"
        )


def _resolve_device(device: str | torch.device) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
