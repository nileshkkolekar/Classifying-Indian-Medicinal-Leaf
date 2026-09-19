"""Transform pipelines for the training and evaluation splits.

Augmentation choices are dataset-specific: leaves are orientation-agnostic,
so vertical flips and large rotations are safe, while hue jitter is kept
small because colour genuinely distinguishes species (Tulsi's purple-tinged
leaves against Mint's bright green).

Every component is a module-level class, never a lambda, so the pipelines
survive pickling — required by DataLoader workers on Windows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from PIL import Image
from torchvision.transforms import v2

from medicinal_leaf.preprocessing.image import prepare_image

if TYPE_CHECKING:
    from medicinal_leaf.config.settings import AugmentationConfig, PreprocessingConfig


class LeafPrepare:
    """Deterministic front half of every pipeline.

    Wraps :func:`prepare_image` — optional background removal followed by the
    configured resize — so that training and inference share one code path.
    """

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config

    def __call__(self, image: Image.Image) -> Image.Image:
        return prepare_image(image, self.config)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(size={self.config.image_size}, "
            f"strategy={self.config.resize_strategy}, segment={self.config.segment_leaf})"
        )


def _to_normalized_tensor(config: PreprocessingConfig) -> list[v2.Transform]:
    """Shared tail: PIL -> float tensor in [0, 1] -> normalised."""
    return [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(config.normalize_mean), std=list(config.normalize_std)),
    ]


def build_train_transform(
    config: PreprocessingConfig,
    augmentation: AugmentationConfig,
) -> v2.Compose:
    """Stochastic pipeline for the training split."""
    steps: list[v2.Transform] = [LeafPrepare(config)]

    if augmentation.enabled:
        steps.append(
            v2.RandomResizedCrop(
                size=config.image_size,
                scale=tuple(augmentation.random_resized_crop_scale),
                antialias=True,
            )
        )
        if augmentation.horizontal_flip > 0:
            steps.append(v2.RandomHorizontalFlip(p=augmentation.horizontal_flip))
        if augmentation.vertical_flip > 0:
            steps.append(v2.RandomVerticalFlip(p=augmentation.vertical_flip))
        if augmentation.rotation_degrees > 0:
            steps.append(
                v2.RandomRotation(
                    degrees=augmentation.rotation_degrees,
                    fill=list(config.pad_color),
                )
            )
        if any(
            (
                augmentation.color_jitter_brightness,
                augmentation.color_jitter_contrast,
                augmentation.color_jitter_saturation,
                augmentation.color_jitter_hue,
            )
        ):
            steps.append(
                v2.ColorJitter(
                    brightness=augmentation.color_jitter_brightness,
                    contrast=augmentation.color_jitter_contrast,
                    saturation=augmentation.color_jitter_saturation,
                    hue=augmentation.color_jitter_hue,
                )
            )

    steps.extend(_to_normalized_tensor(config))

    # Erasing works on tensors, so it has to follow the conversion above.
    if augmentation.enabled and augmentation.random_erasing > 0:
        steps.append(v2.RandomErasing(p=augmentation.random_erasing))

    return v2.Compose(steps)


def build_eval_transform(config: PreprocessingConfig) -> v2.Compose:
    """Deterministic pipeline for validation, test and inference."""
    return v2.Compose([LeafPrepare(config), *_to_normalized_tensor(config)])


def build_transforms(
    config: PreprocessingConfig,
    augmentation: AugmentationConfig,
) -> dict[str, v2.Compose]:
    """Both pipelines keyed by split name."""
    train = build_train_transform(config, augmentation)
    evaluate = build_eval_transform(config)
    return {"train": train, "val": evaluate, "test": evaluate}
