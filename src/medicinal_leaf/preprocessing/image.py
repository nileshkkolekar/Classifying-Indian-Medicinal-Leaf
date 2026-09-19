"""Geometry and colour operations shared by training and inference.

Everything here is deterministic. If train and inference ever disagree about
how an image reaches the network, accuracy drops in a way that is very hard
to diagnose — so both paths call :func:`prepare_image`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageOps

if TYPE_CHECKING:
    import torch

    from medicinal_leaf.config.settings import PreprocessingConfig


def load_image(path: str | Path) -> Image.Image:
    """Open an image as RGB with EXIF rotation already applied.

    Phone cameras store orientation in EXIF rather than in the pixels; without
    ``exif_transpose`` a portrait photo arrives sideways.
    """
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def letterbox(
    image: Image.Image,
    size: int,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Resize to fit a ``size x size`` square, padding to preserve aspect ratio.

    Preferred over a plain resize for leaves: shape is a discriminative
    feature, and squashing a long Neem leaflet into a square distorts exactly
    the cue the model should be using.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")

    width, height = image.size
    scale = size / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(new_size, Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (size, size), fill)
    offset = ((size - new_size[0]) // 2, (size - new_size[1]) // 2)
    canvas.paste(resized, offset)
    return canvas


def resize_square(image: Image.Image, size: int) -> Image.Image:
    """Resize to ``size x size``, ignoring aspect ratio."""
    return image.resize((size, size), Image.Resampling.BILINEAR)


def center_crop_square(image: Image.Image, size: int) -> Image.Image:
    """Scale the shortest side to ``size``, then crop the centre square."""
    width, height = image.size
    scale = size / min(width, height)
    resized = image.resize(
        (max(size, round(width * scale)), max(size, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    new_width, new_height = resized.size
    left = (new_width - size) // 2
    top = (new_height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def apply_resize_strategy(
    image: Image.Image,
    size: int,
    strategy: str = "letterbox",
    pad_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Dispatch to the configured resize strategy."""
    if strategy == "letterbox":
        return letterbox(image, size, pad_color)
    if strategy == "resize":
        return resize_square(image, size)
    if strategy == "center_crop":
        return center_crop_square(image, size)
    raise ValueError(
        f"Unknown resize strategy {strategy!r}; expected letterbox, resize or center_crop."
    )


def prepare_image(
    source: str | Path | Image.Image,
    config: PreprocessingConfig,
) -> Image.Image:
    """Take a path or PIL image to the exact RGB square the network expects.

    Optionally removes the background first — see
    :func:`medicinal_leaf.preprocessing.segmentation.segment_leaf`.
    """
    image = source if isinstance(source, Image.Image) else load_image(source)
    if image.mode != "RGB":
        image = image.convert("RGB")

    if config.segment_leaf:
        # Imported here so that plain resizing does not require OpenCV.
        from medicinal_leaf.preprocessing.segmentation import segment_leaf

        image = segment_leaf(image, pad=config.segment_pad, background=config.pad_color)

    return apply_resize_strategy(
        image,
        config.image_size,
        config.resize_strategy,
        config.pad_color,
    )


def denormalize(
    tensor: torch.Tensor,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Invert normalisation and return an ``HWC`` uint8 array for plotting."""
    import torch

    array = tensor.detach().cpu()
    if array.ndim != 3:
        raise ValueError(f"Expected a CHW tensor, got shape {tuple(array.shape)}")

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    array = (array * std_t + mean_t).clamp(0, 1)
    return (array.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
