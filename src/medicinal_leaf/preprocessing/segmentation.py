"""Isolate the leaf from its background.

Field photographs of medicinal plants carry a lot of context — soil, hands,
tiles, other foliage — and a classifier trained on the raw frame happily
learns the background instead of the leaf. Masking to the leaf and cropping
to its bounding box removes that shortcut.

The approach is classical rather than learned: an Excess Green vegetation
index, Otsu thresholding, morphological cleanup, then the largest connected
component. That is cheap, has no training requirement, and degrades
gracefully — when the mask looks implausible the original image is returned
unchanged.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

#: A mask covering less/more than this fraction is treated as a failure.
MIN_PLAUSIBLE_COVERAGE = 0.02
MAX_PLAUSIBLE_COVERAGE = 0.98


def excess_green(rgb: np.ndarray) -> np.ndarray:
    """Excess Green index ``2G - R - B`` over chromatic-normalised channels.

    Normalising by the per-pixel sum makes the index largely invariant to
    illumination, so a leaf in shade still separates from bright soil.
    """
    array = rgb.astype(np.float32)
    total = array.sum(axis=2, keepdims=True)
    # Avoid dividing by zero on pure-black pixels.
    normalised = array / np.clip(total, 1e-6, None)
    red, green, blue = normalised[..., 0], normalised[..., 1], normalised[..., 2]
    return 2.0 * green - red - blue


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected blob — the leaf, not the clutter."""
    # `connectivity` must be passed by keyword: OpenCV's signature puts the
    # optional output arrays (labels, stats, centroids) first, so a bare
    # positional 8 binds to `labels`, not to the connectivity we want.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:  # background only
        return mask
    # Row 0 is the background component; pick the largest of the rest.
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def leaf_mask(
    rgb: np.ndarray,
    *,
    close_kernel: int = 7,
    open_kernel: int = 5,
    keep_largest: bool = True,
) -> np.ndarray:
    """Return a ``uint8`` mask (1 = leaf) for an ``HWC`` RGB array."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB array, got shape {rgb.shape}")

    index = excess_green(rgb)
    # Otsu needs 8-bit input, so rescale the index to 0..255 first.
    spread = float(index.max() - index.min())
    if spread < 1e-6:
        return np.ones(rgb.shape[:2], dtype=np.uint8)
    scaled = ((index - index.min()) / spread * 255).astype(np.uint8)

    _, binary = cv2.threshold(scaled, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if keep_largest:
        binary = _largest_component(binary)

    return binary.astype(np.uint8)


def mask_coverage(mask: np.ndarray) -> float:
    """Fraction of the frame the mask occupies."""
    return float(mask.sum()) / float(mask.size) if mask.size else 0.0


def bounding_box(mask: np.ndarray, pad: int = 0) -> tuple[int, int, int, int]:
    """Tight ``(left, top, right, bottom)`` box around the mask, padded."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return 0, 0, mask.shape[1], mask.shape[0]

    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    return (
        max(0, int(left) - pad),
        max(0, int(top) - pad),
        min(mask.shape[1], int(right) + 1 + pad),
        min(mask.shape[0], int(bottom) + 1 + pad),
    )


def apply_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    background: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Replace everything outside the mask with a flat colour."""
    out = np.empty_like(rgb)
    out[:] = np.asarray(background, dtype=rgb.dtype)
    selection = mask.astype(bool)
    out[selection] = rgb[selection]
    return out


def segment_leaf(
    image: Image.Image,
    *,
    pad: int = 8,
    background: tuple[int, int, int] = (0, 0, 0),
    crop: bool = True,
) -> Image.Image:
    """Mask out the background and crop to the leaf.

    Returns the input untouched when the mask is implausible (almost empty or
    almost the whole frame), which is the right call for a close-up shot that
    is already nothing but leaf.
    """
    rgb = np.asarray(image.convert("RGB"))
    mask = leaf_mask(rgb)

    coverage = mask_coverage(mask)
    if not MIN_PLAUSIBLE_COVERAGE <= coverage <= MAX_PLAUSIBLE_COVERAGE:
        logger.debug("Segmentation skipped: implausible coverage %.3f", coverage)
        return image

    masked = apply_mask(rgb, mask, background)
    if crop:
        left, top, right, bottom = bounding_box(mask, pad)
        masked = masked[top:bottom, left:right]
        if masked.size == 0:  # pragma: no cover - defensive
            return image

    return Image.fromarray(masked)
