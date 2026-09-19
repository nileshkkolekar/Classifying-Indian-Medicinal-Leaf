"""Resizing, segmentation and transform pipelines."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from medicinal_leaf.config.settings import AugmentationConfig, PreprocessingConfig
from medicinal_leaf.preprocessing.augmentation import (
    build_eval_transform,
    build_train_transform,
    build_transforms,
)
from medicinal_leaf.preprocessing.image import (
    apply_resize_strategy,
    center_crop_square,
    denormalize,
    letterbox,
    load_image,
    prepare_image,
    resize_square,
)
from medicinal_leaf.preprocessing.segmentation import (
    bounding_box,
    excess_green,
    leaf_mask,
    mask_coverage,
    segment_leaf,
)

# ── Geometry ─────────────────────────────────────────────────────────────


def test_letterbox_is_square():
    image = Image.new("RGB", (200, 100), (255, 0, 0))
    assert letterbox(image, 64).size == (64, 64)


def test_letterbox_preserves_aspect_ratio():
    image = Image.new("RGB", (200, 100), (255, 0, 0))
    array = np.asarray(letterbox(image, 64, fill=(0, 0, 0)))

    # A 2:1 image in a square canvas occupies the middle half, padded above
    # and below.
    non_black_rows = np.where(array.any(axis=(1, 2)))[0]
    assert len(non_black_rows) == pytest.approx(32, abs=2)


def test_letterbox_pads_with_the_requested_colour():
    image = Image.new("RGB", (100, 50), (255, 255, 255))
    array = np.asarray(letterbox(image, 40, fill=(0, 0, 255)))
    assert tuple(array[0, 0]) == (0, 0, 255)


def test_letterbox_rejects_nonpositive_size():
    with pytest.raises(ValueError, match="must be positive"):
        letterbox(Image.new("RGB", (10, 10)), 0)


def test_resize_square_ignores_aspect_ratio():
    assert resize_square(Image.new("RGB", (200, 50)), 32).size == (32, 32)


def test_center_crop_square():
    assert center_crop_square(Image.new("RGB", (200, 100)), 64).size == (64, 64)


@pytest.mark.parametrize("strategy", ["letterbox", "resize", "center_crop"])
def test_every_strategy_returns_the_requested_size(strategy):
    image = Image.new("RGB", (123, 77), (10, 200, 10))
    assert apply_resize_strategy(image, 48, strategy).size == (48, 48)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="Unknown resize strategy"):
        apply_resize_strategy(Image.new("RGB", (10, 10)), 8, "magic")


def test_load_image_always_returns_rgb(tmp_path):
    path = tmp_path / "grey.png"
    Image.new("L", (20, 20), 128).save(path)
    assert load_image(path).mode == "RGB"


# ── Normalisation ────────────────────────────────────────────────────────


def test_denormalize_inverts_normalisation():
    mean = (0.5, 0.5, 0.5)
    std = (0.25, 0.25, 0.25)
    original = torch.rand(3, 16, 16)

    normalized = (original - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
    recovered = denormalize(normalized, mean, std)

    expected = (original.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    assert np.abs(recovered.astype(int) - expected.astype(int)).max() <= 1


def test_denormalize_rejects_batched_input():
    with pytest.raises(ValueError, match="CHW tensor"):
        denormalize(torch.rand(2, 3, 16, 16))


# ── Segmentation ─────────────────────────────────────────────────────────


def test_excess_green_is_high_on_green():
    green = np.full((4, 4, 3), (0, 255, 0), dtype=np.uint8)
    brown = np.full((4, 4, 3), (139, 90, 43), dtype=np.uint8)
    assert excess_green(green).mean() > excess_green(brown).mean()


def test_leaf_mask_finds_the_blob(leaf_photo):
    mask = leaf_mask(np.asarray(leaf_photo))
    coverage = mask_coverage(mask)
    # The ellipse covers roughly 20% of the frame.
    assert 0.10 < coverage < 0.35


def test_leaf_mask_centre_is_inside_the_leaf(leaf_photo):
    mask = leaf_mask(np.asarray(leaf_photo))
    assert mask[60, 80] == 1
    assert mask[2, 2] == 0


def test_leaf_mask_rejects_non_rgb():
    with pytest.raises(ValueError, match="HxWx3"):
        leaf_mask(np.zeros((10, 10), dtype=np.uint8))


def test_segment_leaf_crops_to_the_bounding_box(leaf_photo):
    segmented = segment_leaf(leaf_photo, pad=0)
    assert segmented.size[0] < leaf_photo.size[0]
    assert segmented.size[1] < leaf_photo.size[1]


def test_segment_leaf_returns_input_when_mask_is_implausible():
    flat = Image.new("RGB", (40, 40), (34, 160, 40))  # nothing but leaf
    assert segment_leaf(flat).size == flat.size


def test_bounding_box_falls_back_to_full_frame_for_empty_mask():
    empty = np.zeros((20, 30), dtype=np.uint8)
    assert bounding_box(empty) == (0, 0, 30, 20)


# ── Pipelines ────────────────────────────────────────────────────────────


def test_prepare_image_returns_configured_size(leaf_photo):
    config = PreprocessingConfig(image_size=48, segment_leaf=False)
    assert prepare_image(leaf_photo, config).size == (48, 48)


def test_prepare_image_with_segmentation(leaf_photo):
    config = PreprocessingConfig(image_size=48, segment_leaf=True, segment_pad=4)
    assert prepare_image(leaf_photo, config).size == (48, 48)


def test_eval_transform_produces_a_normalised_tensor(leaf_photo):
    config = PreprocessingConfig(image_size=32)
    tensor = build_eval_transform(config)(leaf_photo)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    # Normalised data straddles zero rather than sitting in [0, 1].
    assert tensor.min() < 0


def test_eval_transform_is_deterministic(leaf_photo):
    transform = build_eval_transform(PreprocessingConfig(image_size=32))
    assert torch.allclose(transform(leaf_photo), transform(leaf_photo))


def test_train_transform_is_stochastic(leaf_photo):
    transform = build_train_transform(
        PreprocessingConfig(image_size=32), AugmentationConfig(enabled=True)
    )
    torch.manual_seed(0)
    first = transform(leaf_photo)
    second = transform(leaf_photo)

    assert first.shape == second.shape == (3, 32, 32)
    assert not torch.allclose(first, second)


def test_disabling_augmentation_makes_training_deterministic(leaf_photo):
    transform = build_train_transform(
        PreprocessingConfig(image_size=32), AugmentationConfig(enabled=False)
    )
    assert torch.allclose(transform(leaf_photo), transform(leaf_photo))


def test_build_transforms_shares_one_eval_pipeline():
    config = PreprocessingConfig(image_size=32)
    pipelines = build_transforms(config, AugmentationConfig())
    assert set(pipelines) == {"train", "val", "test"}
    assert pipelines["val"] is pipelines["test"]


def test_transforms_survive_pickling(leaf_photo):
    """DataLoader workers on Windows pickle the transform — lambdas would break."""
    import pickle

    transform = build_train_transform(PreprocessingConfig(image_size=32), AugmentationConfig())
    restored = pickle.loads(pickle.dumps(transform))
    assert restored(leaf_photo).shape == (3, 32, 32)
