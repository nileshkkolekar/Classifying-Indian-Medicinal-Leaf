"""Shared fixtures.

Tests build their own synthetic image trees rather than reading ``Data/``, so
the suite runs on a fresh clone and in CI where the dataset is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

CLASS_SPECS: dict[str, int] = {
    "Aloevera": 12,
    "Amla": 10,
    "Mint": 14,
    "Neem": 11,
    "Tulsi": 13,
}


def write_noise_image(path: Path, size: tuple[int, int], seed: int) -> Path:
    """Write a unique random JPEG so duplicate detection stays quiet."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(array).save(path, quality=95)
    return path


@pytest.fixture
def image_root(tmp_path: Path) -> Path:
    """A folder-per-class tree of small, unique JPEGs."""
    root = tmp_path / "Data"
    seed = 0
    for class_name, count in CLASS_SPECS.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for i in range(count):
            # Vary dimensions so aspect-ratio logic is actually exercised.
            size = (64 + (i % 3) * 16, 48 + (i % 4) * 8)
            write_noise_image(class_dir / f"{class_name.lower()}_{i:03d}.jpg", size, seed)
            seed += 1
    return root


@pytest.fixture
def index_frame(image_root: Path):
    """The index built from :func:`image_root`."""
    from medicinal_leaf.data.ingestion import build_index

    return build_index(image_root)


@pytest.fixture
def split_frame_fixture(index_frame):
    """The index with a ``split`` column assigned."""
    from medicinal_leaf.data.splitting import stratified_split

    return stratified_split(index_frame, seed=7)


@pytest.fixture
def leaf_photo() -> Image.Image:
    """A green blob on brown ground — a leaf as far as the segmenter cares."""
    canvas = np.zeros((120, 160, 3), dtype=np.uint8)
    canvas[:, :] = (139, 90, 43)  # soil brown

    yy, xx = np.mgrid[0:120, 0:160]
    ellipse = ((xx - 80) / 45.0) ** 2 + ((yy - 60) / 28.0) ** 2 <= 1.0
    canvas[ellipse] = (34, 160, 40)  # leaf green

    return Image.fromarray(canvas)


@pytest.fixture
def preprocessing_config():
    from medicinal_leaf.config.settings import PreprocessingConfig

    return PreprocessingConfig(image_size=32, segment_leaf=False)


@pytest.fixture
def augmentation_config():
    from medicinal_leaf.config.settings import AugmentationConfig

    return AugmentationConfig(enabled=True)
