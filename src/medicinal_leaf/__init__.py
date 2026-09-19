"""Indian medicinal leaf classification.

A folder-per-class image tree is turned into a validated manifest, split
without leakage, fed through a preprocessing/augmentation pipeline into a
fine-tuned CNN, and served behind a small prediction API.

The public surface is deliberately thin — reach for the submodules directly
for anything beyond the entry points re-exported here.
"""

from medicinal_leaf.config.settings import Settings, load_settings

__all__ = ["Settings", "__version__", "load_settings"]

__version__ = "0.1.0"
