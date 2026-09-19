# Indian Medicinal Leaf Classification

Image classification for five Indian medicinal plant species — **Aloevera,
Amla, Mint, Neem, Tulsi** — from a folder-per-class photo collection to a
trained checkpoint and a prediction API.

The pipeline is built around the failure modes that make small image datasets
lie: duplicate images leaking across splits, backgrounds the model learns
instead of the leaf, and accuracy figures that hide an ignored class. See
[docs/architecture.md](docs/architecture.md) for the reasoning.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere

pip install -e ".[dev,notebook]"
# CPU-only torch, if you have no GPU:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

cp .env.example .env            # optional; every key has a default

leaf-train prepare              # index → validate → split → manifest
leaf-train fit                  # fine-tune against the manifest
leaf-train evaluate             # score the held-out test split
```

`prepare` refuses to continue when it finds something that would invalidate
the results — duplicates, a class below the minimum size, missing files — and
prints the reason.

## Dataset

Place the images under `Data/`, one directory per species:

```
Data/
├── Aloevera/   aloevera_001.jpg ...
├── Amla/
├── Mint/
├── Neem/
└── Tulsi/
```

The images are not tracked in git (see [.gitignore](.gitignore)); only the
manifest describing them is reproducible from code.

**Attribution.** The image dataset is third-party. Record its original source,
license and usage terms here before this project is shared or submitted.

<!-- TODO: dataset source URL, license, and citation -->

> **Disclaimer.** Species identifications produced by this model are for
> botanical and educational purposes only. They are not a substitute for
> expert botanical verification, and must not be used as medical, dosage, or
> treatment guidance.

## Repository layout

```
src/medicinal_leaf/
├── config/settings.py          typed config: YAML + .env + environment
├── data/
│   ├── ingestion.py            directory scan → index frame; LeafDataset
│   ├── validation.py           duplicates, geometry, balance, decodability
│   ├── splitting.py            stratified seeded splits + leakage checks
│   └── manifest.py             CSV + sidecar, label mapping, fingerprint
├── preprocessing/
│   ├── image.py                letterbox / resize / crop, normalisation
│   ├── segmentation.py         Excess-Green leaf/background separation
│   └── augmentation.py         train and eval transform pipelines
├── models/
│   ├── model.py                LeafClassifier: timm backbone + linear head
│   └── factory.py              model/optimiser/scheduler/loss, checkpoints
├── training/
│   ├── trainer.py              fit loop, AMP, early stopping, checkpointing
│   └── train.py                the `leaf-train` CLI
├── evaluation/
│   ├── metrics.py              macro-F1 and friends, confusion matrix
│   └── error_analysis.py       what it got wrong, and how confidently
└── inference/predictor.py      LeafPredictor.from_checkpoint(...)
```

Supporting directories: [configs/](configs/) (`development.yaml`,
`production.yaml`), [notebooks/](notebooks/) (EDA), [tests/](tests/),
[docs/](docs/).

## Configuration

Settings resolve in this order, first match winning:

1. keyword arguments to `load_settings(...)`
2. environment variables — nested keys use `__`
3. `.env` at the repository root
4. `configs/<MLC_ENV>.yaml`
5. defaults in [settings.py](src/medicinal_leaf/config/settings.py)

```bash
MLC_ENV=production leaf-train fit          # switch config file
MLC_TRAINING__EPOCHS=40 leaf-train fit     # override one key
leaf-train fit --backbone efficientnet_b0 --epochs 40
```

`development.yaml` is the fast loop: resnet18, 5 epochs, 224px, no workers.
`production.yaml` is the real run: efficientnet_b0, 60 epochs, 320px, leaf
segmentation, mixed precision, class weighting.

## Using a trained model

```python
from medicinal_leaf.inference.predictor import LeafPredictor

predictor = LeafPredictor.from_checkpoint("artifacts/checkpoints/best.pt")
prediction = predictor.predict("path/to/leaf.jpg")

print(prediction)  # Tulsi (94.1%)
print(prediction.top_k(3))  # [('Tulsi', 0.941), ('Mint', 0.041), ...]
print(prediction.is_confident(0.7))  # True
```

The predictor rebuilds its preprocessing from the checkpoint's own metadata,
so a model trained at 320px with segmentation keeps behaving that way no
matter what the config files say later.

## Development

```bash
pytest                      # full suite
pytest -m "not slow"        # skip the end-to-end run
pytest --cov                # with coverage

ruff check . && ruff format .
mypy
```

Tests build their own synthetic image trees in `tmp_path`, so the suite runs
on a fresh clone with no dataset and never touches the network — models in
tests are constructed with `pretrained=False`.

[CI](.github/workflows/ci.yml) runs lint, format, type-check and tests on
Python 3.11 and 3.12 against CPU-only PyTorch.

## Notes

- Dependency pins in [requirements.txt](requirements.txt) were captured from a
  resolved install on Python 3.11 (Windows) and verified by a full test run.
  `pyproject.toml` carries the looser ranges used to install the package.
- `notebooks/01_eda.ipynb` covers class balance, image dimensions, aspect
  ratios, colour modes and duplicate detection.
