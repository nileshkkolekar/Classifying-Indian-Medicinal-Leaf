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

pip install -e ".[dev,notebook,serve]"
# CPU-only torch, if you have no GPU:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

cp .env.example .env            # optional; every key has a default

leaf-train prepare              # index → validate → split → manifest
leaf-train fit                  # fine-tune against the manifest
leaf-train evaluate             # score the held-out test split

leaf-api                        # serve the API on http://127.0.0.1:8000
streamlit run src/medicinal_leaf/ui/streamlit_app.py    # then the UI
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
├── inference/predictor.py      LeafPredictor.from_checkpoint(...)
├── api/
│   ├── schemas.py              wire contract: verdicts, results, thresholds
│   ├── service.py              upload limits, classify/flag/decline policy
│   └── app.py                  FastAPI endpoints
└── ui/streamlit_app.py         Streamlit client for the API
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

## Web application

Two processes: a FastAPI service that owns the model, and a Streamlit UI that
talks to it over HTTP. Start the API first.

```bash
leaf-api                                                # :8000, docs at /docs
streamlit run src/medicinal_leaf/ui/streamlit_app.py    # :8501
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness, whether a model is loaded, active thresholds |
| `POST /predict` | One image → species + confidence |
| `POST /predict/batch` | A small ZIP → one row per image, answered immediately |
| `POST /jobs` | A large ZIP → `202` + job id, processed in the background |
| `GET /jobs/{id}` | State, progress and running counts |
| `GET /jobs/{id}/results` | Download the results CSV |
| `GET /jobs` · `DELETE /jobs/{id}` | List recent jobs; cancel or delete one |

Every prediction gets one of four verdicts, and the response always carries a
confidence score and the thresholds it was judged against:

- **classified** — above the review threshold.
- **needs_review** — plausible but under the bar; shown, flagged, and
  highlighted in the UI so it can be routed to a human.
- **unable_to_classify** — under the floor, so **no species is named at all**.
  A confident-looking wrong answer is worse than an honest refusal.
- **error** — the file could not be decoded. One bad file in a ZIP never
  fails the whole upload.

Both thresholds are configuration, not constants, and can be overridden per
request:

```bash
curl -F file=@leaf.jpg "http://127.0.0.1:8000/predict?review_threshold=0.9"
```

Bulk results are exportable as CSV. Archives are bounded on entry count,
per-file size, total uncompressed size, and compression ratio — an endpoint
that unpacks ZIPs is the obvious denial-of-service target.

### Bulk at scale

Thousands of images take minutes of CPU, which outlives any sane HTTP
timeout, so large archives go through a queue instead:

```bash
curl -F file=@leaves.zip http://127.0.0.1:8000/jobs      # → 202 + job_id
curl http://127.0.0.1:8000/jobs/<job_id>                 # poll
curl -o results.csv http://127.0.0.1:8000/jobs/<job_id>/results
```

The UI's bulk tab does the same thing with a live progress bar; pick
**Queued** rather than **Immediate**.

| | `POST /predict/batch` | `POST /jobs` |
| --- | --- | --- |
| Archive | 100 MB | 5 GB (10 GB prod) |
| Images | 200 | 20,000 (50,000 prod) |
| Answer | in the response | poll, then download CSV |

Images are decoded, predicted and released **a chunk at a time**, budgeted by
megapixels rather than file count — decoded RGB runs about 10× its compressed
size, so counting files bounds nothing. Peak memory tracks the chunk, not the
upload: a 5 GB archive costs the same as a 50 MB one.

The uploaded archive is streamed to disk rather than held in memory, and
deleted the moment the job ends (NFR-8). Results stream to CSV as they are
produced, so a crash leaves the completed rows intact.

Two limits worth knowing: **one worker processes jobs sequentially** (torch
already saturates the cores, so parallel jobs would only slow each other),
and **the queue is in-process** — jobs survive a restart but are not shared
between replicas. More than one API instance needs SQS or Redis behind the
same interface. The browser upload is also capped well below the API's own
limit, because Streamlit buffers uploads in memory; genuinely huge archives
should be POSTed to `/jobs` directly.

If no checkpoint exists yet, the API still starts and `/health` reports
`degraded`; prediction endpoints return `503` with instructions rather than
the process crash-looping.

## Docker

One image, two entry points — the command decides whether a container is the
API or the UI, so there is a single build and a single ECR repository.

```bash
docker compose up --build
# API  http://localhost:8000/docs
# UI   http://localhost:8501
```

Put a checkpoint at `./artifacts/checkpoints/best.pt` first, or the API
reports `degraded`. The image installs **CPU-only** torch: the default Linux
wheels bundle CUDA and add roughly 2 GB for hardware Fargate does not have.

## AWS

Optional — with no `aws.*` configuration everything runs from local disk.

```bash
leaf-train prepare --sync     # mirror the dataset from S3 first (FR-1)
```

Set `MLC_AWS__CHECKPOINT_URI` and the API downloads its model at startup
instead of baking it into the image.

Deployment targets ECS Fargate, with GitHub Actions authenticating through
**OIDC** — no AWS keys are stored in the repository. CD triggers on CI
succeeding on `main`, so a broken merge never reaches AWS.

Full setup, IAM policies and the required GitHub secrets/variables:
[docs/aws-deployment.md](docs/aws-deployment.md).

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
