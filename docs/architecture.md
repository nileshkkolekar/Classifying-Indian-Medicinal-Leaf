# Architecture

## The shape of the problem

Five medicinal species, a few hundred field photographs, and a hard
requirement that the reported accuracy be real. At this scale the model is
not the risk — a pretrained backbone will fit almost anything. The risks are
that duplicate images straddle the train/test boundary, that the classifier
learns the background instead of the leaf, and that a single accuracy number
hides a species the model never predicts.

The pipeline is arranged around those three risks.

## Flow

```mermaid
flowchart TD
    A[Data/ — one folder per species] --> B[data.ingestion<br/>header scan → index frame]
    B --> C[data.validation<br/>duplicates, geometry, balance]
    C -->|errors| X[stop]
    C -->|clean| D[data.splitting<br/>stratified, seeded]
    D --> E[data.manifest<br/>CSV + sidecar + fingerprint]

    E --> F[preprocessing<br/>segment → resize → normalise]
    F --> G[models.factory<br/>backbone + head + optimiser]
    G --> H[training.trainer<br/>fit, early stop, checkpoint]

    H --> I[evaluation.metrics<br/>macro-F1, confusion]
    H --> J[inference.predictor]
    I --> K[evaluation.error_analysis<br/>confusions, confident errors]

    J --> L[api.service<br/>upload limits, verdict policy]
    L --> M[api.app<br/>FastAPI: /predict, /predict/batch]
    M -->|HTTP| N[ui.streamlit_app<br/>upload, results table, CSV]
```

## Module responsibilities

| Module | Owns | Deliberately does not |
| --- | --- | --- |
| `config.settings` | Typed config from YAML + env + kwargs | Know anything about torch |
| `data.ingestion` | Directory scan, index frame, `LeafDataset` | Judge whether the data is good |
| `data.validation` | Duplicates, geometry, balance, decodability | Modify or drop rows |
| `data.splitting` | Stratified, seeded splits; leakage checks | Persist anything |
| `data.manifest` | CSV + sidecar IO, label mapping, fingerprint | Decide the split |
| `preprocessing.image` | Deterministic geometry and colour ops | Randomness |
| `preprocessing.segmentation` | Leaf/background separation | Learned models |
| `preprocessing.augmentation` | Train/eval transform pipelines | Image IO |
| `models.model` | `LeafClassifier`: backbone + head | Config parsing |
| `models.factory` | Build model/optimiser/scheduler/loss, checkpoint IO | The loop itself |
| `training.trainer` | Epochs, AMP, early stopping, best-checkpoint | Config branching |
| `training.train` | CLI wiring | Business logic |
| `evaluation.metrics` | Scalar and per-class metrics, confusion plot | Reading data |
| `evaluation.error_analysis` | What went wrong and how confidently | Computing headline metrics |
| `inference.predictor` | Load a checkpoint, classify images | Reading `configs/` |
| `data.s3` | Dataset and checkpoint transfer, key safety | Holding credentials |
| `api.schemas` | The wire contract: verdicts, results, thresholds | Any logic |
| `api.service` | Upload limits, ZIP safety, verdict policy | Knowing about HTTP |
| `api.app` | Routing, multipart parsing, status codes | Deciding verdicts |
| `ui.streamlit_app` | Upload, display, CSV export | Loading a model |

## Key decisions

### The manifest is the contract

Every stage after `prepare` reads `artifacts/manifest.csv`, never the image
tree. The sidecar records the label mapping and a fingerprint over
`(path, class, split)`; checkpoints store that fingerprint, so `evaluate`
warns when a model is being scored against a split it may have trained on.
Without this, "I re-ran prepare and the numbers changed" is undiagnosable.

### Preprocessing travels with the checkpoint

`CheckpointMeta` stores image size, resize strategy, normalisation constants
and whether segmentation was on. `LeafPredictor` rebuilds its transform from
that metadata rather than from `configs/`, so editing a YAML file cannot
silently change how a deployed model sees an image. Train/serve skew of this
kind is quiet and expensive.

### Letterbox, not squash

Leaf shape is discriminative — a Neem leaflet is long and narrow, an Amla
leaf small and round. Squashing every image into a square destroys exactly
the cue the model should use, so the default resize pads instead.

### Classical segmentation, not learned

Background removal uses Excess Green (`2G − R − B` over chromatic-normalised
channels), Otsu thresholding, morphological cleanup and the largest connected
component. It needs no training data and no extra model at inference. When
the resulting mask is implausible — under 2% or over 98% coverage, which is
what a close-up of nothing but leaf produces — the original image is returned
unchanged. A learned segmenter would be more accurate and far more machinery
than this dataset justifies.

### Macro-F1 is the headline metric

With uneven class counts, accuracy is dominated by the largest species and a
model that ignores the smallest one still scores well. Early stopping
monitors `val_macro_f1` by default for the same reason.

### Frozen backbones stay in eval mode

`LeafClassifier.train()` keeps a frozen backbone in eval mode so its
BatchNorm running statistics do not drift while only the head is learning —
otherwise "frozen" is not actually frozen.

### The service may decline to answer

A softmax always produces a winner. On a photo of a hand, a screenshot, or a
species the model has never seen, that winner is noise wearing the costume of
an answer — and a UI that prints "Neem — 31%" invites someone to believe it.

So there are three bands rather than two. Above `review_threshold` the result
stands; between the two thresholds it is returned but flagged, highlighted in
the UI and counted separately so it can be routed to a reviewer; below
`unknown_threshold` the label is withheld entirely and the response says
`unable_to_classify`. The confidence and the full distribution come back in
every case, so a caller is free to apply its own policy — what the service
refuses to do is *assert* a species it cannot support.

Both thresholds live in `ServingConfig` and can be overridden per request.
They are configuration precisely because the right bar depends on what the
answer is used for, and that is not a decision the code should freeze.

### Upload handling is hostile-input territory

The prediction endpoints are the only place the system accepts bytes from
outside, and the ZIP endpoint decompresses them. The limits in
`api.service.ZipLimits` — entry count, per-file size, total uncompressed
size, and compression ratio — are checked against the central directory
*before* anything is decompressed, and the read itself is capped in case the
header lied. A ZIP that expands to gigabytes is a few lines to construct.

Nothing is written to disk at any point (NFR-8): uploads are decoded from
memory and dropped when the request ends. Archive member names are sanitised
for display only — traversal is structurally impossible because no path from
the archive is ever used to open a file.

### S3 is optional, and credentials are never configuration

`AWSConfig` carries locations — bucket URIs and a region — and nothing else.
Credentials are left entirely to boto3's own resolution chain, which means an
ECS task role in production and an environment or profile locally. There is
no field that could hold a secret, so none can be committed (NFR-4).

Object keys get the same treatment as archive members: a key like
`../../.ssh/authorized_keys` would otherwise escape the download directory,
so `safe_destination` resolves and verifies containment before any write.

Everything is optional. With no `aws.*` set, the pipeline reads local disk
and the S3 module is never imported — boto3 lives in its own extra.

## Data contracts

**Index frame** (`data.ingestion.build_index`):

| Column | Type | Notes |
| --- | --- | --- |
| `file_path` | str | Absolute |
| `class_name` | str | Source directory name |
| `width`, `height` | int | From the header, pixels never decoded |
| `aspect_ratio` | float | `width / height` |
| `mode` | str | PIL mode before RGB conversion |
| `size_bytes` | int | On-disk size |

**Manifest** adds `split` (`train`/`val`/`test`) and `label_idx` (the
alphabetical class index).

**Prediction frame** (`evaluation.error_analysis.collect_predictions`):
`file_path`, `true_idx`, `predicted_idx`, `confidence`, `true_class_prob`,
`true_label`, `predicted_label`, `correct`.

## Configuration layering

First match wins:

1. keyword arguments to `load_settings(...)`
2. environment variables — `MLC_TRAINING__EPOCHS=40`
3. `.env` at the repository root
4. `configs/<MLC_ENV>.yaml`
5. defaults in `config/settings.py`

Relative paths in config resolve against the repository root, so the same
YAML works regardless of the working directory.

## Extension points

- **A different backbone** — set `model.backbone` to any `timm` model name.
  Nothing else changes; the head is sized from `backbone.num_features`.
- **Cross-validation** — `stratified_split` returns a frame rather than
  writing one, so a fold loop can wrap it and call `write_manifest` per fold.
- **A learned segmenter** — replace `segmentation.leaf_mask`; the
  plausibility fallback in `segment_leaf` already handles its failures.
- **Test-time augmentation** — average `LeafPredictor.predict_batch` over
  flipped copies; the transform is deterministic, so flips must be explicit.
- **Embeddings** — `LeafPredictor.embed` exposes pooled backbone features for
  similarity search or clustering.

## Known limitations

- Duplicate detection is byte-level. Re-encoded or resized near-duplicates
  slip through; catching those needs perceptual hashing or embedding
  distances.
- The split is random within each class. If the same physical plant was
  photographed several times, those frames can land in different splits and
  inflate test scores. Grouping by specimen would need metadata the dataset
  does not currently carry.
- `warmup_epochs` steps per epoch, not per batch, which is coarse for short
  schedules.
