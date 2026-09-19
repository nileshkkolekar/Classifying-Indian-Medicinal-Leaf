"""End-to-end: index -> validate -> split -> manifest -> train -> predict.

Everything runs on a synthetic dataset at 32px with an untrained resnet18, so
the whole file finishes in seconds and never hits the network. The assertions
are about the pipeline holding together, not about accuracy — random noise is
not learnable.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from medicinal_leaf.config.settings import (
    AugmentationConfig,
    ModelConfig,
    PreprocessingConfig,
    TrainingConfig,
)
from medicinal_leaf.data.ingestion import LeafDataset, build_index
from medicinal_leaf.data.manifest import read_manifest, split_frame, write_manifest
from medicinal_leaf.data.splitting import assert_no_leakage, stratified_split
from medicinal_leaf.data.validation import raise_for_errors, validate_index
from medicinal_leaf.evaluation.error_analysis import collect_predictions, summarize_errors
from medicinal_leaf.inference.predictor import LeafPredictor
from medicinal_leaf.models.factory import (
    CheckpointMeta,
    build_criterion,
    build_model,
    build_optimizer,
)
from medicinal_leaf.preprocessing.augmentation import build_eval_transform, build_train_transform
from medicinal_leaf.training.trainer import Trainer, set_seed

pytestmark = [pytest.mark.integration, pytest.mark.slow]

IMAGE_SIZE = 32


@pytest.fixture
def prepared(image_root, tmp_path):
    """Run the data half of the pipeline and return the manifest location."""
    frame = build_index(image_root)

    report = validate_index(frame, min_images_per_class=5)
    raise_for_errors(report)

    split = stratified_split(frame, seed=13)
    assert_no_leakage(split)

    manifest_path = tmp_path / "artifacts" / "manifest.csv"
    meta = write_manifest(split, manifest_path)
    return manifest_path, meta


def test_data_pipeline_produces_a_usable_manifest(prepared):
    manifest_path, meta = prepared
    frame, loaded = read_manifest(manifest_path)

    assert loaded.num_classes == 5
    assert set(frame["split"]) == {"train", "val", "test"}
    assert loaded.fingerprint == meta.fingerprint


def test_dataset_yields_batches_the_model_accepts(prepared):
    manifest_path, meta = prepared
    frame, _ = read_manifest(manifest_path)

    transform = build_eval_transform(PreprocessingConfig(image_size=IMAGE_SIZE))
    dataset = LeafDataset(split_frame(frame, "train"), meta.class_to_idx, transform)
    images, targets = next(iter(DataLoader(dataset, batch_size=4)))

    assert images.shape == (4, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert targets.dtype == torch.int64
    assert targets.max() < meta.num_classes


def test_class_weights_are_normalised(prepared):
    manifest_path, meta = prepared
    frame, _ = read_manifest(manifest_path)

    dataset = LeafDataset(split_frame(frame, "train"), meta.class_to_idx)
    weights = dataset.class_weights()

    assert weights.shape == (5,)
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-5)


def test_dataset_rejects_unknown_labels(prepared):
    manifest_path, meta = prepared
    frame, _ = read_manifest(manifest_path)

    partial = {k: v for k, v in meta.class_to_idx.items() if k != "Mint"}
    with pytest.raises(KeyError, match="Mint"):
        LeafDataset(frame, partial)


def test_training_runs_and_saves_a_checkpoint(prepared, tmp_path):
    manifest_path, manifest_meta = prepared
    frame, _ = read_manifest(manifest_path)
    set_seed(0)

    preprocessing = PreprocessingConfig(image_size=IMAGE_SIZE)
    train_transform = build_train_transform(preprocessing, AugmentationConfig(enabled=True))
    eval_transform = build_eval_transform(preprocessing)

    train_set = LeafDataset(
        split_frame(frame, "train"), manifest_meta.class_to_idx, train_transform
    )
    val_set = LeafDataset(split_frame(frame, "val"), manifest_meta.class_to_idx, eval_transform)

    training = TrainingConfig(
        epochs=2,
        batch_size=8,
        num_workers=0,
        device="cpu",
        early_stopping_patience=None,
        checkpoint_dir=tmp_path / "checkpoints",
        run_dir=tmp_path / "runs",
    )

    model = build_model(ModelConfig(backbone="resnet18", pretrained=False), num_classes=5)
    trainer = Trainer(
        model,
        build_optimizer(model, training),
        build_criterion(training),
        training,
        class_names=manifest_meta.class_names,
        checkpoint_meta=CheckpointMeta(
            backbone="resnet18",
            num_classes=5,
            class_names=manifest_meta.class_names,
            image_size=IMAGE_SIZE,
            resize_strategy="letterbox",
            normalize_mean=list(preprocessing.normalize_mean),
            normalize_std=list(preprocessing.normalize_std),
            segment_leaf=False,
            manifest_fingerprint=manifest_meta.fingerprint,
        ),
    )

    history = trainer.fit(
        DataLoader(train_set, batch_size=8, shuffle=True),
        DataLoader(val_set, batch_size=8),
    )

    assert len(history) == 2
    assert all(0.0 <= r.val_accuracy <= 1.0 for r in history)
    assert trainer.checkpoint_path.is_file()
    assert trainer.history_frame().shape[0] == 2


def test_predictor_loads_the_checkpoint_and_classifies(prepared, tmp_path, image_root):
    """The trained checkpoint should serve predictions with no config in sight."""
    manifest_path, manifest_meta = prepared
    frame, _ = read_manifest(manifest_path)

    preprocessing = PreprocessingConfig(image_size=IMAGE_SIZE)
    training = TrainingConfig(
        epochs=1,
        batch_size=8,
        num_workers=0,
        device="cpu",
        early_stopping_patience=None,
        checkpoint_dir=tmp_path / "checkpoints",
        run_dir=tmp_path / "runs",
    )

    model = build_model(ModelConfig(backbone="resnet18", pretrained=False), num_classes=5)
    train_set = LeafDataset(
        split_frame(frame, "train"),
        manifest_meta.class_to_idx,
        build_eval_transform(preprocessing),
    )
    trainer = Trainer(
        model,
        build_optimizer(model, training),
        build_criterion(training),
        training,
        class_names=manifest_meta.class_names,
        checkpoint_meta=CheckpointMeta(
            backbone="resnet18",
            num_classes=5,
            class_names=manifest_meta.class_names,
            image_size=IMAGE_SIZE,
            resize_strategy="letterbox",
            normalize_mean=list(preprocessing.normalize_mean),
            normalize_std=list(preprocessing.normalize_std),
            segment_leaf=False,
        ),
    )
    loader = DataLoader(train_set, batch_size=8)
    trainer.fit(loader, loader)

    predictor = LeafPredictor.from_checkpoint(trainer.checkpoint_path, device="cpu")
    sample = next((image_root / "Mint").glob("*.jpg"))

    prediction = predictor.predict(sample)
    assert prediction.label in manifest_meta.class_names
    assert 0.0 <= prediction.confidence <= 1.0
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert len(prediction.top_k(3)) == 3

    batch = predictor.predict_batch(sorted((image_root / "Mint").glob("*.jpg"))[:4])
    assert len(batch) == 4
    assert all(p.source is not None for p in batch)


def test_error_analysis_over_a_loader(prepared, tmp_path):
    manifest_path, manifest_meta = prepared
    frame, _ = read_manifest(manifest_path)

    transform = build_eval_transform(PreprocessingConfig(image_size=IMAGE_SIZE))
    dataset = LeafDataset(
        split_frame(frame, "test"), manifest_meta.class_to_idx, transform, return_path=True
    )
    model = build_model(ModelConfig(backbone="resnet18", pretrained=False), num_classes=5)

    predictions = collect_predictions(
        model, DataLoader(dataset, batch_size=8), manifest_meta.class_names, "cpu"
    )

    assert len(predictions) == len(dataset)
    assert predictions["file_path"].str.len().gt(0).all()
    assert predictions["confidence"].between(0, 1).all()
    assert isinstance(summarize_errors(predictions), str)
