"""Model construction, freezing and checkpoint round-trips.

Every model here is built with ``pretrained=False`` so the suite never
touches the network.
"""

from __future__ import annotations

import pytest
import torch

from medicinal_leaf.config.settings import ModelConfig, TrainingConfig
from medicinal_leaf.models.factory import (
    CheckpointMeta,
    build_criterion,
    build_model,
    build_optimizer,
    build_scheduler,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
)
from medicinal_leaf.models.model import LeafClassifier

BACKBONE = "resnet18"


@pytest.fixture
def model() -> LeafClassifier:
    return LeafClassifier(BACKBONE, num_classes=5, pretrained=False)


@pytest.fixture
def meta() -> CheckpointMeta:
    return CheckpointMeta(
        backbone=BACKBONE,
        num_classes=5,
        class_names=["Aloevera", "Amla", "Mint", "Neem", "Tulsi"],
        image_size=32,
        resize_strategy="letterbox",
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        segment_leaf=False,
    )


def test_forward_shape(model):
    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 5)


def test_forward_features_shape(model):
    features = model.forward_features(torch.randn(2, 3, 32, 32))
    assert features.shape == (2, model.num_features)


def test_predict_proba_sums_to_one(model):
    probabilities = model.predict_proba(torch.randn(3, 3, 32, 32))
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(3), atol=1e-5)


def test_predict_proba_restores_training_mode(model):
    model.train()
    model.predict_proba(torch.randn(1, 3, 32, 32))
    assert model.training


def test_too_few_classes_is_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        LeafClassifier(BACKBONE, num_classes=1, pretrained=False)


def test_freezing_leaves_only_the_head_trainable(model):
    model.freeze_backbone()

    assert model.backbone_frozen
    head_params = sum(p.numel() for p in model.head.parameters())
    assert model.trainable_parameters == head_params


def test_unfreezing_restores_everything(model):
    model.freeze_backbone()
    model.unfreeze_backbone()

    assert not model.backbone_frozen
    total, trainable = count_parameters(model)
    assert total == trainable


def test_frozen_backbone_stays_in_eval_mode(model):
    """Otherwise BatchNorm keeps updating statistics it should not touch."""
    model.freeze_backbone()
    model.train()
    assert not model.backbone.training


def test_build_model_honours_config():
    config = ModelConfig(backbone=BACKBONE, pretrained=False, dropout=0.5)
    built = build_model(config, num_classes=3)
    assert built.num_classes == 3
    # Batch of 1 only works in eval mode — BatchNorm needs more than one
    # value per channel while training. This is the single-image inference
    # path that LeafPredictor relies on.
    built.eval()
    assert built(torch.randn(1, 3, 32, 32)).shape == (1, 3)


def test_batch_of_one_fails_in_train_mode(model):
    """Guards the reason build_dataloader drops a trailing batch of 1."""
    model.train()
    with pytest.raises(ValueError, match="more than 1 value per channel"):
        model(torch.randn(1, 3, 32, 32))


@pytest.mark.parametrize("name", ["adamw", "adam", "sgd"])
def test_every_optimizer_builds(model, name):
    optimizer = build_optimizer(model, TrainingConfig(optimizer=name))
    assert optimizer.param_groups[0]["lr"] > 0


def test_optimizer_only_sees_trainable_parameters(model):
    model.freeze_backbone()
    optimizer = build_optimizer(model, TrainingConfig())
    counted = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
    assert counted == model.trainable_parameters


@pytest.mark.parametrize("name", ["cosine", "step", "plateau", "none"])
def test_every_scheduler_builds(model, name):
    optimizer = build_optimizer(model, TrainingConfig())
    scheduler = build_scheduler(optimizer, TrainingConfig(scheduler=name))
    assert (scheduler is None) == (name == "none")


def test_warmup_is_prepended(model):
    optimizer = build_optimizer(model, TrainingConfig())
    config = TrainingConfig(scheduler="cosine", warmup_epochs=2, epochs=10)
    scheduler = build_scheduler(optimizer, config)
    # Warmup starts well below the configured learning rate.
    assert optimizer.param_groups[0]["lr"] < config.learning_rate
    assert scheduler is not None


def test_criterion_applies_label_smoothing():
    criterion = build_criterion(TrainingConfig(label_smoothing=0.1))
    assert criterion.label_smoothing == pytest.approx(0.1)


def test_criterion_applies_class_weights():
    weights = torch.tensor([1.0, 2.0, 3.0, 1.0, 1.0])
    criterion = build_criterion(TrainingConfig(class_weighting=True), class_weights=weights)
    assert torch.equal(criterion.weight, weights)


def test_checkpoint_round_trip(model, meta, tmp_path):
    path = save_checkpoint(tmp_path / "best.pt", model, meta)
    restored, restored_meta = load_checkpoint(path)

    assert restored_meta.backbone == meta.backbone
    assert restored_meta.class_names == meta.class_names

    sample = torch.randn(1, 3, 32, 32)
    model.eval()
    assert torch.allclose(model(sample), restored(sample), atol=1e-5)


def test_checkpoint_writes_a_readable_sidecar(model, meta, tmp_path):
    path = save_checkpoint(tmp_path / "best.pt", model, meta)
    sidecar = path.with_suffix(".meta.json")
    assert sidecar.is_file()
    assert "resnet18" in sidecar.read_text(encoding="utf-8")


def test_checkpoint_meta_rebuilds_preprocessing(meta):
    config = meta.preprocessing_config()
    assert config.image_size == 32
    assert config.resize_strategy == "letterbox"
    assert config.segment_leaf is False


def test_loading_a_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nothing.pt")
