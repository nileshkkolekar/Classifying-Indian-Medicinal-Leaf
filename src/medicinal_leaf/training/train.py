"""Command-line entry points for the pipeline.

    leaf-train prepare     # index -> validate -> split -> manifest
    leaf-train fit         # train against the manifest
    leaf-train evaluate    # score a checkpoint on a held-out split

Each command builds its own :class:`Settings`, so flags stay thin and the
YAML/env layers remain the source of truth.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from torch.utils.data import DataLoader

from medicinal_leaf.config.settings import Settings, load_settings
from medicinal_leaf.data.ingestion import LeafDataset, build_index_from_settings
from medicinal_leaf.data.manifest import read_manifest, split_frame, write_manifest
from medicinal_leaf.data.splitting import assert_no_leakage, split_summary, stratified_split
from medicinal_leaf.data.validation import raise_for_errors, validate_index
from medicinal_leaf.evaluation.error_analysis import (
    collect_predictions,
    save_report,
    summarize_errors,
)
from medicinal_leaf.evaluation.metrics import (
    classification_report_text,
    compute_metrics,
    plot_confusion_matrix,
)
from medicinal_leaf.models.factory import (
    CheckpointMeta,
    build_criterion,
    build_model,
    build_optimizer,
    build_scheduler,
    count_parameters,
    load_checkpoint,
)
from medicinal_leaf.preprocessing.augmentation import build_eval_transform, build_train_transform
from medicinal_leaf.training.trainer import Trainer, set_seed

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, help="Indian medicinal leaf classification pipeline.")

EnvOption = Annotated[
    str | None,
    typer.Option("--env", "-e", help="Config to load: development or production."),
]


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def build_dataloader(
    dataset: LeafDataset,
    settings: Settings,
    *,
    shuffle: bool,
) -> DataLoader:
    """Wrap a dataset with the configured worker and memory settings."""
    workers = settings.training.num_workers
    return DataLoader(
        dataset,
        batch_size=settings.training.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=settings.training.pin_memory,
        persistent_workers=workers > 0,
        # A trailing batch of 1 breaks BatchNorm in train mode.
        drop_last=shuffle and len(dataset) % settings.training.batch_size == 1,
    )


@app.command()
def prepare(
    env: EnvOption = None,
    check_decodable: Annotated[
        bool, typer.Option(help="Fully decode every image (slow, thorough).")
    ] = False,
    verbose: bool = False,
) -> None:
    """Index the image tree, validate it, split it and write the manifest."""
    setup_logging(verbose)
    settings = load_settings(env)
    settings.ensure_directories()
    typer.echo(f"Scanning {settings.data.raw_dir} ...")

    frame = build_index_from_settings(settings)

    report = validate_index(
        frame,
        min_side=settings.data.min_side,
        max_aspect_ratio=settings.data.max_aspect_ratio,
        min_images_per_class=settings.data.min_images_per_class,
        max_imbalance_ratio=settings.data.max_imbalance_ratio,
        check_decodable=check_decodable,
    )
    typer.echo(report.summary())
    for warning in report.warnings[:10]:
        typer.echo(f"  {warning}")
    raise_for_errors(report)

    split = stratified_split(
        frame,
        train_size=settings.data.train_size,
        val_size=settings.data.val_size,
        test_size=settings.data.test_size,
        seed=settings.seed,
    )
    assert_no_leakage(split)

    meta = write_manifest(split, settings.data.manifest_path)
    typer.echo("")
    typer.echo(split_summary(split).to_string())
    typer.echo("")
    typer.secho(
        f"Manifest written to {settings.data.manifest_path} "
        f"({meta.n_images} images, fingerprint {meta.fingerprint})",
        fg=typer.colors.GREEN,
    )


@app.command()
def fit(
    env: EnvOption = None,
    epochs: Annotated[int | None, typer.Option(help="Override training.epochs.")] = None,
    batch_size: Annotated[int | None, typer.Option(help="Override training.batch_size.")] = None,
    backbone: Annotated[str | None, typer.Option(help="Override model.backbone.")] = None,
    verbose: bool = False,
) -> None:
    """Fine-tune the classifier against the prepared manifest."""
    setup_logging(verbose)
    settings = load_settings(env)
    if epochs is not None:
        settings.training.epochs = epochs
    if batch_size is not None:
        settings.training.batch_size = batch_size
    if backbone is not None:
        settings.model.backbone = backbone

    settings.ensure_directories()
    set_seed(settings.seed)
    typer.echo(f"Configuration: {settings.summary()}")

    frame, manifest_meta = read_manifest(settings.data.manifest_path)
    class_names = manifest_meta.class_names

    train_transform = build_train_transform(settings.preprocessing, settings.augmentation)
    eval_transform = build_eval_transform(settings.preprocessing)

    train_set = LeafDataset(
        split_frame(frame, "train"), manifest_meta.class_to_idx, train_transform
    )
    val_set = LeafDataset(split_frame(frame, "val"), manifest_meta.class_to_idx, eval_transform)

    train_loader = build_dataloader(train_set, settings, shuffle=True)
    val_loader = build_dataloader(val_set, settings, shuffle=False)
    typer.echo(f"Train: {len(train_set)} images · Val: {len(val_set)} images")

    model = build_model(settings.model, num_classes=len(class_names))
    total, trainable = count_parameters(model)
    typer.echo(f"{settings.model.backbone}: {total:,} parameters ({trainable:,} trainable)")

    optimizer = build_optimizer(model, settings.training)
    scheduler = build_scheduler(optimizer, settings.training)
    criterion = build_criterion(
        settings.training,
        class_weights=train_set.class_weights() if settings.training.class_weighting else None,
    )

    checkpoint_meta = CheckpointMeta(
        backbone=settings.model.backbone,
        num_classes=len(class_names),
        class_names=class_names,
        image_size=settings.preprocessing.image_size,
        resize_strategy=settings.preprocessing.resize_strategy,
        normalize_mean=list(settings.preprocessing.normalize_mean),
        normalize_std=list(settings.preprocessing.normalize_std),
        segment_leaf=settings.preprocessing.segment_leaf,
        dropout=settings.model.dropout,
        manifest_fingerprint=manifest_meta.fingerprint,
    )

    trainer = Trainer(
        model,
        optimizer,
        criterion,
        settings.training,
        class_names=class_names,
        scheduler=scheduler,
        checkpoint_meta=checkpoint_meta,
        unfreeze_after_epoch=settings.model.unfreeze_after_epoch,
    )
    history = trainer.fit(train_loader, val_loader)

    history_path = settings.training.run_dir / "history.csv"
    trainer.history_frame().to_csv(history_path, index=False)

    typer.echo("")
    typer.secho(
        f"Best {settings.training.early_stopping_metric}="
        f"{trainer.best_metric:.4f} at epoch {trainer.best_epoch} "
        f"after {len(history)} epoch(s)",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Checkpoint: {trainer.checkpoint_path}")
    typer.echo(f"History:    {history_path}")


@app.command()
def evaluate(
    env: EnvOption = None,
    checkpoint: Annotated[
        Path | None, typer.Option(help="Defaults to the best checkpoint.")
    ] = None,
    split: Annotated[str, typer.Option(help="Which split to score.")] = "test",
    verbose: bool = False,
) -> None:
    """Score a checkpoint on a held-out split and write an error report."""
    setup_logging(verbose)
    settings = load_settings(env)
    settings.ensure_directories()

    checkpoint_path = checkpoint or settings.training.checkpoint_dir / "best.pt"
    device = settings.training.resolved_device()
    model, meta = load_checkpoint(checkpoint_path, device)

    frame, manifest_meta = read_manifest(settings.data.manifest_path)
    if manifest_meta.fingerprint != meta.manifest_fingerprint:
        typer.secho(
            "Warning: the manifest has changed since this checkpoint was trained "
            f"({meta.manifest_fingerprint} -> {manifest_meta.fingerprint}). "
            "Test rows may have been seen during training.",
            fg=typer.colors.YELLOW,
        )

    # Reuse the checkpoint's own preprocessing, not whatever the config says now.
    transform = build_eval_transform(meta.preprocessing_config())
    dataset = LeafDataset(
        split_frame(frame, split),
        manifest_meta.class_to_idx,
        transform,
        return_path=True,
    )
    loader = build_dataloader(dataset, settings, shuffle=False)

    predictions = collect_predictions(model, loader, meta.class_names, device)
    y_true = predictions["true_idx"].to_numpy()
    y_pred = predictions["predicted_idx"].to_numpy()
    metrics = compute_metrics(y_true, y_pred, meta.class_names)

    typer.echo("")
    typer.echo(classification_report_text(y_true, y_pred, meta.class_names))
    typer.echo(summarize_errors(predictions))

    out_dir = settings.training.run_dir / f"eval_{split}"
    written = save_report(predictions, out_dir)
    plot_confusion_matrix(metrics, path=out_dir / "confusion_matrix.png")

    typer.echo("")
    typer.secho(
        f"{split}: accuracy {metrics.accuracy:.4f} · macro-F1 {metrics.macro_f1:.4f}",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Report written to {out_dir} ({len(written)} files)")


if __name__ == "__main__":
    app()
