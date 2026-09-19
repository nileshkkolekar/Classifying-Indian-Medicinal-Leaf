"""Look at what the model got wrong, and how sure it was.

Aggregate metrics say *how much* is wrong; this module says *what*. The two
questions worth asking of a leaf classifier are which species pairs it
confuses (usually the visually similar ones) and whether its mistakes are
hesitant or confident — confident errors normally mean mislabelled data or a
background the model latched onto.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from medicinal_leaf.preprocessing.image import load_image

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    """Run the model over a loader and return one row per sample.

    Columns: ``file_path``, ``true_label``, ``predicted_label``,
    ``true_idx``, ``predicted_idx``, ``confidence``, ``true_class_prob``,
    ``correct``. The loader's dataset should have ``return_path=True``;
    without it the path column is filled with an empty string.
    """
    model.eval()
    model.to(device)

    paths: list[str] = []
    truths: list[int] = []
    preds: list[int] = []
    confidences: list[float] = []
    true_probs: list[float] = []

    for batch in loader:
        if len(batch) == 3:
            images, targets, batch_paths = batch
        else:
            images, targets = batch
            batch_paths = [""] * len(targets)

        images = images.to(device, non_blocking=True)
        probabilities = torch.softmax(model(images), dim=1).cpu()

        confidence, predicted = probabilities.max(dim=1)
        targets_cpu = targets.cpu()

        paths.extend(batch_paths)
        truths.extend(targets_cpu.tolist())
        preds.extend(predicted.tolist())
        confidences.extend(confidence.tolist())
        true_probs.extend(probabilities[torch.arange(len(targets_cpu)), targets_cpu].tolist())

    frame = pd.DataFrame(
        {
            "file_path": paths,
            "true_idx": truths,
            "predicted_idx": preds,
            "confidence": confidences,
            "true_class_prob": true_probs,
        }
    )
    frame["true_label"] = frame["true_idx"].map(dict(enumerate(class_names)))
    frame["predicted_label"] = frame["predicted_idx"].map(dict(enumerate(class_names)))
    frame["correct"] = frame["true_idx"] == frame["predicted_idx"]

    logger.info(
        "Collected %d predictions (%d incorrect)", len(frame), int((~frame["correct"]).sum())
    )
    return frame


def top_confusions(frame: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Most frequent ``(true, predicted)`` mistakes, with mean confidence."""
    errors = frame[~frame["correct"]]
    if errors.empty:
        return pd.DataFrame(columns=["true_label", "predicted_label", "count", "mean_confidence"])

    grouped = (
        errors.groupby(["true_label", "predicted_label"])
        .agg(count=("correct", "size"), mean_confidence=("confidence", "mean"))
        .reset_index()
        .sort_values(["count", "mean_confidence"], ascending=False)
    )
    return grouped.head(n).reset_index(drop=True)


def per_class_error_rate(frame: pd.DataFrame) -> pd.DataFrame:
    """Support, error count and error rate for each true class."""
    grouped = (
        frame.groupby("true_label")
        .agg(
            support=("correct", "size"),
            errors=("correct", lambda s: int((~s).sum())),
            mean_true_prob=("true_class_prob", "mean"),
        )
        .assign(error_rate=lambda df: df["errors"] / df["support"])
    )
    return grouped.sort_values("error_rate", ascending=False).round(4)


def most_confident_errors(frame: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Wrong predictions the model was most sure about — inspect these first."""
    errors = frame[~frame["correct"]]
    return errors.nlargest(n, "confidence").reset_index(drop=True)


def least_confident_correct(frame: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Right answers that were nearly wrong — the decision boundary."""
    correct = frame[frame["correct"]]
    return correct.nsmallest(n, "confidence").reset_index(drop=True)


def export_error_grid(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    n: int = 12,
    columns: int = 4,
    by: str = "confidence",
) -> Figure:
    """Save a contact sheet of misclassified images labelled ``true -> pred``."""
    import matplotlib.pyplot as plt

    errors = frame[~frame["correct"]]
    errors = errors[errors["file_path"].astype(bool)]
    if errors.empty:
        raise ValueError("No misclassified samples with file paths to plot.")

    selection = errors.nlargest(min(n, len(errors)), by)
    rows = int(np.ceil(len(selection) / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.4 * rows))
    flat_axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(flat_axes, selection.iterrows(), strict=False):
        try:
            ax.imshow(load_image(row["file_path"]))
        except OSError as exc:  # pragma: no cover - unreadable file at plot time
            logger.warning("Could not render %s: %s", row["file_path"], exc)
            ax.text(0.5, 0.5, "unreadable", ha="center", va="center")
        ax.set_title(
            f"{row['true_label']} → {row['predicted_label']}\n{row['confidence']:.2f}",
            fontsize=9,
        )
        ax.axis("off")

    for ax in flat_axes[len(selection) :]:
        ax.axis("off")

    fig.suptitle(f"Misclassified samples (top {len(selection)} by {by})")
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Wrote error grid: %s", path)
    return fig


def summarize_errors(frame: pd.DataFrame, n_confusions: int = 5) -> str:
    """Short text digest suitable for a run log or PR comment."""
    total = len(frame)
    wrong = int((~frame["correct"]).sum())
    lines = [
        f"{wrong}/{total} incorrect ({wrong / total:.2%})" if total else "no predictions",
        "",
        "Per-class error rate:",
        per_class_error_rate(frame).to_string(),
    ]

    confusions = top_confusions(frame, n_confusions)
    if not confusions.empty:
        lines += ["", f"Top {len(confusions)} confusions:", confusions.to_string(index=False)]

    return "\n".join(lines)


def save_report(frame: pd.DataFrame, directory: str | Path) -> dict[str, Path]:
    """Persist the prediction frame and its digests. Returns the paths written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    written: dict[str, Any] = {}
    predictions_path = directory / "predictions.csv"
    frame.to_csv(predictions_path, index=False)
    written["predictions"] = predictions_path

    confusions_path = directory / "top_confusions.csv"
    top_confusions(frame, n=50).to_csv(confusions_path, index=False)
    written["confusions"] = confusions_path

    summary_path = directory / "error_summary.txt"
    summary_path.write_text(summarize_errors(frame), encoding="utf-8")
    written["summary"] = summary_path

    return written
