"""Classification metrics.

Macro-F1 is the headline number rather than accuracy. With uneven class
counts, accuracy is dominated by whichever species happens to be
over-represented, and a model that quietly ignores the smallest class can
still post a respectable score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

ArrayLike = np.ndarray | list[int]


@dataclass(slots=True)
class ClassificationMetrics:
    """Scalar summary plus the per-class breakdown and confusion matrix."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    class_names: list[str]
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=int))
    loss: float | None = None
    top_k: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        """Scalars only — what gets logged per epoch."""
        payload: dict[str, float] = {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
        }
        if self.loss is not None:
            payload["loss"] = self.loss
        payload.update({f"top_{k}_accuracy": v for k, v in self.top_k.items()})
        return payload

    def to_frame(self) -> pd.DataFrame:
        """Per-class precision / recall / F1 / support."""
        return pd.DataFrame(self.per_class).T[["precision", "recall", "f1", "support"]]

    def confusion_frame(self, normalize: bool = False) -> pd.DataFrame:
        matrix = self.confusion.astype(float)
        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
        return pd.DataFrame(matrix, index=self.class_names, columns=self.class_names)

    def to_markdown(self) -> str:
        header = (
            f"**accuracy** {self.accuracy:.4f} · "
            f"**macro-F1** {self.macro_f1:.4f} · "
            f"**balanced acc** {self.balanced_accuracy:.4f}"
        )
        if self.loss is not None:
            header += f" · **loss** {self.loss:.4f}"
        return f"{header}\n\n{self.to_frame().round(4).to_markdown()}"

    def __str__(self) -> str:
        parts = [
            f"acc={self.accuracy:.4f}",
            f"macro_f1={self.macro_f1:.4f}",
            f"bal_acc={self.balanced_accuracy:.4f}",
        ]
        if self.loss is not None:
            parts.insert(0, f"loss={self.loss:.4f}")
        return " ".join(parts)


def top_k_accuracy(probabilities: np.ndarray, y_true: ArrayLike, k: int = 3) -> float:
    """Fraction of samples whose true label is among the ``k`` highest scores."""
    probs = np.asarray(probabilities)
    truth = np.asarray(y_true)
    if probs.ndim != 2:
        raise ValueError(f"Expected 2-D probabilities, got shape {probs.shape}")

    k = min(k, probs.shape[1])
    top = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float(np.mean([truth[i] in top[i] for i in range(len(truth))]))


def compute_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    class_names: list[str],
    *,
    loss: float | None = None,
    probabilities: np.ndarray | None = None,
    top_k: tuple[int, ...] = (3,),
) -> ClassificationMetrics:
    """Compute every metric in one pass over the predictions."""
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true {truth.shape} vs y_pred {pred.shape}")
    if truth.size == 0:
        raise ValueError("Cannot compute metrics over zero predictions.")

    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, pred, labels=labels, zero_division=0
    )

    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }

    top_k_scores: dict[int, float] = {}
    if probabilities is not None:
        top_k_scores = {k: top_k_accuracy(probabilities, truth, k) for k in top_k}

    return ClassificationMetrics(
        accuracy=float(accuracy_score(truth, pred)),
        balanced_accuracy=float(balanced_accuracy_score(truth, pred)),
        macro_f1=float(f1_score(truth, pred, average="macro", labels=labels, zero_division=0)),
        weighted_f1=float(
            f1_score(truth, pred, average="weighted", labels=labels, zero_division=0)
        ),
        macro_precision=float(np.mean(precision)),
        macro_recall=float(np.mean(recall)),
        class_names=list(class_names),
        per_class=per_class,
        confusion=confusion_matrix(truth, pred, labels=labels),
        loss=loss,
        top_k=top_k_scores,
    )


def classification_report_text(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    class_names: list[str],
) -> str:
    """scikit-learn's text report, for pasting into a run log."""
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )


def plot_confusion_matrix(
    metrics: ClassificationMetrics,
    *,
    normalize: bool = True,
    path: str | Path | None = None,
    title: str = "Confusion matrix",
    **heatmap_kwargs: Any,
) -> Figure:
    """Render the confusion matrix, optionally saving it.

    Row-normalised by default: what matters is the share of each true class
    that went astray, not the raw count, which just re-reads class sizes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    frame = metrics.confusion_frame(normalize=normalize)
    fig, ax = plt.subplots(figsize=(1.4 * len(metrics.class_names) + 3,) * 2)

    sns.heatmap(
        frame,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        cbar=False,
        square=True,
        vmin=0,
        vmax=1 if normalize else None,
        ax=ax,
        **heatmap_kwargs,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title} (macro-F1 {metrics.macro_f1:.3f})")
    fig.tight_layout()

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    return fig
