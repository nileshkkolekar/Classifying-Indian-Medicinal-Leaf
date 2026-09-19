"""Metrics and failure-mode analysis."""

from medicinal_leaf.evaluation.error_analysis import (
    collect_predictions,
    export_error_grid,
    most_confident_errors,
    per_class_error_rate,
    summarize_errors,
    top_confusions,
)
from medicinal_leaf.evaluation.metrics import (
    ClassificationMetrics,
    classification_report_text,
    compute_metrics,
    plot_confusion_matrix,
    top_k_accuracy,
)

__all__ = [
    "ClassificationMetrics",
    "classification_report_text",
    "collect_predictions",
    "compute_metrics",
    "export_error_grid",
    "most_confident_errors",
    "per_class_error_rate",
    "plot_confusion_matrix",
    "summarize_errors",
    "top_confusions",
    "top_k_accuracy",
]
