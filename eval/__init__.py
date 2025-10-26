"""
Model Evaluation and Benchmarking

Comprehensive evaluation suite for comparing models.
"""

from .evaluate import evaluate_model, benchmark_models
from .metrics import compute_metrics, plot_confusion_matrix, plot_roc_curve

__all__ = [
    "evaluate_model",
    "benchmark_models",
    "compute_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
]
