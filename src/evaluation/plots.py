"""
Plotting utilities for model evaluation.

All functions return a matplotlib Figure so callers can either:
  - save it to disk (results/figures/)
  - log it as an MLflow artefact via mlflow.log_figure()
"""

from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str = "Model",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot a single ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    return fig or ax.get_figure()


def plot_roc_comparison(
    results: Dict[str, Dict[str, np.ndarray]],
) -> plt.Figure:
    """Overlay ROC curves for multiple models.

    Args:
        results: Mapping of model_name → {"y_true": ..., "y_prob": ...}

    Returns:
        Figure with all curves overlaid.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, data in results.items():
        plot_roc_curve(data["y_true"], data["y_prob"], label=name, ax=ax)
    ax.set_title("ROC Curve Comparison")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> plt.Figure:
    """Plot a normalised confusion matrix heatmap."""
    labels = labels or ["Non-AMP", "AMP"]
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (normalised)")
    fig.tight_layout()
    return fig


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metrics_to_plot: Optional[List[str]] = None,
) -> plt.Figure:
    """Bar chart comparing key metrics across multiple models.

    Args:
        metrics_dict: Mapping of model_name → metrics dict (from compute_metrics).
        metrics_to_plot: Subset of metric keys to show. Defaults to main 5.

    Returns:
        Figure with grouped bar chart.
    """
    import pandas as pd

    metrics_to_plot = metrics_to_plot or ["accuracy", "f1", "mcc", "roc_auc", "pr_auc"]
    df = pd.DataFrame(metrics_dict).T[metrics_to_plot]

    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind="bar", ax=ax, edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend(loc="lower right")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig

