"""
Evaluation metrics for binary AMP classification.

All functions accept numpy arrays and return plain Python floats / dicts
so they can be passed directly to mlflow.log_metrics().
"""

from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute the standard AMP prediction benchmark metrics.

    Args:
        y_true: Ground-truth binary labels (0 / 1).
        y_pred: Hard binary predictions (0 / 1).
        y_prob: Probability of the positive class in [0, 1].

    Returns:
        Dictionary with keys: accuracy, f1, mcc, roc_auc, pr_auc,
        sensitivity (recall), specificity, precision.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
    }

