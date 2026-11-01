"""Evaluation module for binary classification models.
Calculates comprehensive performance metrics: Accuracy, Precision, Recall, F1,
ROC-AUC, Confusion Matrix, Specificity, Sensitivity, Brier Score, and Log Loss.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import METRICS_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, Any]:
    """Compute comprehensive classification evaluation metrics for binary models.

    Args:
        y_true: True binary targets (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_proba: Predicted probability of positive class (Heart_Disease = 1).

    Returns:
        dict: Evaluated performance metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except (ValueError, TypeError):
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    # Binary confusion matrix extraction: [[TN, FP], [FN, TP]]
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    else:
        # Edge case if only single class present
        sensitivity = rec
        specificity = 0.0

    # Calibration & probability metrics
    try:
        brier = float(brier_score_loss(y_true, y_proba))
    except (ValueError, TypeError):
        brier = 0.0

    try:
        # Clip probabilities to prevent log_loss numerical infinity
        eps = 1e-15
        clipped_proba = np.clip(y_proba, eps, 1 - eps)
        ll = float(log_loss(y_true, clipped_proba))
    except (ValueError, TypeError):
        ll = 0.0

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "confusion_matrix": cm.tolist(),
    }

    return metrics


def save_metrics(metrics: Dict[str, Any], filepath: Path = METRICS_PATH) -> None:
    """Save metrics dictionary to a JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Saved model metrics to %s", filepath)


def load_metrics(filepath: Path = METRICS_PATH) -> Dict[str, Any]:
    """Load model metrics from JSON file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Metrics file does not exist at {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
