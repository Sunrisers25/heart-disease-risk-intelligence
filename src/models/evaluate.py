"""Model evaluation module.
Calculates accuracy, precision, recall, f1, roc-auc, and confusion matrix.
Logs and serializes metrics to JSON.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
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
    """Compute standard classification evaluation metrics.

    Args:
        y_true: True binary targets (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_proba: Predicted probability of positive class (Heart_Disease = Yes).

    Returns:
        dict: Evaluated metrics and confusion matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
    }

    return metrics


def save_metrics(metrics: Dict[str, Any], filepath: Path = METRICS_PATH) -> None:
    """Save metrics dictionary to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Saved model metrics to %s", filepath)


def load_metrics(filepath: Path = METRICS_PATH) -> Dict[str, Any]:
    """Load model metrics from JSON file."""
    if not filepath.exists():
        raise FileNotFoundError(f"Metrics file does not exist at {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
