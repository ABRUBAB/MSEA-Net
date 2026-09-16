from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
import torch.nn.functional as F
import torch


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int = 8,
) -> Dict[str, float]:
    """
    Compute comprehensive clinical diagnostic evaluation metrics.
    """
    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))

    # Multi-class AUROC and PR-AUC
    try:
        # Binarize targets for multiclass AUC
        y_true_onehot = np.eye(num_classes)[y_true]
        metrics['auc_macro'] = float(roc_auc_score(y_true_onehot, y_probs, multi_class='ovr', average='macro'))
        metrics['pr_auc_macro'] = float(average_precision_score(y_true_onehot, y_probs, average='macro'))
    except Exception:
        metrics['auc_macro'] = 0.0
        metrics['pr_auc_macro'] = 0.0

    # Multi-class Brier score
    try:
        y_true_onehot = np.eye(num_classes)[y_true]
        brier = np.mean(np.sum((y_probs - y_true_onehot) ** 2, axis=1))
        metrics['brier_score'] = float(brier)
    except Exception:
        metrics['brier_score'] = 0.0

    return metrics
