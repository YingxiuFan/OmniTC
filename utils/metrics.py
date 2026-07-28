# -*- coding: utf-8 -*-
"""
Metrics Utilities (Simplified)
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_recall_fscore_support, confusion_matrix
)
from typing import Dict


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics = {}

    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        metrics['auc_roc'] = np.nan
        metrics['auc_pr'] = 1.0 if unique_labels[0] == 1 else 0.0
    else:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc_roc'] = np.nan
        try:
            metrics['auc_pr'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['auc_pr'] = 0.0

    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['true_positive'] = tp
    metrics['true_negative'] = tn
    metrics['false_positive'] = fp
    metrics['false_negative'] = fn

    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denominator > 0:
        metrics['mcc'] = (tp * tn - fp * fn) / denominator
    else:
        metrics['mcc'] = 0.0

    metrics['auprc'] = metrics['auc_pr']

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    print(f"\n{prefix}Metrics:")
    print(f"  - AUC-ROC:  {metrics['auc_roc']:.4f}")
    print(f"  - AUPRC:    {metrics['auprc']:.4f}")
    print(f"  - MCC:      {metrics['mcc']:.4f}")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")
    print(f"  - F1 Score: {metrics['f1']:.4f}")
