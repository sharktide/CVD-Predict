#!/usr/bin/env python3
"""
OHCA Model Evaluation — Core Metrics
=====================================
Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC with
per-class breakdown and confidence intervals via bootstrap.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

DATA_DIR = Path(__file__).parent / 'data'
MODEL_PATH = Path(__file__).parent / 'model.keras'
N_BOOTSTRAP = 1000


def load_data():
    return {
        'X_ecg': np.load(DATA_DIR / 'X_ecg_test.npy'),
        'X_acc': np.load(DATA_DIR / 'X_acc_test.npy'),
        'X_bio': np.load(DATA_DIR / 'X_bio_test.npy'),
        'y':     np.load(DATA_DIR / 'y_test.npy'),
    }


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=N_BOOTSTRAP, ci=0.95):
    """Compute metric with bootstrap confidence interval."""
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        scores.append(metric_fn(y_true[idx], y_prob[idx]))
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return np.mean(scores), lo, hi


def main():
    data = load_data()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    y_prob = model.predict(
        [data['X_ecg'], data['X_acc'], data['X_bio']]
    ).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = data['y']

    print("=" * 60)
    print("  OHCA MODEL — CORE EVALUATION METRICS")
    print("=" * 60)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    # Core Metrics
    metrics = {
        'Accuracy':  (accuracy_score(y_true, y_pred), y_true, y_pred),
        'Precision': (precision_score(y_true, y_pred, zero_division=0), y_true, y_pred),
        'Recall':    (recall_score(y_true, y_pred, zero_division=0), y_true, y_pred),
        'F1-Score':  (f1_score(y_true, y_pred, zero_division=0), y_true, y_pred),
        'ROC-AUC':   (roc_auc_score(y_true, y_prob), y_true, y_prob),
    }

    print("\nMetric Breakdown:")
    for name, (val, yt, yp) in metrics.items():
        metric_fn = lambda yt, yp: accuracy_score(yt, (yp >= 0.5).astype(int)) if name != 'ROC-AUC' else roc_auc_score(yt, yp)
        mean_ci, lo, hi = bootstrap_ci(yt, yp, metric_fn)
        print(f"  {name:12s}: {val:.4f}  [95% CI: {lo:.4f} – {hi:.4f}]")

    # Full classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Stable', 'Pre-Arrest']))

    # Per-class analysis
    print("Per-class analysis:")
    for cls, label in [(0, 'Stable'), (1, 'Pre-Arrest')]:
        mask = y_true == cls
        mean_prob = y_prob[mask].mean()
        std_prob = y_prob[mask].std()
        print(f"  {label:12s}: mean_prob={mean_prob:.4f} ± {std_prob:.4f}  n={mask.sum()}")

    if accuracy_score(y_true, y_pred) > 0.93:
        print("\n⚠  WARNING: Accuracy > 93%. Verify data generation overlap.")


if __name__ == '__main__':
    main()
