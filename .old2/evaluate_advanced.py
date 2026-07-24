#!/usr/bin/env python3
"""
OHCA Model — Advanced Evaluation & Visualizations
==================================================
Generates comprehensive diagnostic plots:
  1. ROC Curve + AUC
  2. Precision-Recall Curve
  3. Probability Distribution by Class
  4. Calibration Curve
  5. Feature Importance via Grad-CAM on ECG
  6. Per-Window Prediction Heatmap
  7. Confusion Matrix Heatmap
"""

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix
)
from scipy.ndimage import gaussian_filter1d

DATA_DIR = Path(__file__).parent / 'data'
MODEL_PATH = Path(__file__).parent / 'models_v3/best_v3.keras'
OUT_DIR = Path(__file__).parent / 'eval_plots'

ECG_FS = 130
ACC_FS = 25


def load_data():
    return {
        'X_ecg': np.load(DATA_DIR / 'X_ecg_test.npy'),
        'X_acc': np.load(DATA_DIR / 'X_acc_test.npy'),
        'X_bio': np.load(DATA_DIR / 'X_bio_test.npy'),
        'y':     np.load(DATA_DIR / 'y_test.npy'),
    }


def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — OHCA Prediction')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'roc_curve.png', dpi=150)
    plt.close()
    print("  Saved: roc_curve.png")


def plot_pr_curve(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'AP = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve — OHCA Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'pr_curve.png', dpi=150)
    plt.close()
    print("  Saved: pr_curve.png")


def plot_probability_distribution(y_true, y_prob):
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 50)
    plt.hist(y_prob[y_true == 0], bins=bins, alpha=0.6, label='Stable', color='blue', density=True)
    plt.hist(y_prob[y_true == 1], bins=bins, alpha=0.6, label='Pre-Arrest', color='red', density=True)
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'probability_distribution.png', dpi=150)
    plt.close()
    print("  Saved: probability_distribution.png")


def plot_calibration(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_means = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append(y_prob[mask].mean())
            bin_means.append(y_true[mask].mean())
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, bin_means, 'o-', color='teal', lw=2, label='Model')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve — OHCA Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'calibration_curve.png', dpi=150)
    plt.close()
    print("  Saved: calibration_curve.png")


def plot_confusion_heatmap(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar(im)
    classes = ['Stable', 'Pre-Arrest']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'confusion_matrix.png', dpi=150)
    plt.close()
    print("  Saved: confusion_matrix.png")


def compute_gradcam_ecg(model, ecg_batch, layer_name=None):
    """Compute Grad-CAM saliency for the ECG branch."""
    if layer_name is None:
        for layer in model.layers:
            if 'conv1d' in layer.name.lower():
                layer_name = layer.name
                break
    if layer_name is None:
        print("  ⚠  No Conv1D layer found for Grad-CAM")
        return None

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    n = ecg_batch.shape[0]
    dummy_acc = np.zeros((n, 250, 3), dtype=np.float32)
    dummy_bio = np.zeros((n, 3), dtype=np.float32)
    inputs_dict = {
        'ecg_input': tf.convert_to_tensor(ecg_batch),
        'acc_input': tf.convert_to_tensor(dummy_acc),
        'bio_input': tf.convert_to_tensor(dummy_bio),
    }

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inputs_dict)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap.numpy().flatten()
    return gaussian_filter1d(heatmap, sigma=5)


def plot_ecg_gradcam(model, ecg_data, y_true, n_samples=4):
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        idx = np.random.randint(len(ecg_data))
        ecg = ecg_data[idx:idx+1]
        label = 'Pre-Arrest' if y_true[idx] == 1 else 'Stable'
        heatmap = compute_gradcam_ecg(model, ecg)
        t = np.arange(1300) / ECG_FS

        ax.plot(t, ecg[0].flatten(), color='black', lw=0.8, label='ECG')
        if heatmap is not None:
            ax2 = ax.twinx()
            ax2.fill_between(t, 0, heatmap, alpha=0.4, color='red', label='Saliency')
            ax2.set_ylabel('Saliency', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        ax.set_title(f'Sample {idx} — {label}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (mV)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'ecg_gradcam.png', dpi=150)
    plt.close()
    print("  Saved: ecg_gradcam.png")


def plot_sample_predictions(y_true, y_prob, n=20):
    fig, ax = plt.subplots(figsize=(12, 4))
    indices = np.arange(n)
    colors = ['blue' if yt == 0 else 'red' for yt in y_true[:n]]
    ax.bar(indices, y_prob[:n], color=colors, alpha=0.7)
    ax.axhline(y=0.5, color='black', linestyle='--', label='Threshold')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Predicted Probability')
    ax.set_title('Per-Window Predictions (Blue=Stable, Red=Pre-Arrest)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'sample_predictions.png', dpi=150)
    plt.close()
    print("  Saved: sample_predictions.png")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    data = load_data()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print("=" * 60)
    print("  OHCA MODEL — ADVANCED EVALUATION")
    print("=" * 60)

    y_prob = model.predict(
        [data['X_ecg'], data['X_acc'], data['X_bio']]
    ).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = data['y']

    print("\nGenerating plots...")
    plot_roc(y_true, y_prob)
    plot_pr_curve(y_true, y_prob)
    plot_probability_distribution(y_true, y_prob)
    plot_calibration(y_true, y_prob)
    plot_confusion_heatmap(y_true, y_pred)
    plot_ecg_gradcam(model, data['X_ecg'], y_true)
    plot_sample_predictions(y_true, y_prob)

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
