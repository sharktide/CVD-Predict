#!/usr/bin/env python3
"""Train CVD Watch Model v13 — Multi-horizon (1h, 6h, 24h) with accelerometer.

Uses wristppg/ as primary synthetic data source with 3-axis accelerometer.
Evaluates on real MIMIC/MMASH test set and wristppg synthetic test set.
Logs everything to TensorBoard with comprehensive graphs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_pipeline_v12 import (
    PPG_LENGTH, FS_TARGET,
    generate_wristppg_synthetic, extract_ppg_features, extract_accel_features,
    load_real_data_by_patient, patient_level_split, flatten_patients,
    build_arrays, build_biodata_array,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VERSION = "v13"
OUT_DIR = Path(f"production/cvd_risk_{VERSION}_watch")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR = OUT_DIR / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_test(n=60, seed=99):
    """Generate wristppg synthetic test set."""
    from wristppg import WristPPGSimulator
    rng = np.random.default_rng(seed)
    ppgs, accels, labels = [], [], []

    at_risk_profiles = ["shock", "hfref", "afib_isolated", "hypovolemia", "sepsis_warm"]

    for _ in range(n // 2):
        try:
            sim = WristPPGSimulator(seed=int(rng.integers(0, 2**31)))
            result = sim.generate(profile="healthy", duration_s=60.0, activity="rest")
            ppgs.append(result.ppg[:PPG_LENGTH])
            accels.append(result.accel[:PPG_LENGTH])
            labels.append(0)
        except Exception:
            continue

    for _ in range(n // 2):
        try:
            sim = WristPPGSimulator(seed=int(rng.integers(0, 2**31)))
            prof = rng.choice(at_risk_profiles)
            result = sim.generate(profile=prof, duration_s=60.0, activity="rest")
            ppgs.append(result.ppg[:PPG_LENGTH])
            accels.append(result.accel[:PPG_LENGTH])
            labels.append(1)
        except Exception:
            continue

    return ppgs, accels, np.array(labels, dtype=np.float32)


def extract_features_for_signal(ppg, accel=None, fs=25):
    """Extract combined PPG + accel features."""
    feats = extract_ppg_features(ppg, fs=fs)
    if accel is not None and accel.ndim == 2 and accel.shape[-1] == 3:
        accel_feats = extract_accel_features(accel, fs=fs)
        feats.update(accel_feats)
    return feats


def plot_training_curves(history, out_dir):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(history.history["loss"], label="Train", linewidth=2)
    axes[0, 0].plot(history.history["val_loss"], label="Val", linewidth=2)
    axes[0, 0].set_title("Loss", fontsize=14)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # AUC
    axes[0, 1].plot(history.history["auc"], label="Train", linewidth=2)
    axes[0, 1].plot(history.history["val_auc"], label="Val", linewidth=2)
    axes[0, 1].set_title("AUROC", fontsize=14)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Precision
    axes[1, 0].plot(history.history["precision"], label="Train", linewidth=2)
    axes[1, 0].plot(history.history["val_precision"], label="Val", linewidth=2)
    axes[1, 0].set_title("Precision", fontsize=14)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Recall
    axes[1, 1].plot(history.history["recall"], label="Train", linewidth=2)
    axes[1, 1].plot(history.history["val_recall"], label="Val", linewidth=2)
    axes[1, 1].set_title("Recall", fontsize=14)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"CVD {VERSION.upper()} — Training Curves", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved training curves to %s", out_dir / "training_curves.png")


def plot_roc_curve(y_true, y_prob, out_dir, label="test"):
    """Plot and save ROC curve."""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUROC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"CVD {VERSION.upper()} — ROC Curve ({label})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"roc_curve_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, out_dir, label="test"):
    """Plot and save confusion matrix."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"CVD {VERSION.upper()} — Confusion Matrix ({label})", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)
    classes = ["Healthy", "At-Risk"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=16)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_matrix_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_probability_distribution(y_true, y_prob, out_dir, label="test"):
    """Plot probability distributions for each class."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_prob[y_true == 0], bins=30, alpha=0.6, label="Healthy", color="green", density=True)
    ax.hist(y_prob[y_true == 1], bins=30, alpha=0.6, label="At-Risk", color="red", density=True)
    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"CVD {VERSION.upper()} — Probability Distribution ({label})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"prob_distribution_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_bar(metrics_dict, out_dir):
    """Plot comparison bar chart of metrics."""
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values, color=["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"])
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"CVD {VERSION.upper()} — Test Metrics", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def make_horizon_labels(binary_labels, profiles=None, rng=None):
    """Derive multi-horizon labels from binary labels + profile info.

    Returns (N, 3) array with columns [1h, 6h, 24h].
    """
    n = len(binary_labels)
    labels_1h = binary_labels.copy()
    labels_6h = binary_labels.copy()
    labels_24h = binary_labels.copy()

    # For at-risk: all horizons positive
    # For borderline/healthy: only 24h may be positive (early warning)
    for i in range(n):
        if binary_labels[i] == 0:
            labels_1h[i] = 0
            labels_6h[i] = 0
            labels_24h[i] = 0
        else:
            # At-risk: high confidence at 1h for severe, lower for mild
            if profiles is not None and i < len(profiles):
                prof = profiles[i]
                if prof in ("shock", "hfref"):
                    labels_1h[i] = 1
                    labels_6h[i] = 1
                    labels_24h[i] = 1
                elif prof in ("afib_isolated", "hypovolemia", "sepsis_warm"):
                    labels_1h[i] = 0
                    labels_6h[i] = 1
                    labels_24h[i] = 1
                else:
                    labels_1h[i] = 0
                    labels_6h[i] = 0
                    labels_24h[i] = 1

    return np.stack([labels_1h, labels_6h, labels_24h], axis=-1).astype(np.float32)


def generate_synthetic_test(n=60, seed=99):
    """Generate wristppg synthetic test set with horizon labels."""
    from wristppg import WristPPGSimulator
    rng = np.random.default_rng(seed)
    ppgs, accels, labels, profiles = [], [], [], []

    at_risk_profiles = ["shock", "hfref", "afib_isolated", "hypovolemia", "sepsis_warm"]

    for _ in range(n // 2):
        try:
            sim = WristPPGSimulator(seed=int(rng.integers(0, 2**31)))
            result = sim.generate(profile="healthy", duration_s=60.0, activity="rest")
            ppgs.append(result.ppg[:PPG_LENGTH])
            accels.append(result.accel[:PPG_LENGTH])
            labels.append(0)
            profiles.append("healthy")
        except Exception:
            continue

    for _ in range(n // 2):
        try:
            sim = WristPPGSimulator(seed=int(rng.integers(0, 2**31)))
            prof = rng.choice(at_risk_profiles)
            result = sim.generate(profile=prof, duration_s=60.0, activity="rest")
            ppgs.append(result.ppg[:PPG_LENGTH])
            accels.append(result.accel[:PPG_LENGTH])
            labels.append(1)
            profiles.append(prof)
        except Exception:
            continue

    labels_arr = np.array(labels, dtype=np.float32)
    horizon_labels = make_horizon_labels(labels_arr, profiles)
    return ppgs, accels, labels_arr, horizon_labels


def extract_features_for_signal(ppg, accel=None, fs=25):
    """Extract combined PPG + accel features."""
    feats = extract_ppg_features(ppg, fs=fs)
    if accel is not None and accel.ndim == 2 and accel.shape[-1] == 3:
        accel_feats = extract_accel_features(accel, fs=fs)
        feats.update(accel_feats)
    return feats


def plot_training_curves(history, out_dir):
    """Plot and save training curves for multi-horizon model."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss
    axes[0, 0].plot(history.history["loss"], label="Train", linewidth=2)
    axes[0, 0].plot(history.history["val_loss"], label="Val", linewidth=2)
    axes[0, 0].set_title("Total Loss", fontsize=14)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Per-horizon AUC
    for i, h in enumerate(["1h", "6h", "24h"]):
        key = f"horizon_{h}_auc"
        val_key = f"val_horizon_{h}_auc"
        row, col = (0, 1) if i < 2 else (1, 0)
        if key in history.history:
            axes[row, col].plot(history.history[key], label="Train", linewidth=2)
            axes[row, col].plot(history.history[val_key], label="Val", linewidth=2)
        axes[row, col].set_title(f"AUC ({h})", fontsize=14)
        axes[row, col].set_xlabel("Epoch")
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)

    # Per-horizon loss
    for i, h in enumerate(["1h", "6h", "24h"]):
        key = f"horizon_{h}_loss"
        val_key = f"val_horizon_{h}_loss"
        row, col = (1, 1) if i == 0 else ((1, 2) if i == 1 else (0, 2))
        if key in history.history:
            axes[row, col].plot(history.history[key], label="Train", linewidth=2)
            axes[row, col].plot(history.history[val_key], label="Val", linewidth=2)
        axes[row, col].set_title(f"Loss ({h})", fontsize=14)
        axes[row, col].set_xlabel("Epoch")
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)

    plt.suptitle(f"CVD {VERSION.upper()} — Training Curves (Multi-Horizon)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved training curves to %s", out_dir / "training_curves.png")


def plot_roc_curves_per_horizon(y_true_dict, y_prob_dict, out_dir, label="test"):
    """Plot ROC curves for each horizon on one figure."""
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#FF9800", "#F44336"]
    for (h, y_true, y_prob), color in zip(
        [("1h", y_true_dict["1h"], y_prob_dict["1h"]),
         ("6h", y_true_dict["6h"], y_prob_dict["6h"]),
         ("24h", y_true_dict["24h"], y_prob_dict["24h"])], colors):
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, linewidth=2, color=color, label=f"{h} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"CVD {VERSION.upper()} — ROC Curves ({label})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"roc_curves_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()


def train_v13():
    logger.info("=" * 60)
    logger.info(f"TRAINING {VERSION.upper()} — Multi-Horizon + Accelerometer Watch Model")
    logger.info("=" * 60)

    # [1] Generate wristppg synthetic training data
    logger.info("\n[1/8] Generating wristppg synthetic training data...")
    synth_ppgs, synth_accels, synth_feats, synth_labels, synth_sev = \
        generate_wristppg_synthetic(n_healthy=50, n_at_risk=50, n_borderline=20, seed=42)

    # Extract features for synthetic data
    synth_feat_dicts = []
    for i in range(len(synth_ppgs)):
        feats = extract_features_for_signal(synth_ppgs[i], synth_accels[i])
        feats["base_hr"] = synth_feats[i].get("base_hr", 70.0)
        for k, v in synth_feats[i].items():
            if k.startswith("wppg_"):
                feats[k] = v
        synth_feat_dicts.append(feats)

    # [2] Load real data
    logger.info("\n[2/8] Loading real data by patient...")
    patient_groups = load_real_data_by_patient()
    train_p, val_p, test_p = patient_level_split(patient_groups)

    train_sigs, train_accels, train_feats_real, y_train_real = flatten_patients(patient_groups, train_p)
    val_sigs, val_accels, val_feats_real, y_val_real = flatten_patients(patient_groups, val_p)
    test_sigs, test_accels, test_feats_real, y_test_real = flatten_patients(patient_groups, test_p)

    train_feat_dicts_real = [extract_features_for_signal(s, a) for s, a in zip(train_sigs, train_accels)]
    val_feat_dicts_real = [extract_features_for_signal(s, a) for s, a in zip(val_sigs, val_accels)]
    test_feat_dicts_real = [extract_features_for_signal(s, a) for s, a in zip(test_sigs, test_accels)]

    # [3] Combine real train + synthetic
    logger.info("\n[3/8] Combining real + synthetic training data...")
    all_train_sigs = train_sigs + synth_ppgs
    all_train_accels = train_accels + synth_accels
    all_train_feats = train_feat_dicts_real + synth_feat_dicts
    all_train_labels = np.concatenate([y_train_real, synth_labels])

    # Derive multi-horizon labels
    synth_profiles_list = ["healthy"] * 300 + ["shock"] * 50 + ["hfref"] * 50 + ["afib_isolated"] * 50 + \
                          ["hypovolemia"] * 50 + ["sepsis_warm"] * 50 + ["hypertension"] * 25 + \
                          ["diabetes"] * 25 + ["aging"] * 25 + ["hfpef"] * 25 + ["pad"] * 25 + \
                          ["arterial_stiffness_isolated"] * 25 + ["hypertension"] * 25 + \
                          ["diabetes"] * 25 + ["aging"] * 25 + ["hfpef"] * 25
    # Pad to match length
    while len(synth_profiles_list) < len(synth_labels):
        synth_profiles_list.append("hypertension")
    real_profiles_list = ["CONTROL" if l == 0 else "EVENT" for l in y_train_real]
    all_profiles = real_profiles_list + synth_profiles_list[:len(synth_labels)]
    y_train_horizon = make_horizon_labels(all_train_labels, all_profiles)

    # Unified feature columns
    all_feat_dicts = all_train_feats + val_feat_dicts_real + test_feat_dicts_real
    feature_cols = sorted(set().union(*[f.keys() for f in all_feat_dicts]))
    logger.info("Unified feature columns: %d", len(feature_cols))

    # [4] Build arrays
    logger.info("\n[4/8] Building arrays...")
    X_train_ppg, X_train_accel, X_train_feat = build_arrays(
        all_train_sigs, all_train_accels, all_train_feats, feature_cols)
    X_val_ppg, X_val_accel, X_val_feat = build_arrays(
        val_sigs, val_accels, val_feat_dicts_real, feature_cols)
    X_test_ppg, X_test_accel, X_test_feat = build_arrays(
        test_sigs, test_accels, test_feat_dicts_real, feature_cols)

    n_biodata = 9
    X_train_bio = build_biodata_array(len(X_train_ppg))
    X_val_bio = build_biodata_array(len(X_val_ppg), seed=43)
    X_test_bio = build_biodata_array(len(X_test_ppg), seed=44)

    # Val horizon labels
    y_val_horizon = make_horizon_labels(y_val_real)

    logger.info("Train: %d | Val: %d | Test: %d", len(all_train_labels), len(y_val_real), len(y_test_real))

    # [5] Build model
    logger.info("\n[5/8] Building model...")
    from src.model_v13 import build_v13
    model = build_v13(
        ppg_input_shape=(PPG_LENGTH, 1),
        accel_input_shape=(PPG_LENGTH, 3),
        hrv_feature_dim=X_train_feat.shape[1],
        biodata_dim=n_biodata,
    )
    model.summary(print_fn=logger.info)

    n_h = int((all_train_labels == 0).sum())
    n_e = int((all_train_labels == 1).sum())
    cw = {0: (n_h + n_e) / (2 * n_h), 1: (n_h + n_e) / (2 * n_e)}

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
        loss=["binary_crossentropy", "binary_crossentropy", "binary_crossentropy"],
        loss_weights=[1.0, 1.0, 1.0],
        metrics=[
            [tf.keras.metrics.AUC(name="auc")],
            [tf.keras.metrics.AUC(name="auc")],
            [tf.keras.metrics.AUC(name="auc")],
        ],
    )

    # [6] Train
    logger.info("\n[6/8] Training...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                          restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(OUT_DIR / "best_model.keras"),
                                            monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7,
                                              min_lr=1e-6),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(LOG_DIR), histogram_freq=1, write_graph=True,
            write_images=True, update_freq="epoch", profile_batch=0),
        tf.keras.callbacks.CSVLogger(str(OUT_DIR / "training_log.csv"), append=False),
    ]

    history = model.fit(
        {"ppg_input": X_train_ppg, "accel_input": X_train_accel,
         "feature_input": X_train_feat, "biodata_input": X_train_bio},
        [y_train_horizon[:, 0], y_train_horizon[:, 1], y_train_horizon[:, 2]],
        validation_data=(
            {"ppg_input": X_val_ppg, "accel_input": X_val_accel,
             "feature_input": X_val_feat, "biodata_input": X_val_bio},
            [y_val_horizon[:, 0], y_val_horizon[:, 1], y_val_horizon[:, 2]],
        ),
        epochs=60, batch_size=32, callbacks=callbacks,
    )

    # [7] Evaluate on REAL test set
    logger.info("\n[7/8] Evaluating on HELD-OUT REAL test set...")
    preds_real = model(
        {"ppg_input": X_test_ppg, "accel_input": X_test_accel,
         "feature_input": X_test_feat, "biodata_input": X_test_bio},
        training=False,
    )
    preds_real = [np.array(p).flatten() for p in preds_real]

    from sklearn.metrics import roc_auc_score

    metrics_real = {}
    for i, h in enumerate(["1h", "6h", "24h"]):
        y_prob = preds_real[i]
        y_true = y_test_real  # binary labels as proxy for real data
        if len(np.unique(y_true)) > 1:
            metrics_real[f"auroc_{h}"] = float(roc_auc_score(y_true, y_prob))
        else:
            metrics_real[f"auroc_{h}"] = float('nan')
    logger.info("  REAL TEST: AUROC 1h=%.4f 6h=%.4f 24h=%.4f",
                metrics_real["auroc_1h"], metrics_real["auroc_6h"], metrics_real["auroc_24h"])

    # Plot per-horizon ROC
    y_prob_dict = {"1h": preds_real[0], "6h": preds_real[1], "24h": preds_real[2]}
    y_true_dict = {"1h": y_test_real, "6h": y_test_real, "24h": y_test_real}
    plot_roc_curves_per_horizon(y_true_dict, y_prob_dict, GRAPHS_DIR, label="real_test")
    plot_training_curves(history, GRAPHS_DIR)

    # [8] Evaluate on wristppg synthetic test set
    logger.info("\n[8/8] Evaluating on wristppg SYNTHETIC test set...")
    synth_test_ppgs, synth_test_accels, synth_test_labels, synth_test_horizon = \
        generate_synthetic_test(n=60, seed=99)
    synth_test_feats = [extract_features_for_signal(p, a) for p, a in zip(synth_test_ppgs, synth_test_accels)]

    X_synth_test_ppg, X_synth_test_accel, X_synth_test_feat = build_arrays(
        synth_test_ppgs, synth_test_accels, synth_test_feats, feature_cols)
    X_synth_test_bio = build_biodata_array(len(X_synth_test_ppg), seed=100)

    preds_synth = model(
        {"ppg_input": X_synth_test_ppg, "accel_input": X_synth_test_accel,
         "feature_input": X_synth_test_feat, "biodata_input": X_synth_test_bio},
        training=False,
    )
    preds_synth = [np.array(p).flatten() for p in preds_synth]

    metrics_synth = {}
    for i, h in enumerate(["1h", "6h", "24h"]):
        y_prob = preds_synth[i]
        y_true = synth_test_horizon[:, i]
        if len(np.unique(y_true)) > 1:
            metrics_synth[f"auroc_{h}"] = float(roc_auc_score(y_true, y_prob))
        else:
            metrics_synth[f"auroc_{h}"] = float('nan')
    logger.info("  SYNTHETIC TEST: AUROC 1h=%.4f 6h=%.4f 24h=%.4f",
                metrics_synth["auroc_1h"], metrics_synth["auroc_6h"], metrics_synth["auroc_24h"])

    y_prob_dict_s = {"1h": preds_synth[0], "6h": preds_synth[1], "24h": preds_synth[2]}
    y_true_dict_s = {h: synth_test_horizon[:, i] for i, h in enumerate(["1h", "6h", "24h"])}
    plot_roc_curves_per_horizon(y_true_dict_s, y_prob_dict_s, GRAPHS_DIR, label="synth_test")

    # Save everything
    model.save(str(OUT_DIR / "final_model.keras"))
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(OUT_DIR / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    config = {
        "version": VERSION,
        "description": "Multi-horizon (1h/6h/24h) cardiac arrest prediction with accelerometer",
        "ppg_length": PPG_LENGTH, "sampling_rate_hz": FS_TARGET,
        "feature_columns": feature_cols,
        "architecture": {
            "ppg_branch": "ResNet 1D-CNN (16->32->64) + BiLSTM(32)",
            "accel_branch": "ResNet 1D-CNN (16->32->64) + BiLSTM(16)",
            "cross_attention": "PPG queries accel (32-dim)",
            "outputs": "3 sigmoid heads (1h, 6h, 24h)",
            "total_params": model.count_params(),
        },
        "training": {
            "dataset": "wristppg_synthetic + MIMIC/MMASH real",
            "loss": "sum of 3 binary crossentropies",
            "epochs_trained": len(history.history["loss"]),
        },
        "performance_real_test": metrics_real,
        "performance_synth_test": metrics_synth,
    }

    with open(OUT_DIR / "config.yaml", "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    with open(OUT_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    logger.info("\n" + "=" * 60)
    logger.info(f"{VERSION.upper()} SUMMARY")
    logger.info("=" * 60)
    logger.info("  Real test:     AUROC 1h=%.4f 6h=%.4f 24h=%.4f",
                metrics_real["auroc_1h"], metrics_real["auroc_6h"], metrics_real["auroc_24h"])
    logger.info("  Synthetic test: AUROC 1h=%.4f 6h=%.4f 24h=%.4f",
                metrics_synth["auroc_1h"], metrics_synth["auroc_6h"], metrics_synth["auroc_24h"])
    logger.info("  Saved to %s", OUT_DIR)
    return model, metrics_real, metrics_synth, history


if __name__ == "__main__":
    train_v13()
