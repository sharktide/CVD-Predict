#!/usr/bin/env python3
"""Train CVD Watch Model v14 — Risk + Severity with accelerometer.

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

VERSION = "v14"
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

    # Total loss
    axes[0, 0].plot(history.history["loss"], label="Train", linewidth=2)
    axes[0, 0].plot(history.history["val_loss"], label="Val", linewidth=2)
    axes[0, 0].set_title("Total Loss", fontsize=14)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Risk AUROC
    risk_key = "risk_output_auc"
    val_risk_key = "val_risk_output_auc"
    if risk_key in history.history:
        axes[0, 1].plot(history.history[risk_key], label="Train", linewidth=2)
    if val_risk_key in history.history:
        axes[0, 1].plot(history.history[val_risk_key], label="Val", linewidth=2)
    axes[0, 1].set_title("Risk AUROC", fontsize=14)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Risk loss
    risk_loss_key = "risk_output_loss"
    val_risk_loss_key = "val_risk_output_loss"
    if risk_loss_key in history.history:
        axes[1, 0].plot(history.history[risk_loss_key], label="Train", linewidth=2)
    if val_risk_loss_key in history.history:
        axes[1, 0].plot(history.history[val_risk_loss_key], label="Val", linewidth=2)
    axes[1, 0].set_title("Risk Loss (BCE)", fontsize=14)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Severity accuracy
    sev_key = "severity_output_accuracy"
    val_sev_key = "val_severity_output_accuracy"
    if sev_key in history.history:
        axes[1, 1].plot(history.history[sev_key], label="Train", linewidth=2)
    if val_sev_key in history.history:
        axes[1, 1].plot(history.history[val_sev_key], label="Val", linewidth=2)
    axes[1, 1].set_title("Severity Accuracy", fontsize=14)
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


def train_v14():
    logger.info("=" * 60)
    logger.info(f"TRAINING {VERSION.upper()} — Risk + Severity + Accelerometer Watch Model")
    logger.info("=" * 60)

    # [1] Generate wristppg synthetic training data
    logger.info("\n[1/8] Generating wristppg synthetic training data...")
    synth_ppgs, synth_accels, synth_feats, synth_labels, synth_sev = \
        generate_wristppg_synthetic(n_healthy=50, n_at_risk=50, n_borderline=20, seed=42)

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

    # [3] Combine
    logger.info("\n[3/8] Combining real + synthetic training data...")
    all_train_sigs = train_sigs + synth_ppgs
    all_train_accels = train_accels + synth_accels
    all_train_feats = train_feat_dicts_real + synth_feat_dicts
    all_train_labels = np.concatenate([y_train_real, synth_labels])

    # Severity labels: for real data, use 0 for healthy, 2 for at-risk as proxy
    y_train_severity_real = np.where(y_train_real == 0, 0, 2).astype(np.int32)
    all_train_severity = np.concatenate([y_train_severity_real, synth_sev])

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

    y_val_severity = np.where(y_val_real == 0, 0, 2).astype(np.int32)

    logger.info("Train: %d | Val: %d | Test: %d", len(all_train_labels), len(y_val_real), len(y_test_real))
    logger.info("Severity distribution (train): %s",
                {i: int((all_train_severity == i).sum()) for i in range(4)})

    # [5] Build model
    logger.info("\n[5/8] Building model...")
    from src.model_v14 import build_v14
    model = build_v14(
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
        loss=["binary_crossentropy", "sparse_categorical_crossentropy"],
        loss_weights=[1.0, 0.5],
        metrics=[
            [tf.keras.metrics.AUC(name="auc")],
            [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
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
        [all_train_labels, all_train_severity],
        validation_data=(
            {"ppg_input": X_val_ppg, "accel_input": X_val_accel,
             "feature_input": X_val_feat, "biodata_input": X_val_bio},
            [y_val_real, y_val_severity],
        ),
        epochs=60, batch_size=32, callbacks=callbacks,
    )

    # [7] Evaluate
    logger.info("\n[7/8] Evaluating on REAL test set...")
    preds_real = model(
        {"ppg_input": X_test_ppg, "accel_input": X_test_accel,
         "feature_input": X_test_feat, "biodata_input": X_test_bio},
        training=False,
    )
    y_prob_real = np.array(preds_real[0]).flatten()
    y_sev_pred_real = np.array(preds_real[1])  # (N, 4)

    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                  confusion_matrix, classification_report)

    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.005):
        f = f1_score(y_test_real, (y_prob_real >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    y_pred_real = (y_prob_real >= best_t).astype(int)
    y_sev_classes = np.argmax(y_sev_pred_real, axis=-1)
    y_sev_true = np.where(y_test_real == 0, 0, 2).astype(np.int32)

    metrics_real = {
        "risk_auroc": float(roc_auc_score(y_test_real, y_prob_real)) if len(np.unique(y_test_real)) > 1 else float('nan'),
        "risk_f1": float(best_f1),
        "severity_accuracy": float(accuracy_score(y_sev_true, y_sev_classes)),
        "severity_f1_macro": float(f1_score(y_sev_true, y_sev_classes, average="macro", zero_division=0)),
    }
    logger.info("  REAL TEST: Risk AUROC=%.4f F1=%.4f | Severity Acc=%.4f F1=%.4f",
                metrics_real["risk_auroc"], metrics_real["risk_f1"],
                metrics_real["severity_accuracy"], metrics_real["severity_f1_macro"])

    # Plot
    plot_roc_curve(y_test_real, y_prob_real, GRAPHS_DIR, label="real_test")
    plot_training_curves(history, GRAPHS_DIR)

    # Severity confusion matrix
    cm_sev = confusion_matrix(y_sev_true, y_sev_classes, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_sev, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"{VERSION.upper()} — Severity Confusion Matrix (Real)", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)
    classes = ["None", "Mild", "Moderate", "Severe"]
    ax.set_xticks(range(4))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(4))
    ax.set_yticklabels(classes)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm_sev[i, j]), ha="center", va="center",
                    color="white" if cm_sev[i, j] > cm_sev.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "severity_cm_real.png", dpi=150, bbox_inches="tight")
    plt.close()

    # [8] Synthetic test
    logger.info("\n[8/8] Evaluating on wristppg SYNTHETIC test set...")
    synth_test_ppgs, synth_test_accels, synth_test_labels = generate_synthetic_test(n=60, seed=99)
    synth_test_feats = [extract_features_for_signal(p, a) for p, a in zip(synth_test_ppgs, synth_test_accels)]

    X_synth_test_ppg, X_synth_test_accel, X_synth_test_feat = build_arrays(
        synth_test_ppgs, synth_test_accels, synth_test_feats, feature_cols)
    X_synth_test_bio = build_biodata_array(len(X_synth_test_ppg), seed=100)

    preds_synth = model(
        {"ppg_input": X_synth_test_ppg, "accel_input": X_synth_test_accel,
         "feature_input": X_synth_test_feat, "biodata_input": X_synth_test_bio},
        training=False,
    )
    y_prob_synth = np.array(preds_synth[0]).flatten()
    y_sev_pred_synth = np.argmax(np.array(preds_synth[1]), axis=-1)

    best_f1_s, best_t_s = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.005):
        f = f1_score(synth_test_labels, (y_prob_synth >= t).astype(int), zero_division=0)
        if f > best_f1_s:
            best_f1_s, best_t_s = f, t

    metrics_synth = {
        "risk_auroc": float(roc_auc_score(synth_test_labels, y_prob_synth)),
        "risk_f1": float(best_f1_s),
    }
    logger.info("  SYNTHETIC TEST: Risk AUROC=%.4f F1=%.4f",
                metrics_synth["risk_auroc"], metrics_synth["risk_f1"])

    plot_roc_curve(synth_test_labels, y_prob_synth, GRAPHS_DIR, label="synth_test")

    # Save
    model.save(str(OUT_DIR / "final_model.keras"))
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(OUT_DIR / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    config = {
        "version": VERSION,
        "description": "Risk + Severity classification with accelerometer",
        "ppg_length": PPG_LENGTH, "sampling_rate_hz": FS_TARGET,
        "feature_columns": feature_cols,
        "architecture": {
            "ppg_branch": "ResNet 1D-CNN (16->32->64) + BiLSTM(32)",
            "accel_branch": "ResNet 1D-CNN (16->32->64) + BiLSTM(16)",
            "cross_attention": "PPG queries accel (32-dim)",
            "outputs": "risk (sigmoid) + severity (4-class softmax)",
            "total_params": model.count_params(),
        },
        "training": {
            "loss": "BCE(risk) + 0.5 * CCE(severity)",
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
    logger.info("  Real test:     Risk AUROC=%.4f F1=%.4f | Sev Acc=%.4f",
                metrics_real["risk_auroc"], metrics_real["risk_f1"], metrics_real["severity_accuracy"])
    logger.info("  Synthetic test: Risk AUROC=%.4f F1=%.4f",
                metrics_synth["risk_auroc"], metrics_synth["risk_f1"])
    logger.info("  Saved to %s", OUT_DIR)
    return model, metrics_real, metrics_synth, history


if __name__ == "__main__":
    train_v14()
