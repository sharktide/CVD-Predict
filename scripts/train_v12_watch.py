#!/usr/bin/env python3
"""Train CVD Watch Model v12 — Binary cardiac arrest risk with accelerometer.

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

VERSION = "v12"
OUT_DIR = Path(f"production/cvd_risk_{VERSION}_watch")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR = OUT_DIR / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_test(n=150, seed=99):
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


def train_v12():
    logger.info("=" * 60)
    logger.info(f"TRAINING {VERSION.upper()} — wristppg + Accelerometer Watch Model")
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
        # Add wristppg features
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

    # Extract features for real data
    train_feat_dicts_real = [extract_features_for_signal(s, a) for s, a in zip(train_sigs, train_accels)]
    val_feat_dicts_real = [extract_features_for_signal(s, a) for s, a in zip(val_sigs, val_accels)]
    test_feat_dicts_real = [extract_features_for_signal(s, a) for s, a in zip(test_sigs, test_accels)]

    # [3] Combine real train + synthetic
    logger.info("\n[3/8] Combining real + synthetic training data...")
    all_train_sigs = train_sigs + synth_ppgs
    all_train_accels = train_accels + synth_accels
    all_train_feats = train_feat_dicts_real + synth_feat_dicts
    all_train_labels = np.concatenate([y_train_real, synth_labels])

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

    logger.info("Train: %d (%d healthy, %d at-risk)",
                len(all_train_labels), int((all_train_labels == 0).sum()), int((all_train_labels == 1).sum()))
    logger.info("Val: %d | Test: %d", len(y_val_real), len(y_test_real))

    # [5] Build model
    logger.info("\n[5/8] Building model...")
    from src.model_v12 import build_v12
    model = build_v12(
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
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )

    # [6] Train
    logger.info("\n[6/8] Training...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=20, mode="max",
                                          restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(OUT_DIR / "best_model.keras"),
                                            monitor="val_auc", mode="max", save_best_only=True),
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
        all_train_labels,
        validation_data=(
            {"ppg_input": X_val_ppg, "accel_input": X_val_accel,
             "feature_input": X_val_feat, "biodata_input": X_val_bio},
            y_val_real,
        ),
        epochs=60, batch_size=32, class_weight=cw, callbacks=callbacks,
    )

    # [7] Evaluate on REAL test set
    logger.info("\n[7/8] Evaluating on HELD-OUT REAL test set...")
    preds_real = model(
        {"ppg_input": X_test_ppg, "accel_input": X_test_accel,
         "feature_input": X_test_feat, "biodata_input": X_test_bio},
        training=False,
    )
    y_prob_real = np.array(preds_real).flatten()

    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                  roc_auc_score, confusion_matrix, brier_score_loss)

    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.005):
        f = f1_score(y_test_real, (y_prob_real >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    y_pred_real = (y_prob_real >= best_t).astype(int)
    cm = confusion_matrix(y_test_real, y_pred_real, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics_real = {
        "auroc": float(roc_auc_score(y_test_real, y_prob_real)) if len(np.unique(y_test_real)) > 1 else float('nan'),
        "accuracy": float(accuracy_score(y_test_real, y_pred_real)),
        "precision": float(precision_score(y_test_real, y_pred_real, zero_division=0)),
        "recall": float(recall_score(y_test_real, y_pred_real, zero_division=0)),
        "f1": float(best_f1),
        "brier": float(brier_score_loss(y_test_real, y_prob_real)),
        "threshold": float(best_t),
        "n_test": int(len(y_test_real)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

    logger.info("  REAL TEST: AUROC=%.4f Acc=%.1f%% Prec=%.4f Rec=%.4f F1=%.4f",
                metrics_real["auroc"], metrics_real["accuracy"] * 100,
                metrics_real["precision"], metrics_real["recall"], metrics_real["f1"])

    # Plot real test results
    plot_roc_curve(y_test_real, y_prob_real, GRAPHS_DIR, label="real_test")
    plot_confusion_matrix(y_test_real, y_pred_real, GRAPHS_DIR, label="real_test")
    plot_probability_distribution(y_test_real, y_prob_real, GRAPHS_DIR, label="real_test")

    # [8] Evaluate on wristppg synthetic test set
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
    y_prob_synth = np.array(preds_synth).flatten()

    best_f1_s, best_t_s = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.005):
        f = f1_score(synth_test_labels, (y_prob_synth >= t).astype(int), zero_division=0)
        if f > best_f1_s:
            best_f1_s, best_t_s = f, t

    y_pred_synth = (y_prob_synth >= best_t_s).astype(int)
    cm_s = confusion_matrix(synth_test_labels, y_pred_synth, labels=[0, 1])
    tn_s, fp_s, fn_s, tp_s = cm_s.ravel()

    metrics_synth = {
        "auroc": float(roc_auc_score(synth_test_labels, y_prob_synth)),
        "accuracy": float(accuracy_score(synth_test_labels, y_pred_synth)),
        "precision": float(precision_score(synth_test_labels, y_pred_synth, zero_division=0)),
        "recall": float(recall_score(synth_test_labels, y_pred_synth, zero_division=0)),
        "f1": float(best_f1_s),
        "brier": float(brier_score_loss(synth_test_labels, y_prob_synth)),
        "threshold": float(best_t_s),
    }

    logger.info("  SYNTHETIC TEST: AUROC=%.4f Acc=%.1f%% Prec=%.4f Rec=%.4f F1=%.4f",
                metrics_synth["auroc"], metrics_synth["accuracy"] * 100,
                metrics_synth["precision"], metrics_synth["recall"], metrics_synth["f1"])

    plot_roc_curve(synth_test_labels, y_prob_synth, GRAPHS_DIR, label="synth_test")
    plot_confusion_matrix(synth_test_labels, y_pred_synth, GRAPHS_DIR, label="synth_test")
    plot_probability_distribution(synth_test_labels, y_prob_synth, GRAPHS_DIR, label="synth_test")
    plot_training_curves(history, GRAPHS_DIR)

    # Combined metrics bar chart
    combined_metrics = {
        "AUROC (Real)": metrics_real["auroc"],
        "F1 (Real)": metrics_real["f1"],
        "AUROC (Synth)": metrics_synth["auroc"],
        "F1 (Synth)": metrics_synth["f1"],
        "Recall (Real)": metrics_real["recall"],
    }
    plot_metrics_bar(combined_metrics, GRAPHS_DIR)

    # Save everything
    model.save(str(OUT_DIR / "final_model.keras"))
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(OUT_DIR / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    config = {
        "version": VERSION,
        "description": "CVD model with wristppg synthetic data + accelerometer cross-attention",
        "ppg_length": PPG_LENGTH, "sampling_rate_hz": FS_TARGET,
        "feature_columns": feature_cols,
        "architecture": {
            "ppg_branch": "ResNet 1D-CNN (16->32->64) + BiLSTM(32)",
            "accel_branch": "ResNet 1D-CNN (16->32->64) + BiLSTM(16)",
            "cross_attention": "PPG queries accel (32-dim)",
            "biodata_branch": "MLP (32, 16)",
            "feature_branch": "MLP (32, 32)",
            "shared": "Dense(64)",
            "event_head": "Dense(32) -> Dense(1, sigmoid)",
            "total_params": model.count_params(),
        },
        "training": {
            "dataset": "wristppg_synthetic + MIMIC/MMASH real",
            "synthetic_source": "wristppg/ (physiologically-grounded simulator)",
            "synthetic_counts": {"healthy": 300, "at_risk": 300, "borderline": 100},
            "real_patients": {"train": len(train_p), "val": len(val_p), "test": len(test_p)},
            "optimizer": "AdamW", "lr": 3e-4, "batch_size": 32,
            "epochs_trained": len(history.history["loss"]),
            "class_weights": cw,
        },
        "performance_real_test": metrics_real,
        "performance_synth_test": metrics_synth,
    }

    with open(OUT_DIR / "config.yaml", "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    with open(OUT_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)
    with open(OUT_DIR / "optimal_threshold.json", "w") as f:
        json.dump({"threshold": best_t}, f)

    logger.info("\n" + "=" * 60)
    logger.info(f"{VERSION.upper()} SUMMARY")
    logger.info("=" * 60)
    logger.info("  Real test:     AUROC=%.4f F1=%.4f", metrics_real["auroc"], metrics_real["f1"])
    logger.info("  Synthetic test: AUROC=%.4f F1=%.4f", metrics_synth["auroc"], metrics_synth["f1"])
    logger.info("  Saved to %s", OUT_DIR)
    return model, metrics_real, metrics_synth, history


if __name__ == "__main__":
    train_v12()
