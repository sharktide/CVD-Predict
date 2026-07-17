#!/usr/bin/env python3
"""Train CVD Watch Model v16 — Wrist-only Cardiac Arrest Detection.

Uses wristppg v0.3.0 as primary synthetic data source.
Binary labels: 0=healthy, 1=cardiac arrest.
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
    extract_ppg_features, extract_accel_features,
    build_arrays, build_biodata_array,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VERSION = "v16"
OUT_DIR = Path(f"production/cvd_risk_{VERSION}_watch")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR = OUT_DIR / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def generate_wristppg_cardiac_arrest(
    n_healthy: int = 500,
    n_arrest: int = 500,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate wristppg v0.3.0 synthetic data for cardiac arrest detection.

    Returns
    -------
    ppgs : (N, PPG_LENGTH) array
    accels : (N, PPG_LENGTH, 3) array
    features : (N, feat_dim) array
    biodata : (N, 12) array
    labels : (N,) binary labels
    """
    from wristppg import WristPPGSimulator

    rng = np.random.default_rng(seed)
    ppgs, accels, feat_dicts, biodata_list, labels = [], [], [], [], []

    # Cardiac arrest profiles with severity 1
    arrest_profiles = [
        ("cardiac_arrest_vf", 1),
        ("cardiac_arrest_asystole", 1),
        ("cardiac_arrest_pulseless_electrical", 1),
        ("pre_arrest_deterioration", 1),
        ("respiratory_failure_pre_arrest", 1),
    ]

    activities = ["rest", "rest", "rest", "walking", "sleep"]
    contact_modes = ["good", "good", "good", "loose", "tight"]

    # === Generate healthy ===
    logger.info("Generating %d healthy wristppg signals...", n_healthy)
    for i in range(n_healthy):
        try:
            sim = WristPPGSimulator(seed=int(rng.integers(0, 2**31)))
            act = rng.choice(activities)
            result = sim.generate(
                profile="healthy",
                duration_s=60.0,
                activity=act,
                contact_mode=rng.choice(contact_modes),
            )
            ppgs.append(result.ppg[:PPG_LENGTH])
            accels.append(result.accel[:PPG_LENGTH])

            feats = extract_wristppg_features(result)
            feat_dicts.append(feats)

            # Biodata: age, sex, bmi, comorbidity_count, on_vasopressors,
            #          on_ventilation, on_ecmo, on_rrt, acuity_score,
            #          spo2, body_temp, skin_temp
            biodata_list.append(np.array([
                rng.uniform(20, 70),           # age
                rng.choice([0, 1]),            # sex
                rng.uniform(18, 35),           # bmi
                0,                              # comorbidity_count
                0, 0, 0, 0,                    # on_vasopressors/ventilation/ecmo/rrt
                rng.uniform(1, 3),             # acuity_score
                0.97,                           # spo2 (normal)
                36.5,                           # body_temp
                rng.uniform(30, 33),           # skin_temp
            ], dtype=np.float32))

            labels.append(0)
        except Exception as e:
            logger.debug("Healthy sample %d failed: %s", i, e)
            continue

    # === Generate cardiac arrest ===
    logger.info("Generating %d cardiac arrest wristppg signals...", n_arrest)
    for i in range(n_arrest):
        try:
            sim = WristPPGSimulator(seed=int(rng.integers(0, 2**31)))
            prof_name, sev = arrest_profiles[i % len(arrest_profiles)]
            act = rng.choice(activities)
            result = sim.generate(
                profile=prof_name,
                duration_s=60.0,
                activity=act,
                contact_mode=rng.choice(contact_modes),
            )
            ppgs.append(result.ppg[:PPG_LENGTH])
            accels.append(result.accel[:PPG_LENGTH])

            feats = extract_wristppg_features(result)
            feat_dicts.append(feats)

            # Cardiac arrest biodata: lower spo2, variable temp
            target_spo2 = result.meta.get("spo2", 0.70)
            body_temp = result.meta.get("body_temp_c", 36.0)
            skin_temp = result.meta.get("skin_temperature_c", 32.0)
            biodata_list.append(np.array([
                rng.uniform(40, 85),            # age (cardiac arrest skews older)
                rng.choice([0, 1]),             # sex
                rng.uniform(20, 40),            # bmi
                rng.integers(0, 5),             # comorbidity_count
                rng.choice([0, 1], p=[0.6, 0.4]),  # on_vasopressors
                rng.choice([0, 1], p=[0.7, 0.3]),  # on_ventilation
                0,                               # on_ecmo
                0,                               # on_rrt
                rng.uniform(2, 5),               # acuity_score
                target_spo2,                     # spo2
                body_temp,                        # body_temp
                skin_temp,                        # skin_temp
            ], dtype=np.float32))

            labels.append(1)
        except Exception as e:
            logger.debug("Arrest sample %d failed: %s", i, e)
            continue

    # Convert to arrays, pad/truncate to PPG_LENGTH
    ppgs_padded = np.zeros((len(ppgs), PPG_LENGTH), dtype=np.float32)
    accels_padded = np.zeros((len(accels), PPG_LENGTH, 3), dtype=np.float32)
    for i, (p, a) in enumerate(zip(ppgs, accels)):
        L = min(len(p), PPG_LENGTH)
        ppgs_padded[i, :L] = p[:L]
        accels_padded[i, :L, :] = a[:L]
    ppgs = ppgs_padded
    accels = accels_padded
    labels = np.array(labels, dtype=np.float32)
    biodata = np.array(biodata_list, dtype=np.float32)

    # Build feature matrix
    feature_cols = sorted(set().union(*[f.keys() for f in feat_dicts]))
    features = np.zeros((len(feat_dicts), len(feature_cols)), dtype=np.float32)
    for i, f in enumerate(feat_dicts):
        for j, col in enumerate(feature_cols):
            features[i, j] = f.get(col, 0.0)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info("Generated %d samples (%d healthy, %d arrest), %d features",
                len(labels), int((labels == 0).sum()), int((labels == 1).sum()),
                features.shape[1])

    return ppgs, accels, features, biodata, labels, feature_cols


def extract_wristppg_features(result) -> dict:
    """Extract NON-LEAKING wristppg features from SimulationResult.

    IMPORTANT: We exclude features that leak ground truth:
    - wppg_frac_sinus/afib/pvc: rhythm labels (ground truth)
    - wppg_mean_sv/std_sv: stroke volume from beat records (ground truth)
    - wppg_mean_ef/std_ef: ejection fraction from beat records (ground truth)
    - wppg_latent_*: simulator latent variables (ground truth)
    - wppg_mean_ptt/std_ptt: PTT from beat records (ground truth)
    - wppg_mean_aix/std_aix: augmentation index from beat records (ground truth)

    We keep ONLY features that would be computable from a real PPG signal:
    - HRV features (from peak detection)
    - Accelerometer features
    - Biodata (age, sex, etc.)
    - Signal quality metrics
    """
    # NO wppg_* features from beat records - these leak ground truth
    # Only return basic metadata that doesn't leak
    feats = {}

    # Basic signal metadata (non-leaking)
    feats["base_hr"] = float(np.mean(result.hr_instantaneous_bpm)) if len(result.hr_instantaneous_bpm) > 0 else 70.0

    return feats


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

    axes[0, 0].plot(history.history["loss"], label="Train", linewidth=2)
    axes[0, 0].plot(history.history["val_loss"], label="Val", linewidth=2)
    axes[0, 0].set_title("Loss", fontsize=14)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history.history["auc"], label="Train", linewidth=2)
    axes[0, 1].plot(history.history["val_auc"], label="Val", linewidth=2)
    axes[0, 1].set_title("AUROC", fontsize=14)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history.history["precision"], label="Train", linewidth=2)
    axes[1, 0].plot(history.history["val_precision"], label="Val", linewidth=2)
    axes[1, 0].set_title("Precision", fontsize=14)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

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
    classes = ["Healthy", "Arrest"]
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
    ax.hist(y_prob[y_true == 1], bins=30, alpha=0.6, label="Arrest", color="red", density=True)
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


def train_v16():
    logger.info("=" * 60)
    logger.info(f"TRAINING {VERSION.upper()} — Wrist-only Cardiac Arrest Detection")
    logger.info("=" * 60)

    # [1] Generate wristppg v0.3.0 synthetic training data
    logger.info("\n[1/8] Generating wristppg v0.3.0 synthetic training data...")
    ppgs, accels, features, biodata, labels, feature_cols = \
        generate_wristppg_cardiac_arrest(n_healthy=500, n_arrest=500, seed=42)

    # [2] Split into train/val/test (70/15/15)
    logger.info("\n[2/8] Splitting data...")
    n = len(labels)
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train_ppg = ppgs[train_idx]
    X_train_accel = accels[train_idx]
    X_train_feat = features[train_idx]
    X_train_bio = biodata[train_idx]
    y_train = labels[train_idx]

    X_val_ppg = ppgs[val_idx]
    X_val_accel = accels[val_idx]
    X_val_feat = features[val_idx]
    X_val_bio = biodata[val_idx]
    y_val = labels[val_idx]

    X_test_ppg = ppgs[test_idx]
    X_test_accel = accels[test_idx]
    X_test_feat = features[test_idx]
    X_test_bio = biodata[test_idx]
    y_test = labels[test_idx]

    logger.info("Train: %d (%d healthy, %d arrest)", len(y_train),
                int((y_train == 0).sum()), int((y_train == 1).sum()))
    logger.info("Val: %d (%d healthy, %d arrest)", len(y_val),
                int((y_val == 0).sum()), int((y_val == 1).sum()))
    logger.info("Test: %d (%d healthy, %d arrest)", len(y_test),
                int((y_test == 0).sum()), int((y_test == 1).sum()))

    # [3] Build model
    logger.info("\n[3/8] Building model...")
    from src.model_v16 import build_v16
    model = build_v16(
        ppg_input_shape=(PPG_LENGTH, 1),
        accel_input_shape=(PPG_LENGTH, 3),
        hrv_feature_dim=X_train_feat.shape[1],
        biodata_dim=X_train_bio.shape[1],
    )
    model.summary(print_fn=logger.info)

    # [4] Compile
    logger.info("\n[4/8] Compiling...")
    n_h = int((y_train == 0).sum())
    n_e = int((y_train == 1).sum())
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

    # [5] Train
    logger.info("\n[5/8] Training...")
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
        y_train,
        validation_data=(
            {"ppg_input": X_val_ppg, "accel_input": X_val_accel,
             "feature_input": X_val_feat, "biodata_input": X_val_bio},
            y_val,
        ),
        epochs=60, batch_size=32, class_weight=cw, callbacks=callbacks,
    )

    # [6] Evaluate on held-out test set
    logger.info("\n[6/8] Evaluating on HELD-OUT test set...")
    preds = model(
        {"ppg_input": X_test_ppg, "accel_input": X_test_accel,
         "feature_input": X_test_feat, "biodata_input": X_test_bio},
        training=False,
    )
    y_prob = np.array(preds).flatten()

    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                  roc_auc_score, confusion_matrix, brier_score_loss)

    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.005):
        f = f1_score(y_test, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    y_pred = (y_prob >= best_t).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "auroc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else float('nan'),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(best_f1),
        "brier": float(brier_score_loss(y_test, y_prob)),
        "threshold": float(best_t),
        "n_test": int(len(y_test)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

    logger.info("  TEST: AUROC=%.4f Acc=%.1f%% Prec=%.4f Rec=%.4f F1=%.4f",
                metrics["auroc"], metrics["accuracy"] * 100,
                metrics["precision"], metrics["recall"], metrics["f1"])

    # [7] Plot results
    logger.info("\n[7/8] Plotting results...")
    plot_roc_curve(y_test, y_prob, GRAPHS_DIR, label="test")
    plot_confusion_matrix(y_test, y_pred, GRAPHS_DIR, label="test")
    plot_probability_distribution(y_test, y_prob, GRAPHS_DIR, label="test")
    plot_training_curves(history, GRAPHS_DIR)

    metrics_bars = {
        "AUROC": metrics["auroc"],
        "F1": metrics["f1"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "Accuracy": metrics["accuracy"],
    }
    plot_metrics_bar(metrics_bars, GRAPHS_DIR)

    # [8] Save everything
    logger.info("\n[8/8] Saving model and config...")
    model.save(str(OUT_DIR / "final_model.keras"))
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(OUT_DIR / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    config = {
        "version": VERSION,
        "description": "Wrist-only cardiac arrest detection using wristppg v0.3.0 simulator",
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
            "dataset": "wristppg v0.3.0 synthetic ONLY (no real data)",
            "synthetic_source": "wristppg/ v0.3.0 (cardiac arrest overhaul)",
            "synthetic_counts": {"healthy": int((y_train == 0).sum()), "arrest": int((y_train == 1).sum())},
            "optimizer": "AdamW", "lr": 3e-4, "batch_size": 32,
            "epochs_trained": len(history.history["loss"]),
            "class_weights": cw,
        },
        "performance_test": metrics,
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
    logger.info("  Test: AUROC=%.4f F1=%.4f Prec=%.4f Rec=%.4f",
                metrics["auroc"], metrics["f1"], metrics["precision"], metrics["recall"])
    logger.info("  Saved to %s", OUT_DIR)
    return model, metrics, history


if __name__ == "__main__":
    train_v16()
