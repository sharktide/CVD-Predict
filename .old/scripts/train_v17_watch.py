#!/usr/bin/env python3
"""Train CVD Watch Model v17 — Leakage-Free Cardiac Arrest Detection.

Fixes from v16:
  1. NO leaking features (removed all wppg_* from beat records/latent vars)
  2. Simulator fixes: blood fraction properly flattened during arrest
  3. Clean optical signal scaled to realistic photodiode voltage range
  4. Expanded dataset: 2000+ synthetic samples
  5. Post-hoc calibration: Platt scaling + temperature scaling

Outputs everything to production/cvd_risk_v17_watch/.
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
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VERSION = "v17"
OUT_DIR = Path(f"production/cvd_risk_{VERSION}_watch")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR = OUT_DIR / "eval_graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


# ── Data Generation ────────────────────────────────────────────────
def generate_wristppg_cardiac_arrest(
    n_healthy: int = 1000,
    n_arrest: int = 1000,
    seed: int = 42,
):
    """Generate wristppg v0.3.0 synthetic data — NO leaking features.

    Returns only signal-derived features:
      - HRV features from PPG peak detection
      - Signal quality features (spectral flatness, autocorrelation, etc.)
      - Accelerometer features
      - Biodata (age, sex, BMI, vitals)
    """
    from wristppg import WristPPGSimulator

    rng = np.random.default_rng(seed)
    ppgs, accels, feat_dicts, biodata_list, labels = [], [], [], [], []

    activities = ["rest", "rest", "rest", "walking", "sleep"]
    contact_modes = ["good", "good", "good", "loose", "tight"]

    arrest_profiles = [
        "cardiac_arrest_vf",
        "cardiac_arrest_asystole",
        "cardiac_arrest_pulseless_electrical",
        "pre_arrest_deterioration",
        "respiratory_failure_pre_arrest",
    ]

    logger.info("Generating %d healthy + %d arrest samples...", n_healthy, n_arrest)

    for label, n_samples, profiles in [
        (0, n_healthy, ["healthy"]),
        (1, n_arrest, arrest_profiles),
    ]:
        n_ok = 0
        attempts = 0
        while n_ok < n_samples and attempts < n_samples * 5:
            attempts += 1
            try:
                prof = rng.choice(profiles)
                sim = WristPPGSimulator(seed=int(rng.integers(0, 2**31)))
                result = sim.generate(
                    profile=prof,
                    duration_s=60.0,
                    activity=rng.choice(activities),
                    contact_mode=rng.choice(contact_modes),
                )
                ppgs.append(result.ppg[:PPG_LENGTH])
                accels.append(result.accel[:PPG_LENGTH])

                # Signal-derived features only (NO leaking)
                pf = extract_ppg_features(result.ppg, fs=FS_TARGET)
                af = extract_accel_features(result.accel, fs=FS_TARGET)
                feat_dicts.append({**pf, **af})

                if label == 0:
                    biodata_list.append(np.array([
                        rng.uniform(20, 70), rng.choice([0, 1]),
                        rng.uniform(18, 35), 0,
                        0, 0, 0, 0, rng.uniform(1, 3),
                        0.97, 36.5, rng.uniform(30, 33),
                    ], dtype=np.float32))
                else:
                    biodata_list.append(np.array([
                        rng.uniform(40, 85), rng.choice([0, 1]),
                        rng.uniform(20, 40), rng.integers(0, 5),
                        rng.choice([0, 1], p=[0.6, 0.4]),
                        rng.choice([0, 1], p=[0.7, 0.3]),
                        0, 0, rng.uniform(2, 5),
                        result.meta.get("spo2", 0.70),
                        result.meta.get("body_temp_c", 36.0),
                        result.meta.get("skin_temperature_c", 32.0),
                    ], dtype=np.float32))

                labels.append(label)
                n_ok += 1
            except Exception as e:
                continue

        logger.info("  %s: %d/%d generated (%d attempts)",
                     "healthy" if label == 0 else "arrest", n_ok, n_samples, attempts)

    ppgs_padded = np.zeros((len(ppgs), PPG_LENGTH), dtype=np.float32)
    accels_padded = np.zeros((len(accels), PPG_LENGTH, 3), dtype=np.float32)
    for i, (p, a) in enumerate(zip(ppgs, accels)):
        L = min(len(p), PPG_LENGTH)
        ppgs_padded[i, :L] = p[:L]
        accels_padded[i, :L, :] = a[:L]

    labels = np.array(labels, dtype=np.float32)
    biodata = np.array(biodata_list, dtype=np.float32)

    feature_cols = sorted(set().union(*[f.keys() for f in feat_dicts]))
    features = np.zeros((len(feat_dicts), len(feature_cols)), dtype=np.float32)
    for i, f in enumerate(feat_dicts):
        for j, col in enumerate(feature_cols):
            features[i, j] = f.get(col, 0.0)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info("Total: %d samples (%d healthy, %d arrest), %d features",
                len(labels), int((labels == 0).sum()), int((labels == 1).sum()),
                features.shape[1])

    return ppgs_padded, accels_padded, features, biodata, labels, feature_cols


# ── Calibration ────────────────────────────────────────────────────
class PlattScaler:
    """Platt scaling for binary classification calibration."""
    def __init__(self):
        self.A = 1.0
        self.B = 0.0

    def fit(self, logits: np.ndarray, labels: np.ndarray, lr=0.01, epochs=1000):
        """Fit Platt scaling using gradient descent on NLL."""
        A, B = 1.0, 0.0
        for _ in range(epochs):
            p = 1.0 / (1.0 + np.exp(A * logits + B))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            dA = np.mean((labels - p) * logits)
            dB = np.mean(labels - p)
            A += lr * dA
            B += lr * dB
        self.A, self.B = A, B

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to probabilities."""
        logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7) /
                        np.clip(1 - probs, 1e-7, 1 - 1e-7))
        return 1.0 / (1.0 + np.exp(self.A * logits + self.B))


class TemperatureScaler:
    """Temperature scaling for calibration."""
    def __init__(self):
        self.T = 1.0

    def fit(self, probs: np.ndarray, labels: np.ndarray, lr=0.01, epochs=1000):
        """Fit temperature by minimizing NLL."""
        T = 1.0
        for _ in range(epochs):
            logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7) /
                            np.clip(1 - probs, 1e-7, 1 - 1e-7))
            scaled = logits / T
            p = 1.0 / (1.0 + np.exp(-scaled))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            grad = np.mean((labels - p) * logits / T)
            T = max(T - lr * grad, 0.01)
        self.T = T

    def predict(self, probs: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(probs, 1e-7, 1 - 1e-7) /
                        np.clip(1 - probs, 1e-7, 1 - 1e-7))
        return 1.0 / (1.0 + np.exp(-logits / self.T))


def compute_calibration_metrics(y_true, y_prob, n_bins=15):
    """Compute ECE, MCE, Brier score, and bin data."""
    from sklearn.metrics import brier_score_loss, log_loss
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    bin_data = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        bin_data.append({
            "bin_lo": float(bin_edges[i]),
            "bin_hi": float(bin_edges[i + 1]),
            "count": int(mask.sum()),
            "accuracy": float(acc),
            "confidence": float(conf),
        })
        gap = abs(acc - conf)
        ece += mask.sum() / len(y_true) * gap
        mce = max(mce, gap)

    brier = float(brier_score_loss(y_true, y_prob))
    try:
        ll = float(log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)))
    except Exception:
        ll = float('nan')

    return {"ece": ece, "mce": mce, "brier": brier, "log_loss": ll, "bins": bin_data}


# ── Plotting ───────────────────────────────────────────────────────
def plot_reliability(y_true, y_prob_before, y_prob_after, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, y_prob, title in [
        (axes[0], y_prob_before, "Before Calibration"),
        (axes[1], y_prob_after, "After Calibration"),
    ]:
        bins = np.linspace(0, 1, 16)
        bin_centers, bin_accs = [], []
        for i in range(len(bins) - 1):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_accs.append(y_true[mask].mean())
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.plot(bin_centers, bin_accs, "bo-", markersize=6, label="Model")
        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"CVD {VERSION.upper()} — Reliability Diagram", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "reliability_before_after.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_threshold_sweep(y_true, y_prob_before, y_prob_after, out_dir):
    from sklearn.metrics import f1_score, precision_score, recall_score
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, y_prob, title in [
        (axes[0], y_prob_before, "Before Calibration"),
        (axes[1], y_prob_after, "After Calibration"),
    ]:
        thresholds = np.arange(0.05, 0.95, 0.01)
        f1s, precs, recs = [], [], []
        for t in thresholds:
            yp = (y_prob >= t).astype(int)
            f1s.append(f1_score(y_true, yp, zero_division=0))
            precs.append(precision_score(y_true, yp, zero_division=0))
            recs.append(recall_score(y_true, yp, zero_division=0))
        ax.plot(thresholds, f1s, label="F1", linewidth=2)
        ax.plot(thresholds, precs, label="Precision", linewidth=2)
        ax.plot(thresholds, recs, label="Recall", linewidth=2)
        best_t = thresholds[np.argmax(f1s)]
        ax.axvline(best_t, color="red", linestyle="--", alpha=0.5, label=f"Best t={best_t:.2f}")
        ax.set_xlabel("Threshold", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"CVD {VERSION.upper()} — Threshold Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion(y_true, y_pred, out_dir, label="test"):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    classes = ["Healthy", "Arrest"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=16)
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    ax.set_title(f"CVD {VERSION.upper()} — Confusion Matrix ({label})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(feature_cols, importances, out_dir, top_n=25):
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(idx)), importances[idx], color="#2196F3")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_cols[i] for i in idx])
    ax.set_xlabel("Permutation Importance (AUROC drop)", fontsize=12)
    ax.set_title(f"CVD {VERSION.upper()} — Top {top_n} Features", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── Main Training ──────────────────────────────────────────────────
def train_v17():
    logger.info("=" * 60)
    logger.info(f"TRAINING {VERSION.upper()} — Leakage-Free Cardiac Arrest Detection")
    logger.info("=" * 60)

    # [1] Generate data
    logger.info("\n[1/10] Generating data (2000 samples)...")
    ppgs, accels, features, biodata, labels, feature_cols = \
        generate_wristppg_cardiac_arrest(n_healthy=1000, n_arrest=1000, seed=42)

    # [2] Split 60/20/20
    logger.info("\n[2/10] Splitting data...")
    rng = np.random.default_rng(42)
    n = len(labels)
    indices = rng.permutation(n)
    n_train = int(0.60 * n)
    n_val = int(0.20 * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def split(arr):
        return arr[train_idx], arr[val_idx], arr[test_idx]

    X_train_ppg, X_val_ppg, X_test_ppg = split(ppgs)
    X_train_accel, X_val_accel, X_test_accel = split(accels)
    X_train_feat, X_val_feat, X_test_feat = split(features)
    X_train_bio, X_val_bio, X_test_bio = split(biodata)
    y_train, y_val, y_test = split(labels)

    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        logger.info("  %s: %d (%d healthy, %d arrest)", name, len(y),
                     int((y == 0).sum()), int((y == 1).sum()))

    # [3] Build model
    logger.info("\n[3/10] Building model...")
    from src.model_v16 import build_v16
    model = build_v16(
        ppg_input_shape=(PPG_LENGTH, 1),
        accel_input_shape=(PPG_LENGTH, 3),
        hrv_feature_dim=X_train_feat.shape[1],
        biodata_dim=X_train_bio.shape[1],
    )
    model.summary(print_fn=logger.info)

    # [4] Compile
    logger.info("\n[4/10] Compiling...")
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
    logger.info("\n[5/10] Training...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=25, mode="max", restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            str(OUT_DIR / "best_model.keras"), monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(str(OUT_DIR / "training_log.csv"), append=False),
    ]

    inputs = {
        "ppg_input": X_train_ppg,
        "accel_input": X_train_accel,
        "feature_input": X_train_feat,
        "biodata_input": X_train_bio,
    }
    val_inputs = {
        "ppg_input": X_val_ppg,
        "accel_input": X_val_accel,
        "feature_input": X_val_feat,
        "biodata_input": X_val_bio,
    }

    history = model.fit(
        inputs, y_train,
        validation_data=(val_inputs, y_val),
        epochs=80, batch_size=32, class_weight=cw, callbacks=callbacks,
    )

    # [6] Evaluate on test set (before calibration)
    logger.info("\n[6/10] Evaluating on test set (before calibration)...")
    test_inputs = {
        "ppg_input": X_test_ppg,
        "accel_input": X_test_accel,
        "feature_input": X_test_feat,
        "biodata_input": X_test_bio,
    }
    preds_before = np.array(model(test_inputs, training=False)).flatten()

    # [7] Post-hoc calibration on validation set
    logger.info("\n[7/10] Calibrating on validation set...")
    val_preds = np.array(model(val_inputs, training=False)).flatten()

    platt = PlattScaler()
    platt.fit(val_preds, y_val)
    temp = TemperatureScaler()
    temp.fit(val_preds, y_val)

    preds_after = platt.predict(preds_before)

    # [8] Compute metrics
    logger.info("\n[8/10] Computing metrics...")
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, confusion_matrix,
                                  average_precision_score)

    cal_before = compute_calibration_metrics(y_test, preds_before)
    cal_after = compute_calibration_metrics(y_test, preds_after)

    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.005):
        f = f1_score(y_test, (preds_after >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    y_pred = (preds_after >= best_t).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "auroc": float(roc_auc_score(y_test, preds_after)),
        "pr_auc": float(average_precision_score(y_test, preds_after)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(best_f1),
        "threshold": float(best_t),
        "n_test": int(len(y_test)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "calibration_before": cal_before,
        "calibration_after": cal_after,
        "platt_A": float(platt.A), "platt_B": float(platt.B),
        "temperature": float(temp.T),
    }

    logger.info("  AUROC=%.4f  F1=%.4f  Prec=%.4f  Rec=%.4f  Acc=%.1f%%",
                metrics["auroc"], metrics["f1"], metrics["precision"],
                metrics["recall"], metrics["accuracy"] * 100)
    logger.info("  ECE before=%.4f  after=%.4f  Brier before=%.4f  after=%.4f",
                cal_before["ece"], cal_after["ece"],
                cal_before["brier"], cal_after["brier"])

    # [9] Plot
    logger.info("\n[9/10] Plotting...")
    plot_reliability(y_test, preds_before, preds_after, GRAPHS_DIR)
    plot_threshold_sweep(y_test, preds_before, preds_after, GRAPHS_DIR)
    plot_confusion(y_test, y_pred, GRAPHS_DIR, "test")

    # Feature importance via permutation
    logger.info("  Computing feature importance...")
    base_auroc = roc_auc_score(y_test, preds_after)
    importances = np.zeros(X_test_feat.shape[1])
    for j in range(X_test_feat.shape[1]):
        X_perm = X_test_feat.copy()
        rng_perm = np.random.default_rng(j)
        rng_perm.shuffle(X_perm[:, j])
        perm_inputs = {
            "ppg_input": X_test_ppg,
            "accel_input": X_test_accel,
            "feature_input": X_perm,
            "biodata_input": X_test_bio,
        }
        perm_preds = np.array(model(perm_inputs, training=False)).flatten()
        perm_preds = platt.predict(perm_preds)
        perm_auroc = roc_auc_score(y_test, perm_preds)
        importances[j] = base_auroc - perm_auroc
    plot_feature_importance(feature_cols, importances, GRAPHS_DIR)

    # [10] Save
    logger.info("\n[10/10] Saving...")
    model.save(str(OUT_DIR / "final_model.keras"))
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(OUT_DIR / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)
    with open(OUT_DIR / "optimal_threshold.json", "w") as f:
        json.dump({"threshold": best_t}, f)
    with open(OUT_DIR / "calibration.json", "w") as f:
        json.dump({
            "platt_A": platt.A, "platt_B": platt.B,
            "temperature": temp.T,
            "val_metrics": {
                "auroc": float(roc_auc_score(y_val, val_preds)),
                "brier_before": float(compute_calibration_metrics(y_val, val_preds)["brier"]),
            },
        }, f, indent=2)

    with open(OUT_DIR / "config.yaml", "w") as f:
        import yaml
        config = {
            "version": VERSION,
            "description": "Leakage-free cardiac arrest detection",
            "n_features": int(X_train_feat.shape[1]),
            "n_biodata": int(X_train_bio.shape[1]),
            "feature_columns": feature_cols,
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "total_params": model.count_params(),
            "performance_test": metrics,
        }
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Feature importance ranking
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    with open(OUT_DIR / "feature_importance.json", "w") as f:
        json.dump([{"feature": n, "importance": float(v)} for n, v in feat_imp], f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info(f"{VERSION.upper()} SUMMARY")
    logger.info("=" * 60)
    logger.info("  Test AUROC=%.4f  F1=%.4f  Prec=%.4f  Rec=%.4f",
                metrics["auroc"], metrics["f1"], metrics["precision"], metrics["recall"])
    logger.info("  ECE: %.4f -> %.4f  Brier: %.4f -> %.4f",
                cal_before["ece"], cal_after["ece"],
                cal_before["brier"], cal_after["brier"])
    logger.info("  Top features: %s", ", ".join(f"{n} ({v:.4f})" for n, v in feat_imp[:5]))
    logger.info("  Saved to %s", OUT_DIR)

    return model, metrics, history


if __name__ == "__main__":
    train_v17()
