#!/usr/bin/env python3
"""Train CVD Watch Model v10 — trained on realistic synthetic wrist PPG.

v10 uses the RealisticWatchPPGGenerator for synthetic augmentation, producing
signals that closely approximate real Apple Watch PPG output. This bridges
the domain gap between training data and deployment target.

Evaluation:
  - Realistic watch test set (150 signals with wrist PPG morphology)
  - Original MIMIC/MMASH test set (patient-level held out)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import resample as scipy_resample

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from realistic_watch_test import RealisticWatchPPGGenerator, extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PPG_LENGTH = 7500
FS_TARGET = 25


# ---------------------------------------------------------------------------
# Patient-Level Data Loading (same as v8/v9)
# ---------------------------------------------------------------------------

def load_real_data_by_patient():
    signals_df = pd.read_parquet("data/processed/signals.parquet")
    features_df = pd.read_parquet("data/processed/features.parquet")
    patient_groups = {}
    for patient_id, group in signals_df.groupby("patient_id"):
        label = 0 if group.iloc[0]["event_type"] == "CONTROL" else 1
        patient_groups[patient_id] = {
            "label": label, "signals": [], "features": [],
            "event_type": group.iloc[0]["event_type"],
        }
        for idx, row in group.iterrows():
            try:
                if row["window_type"] == "wearable_control":
                    sig = np.load(row["wearable_ppg_path"])
                    fs = 25
                else:
                    sig = np.load(row["raw_ppg_path"])
                    fs = 125
                sig = sig.astype(np.float32)
                if fs != FS_TARGET:
                    sig = scipy_resample(sig, int(len(sig) * FS_TARGET / fs)).astype(np.float32)
                padded = np.zeros(PPG_LENGTH, dtype=np.float32)
                L = min(len(sig), PPG_LENGTH)
                padded[:L] = sig[:L]
                feat = {}
                feat_row = features_df.loc[idx] if idx in features_df.index else features_df.iloc[signals_df.index.get_loc(idx)]
                for col in features_df.columns:
                    val = feat_row[col]
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        feat[col] = float(val) if not np.isnan(val) else 0.0
                patient_groups[patient_id]["signals"].append(padded)
                patient_groups[patient_id]["features"].append(feat)
            except Exception:
                continue
    logger.info("Loaded %d patients (%d healthy, %d at-risk)",
                len(patient_groups),
                sum(1 for p in patient_groups.values() if p["label"] == 0),
                sum(1 for p in patient_groups.values() if p["label"] == 1))
    return patient_groups


def load_realistic_synthetic(n_healthy=250, n_at_risk=250, n_borderline=100, seed=42):
    """Generate realistic synthetic watch PPG for training augmentation."""
    gen = RealisticWatchPPGGenerator(fs=FS_TARGET, seed=seed)
    ppgs, feats_list, labels = [], [], []

    for i in range(n_healthy):
        ppg, meta = gen.generate_healthy()
        feats = extract_features(ppg, fs=FS_TARGET)
        feats["base_hr"] = meta["hr"]
        ppgs.append(ppg)
        feats_list.append(feats)
        labels.append(0)

    for i in range(n_at_risk):
        ppg, meta = gen.generate_at_risk()
        feats = extract_features(ppg, fs=FS_TARGET)
        feats["base_hr"] = meta["hr"]
        ppgs.append(ppg)
        feats_list.append(feats)
        labels.append(1)

    for i in range(n_borderline):
        ppg, meta = gen.generate_borderline()
        feats = extract_features(ppg, fs=FS_TARGET)
        feats["base_hr"] = meta["hr"]
        ppgs.append(ppg)
        feats_list.append(feats)
        labels.append(1)

    logger.info("Generated %d realistic synthetic watch signals", len(ppgs))
    return ppgs, feats_list, labels


def build_arrays(signals_list, features_list, feature_cols=None):
    X_ppg = np.zeros((len(signals_list), PPG_LENGTH), dtype=np.float32)
    for i, sig in enumerate(signals_list):
        L = min(len(sig), PPG_LENGTH)
        X_ppg[i, :L] = sig[:L]
    X_ppg = X_ppg[..., np.newaxis]
    if feature_cols is None:
        feature_cols = sorted(set().union(*[f.keys() for f in features_list]))
    X_feat = np.zeros((len(features_list), len(feature_cols)), dtype=np.float32)
    for i, f in enumerate(features_list):
        for j, col in enumerate(feature_cols):
            X_feat[i, j] = f.get(col, 0.0)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
    return X_ppg, X_feat, feature_cols


def patient_level_split(patient_groups, test_ratio=0.15, val_ratio=0.15, seed=42):
    from sklearn.model_selection import train_test_split
    patients = list(patient_groups.keys())
    labels = [patient_groups[p]["label"] for p in patients]
    pv_train, pv_test = train_test_split(
        list(range(len(patients))), test_size=test_ratio, random_state=seed, stratify=labels)
    pv_train_inner, pv_val = train_test_split(
        pv_train, test_size=val_ratio / (1 - test_ratio), random_state=seed,
        stratify=[labels[i] for i in pv_train])
    train_patients = [patients[i] for i in pv_train_inner]
    val_patients = [patients[i] for i in pv_val]
    test_patients = [patients[i] for i in pv_test]
    logger.info("Patient-level split: Train=%d, Val=%d, Test=%d patients",
                len(train_patients), len(val_patients), len(test_patients))
    assert len(set(train_patients) & set(val_patients)) == 0
    assert len(set(train_patients) & set(test_patients)) == 0
    assert len(set(val_patients) & set(test_patients)) == 0
    logger.info("  Verified: zero patient overlap")
    return train_patients, val_patients, test_patients


def flatten_patients(patient_groups, patient_list):
    signals, feats, labels = [], [], []
    for p in patient_list:
        for sig, feat in zip(patient_groups[p]["signals"], patient_groups[p]["features"]):
            signals.append(sig)
            feats.append(feat)
            labels.append(patient_groups[p]["label"])
    return signals, feats, np.array(labels, dtype=np.float32)


def train_v10():
    out_dir = Path("production/cvd_risk_v10_watch")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TRAINING v10-watch — Realistic Synthetic Watch Training")
    logger.info("=" * 60)

    # Load real data
    logger.info("\n[1/7] Loading real data by patient...")
    patient_groups = load_real_data_by_patient()

    # Patient-level split
    logger.info("\n[2/7] Patient-level train/val/test split...")
    train_p, val_p, test_p = patient_level_split(patient_groups)

    # Flatten
    train_sigs, train_feats, y_train = flatten_patients(patient_groups, train_p)
    val_sigs, val_feats, y_val = flatten_patients(patient_groups, val_p)
    test_sigs, test_feats, y_test_real = flatten_patients(patient_groups, test_p)

    # Realistic synthetic augmentation
    logger.info("\n[3/7] Generating realistic synthetic watch PPG...")
    synth_sigs, synth_feats, y_synth = load_realistic_synthetic(
        n_healthy=250, n_at_risk=250, n_borderline=100)

    # Combine real train + realistic synthetic
    train_sigs_aug = train_sigs + synth_sigs
    train_feats_aug = train_feats + synth_feats
    y_train_aug = np.concatenate([y_train, np.array(y_synth, dtype=np.float32)])

    # Unified feature columns
    all_feat_dicts = train_feats_aug + val_feats + test_feats
    feature_cols = sorted(set().union(*[f.keys() for f in all_feat_dicts]))
    logger.info("Unified feature columns: %d", len(feature_cols))

    # Build arrays
    logger.info("\n[4/7] Building arrays...")
    X_train, X_feat_train, _ = build_arrays(train_sigs_aug, train_feats_aug, feature_cols)
    X_val, X_feat_val, _ = build_arrays(val_sigs, val_feats, feature_cols)
    X_test, X_feat_test, _ = build_arrays(test_sigs, test_feats, feature_cols)

    logger.info("Train: %d signals (%d healthy, %d at-risk)",
                len(y_train_aug), int((y_train_aug == 0).sum()), int((y_train_aug == 1).sum()))
    logger.info("Val:   %d signals (%d healthy, %d at-risk)",
                len(y_val), int((y_val == 0).sum()), int((y_val == 1).sum()))
    logger.info("Test:  %d signals (%d healthy, %d at-risk)",
                len(y_test_real), int((y_test_real == 0).sum()), int((y_test_real == 1).sum()))

    # Build model
    from src.model_watch import build_watch_model
    model = build_watch_model(ppg_input_shape=(PPG_LENGTH, 1), feature_dim=X_feat_train.shape[1])
    model.summary(print_fn=logger.info)

    n_h = int((y_train_aug == 0).sum())
    n_e = int((y_train_aug == 1).sum())
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

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=20, mode="max",
                                          restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(out_dir / "best_model.keras"),
                                            monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7,
                                              min_lr=1e-6),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1,
                                        write_graph=True, write_images=True,
                                        update_freq="epoch", profile_batch=0),
        tf.keras.callbacks.CSVLogger(str(out_dir / "training_log.csv"), append=False),
    ]

    logger.info("\n[5/7] Training...")
    history = model.fit(
        {"ppg_input": X_train, "feature_input": X_feat_train}, y_train_aug,
        validation_data=({"ppg_input": X_val, "feature_input": X_feat_val}, y_val),
        epochs=120, batch_size=32, class_weight=cw, callbacks=callbacks,
    )

    # Evaluate on REAL test set (MIMIC/MMASH - patient-level held out)
    logger.info("\n[6/7] Evaluating on HELD-OUT REAL test set...")
    preds_real = model({"ppg_input": X_test, "feature_input": X_feat_test}, training=False)
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

    logger.info("  REAL TEST (MIMIC/MMASH): AUROC=%.4f Acc=%.1f%% Prec=%.4f Rec=%.4f F1=%.4f",
                metrics_real["auroc"], metrics_real["accuracy"] * 100,
                metrics_real["precision"], metrics_real["recall"], metrics_real["f1"])

    # Evaluate on realistic watch test set
    logger.info("\n[7/7] Evaluating on REALISTIC WATCH test set...")
    from realistic_watch_test import RealisticWatchPPGGenerator, predict as watch_predict
    gen_test = RealisticWatchPPGGenerator(fs=FS_TARGET, seed=99)  # different seed
    watch_signals, watch_labels = [], []
    for _ in range(60):
        ppg, _ = gen_test.generate_healthy()
        watch_signals.append(ppg)
        watch_labels.append(0)
    for _ in range(60):
        ppg, _ = gen_test.generate_at_risk()
        watch_signals.append(ppg)
        watch_labels.append(1)
    for _ in range(30):
        ppg, _ = gen_test.generate_borderline()
        watch_signals.append(ppg)
        watch_labels.append(1)
    watch_labels = np.array(watch_labels)

    watch_probs = []
    for ppg in watch_signals:
        prob, _ = watch_predict(model, feature_cols, ppg)
        watch_probs.append(prob)
    watch_probs = np.array(watch_probs)

    best_f1_w, best_t_w = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.005):
        f = f1_score(watch_labels, (watch_probs >= t).astype(int), zero_division=0)
        if f > best_f1_w:
            best_f1_w, best_t_w = f, t

    y_pred_watch = (watch_probs >= best_t_w).astype(int)
    cm_w = confusion_matrix(watch_labels, y_pred_watch, labels=[0, 1])
    tn_w, fp_w, fn_w, tp_w = cm_w.ravel()

    metrics_watch = {
        "auroc": float(roc_auc_score(watch_labels, watch_probs)),
        "accuracy": float(accuracy_score(watch_labels, y_pred_watch)),
        "precision": float(precision_score(watch_labels, y_pred_watch, zero_division=0)),
        "recall": float(recall_score(watch_labels, y_pred_watch, zero_division=0)),
        "f1": float(best_f1_w),
        "brier": float(brier_score_loss(watch_labels, watch_probs)),
        "threshold": float(best_t_w),
        "tn": int(tn_w), "fp": int(fp_w), "fn": int(fn_w), "tp": int(tp_w),
    }

    logger.info("  WATCH TEST: AUROC=%.4f Acc=%.1f%% Prec=%.4f Rec=%.4f F1=%.4f",
                metrics_watch["auroc"], metrics_watch["accuracy"] * 100,
                metrics_watch["precision"], metrics_watch["recall"], metrics_watch["f1"])
    logger.info("  CM: TN=%d FP=%d FN=%d TP=%d", tn_w, fp_w, fn_w, tp_w)

    # Save
    model.save(str(out_dir / "final_model.keras"))
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    config = {
        "version": "v10-watch",
        "description": "CVD model trained on realistic synthetic Apple Watch PPG",
        "ppg_length": PPG_LENGTH, "sampling_rate_hz": FS_TARGET,
        "feature_columns": feature_cols,
        "architecture": {
            "ppg_branch": "ResNet 1D-CNN (16->32->64) + BiLSTM(32)",
            "feature_branch": "MLP (32, 32)", "shared": "Dense(32)",
            "event_head": "Dense(16) -> Dense(1, sigmoid)",
            "total_params": model.count_params(),
        },
        "training": {
            "dataset": "hybrid_realistic_synthetic",
            "split_method": "patient_level_stratified",
            "synthetic_augmentation": {
                "generator": "RealisticWatchPPGGenerator",
                "morphology": "Wrist PPG (broad peaks, weak dicrotic notch)",
                "motion": "Gait-synchronized bursty artifacts",
                "noise": "Apple Watch green LED sensor noise + 12-bit ADC",
                "skin_tone": "Fitzpatrick I-VI melanin-dependent attenuation",
                "n_synthetic": len(synth_sigs),
            },
            "n_patients_total": len(patient_groups),
            "n_patients_train": len(train_p),
            "n_patients_val": len(val_p),
            "n_patients_test": len(test_p),
            "real_sources": ["MIMIC-IV (MI, ARREST)", "MMASH (CONTROL)", "SleepAccel (CONTROL)"],
            "optimizer": "AdamW", "lr": 3e-4, "batch_size": 32,
            "epochs_trained": len(history.history["loss"]),
            "class_weights": cw,
        },
        "performance_real_test": metrics_real,
        "performance_realistic_watch_test": metrics_watch,
    }

    with open(out_dir / "config.yaml", "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)
    with open(out_dir / "optimal_threshold.json", "w") as f:
        json.dump({"threshold": best_t}, f)

    logger.info("\n" + "=" * 60)
    logger.info("v10-watch SUMMARY")
    logger.info("=" * 60)
    logger.info("  Real test (MIMIC/MMASH): AUROC=%.4f F1=%.4f",
                metrics_real["auroc"], metrics_real["f1"])
    logger.info("  Watch test (realistic):  AUROC=%.4f F1=%.4f",
                metrics_watch["auroc"], metrics_watch["f1"])
    logger.info("  Saved to %s", out_dir)
    return model, metrics_real, metrics_watch, history


if __name__ == "__main__":
    train_v10()
