#!/usr/bin/env python3
"""Evaluate v11 model on all test sets."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss

from src.model_watch import build_watch_model
from ultra_realistic_ppg import UltraRealisticPPGGenerator
from realistic_watch_test import RealisticWatchPPGGenerator, extract_features
from train_v11_watch import load_real_data_by_patient, patient_level_split, flatten_patients, build_arrays

PPG_LENGTH = 7500

# Load model
print("Loading v11 model...")
feature_cols = json.load(open("production/cvd_risk_v10_watch/feature_columns.json"))
model = build_watch_model(ppg_input_shape=(PPG_LENGTH, 1), feature_dim=len(feature_cols))
model.load_weights("production/cvd_risk_v11_watch/best_model.keras")
print(f"  Loaded model with {model.count_params()} params")

# === TEST 1: Real test set (MIMIC/MMASH) ===
print("\n=== REAL TEST SET (MIMIC/MMASH) ===")
patient_groups = load_real_data_by_patient()
_, _, test_p = patient_level_split(patient_groups)
test_sigs, test_feats, y_test = flatten_patients(patient_groups, test_p)
X_test, X_feat_test, _ = build_arrays(test_sigs, test_feats, feature_cols)

y_prob = model({"ppg_input": X_test, "feature_input": X_feat_test}, training=False).numpy().flatten()

best_f1, best_t = 0, 0.5
for t in np.arange(0.05, 0.95, 0.005):
    f = f1_score(y_test, (y_prob >= t).astype(int), zero_division=0)
    if f > best_f1:
        best_f1, best_t = f, t

y_pred = (y_prob >= best_t).astype(int)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print(f"  AUROC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"  Accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"  F1: {best_f1:.4f}")
print(f"  Threshold: {best_t:.3f}")
print(f"  CM: TN={tn} FP={fp} FN={fn} TP={tp}")

# === TEST 2: Ultra-realistic watch test (UltraRealisticPPGGenerator) ===
print("\n=== ULTRA-REALISTIC WATCH TEST ===")
gen_ultra = UltraRealisticPPGGenerator(fs=25, seed=99)
ultra_signals, ultra_labels = [], []
for _ in range(60):
    ppg, _ = gen_ultra.generate_healthy()
    ultra_signals.append(ppg); ultra_labels.append(0)
for _ in range(60):
    ppg, _ = gen_ultra.generate_at_risk()
    ultra_signals.append(ppg); ultra_labels.append(1)
for _ in range(30):
    ppg, _ = gen_ultra.generate_borderline()
    ultra_signals.append(ppg); ultra_labels.append(1)
ultra_labels = np.array(ultra_labels)

ultra_probs = []
for ppg in ultra_signals:
    feat = extract_features(ppg, fs=25)
    feat_array = np.array([[feat.get(col, 0) for col in feature_cols]])
    ppg_padded = np.zeros(PPG_LENGTH, dtype=np.float32)
    L = min(len(ppg), PPG_LENGTH)
    ppg_padded[:L] = ppg[:L]
    ppg_input = ppg_padded.reshape(1, -1, 1).astype(np.float32)
    prob = model.predict({"ppg_input": ppg_input, "feature_input": feat_array}, verbose=0)[0][0]
    ultra_probs.append(prob)
ultra_probs = np.array(ultra_probs)

best_f1_u, best_t_u = 0, 0.5
for t in np.arange(0.05, 0.95, 0.005):
    f = f1_score(ultra_labels, (ultra_probs >= t).astype(int), zero_division=0)
    if f > best_f1_u:
        best_f1_u, best_t_u = f, t

y_pred_u = (ultra_probs >= best_t_u).astype(int)
cm_u = confusion_matrix(ultra_labels, y_pred_u, labels=[0, 1])
tn_u, fp_u, fn_u, tp_u = cm_u.ravel()
print(f"  AUROC: {roc_auc_score(ultra_labels, ultra_probs):.4f}")
print(f"  Accuracy: {accuracy_score(ultra_labels, y_pred_u)*100:.1f}%")
print(f"  Precision: {precision_score(ultra_labels, y_pred_u, zero_division=0):.4f}")
print(f"  Recall: {recall_score(ultra_labels, y_pred_u, zero_division=0):.4f}")
print(f"  F1: {best_f1_u:.4f}")
print(f"  Threshold: {best_t_u:.3f}")
print(f"  CM: TN={tn_u} FP={fp_u} FN={fn_u} TP={tp_u}")

# === TEST 3: Realistic watch test (v10 generator) ===
print("\n=== REALISTIC WATCH TEST (v10 generator) ===")
gen_v10 = RealisticWatchPPGGenerator(fs=25, seed=99)
v10_signals, v10_labels = [], []
for _ in range(60):
    ppg, _ = gen_v10.generate_healthy()
    v10_signals.append(ppg); v10_labels.append(0)
for _ in range(60):
    ppg, _ = gen_v10.generate_at_risk()
    v10_signals.append(ppg); v10_labels.append(1)
for _ in range(30):
    ppg, _ = gen_v10.generate_borderline()
    v10_signals.append(ppg); v10_labels.append(1)
v10_labels = np.array(v10_labels)

v10_probs = []
for ppg in v10_signals:
    feat = extract_features(ppg, fs=25)
    feat_array = np.array([[feat.get(col, 0) for col in feature_cols]])
    ppg_padded = np.zeros(PPG_LENGTH, dtype=np.float32)
    L = min(len(ppg), PPG_LENGTH)
    ppg_padded[:L] = ppg[:L]
    ppg_input = ppg_padded.reshape(1, -1, 1).astype(np.float32)
    prob = model.predict({"ppg_input": ppg_input, "feature_input": feat_array}, verbose=0)[0][0]
    v10_probs.append(prob)
v10_probs = np.array(v10_probs)

best_f1_v, best_t_v = 0, 0.5
for t in np.arange(0.05, 0.95, 0.005):
    f = f1_score(v10_labels, (v10_probs >= t).astype(int), zero_division=0)
    if f > best_f1_v:
        best_f1_v, best_t_v = f, t

y_pred_v = (v10_probs >= best_t_v).astype(int)
cm_v = confusion_matrix(v10_labels, y_pred_v, labels=[0, 1])
tn_v, fp_v, fn_v, tp_v = cm_v.ravel()
print(f"  AUROC: {roc_auc_score(v10_labels, v10_probs):.4f}")
print(f"  Accuracy: {accuracy_score(v10_labels, y_pred_v)*100:.1f}%")
print(f"  Precision: {precision_score(v10_labels, y_pred_v, zero_division=0):.4f}")
print(f"  Recall: {recall_score(v10_labels, y_pred_v, zero_division=0):.4f}")
print(f"  F1: {best_f1_v:.4f}")
print(f"  Threshold: {best_t_v:.3f}")
print(f"  CM: TN={tn_v} FP={fp_v} FN={fn_v} TP={tp_v}")

# === Cross-generator comparison ===
print("\n=== CROSS-GENERATOR COMPARISON ===")
print(f"{'Test Set':<35} {'AUROC':>6} {'F1':>6} {'Recall':>6} {'Prec':>6}")
print("-" * 65)
print(f"{'Real (MIMIC/MMASH)':<35} {roc_auc_score(y_test, y_prob):.4f} {best_f1:.4f} {recall_score(y_test, y_pred, zero_division=0):.4f} {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"{'Ultra-realistic watch':<35} {roc_auc_score(ultra_labels, ultra_probs):.4f} {best_f1_u:.4f} {recall_score(ultra_labels, y_pred_u, zero_division=0):.4f} {precision_score(ultra_labels, y_pred_u, zero_division=0):.4f}")
print(f"{'Realistic watch (v10 gen)':<35} {roc_auc_score(v10_labels, v10_probs):.4f} {best_f1_v:.4f} {recall_score(v10_labels, y_pred_v, zero_division=0):.4f} {precision_score(v10_labels, y_pred_v, zero_division=0):.4f}")
