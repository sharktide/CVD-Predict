#!/usr/bin/env python3
"""Comprehensive evaluation of v17 cardiac arrest detection model.

Covers:
  1. Bootstrap confidence intervals (1000 resamples)
  2. Cross-validation (5-fold with different seeds)
  3. Subgroup analysis (profile, activity, contact mode)
  4. Error analysis (misclassified samples)
  5. Threshold analysis (clinical operating points)
  6. ROC/PR curves, prediction distributions
  7. Calibration (reliability diagrams, ECE per subgroup)
  8. Robustness (noise injection, signal degradation, contact loss)
  9. Latency profiling
"""

from __future__ import annotations

import json
import os
import sys
import time
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, brier_score_loss, accuracy_score,
    precision_score, recall_score, log_loss, roc_curve, precision_recall_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_pipeline_v12 import (
    PPG_LENGTH, FS_TARGET,
    extract_ppg_features, extract_accel_features,
)

OUT_DIR = Path("production/cvd_risk_v17_watch")
GRAPHS_DIR = OUT_DIR / "eval_graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

VERSION = "v17"


# ═══════════════════════════════════════════════════════════════════
# DATA GENERATION (with metadata)
# ═══════════════════════════════════════════════════════════════════
def generate_data_with_metadata(n_healthy=500, n_arrest=500, seed=42):
    """Generate data and return metadata for subgroup analysis."""
    from wristppg import WristPPGSimulator

    rng = np.random.default_rng(seed)
    ppgs, accels, feat_dicts, biodata_list, labels = [], [], [], [], []
    metadata = []

    activities = ["rest", "rest", "rest", "walking", "sleep"]
    contact_modes = ["good", "good", "good", "loose", "tight"]

    arrest_profiles = [
        "cardiac_arrest_vf",
        "cardiac_arrest_asystole",
        "cardiac_arrest_pulseless_electrical",
        "pre_arrest_deterioration",
        "respiratory_failure_pre_arrest",
    ]

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
                act = rng.choice(activities)
                cm = rng.choice(contact_modes)
                sim = WristPPGSimulator(seed=int(rng.integers(0, 2**31)))
                result = sim.generate(profile=prof, duration_s=60.0, activity=act, contact_mode=cm)
                ppgs.append(result.ppg[:PPG_LENGTH])
                accels.append(result.accel[:PPG_LENGTH])

                pf = extract_ppg_features(result.ppg, fs=FS_TARGET)
                af = extract_accel_features(result.accel, fs=FS_TARGET)
                feat_dicts.append({**pf, **af})

                if label == 0:
                    biodata_list.append(np.array([
                        rng.uniform(20, 70), rng.choice([0, 1]),
                        rng.uniform(18, 35), 0, 0, 0, 0, 0,
                        rng.uniform(1, 3), 0.97, 36.5, rng.uniform(30, 33),
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
                metadata.append({
                    "profile": prof, "activity": act, "contact_mode": cm,
                    "spo2": result.meta.get("spo2", 0.97),
                    "heart_rate": result.meta.get("heart_rate", 72),
                })
                n_ok += 1
            except Exception:
                continue

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

    return ppgs_padded, accels_padded, features, biodata, labels, feature_cols, metadata


# ═══════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════
def load_model():
    """Rebuild v17 model from source and load weights."""
    import tensorflow as tf
    from src.model_v16 import build_v16

    tmpdir = "/tmp/v17_eval_weights"
    os.makedirs(tmpdir, exist_ok=True)
    with zipfile.ZipFile(str(OUT_DIR / "best_model.keras"), 'r') as z:
        z.extract("model.weights.h5", tmpdir)

    # We need to know feature dims — generate tiny sample
    from wristppg import WristPPGSimulator
    from src.data_pipeline_v12 import PPG_LENGTH, FS_TARGET, extract_ppg_features, extract_accel_features
    rng = np.random.default_rng(99999)
    sim = WristPPGSimulator(seed=42)
    res = sim.generate(profile="healthy", duration_s=60.0, activity="rest", contact_mode="good")
    pf = extract_ppg_features(res.ppg, fs=FS_TARGET)
    af = extract_accel_features(res.accel, fs=FS_TARGET)
    n_feat = len(set(pf.keys()) | set(af.keys()))

    model = build_v16(
        ppg_input_shape=(PPG_LENGTH, 1),
        accel_input_shape=(PPG_LENGTH, 3),
        hrv_feature_dim=n_feat,
        biodata_dim=12,
    )
    model.load_weights(os.path.join(tmpdir, "model.weights.h5"))
    return model


# ═══════════════════════════════════════════════════════════════════
# CALIBRATION
# ═══════════════════════════════════════════════════════════════════
class PlattScaler:
    def __init__(self):
        self.A, self.B = 1.0, 0.0
    def fit(self, probs, labels):
        from sklearn.linear_model import LogisticRegression
        logits = np.log(np.clip(probs, 1e-7, 1-1e-7) / np.clip(1-probs, 1e-7, 1-1e-7))
        lr = LogisticRegression(C=1.0, solver='lbfgs')
        lr.fit(logits.reshape(-1, 1), labels)
        self.A, self.B = lr.coef_[0][0], lr.intercept_[0]
    def predict(self, probs):
        logits = np.log(np.clip(probs, 1e-7, 1-1e-7) / np.clip(1-probs, 1e-7, 1-1e-7))
        return 1.0 / (1.0 + np.exp(-(self.A * logits + self.B)))


def compute_ece(y_true, y_prob, n_bins=15):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    bin_data = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        gap = abs(acc - conf)
        ece += mask.sum() / len(y_true) * gap
        mce = max(mce, gap)
        bin_data.append({
            "bin": f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}",
            "count": int(mask.sum()),
            "accuracy": round(float(acc), 4),
            "confidence": round(float(conf), 4),
            "gap": round(float(gap), 4),
        })
    return ece, mce, bin_data


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════
def bootstrap_ci(y_true, y_prob, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aurocs, f1s, briers, eces = [], [], [], []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aurocs.append(roc_auc_score(yt, yp))
        best_f1 = 0
        for t in np.arange(0.05, 0.95, 0.01):
            f = f1_score(yt, (yp >= t).astype(int), zero_division=0)
            if f > best_f1: best_f1 = f
        f1s.append(best_f1)
        briers.append(brier_score_loss(yt, yp))
        e, _, _ = compute_ece(yt, yp)
        eces.append(e)

    def pct(arr, p): return float(np.percentile(arr, p))
    return {
        "auroc": {"mean": np.mean(aurocs), "std": np.std(aurocs),
                  "ci95": [pct(aurocs, 2.5), pct(aurocs, 97.5)]},
        "f1": {"mean": np.mean(f1s), "std": np.std(f1s),
               "ci95": [pct(f1s, 2.5), pct(f1s, 97.5)]},
        "brier": {"mean": np.mean(briers), "std": np.std(briers),
                  "ci95": [pct(briers, 2.5), pct(briers, 97.5)]},
        "ece": {"mean": np.mean(eces), "std": np.std(eces),
                "ci95": [pct(eces, 2.5), pct(eces, 97.5)]},
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════
def cross_validate(n_folds=5, seed=42):
    import tensorflow as tf
    from src.model_v16 import build_v16

    rng = np.random.default_rng(seed)
    fold_seeds = [int(rng.integers(0, 2**31)) for _ in range(n_folds)]
    fold_results = []

    for fold_i, fold_seed in enumerate(fold_seeds):
        print(f"  Fold {fold_i+1}/{n_folds} (seed={fold_seed})...")
        ppgs, accels, features, biodata, labels, feature_cols, meta = \
            generate_data_with_metadata(n_healthy=250, n_arrest=250, seed=fold_seed)

        # Split 80/20
        n = len(labels)
        idx = rng.permutation(n)
        split = int(0.8 * n)
        train_i, test_i = idx[:split], idx[split:]

        X_train = {"ppg_input": ppgs[train_i], "accel_input": accels[train_i],
                    "feature_input": features[train_i], "biodata_input": biodata[train_i]}
        X_test = {"ppg_input": ppgs[test_i], "accel_input": accels[test_i],
                   "feature_input": features[test_i], "biodata_input": biodata[test_i]}
        y_train, y_test = labels[train_i], labels[test_i]

        model = build_v16(
            ppg_input_shape=(PPG_LENGTH, 1), accel_input_shape=(PPG_LENGTH, 3),
            hrv_feature_dim=features.shape[1], biodata_dim=12,
        )
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )

        n_h = int((y_train == 0).sum())
        n_e = int((y_train == 1).sum())
        cw = {0: (n_h + n_e) / (2 * n_h), 1: (n_h + n_e) / (2 * n_e)}

        model.fit(X_train, y_train, epochs=30, batch_size=32, class_weight=cw,
                  validation_split=0.15,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=10, mode="max",
                                                              restore_best_weights=True)],
                  verbose=0)

        preds = np.array(model(X_test, training=False)).flatten()
        auroc = roc_auc_score(y_test, preds)
        best_f1, best_t = 0, 0.5
        for t in np.arange(0.05, 0.95, 0.005):
            f = f1_score(y_test, (preds >= t).astype(int), zero_division=0)
            if f > best_f1: best_f1, best_t = f, t
        brier = brier_score_loss(y_test, preds)
        ece, mce, _ = compute_ece(y_test, preds)

        fold_results.append({
            "fold": fold_i + 1, "seed": fold_seed,
            "n_train": len(y_train), "n_test": len(y_test),
            "auroc": round(auroc, 4), "f1": round(best_f1, 4),
            "brier": round(brier, 4), "ece": round(ece, 4),
        })
        del model
        tf.keras.backend.clear_session()

    aurocs = [f["auroc"] for f in fold_results]
    f1s = [f["f1"] for f in fold_results]
    summary = {
        "folds": fold_results,
        "mean_auroc": round(np.mean(aurocs), 4),
        "std_auroc": round(np.std(aurocs), 4),
        "mean_f1": round(np.mean(f1s), 4),
        "std_f1": round(np.std(f1s), 4),
    }
    return summary


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: SUBGROUP ANALYSIS
# ═══════════════════════════════════════════════════════════════════
def subgroup_analysis(y_test, preds, metadata_test):
    results = {}

    # By profile
    profiles = {}
    for i, m in enumerate(metadata_test):
        p = m["profile"]
        if p not in profiles: profiles[p] = []
        profiles[p].append(i)
    results["by_profile"] = {}
    for prof, idx in profiles.items():
        yt, yp = y_test[idx], preds[idx]
        if len(np.unique(yt)) < 2:
            results["by_profile"][prof] = {"n": len(idx), "note": "single class"}
            continue
        results["by_profile"][prof] = {
            "n": len(idx),
            "auroc": round(roc_auc_score(yt, yp), 4),
            "mean_pred_arrest": round(yp[yt == 1].mean(), 4) if (yt == 1).any() else None,
            "mean_pred_healthy": round(yp[yt == 0].mean(), 4) if (yt == 0).any() else None,
        }

    # By activity
    activities = {}
    for i, m in enumerate(metadata_test):
        a = m["activity"]
        if a not in activities: activities[a] = []
        activities[a].append(i)
    results["by_activity"] = {}
    for act, idx in activities.items():
        yt, yp = y_test[idx], preds[idx]
        if len(np.unique(yt)) < 2:
            results["by_activity"][act] = {"n": len(idx), "note": "single class"}
            continue
        results["by_activity"][act] = {
            "n": len(idx),
            "auroc": round(roc_auc_score(yt, yp), 4),
            "mean_pred": round(yp.mean(), 4),
        }

    # By contact mode
    contacts = {}
    for i, m in enumerate(metadata_test):
        c = m["contact_mode"]
        if c not in contacts: contacts[c] = []
        contacts[c].append(i)
    results["by_contact"] = {}
    for cm, idx in contacts.items():
        yt, yp = y_test[idx], preds[idx]
        if len(np.unique(yt)) < 2:
            results["by_contact"][cm] = {"n": len(idx), "note": "single class"}
            continue
        results["by_contact"][cm] = {
            "n": len(idx),
            "auroc": round(roc_auc_score(yt, yp), 4),
            "mean_pred": round(yp.mean(), 4),
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════════
def error_analysis(y_test, preds, metadata_test, ppgs_test, threshold=0.5):
    y_pred = (preds >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    errors = []
    for i in range(len(y_test)):
        if y_pred[i] != y_test[i]:
            errors.append({
                "idx": int(i),
                "true": int(y_test[i]),
                "pred": float(preds[i]),
                "profile": metadata_test[i]["profile"],
                "activity": metadata_test[i]["activity"],
                "contact_mode": metadata_test[i]["contact_mode"],
                "ppg_std": float(ppgs_test[i].std()),
                "ppg_range": float(ppgs_test[i].max() - ppgs_test[i].min()),
            })

    # Signal stats for correct vs incorrect
    correct_mask = y_pred == y_test
    incorrect_mask = ~correct_mask
    signal_stats = {
        "correct": {
            "mean_ppg_std": float(ppgs_test[correct_mask].mean(axis=0).std()),
            "median_ppg_std": float(np.median([ppgs_test[i].std() for i in range(len(correct_mask)) if correct_mask[i]])),
        },
        "incorrect": {
            "mean_ppg_std": float(ppgs_test[incorrect_mask].mean(axis=0).std()) if incorrect_mask.any() else None,
            "median_ppg_std": float(np.median([ppgs_test[i].std() for i in range(len(incorrect_mask)) if incorrect_mask[i]])) if incorrect_mask.any() else None,
        },
    }

    return {
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
        "n_errors": len(errors),
        "error_rate": round(len(errors) / len(y_test), 4),
        "errors": errors[:20],  # first 20
        "signal_stats": signal_stats,
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: THRESHOLD ANALYSIS
# ═══════════════════════════════════════════════════════════════════
def threshold_analysis(y_test, preds):
    thresholds = np.arange(0.01, 0.99, 0.005)
    rows = []
    for t in thresholds:
        yp = (preds >= t).astype(int)
        cm = confusion_matrix(y_test, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = f1_score(y_test, yp, zero_division=0)
        youden = sens + spec - 1
        rows.append({
            "threshold": round(float(t), 3),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "ppv": round(ppv, 4),
            "npv": round(npv, 4),
            "f1": round(f1, 4),
            "youden": round(youden, 4),
            "fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn),
        })

    # Find optimal by different criteria
    best_youden = max(rows, key=lambda r: r["youden"])
    best_f1 = max(rows, key=lambda r: r["f1"])
    # High sensitivity operating point (>=99% sensitivity)
    high_sens = [r for r in rows if r["sensitivity"] >= 0.99]
    high_sens_op = min(high_sens, key=lambda r: r["fp"]) if high_sens else None

    return {
        "sweep": rows,
        "best_youden": best_youden,
        "best_f1": best_f1,
        "high_sensitivity": high_sens_op,
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════
def robustness_analysis(model, X_test, y_test):
    results = {}
    base_preds = np.array(model(X_test, training=False)).flatten()
    base_auroc = roc_auc_score(y_test, base_preds)
    results["baseline_auroc"] = round(base_auroc, 4)

    # 1. Gaussian noise on PPG
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
    results["noise_robustness"] = {}
    for nl in noise_levels:
        X_noisy = {k: v.copy() for k, v in X_test.items()}
        X_noisy["ppg_input"] = X_noisy["ppg_input"] + np.random.default_rng(0).normal(0, nl, X_noisy["ppg_input"].shape).astype(np.float32)
        preds = np.array(model(X_noisy, training=False)).flatten()
        auroc = roc_auc_score(y_test, preds)
        results["noise_robustness"][f"noise_{nl}"] = round(auroc, 4)

    # 2. Amplitude scaling (simulate poor contact)
    scales = [0.5, 0.25, 0.1, 0.05]
    results["amplitude_robustness"] = {}
    for s in scales:
        X_scaled = {k: v.copy() for k, v in X_test.items()}
        X_scaled["ppg_input"] = X_scaled["ppg_input"] * s
        preds = np.array(model(X_scaled, training=False)).flatten()
        auroc = roc_auc_score(y_test, preds)
        results["amplitude_robustness"][f"scale_{s}"] = round(auroc, 4)

    # 3. Zero-out accelerometer (simulate motion sensor failure)
    X_no_accel = {k: v.copy() for k, v in X_test.items()}
    X_no_accel["accel_input"] = np.zeros_like(X_no_accel["accel_input"])
    preds = np.array(model(X_no_accel, training=False)).flatten()
    results["no_accelerometer_auroc"] = round(roc_auc_score(y_test, preds), 4)

    # 4. Time shift (circular shift of PPG by random amount)
    results["time_shift_robustness"] = {}
    for shift in [50, 100, 200, 500]:
        X_shifted = {k: v.copy() for k, v in X_test.items()}
        X_shifted["ppg_input"] = np.roll(X_shifted["ppg_input"], shift, axis=1)
        preds = np.array(model(X_shifted, training=False)).flatten()
        auroc = roc_auc_score(y_test, preds)
        results["time_shift_robustness"][f"shift_{shift}"] = round(auroc, 4)

    return results


# ═══════════════════════════════════════════════════════════════════
# SECTION 9: LATENCY
# ═══════════════════════════════════════════════════════════════════
def latency_profile(model, X_test, n_warmup=10, n_runs=100):
    import tensorflow as tf

    # Warmup
    for i in range(n_warmup):
        _ = model({k: v[i:i+1] for k, v in X_test.items()}, training=False)

    # Benchmark single sample
    times_single = []
    for i in range(n_runs):
        idx = i % len(X_test["ppg_input"])
        t0 = time.perf_counter()
        _ = model({k: v[idx:idx+1] for k, v in X_test.items()}, training=False)
        t1 = time.perf_counter()
        times_single.append((t1 - t0) * 1000)

    # Benchmark batch of 32
    batch_X = {k: v[:32] for k, v in X_test.items()}
    times_batch = []
    for _ in range(n_runs // 4):
        t0 = time.perf_counter()
        _ = model(batch_X, training=False)
        t1 = time.perf_counter()
        times_batch.append((t1 - t0) * 1000)

    return {
        "single_sample_ms": {
            "mean": round(np.mean(times_single), 2),
            "std": round(np.std(times_single), 2),
            "p50": round(np.percentile(times_single, 50), 2),
            "p95": round(np.percentile(times_single, 95), 2),
            "p99": round(np.percentile(times_single, 99), 2),
        },
        "batch_32_ms": {
            "mean": round(np.mean(times_batch), 2),
            "std": round(np.std(times_batch), 2),
            "p50": round(np.percentile(times_batch, 50), 2),
            "p95": round(np.percentile(times_batch, 95), 2),
        },
        "per_sample_batch_ms": round(np.mean(times_batch) / 32, 2),
    }


# ═══════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════
def plot_all(y_test, preds_raw, preds_cal, metadata_test, boot_results, cv_results,
             threshold_results, robustness_results, out_dir):
    fig = plt.figure(figsize=(24, 28))
    gs = fig.add_gridspec(5, 4, hspace=0.35, wspace=0.3)

    # 1. ROC Curve
    ax = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_test, preds_raw)
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUROC={boot_results["auroc"]["mean"]:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curve', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 2. PR Curve
    ax = fig.add_subplot(gs[0, 1])
    prec, rec, _ = precision_recall_curve(y_test, preds_raw)
    ax.plot(rec, prec, 'b-', linewidth=2)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('PR Curve', fontweight='bold'); ax.grid(True, alpha=0.3)

    # 3. Prediction Distribution
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(preds_raw[y_test == 0], bins=40, alpha=0.6, label='Healthy', color='green', density=True)
    ax.hist(preds_raw[y_test == 1], bins=40, alpha=0.6, label='Arrest', color='red', density=True)
    ax.set_xlabel('Predicted Probability'); ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 4. Bootstrap CI
    ax = fig.add_subplot(gs[0, 3])
    metrics = ['auroc', 'f1', 'brier', 'ece']
    labels_ci = ['AUROC', 'F1', 'Brier', 'ECE']
    means = [boot_results[m]["mean"] for m in metrics]
    ci_lo = [boot_results[m]["ci95"][0] for m in metrics]
    ci_hi = [boot_results[m]["ci95"][1] for m in metrics]
    errors_lo = [m - l for m, l in zip(means, ci_lo)]
    errors_hi = [h - m for m, h in zip(means, ci_hi)]
    y_pos = range(len(metrics))
    ax.barh(y_pos, means, xerr=[errors_lo, errors_hi], capsize=5, color='steelblue', alpha=0.7)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels_ci)
    ax.set_title('Bootstrap 95% CI', fontweight='bold'); ax.grid(True, alpha=0.3, axis='x')

    # 5. Reliability Diagram (raw)
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, 1, 16)
    for y_p, label, color in [(preds_raw, 'Raw', 'blue'), (preds_cal, 'Platt', 'green')]:
        bin_centers, bin_accs = [], []
        for i in range(len(bins) - 1):
            mask = (y_p >= bins[i]) & (y_p < bins[i+1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i+1]) / 2)
                bin_accs.append(y_test[mask].mean())
        ax.plot(bin_centers, bin_accs, 'o-', color=color, label=label, markersize=4)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('Mean Predicted'); ax.set_ylabel('Fraction Positives')
    ax.set_title('Reliability Diagram', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 6. Threshold: Sensitivity/Specificity
    ax = fig.add_subplot(gs[1, 1])
    t_sweep = threshold_results["sweep"]
    ts = [r["threshold"] for r in t_sweep]
    sens = [r["sensitivity"] for r in t_sweep]
    spec = [r["specificity"] for r in t_sweep]
    f1s = [r["f1"] for r in t_sweep]
    ax.plot(ts, sens, label='Sensitivity', linewidth=2)
    ax.plot(ts, spec, label='Specificity', linewidth=2)
    ax.plot(ts, f1s, label='F1', linewidth=2, linestyle='--')
    bf = threshold_results["best_f1"]
    ax.axvline(bf["threshold"], color='red', linestyle=':', alpha=0.5, label=f'Best F1 t={bf["threshold"]:.2f}')
    ax.set_xlabel('Threshold'); ax.set_ylabel('Score')
    ax.set_title('Threshold Analysis', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 7. Confusion Matrix
    ax = fig.add_subplot(gs[1, 2])
    cm = confusion_matrix(y_test, (preds_raw >= 0.5).astype(int), labels=[0, 1])
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Healthy', 'Arrest'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Healthy', 'Arrest'])
    ax.set_ylabel('True'); ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix (t=0.5)', fontweight='bold')

    # 8. Feature importance
    ax = fig.add_subplot(gs[1, 3])
    fi_path = OUT_DIR / "feature_importance.json"
    if fi_path.exists():
        with open(fi_path) as f:
            fi_data = json.load(f)
        names = [d["feature"] for d in fi_data[:15]]
        imps = [d["importance"] for d in fi_data[:15]]
        ax.barh(range(len(names)), imps, color='steelblue')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel('Permutation Importance')
    ax.set_title('Top 15 Features', fontweight='bold'); ax.grid(True, alpha=0.3, axis='x')

    # 9. Subgroup: By Profile
    ax = fig.add_subplot(gs[2, :2])
    profiles = {}
    for i, m in enumerate(metadata_test):
        p = m["profile"]
        if p not in profiles: profiles[p] = {"y": [], "p": []}
        profiles[p]["y"].append(y_test[i])
        profiles[p]["p"].append(preds_raw[i])
    prof_names = sorted(profiles.keys())
    prof_aurocs = []
    for pn in prof_names:
        yt = np.array(profiles[pn]["y"])
        yp = np.array(profiles[pn]["p"])
        if len(np.unique(yt)) >= 2:
            prof_aurocs.append(roc_auc_score(yt, yp))
        else:
            prof_aurocs.append(0)
    colors = ['green' if 'healthy' in n else 'red' for n in prof_names]
    short_names = [n.replace('cardiac_arrest_', 'CA ').replace('pre_arrest_', 'Pre-').replace('respiratory_failure_', 'Resp ') for n in prof_names]
    ax.barh(range(len(prof_names)), prof_aurocs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(prof_names))); ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel('AUROC'); ax.set_title('AUROC by Profile', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # 10. Subgroup: By Activity
    ax = fig.add_subplot(gs[2, 2])
    acts = {}
    for i, m in enumerate(metadata_test):
        a = m["activity"]
        if a not in acts: acts[a] = {"y": [], "p": []}
        acts[a]["y"].append(y_test[i])
        acts[a]["p"].append(preds_raw[i])
    act_names = sorted(acts.keys())
    act_aurocs = []
    for an in act_names:
        yt, yp = np.array(acts[an]["y"]), np.array(acts[an]["p"])
        act_aurocs.append(roc_auc_score(yt, yp) if len(np.unique(yt)) >= 2 else 0)
    ax.bar(range(len(act_names)), act_aurocs, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(act_names))); ax.set_xticklabels(act_names, fontsize=8)
    ax.set_ylabel('AUROC'); ax.set_title('AUROC by Activity', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 11. Subgroup: By Contact
    ax = fig.add_subplot(gs[2, 3])
    conts = {}
    for i, m in enumerate(metadata_test):
        c = m["contact_mode"]
        if c not in conts: conts[c] = {"y": [], "p": []}
        conts[c]["y"].append(y_test[i])
        conts[c]["p"].append(preds_raw[i])
    con_names = sorted(conts.keys())
    con_aurocs = []
    for cn in con_names:
        yt, yp = np.array(conts[cn]["y"]), np.array(conts[cn]["p"])
        con_aurocs.append(roc_auc_score(yt, yp) if len(np.unique(yt)) >= 2 else 0)
    ax.bar(range(len(con_names)), con_aurocs, color='coral', alpha=0.7)
    ax.set_xticks(range(len(con_names))); ax.set_xticklabels(con_names, fontsize=8)
    ax.set_ylabel('AUROC'); ax.set_title('AUROC by Contact', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 12. Robustness: Noise
    ax = fig.add_subplot(gs[3, 0])
    noise_data = robustness_results.get("noise_robustness", {})
    if noise_data:
        nl = [float(k.split('_')[1]) for k in noise_data.keys()]
        nv = list(noise_data.values())
        ax.plot(nl, nv, 'bo-', linewidth=2)
        ax.axhline(robustness_results["baseline_auroc"], color='red', linestyle='--', alpha=0.5, label='Baseline')
        ax.set_xlabel('Noise σ'); ax.set_ylabel('AUROC')
        ax.set_title('Noise Robustness', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

    # 13. Robustness: Amplitude
    ax = fig.add_subplot(gs[3, 1])
    amp_data = robustness_results.get("amplitude_robustness", {})
    if amp_data:
        al = [float(k.split('_')[1]) for k in amp_data.keys()]
        av = list(amp_data.values())
        ax.plot(al, av, 'go-', linewidth=2)
        ax.axhline(robustness_results["baseline_auroc"], color='red', linestyle='--', alpha=0.5, label='Baseline')
        ax.set_xlabel('Amplitude Scale'); ax.set_ylabel('AUROC')
        ax.set_title('Contact Loss Robustness', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

    # 14. Robustness: Time Shift
    ax = fig.add_subplot(gs[3, 2])
    shift_data = robustness_results.get("time_shift_robustness", {})
    if shift_data:
        sl = [int(k.split('_')[1]) for k in shift_data.keys()]
        sv = list(shift_data.values())
        ax.plot(sl, sv, 'mo-', linewidth=2)
        ax.axhline(robustness_results["baseline_auroc"], color='red', linestyle='--', alpha=0.5, label='Baseline')
        ax.set_xlabel('Time Shift (samples)'); ax.set_ylabel('AUROC')
        ax.set_title('Time Shift Robustness', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

    # 15. Cross-validation results
    ax = fig.add_subplot(gs[3, 3])
    if cv_results and "folds" in cv_results:
        folds = cv_results["folds"]
        fold_aurocs = [f["auroc"] for f in folds]
        ax.bar(range(len(fold_aurocs)), fold_aurocs, color='teal', alpha=0.7)
        ax.axhline(cv_results["mean_auroc"], color='red', linestyle='--',
                    label=f'Mean={cv_results["mean_auroc"]:.4f}±{cv_results["std_auroc"]:.4f}')
        ax.set_xticks(range(len(fold_aurocs)))
        ax.set_xticklabels([f"F{i+1}" for i in range(len(fold_aurocs))])
        ax.set_ylabel('AUROC'); ax.set_title('5-Fold Cross-Validation', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # 16. Calibration table
    ax = fig.add_subplot(gs[4, :2])
    ax.axis('off')
    ece_raw, _, bins_raw = compute_ece(y_test, preds_raw)
    ece_cal, _, bins_cal = compute_ece(y_test, preds_cal)
    table_data = [["Bin", "Count", "Acc (Raw)", "Conf (Raw)", "Gap", "Acc (Platt)", "Conf (Platt)", "Gap"]]
    for br, bc in zip(bins_raw, bins_cal):
        table_data.append([br["bin"], br["count"],
                          f'{br["accuracy"]:.3f}', f'{br["confidence"]:.3f}', f'{br["gap"]:.3f}',
                          f'{bc["accuracy"]:.3f}', f'{bc["confidence"]:.3f}', f'{bc["gap"]:.3f}'])
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(7)
    table.scale(1, 1.2)
    ax.set_title('Calibration Bins: Raw vs Platt', fontweight='bold', fontsize=11)

    # 17. Summary box
    ax = fig.add_subplot(gs[4, 2:])
    ax.axis('off')
    boot = boot_results
    cv = cv_results
    thr = threshold_results
    summary_text = (
        f"V17 COMPREHENSIVE EVALUATION SUMMARY\n"
        f"{'='*50}\n\n"
        f"TEST SET: {len(y_test)} samples ({int((y_test==0).sum())} healthy, {int((y_test==1).sum())} arrest)\n\n"
        f"BOOTSTRAP (1000 resamples):\n"
        f"  AUROC: {boot['auroc']['mean']:.4f} ± {boot['auroc']['std']:.4f}  "
        f"[{boot['auroc']['ci95'][0]:.4f}, {boot['auroc']['ci95'][1]:.4f}]\n"
        f"  F1:    {boot['f1']['mean']:.4f} ± {boot['f1']['std']:.4f}  "
        f"[{boot['f1']['ci95'][0]:.4f}, {boot['f1']['ci95'][1]:.4f}]\n"
        f"  Brier: {boot['brier']['mean']:.4f} ± {boot['brier']['std']:.4f}\n"
        f"  ECE:   {boot['ece']['mean']:.4f} ± {boot['ece']['std']:.4f}\n\n"
        f"5-FOLD CROSS-VALIDATION:\n"
        f"  AUROC: {cv.get('mean_auroc', 'N/A')} ± {cv.get('std_auroc', 'N/A')}\n"
        f"  F1:    {cv.get('mean_f1', 'N/A')} ± {cv.get('std_f1', 'N/A')}\n\n"
        f"OPTIMAL OPERATING POINTS:\n"
        f"  Best F1:     t={thr['best_f1']['threshold']:.2f}  F1={thr['best_f1']['f1']:.4f}  "
        f"Sen={thr['best_f1']['sensitivity']:.3f}  Spec={thr['best_f1']['specificity']:.3f}\n"
        f"  Best Youden: t={thr['best_youden']['threshold']:.2f}  J={thr['best_youden']['youden']:.4f}\n"
    )
    if thr.get("high_sensitivity"):
        hs = thr["high_sensitivity"]
        summary_text += f"  High Sens (≥99%): t={hs['threshold']:.2f}  Spec={hs['specificity']:.3f}  FP={hs['fp']}\n"
    summary_text += (
        f"\nROBUSTNESS:\n"
        f"  No accel: AUROC={robustness_results.get('no_accelerometer_auroc', 'N/A')}\n"
    )
    if "noise_robustness" in robustness_results:
        for k, v in robustness_results["noise_robustness"].items():
            summary_text += f"  PPG noise {k}: AUROC={v}\n"
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f'CVD {VERSION.upper()} — Comprehensive Evaluation Report', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(out_dir / "comprehensive_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved comprehensive_evaluation.png")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  V17 COMPREHENSIVE EVALUATION")
    print("=" * 60)

    # Load model
    print("\n[1/10] Loading model...")
    model = load_model()
    print(f"  Model loaded: {model.count_params()} params")

    # Generate data
    print("\n[2/10] Generating 1000 unseen samples with metadata...")
    ppgs, accels, features, biodata, labels, feature_cols, metadata = \
        generate_data_with_metadata(n_healthy=500, n_arrest=500, seed=7777)
    print(f"  {len(labels)} samples ({int((labels==0).sum())} healthy, {int((labels==1).sum())} arrest)")

    # Split
    rng = np.random.default_rng(7777)
    idx = rng.permutation(len(labels))
    n_cal = int(0.30 * len(labels))
    n_val = int(0.30 * len(labels))
    cal_i, val_i, test_i = idx[:n_cal], idx[n_cal:n_cal+n_val], idx[n_cal+n_val:]

    X_cal = {"ppg_input": ppgs[cal_i], "accel_input": accels[cal_i],
             "feature_input": features[cal_i], "biodata_input": biodata[cal_i]}
    X_val = {"ppg_input": ppgs[val_i], "accel_input": accels[val_i],
             "feature_input": features[val_i], "biodata_input": biodata[val_i]}
    X_test = {"ppg_input": ppgs[test_i], "accel_input": accels[test_i],
              "feature_input": features[test_i], "biodata_input": biodata[test_i]}
    y_cal, y_val, y_test = labels[cal_i], labels[val_i], labels[test_i]
    meta_test = [metadata[i] for i in test_i]

    print(f"  Cal: {len(y_cal)}  Val: {len(y_val)}  Test: {len(y_test)}")

    # Predictions
    print("\n[3/10] Running predictions...")
    preds_cal_set = np.array(model(X_cal, training=False)).flatten()
    preds_val_set = np.array(model(X_val, training=False)).flatten()
    preds_test = np.array(model(X_test, training=False)).flatten()

    # Calibrate (fit on cal set, apply to test)
    platt = PlattScaler()
    platt.fit(preds_cal_set, y_cal)
    preds_cal = platt.predict(preds_test)
    print(f"  Platt: A={platt.A:.4f}, B={platt.B:.4f}")

    # Bootstrap
    print("\n[4/10] Bootstrap confidence intervals (1000 resamples)...")
    y_test_np = np.array(y_test)
    boot_raw = bootstrap_ci(y_test_np, preds_test, n_boot=1000)
    boot_cal = bootstrap_ci(y_test_np, preds_cal, n_boot=1000)
    print(f"  Raw AUROC: {boot_raw['auroc']['mean']:.4f} [{boot_raw['auroc']['ci95'][0]:.4f}, {boot_raw['auroc']['ci95'][1]:.4f}]")
    print(f"  Cal AUROC: {boot_cal['auroc']['mean']:.4f} [{boot_cal['auroc']['ci95'][0]:.4f}, {boot_cal['auroc']['ci95'][1]:.4f}]")

    # Cross-validation
    print("\n[5/10] 5-fold cross-validation...")
    cv_results = cross_validate(n_folds=5, seed=42)
    print(f"  Mean AUROC: {cv_results['mean_auroc']} ± {cv_results['std_auroc']}")
    print(f"  Mean F1:    {cv_results['mean_f1']} ± {cv_results['std_f1']}")

    # Subgroup analysis
    print("\n[6/10] Subgroup analysis...")
    subgroups = subgroup_analysis(y_test_np, preds_test, meta_test)
    print(f"  Profiles: {list(subgroups['by_profile'].keys())}")
    print(f"  Activities: {list(subgroups['by_activity'].keys())}")
    print(f"  Contact: {list(subgroups['by_contact'].keys())}")

    # Error analysis
    print("\n[7/10] Error analysis...")
    errors = error_analysis(y_test_np, preds_test, meta_test, ppgs[test_i])
    print(f"  Errors: {errors['n_errors']}/{len(y_test)} ({errors['error_rate']*100:.1f}%)")
    print(f"  CM: TP={errors['confusion_matrix']['tp']} FP={errors['confusion_matrix']['fp']} "
          f"FN={errors['confusion_matrix']['fn']} TN={errors['confusion_matrix']['tn']}")

    # Threshold analysis
    print("\n[8/10] Threshold analysis...")
    thresholds = threshold_analysis(y_test_np, preds_test)
    bf = thresholds["best_f1"]
    print(f"  Best F1: t={bf['threshold']:.2f} F1={bf['f1']:.4f} Sens={bf['sensitivity']:.3f} Spec={bf['specificity']:.3f}")
    by = thresholds["best_youden"]
    print(f"  Best Youden: t={by['threshold']:.2f} J={by['youden']:.4f}")

    # Robustness
    print("\n[9/10] Robustness analysis...")
    robustness = robustness_analysis(model, X_test, y_test_np)

    # Latency
    print("\n[10/10] Latency profiling...")
    latency = latency_profile(model, X_test)
    print(f"  Single: {latency['single_sample_ms']['mean']:.1f}ms ± {latency['single_sample_ms']['std']:.1f}ms")
    print(f"  Batch:  {latency['per_sample_batch_ms']:.1f}ms/sample")

    # ── Save all results ──
    print("\nSaving results...")
    all_results = {
        "version": VERSION,
        "n_test": int(len(y_test)),
        "bootstrap_raw": boot_raw,
        "bootstrap_calibrated": boot_cal,
        "cross_validation": cv_results,
        "subgroups": subgroups,
        "errors": errors,
        "thresholds": {
            "best_f1": thresholds["best_f1"],
            "best_youden": thresholds["best_youden"],
            "high_sensitivity": thresholds.get("high_sensitivity"),
        },
        "robustness": robustness,
        "latency": latency,
        "calibration": {
            "platt_A": float(platt.A), "platt_B": float(platt.B),
        },
    }
    with open(OUT_DIR / "comprehensive_eval.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Plot
    print("Generating plots...")
    plot_all(y_test_np, preds_test, preds_cal, meta_test, boot_raw, cv_results,
             thresholds, robustness, GRAPHS_DIR)

    print(f"\n{'='*60}")
    print(f"  EVALUATION COMPLETE")
    print(f"  Results: {OUT_DIR}/comprehensive_eval.json")
    print(f"  Graphs:  {GRAPHS_DIR}/comprehensive_evaluation.png")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
