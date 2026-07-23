#!/usr/bin/env python3
"""Ultra-thorough evaluation of CVD Watch Model v16.

Wrist-only cardiac arrest detection model.
Generates comprehensive analysis including calibration, threshold sweep,
subgroup analysis, feature importance, clinical utility, and robustness.
All results saved to production/cvd_risk_v16_watch/eval_graphs/.
"""

from __future__ import annotations

import json
import logging
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

OUT_DIR = Path("production/cvd_risk_v16_watch")
EVAL_DIR = OUT_DIR / "eval_graphs"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "primary": "#2196F3",
    "success": "#4CAF50",
    "warning": "#FF9800",
    "danger": "#F44336",
    "purple": "#9C27B0",
    "teal": "#009688",
    "grey": "#9E9E9E",
}


def generate_test_data(n_healthy=75, n_arrest=75, seed=99):
    """Generate wristppg v0.3.0 test data (same as training but different seed)."""
    from wristppg import WristPPGSimulator

    rng = np.random.default_rng(seed)
    ppgs, accels, feat_dicts, biodata_list, labels = [], [], [], [], []

    arrest_profiles = [
        ("cardiac_arrest_vf", 1),
        ("cardial_arrest_asystole", 1),
        ("cardiac_arrest_pulseless_electrical", 1),
        ("pre_arrest_deterioration", 1),
        ("respiratory_failure_pre_arrest", 1),
    ]

    activities = ["rest", "rest", "rest", "walking", "sleep"]
    contact_modes = ["good", "good", "good", "loose", "tight"]

    # Healthy
    for i in range(n_healthy):
        try:
            sim = WristPPGSimulator(seed=int(rng.integers(0, 2**31)))
            result = sim.generate(
                profile="healthy", duration_s=60.0,
                activity=rng.choice(activities),
                contact_mode=rng.choice(contact_modes),
            )
            ppgs.append(result.ppg[:PPG_LENGTH])
            accels.append(result.accel[:PPG_LENGTH])

            feats = {}
            feats.update(extract_wristppg_features(result))
            feats.update(extract_ppg_features(result.ppg[:PPG_LENGTH], fs=25))
            if result.accel[:PPG_LENGTH].ndim == 2:
                feats.update(extract_accel_features(result.accel[:PPG_LENGTH], fs=25))
            feat_dicts.append(feats)

            biodata_list.append(np.array([
                rng.uniform(20, 70), rng.choice([0, 1]), rng.uniform(18, 35),
                0, 0, 0, 0, 0, rng.uniform(1, 3),
                0.97, 36.5, rng.uniform(30, 33),
            ], dtype=np.float32))
            labels.append(0)
        except Exception:
            continue

    # Cardiac arrest
    for i in range(n_arrest):
        try:
            sim = WristPPGSimulator(seed=int(rng.integers(0, 2**31)))
            prof_name, sev = arrest_profiles[i % len(arrest_profiles)]
            result = sim.generate(
                profile=prof_name, duration_s=60.0,
                activity=rng.choice(activities),
                contact_mode=rng.choice(contact_modes),
            )
            ppgs.append(result.ppg[:PPG_LENGTH])
            accels.append(result.accel[:PPG_LENGTH])

            feats = {}
            feats.update(extract_wristppg_features(result))
            feats.update(extract_ppg_features(result.ppg[:PPG_LENGTH], fs=25))
            if result.accel[:PPG_LENGTH].ndim == 2:
                feats.update(extract_accel_features(result.accel[:PPG_LENGTH], fs=25))
            feat_dicts.append(feats)

            target_spo2 = result.meta.get("spo2", 0.70)
            body_temp = result.meta.get("body_temp_c", 36.0)
            skin_temp = result.meta.get("skin_temperature_c", 32.0)
            biodata_list.append(np.array([
                rng.uniform(40, 85), rng.choice([0, 1]), rng.uniform(20, 40),
                rng.integers(0, 5),
                rng.choice([0, 1], p=[0.6, 0.4]),
                rng.choice([0, 1], p=[0.7, 0.3]),
                0, 0, rng.uniform(2, 5),
                target_spo2, body_temp, skin_temp,
            ], dtype=np.float32))
            labels.append(1)
        except Exception:
            continue

    # Pad signals
    ppgs_padded = np.zeros((len(ppgs), PPG_LENGTH), dtype=np.float32)
    accels_padded = np.zeros((len(accels), PPG_LENGTH, 3), dtype=np.float32)
    for i, (p, a) in enumerate(zip(ppgs, accels)):
        L = min(len(p), PPG_LENGTH)
        ppgs_padded[i, :L] = p[:L]
        accels_padded[i, :L, :] = a[:L]

    labels = np.array(labels, dtype=np.float32)
    biodata = np.array(biodata_list, dtype=np.float32)

    # Build feature matrix
    with open(OUT_DIR / "feature_columns.json") as f:
        feature_cols = json.load(f)

    features = np.zeros((len(feat_dicts), len(feature_cols)), dtype=np.float32)
    for i, f in enumerate(feat_dicts):
        for j, col in enumerate(feature_cols):
            features[i, j] = f.get(col, 0.0)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Metadata for subgroup analysis
    sqi = np.array([d.get("sqi", 0.5) for d in feat_dicts])
    hr = np.array([d.get("base_hr", 70.0) for d in feat_dicts])
    signal_len = np.array([d.get("signal_length", PPG_LENGTH) for d in feat_dicts])

    return {
        "X_ppg": ppgs_padded, "X_accel": accels_padded,
        "X_feat": features, "X_bio": biodata,
        "y": labels, "sqi": sqi, "hr": hr,
        "signal_len": signal_len, "feature_cols": feature_cols,
    }


def extract_wristppg_features(result) -> dict:
    """Extract HRV + wristppg features from SimulationResult."""
    from src.data_pipeline_v12 import extract_wristppg_features as base_extract
    feats = base_extract(result)
    feats["base_hr"] = float(np.mean(result.hr_instantaneous_bpm)) if len(result.hr_instantaneous_bpm) > 0 else 70.0
    feats["wppg_spo2"] = float(result.meta.get("spo2", 0.98))
    feats["wppg_body_temp_c"] = float(result.meta.get("body_temp_c", 36.5))
    feats["wppg_skin_temp_c"] = float(result.meta.get("skin_temperature_c", 32.0))
    feats["wppg_ambient_light"] = float(result.meta.get("ambient_light_fraction", 0.0))
    feats["wppg_wrist_artery_depth"] = float(result.meta.get("wrist_anatomy", {}).get("radial_artery_depth_mm", 2.0))
    feats["wppg_wrist_fat_mm"] = float(result.meta.get("wrist_anatomy", {}).get("subcutaneous_fat_mm", 3.0))
    return feats


def load_model():
    """Build v16 from source and load weights."""
    import zipfile
    import tempfile

    model_path = OUT_DIR / "best_model.keras"
    if not model_path.exists():
        model_path = OUT_DIR / "final_model.keras"

    logger.info("Building v16 model from source...")
    from src.model_v16 import build_v16
    model = build_v16(
        ppg_input_shape=(PPG_LENGTH, 1),
        accel_input_shape=(PPG_LENGTH, 3),
        hrv_feature_dim=26,
        biodata_dim=12,
    )

    dummy_ppg = np.zeros((1, PPG_LENGTH, 1), dtype=np.float32)
    dummy_accel = np.zeros((1, PPG_LENGTH, 3), dtype=np.float32)
    dummy_feat = np.zeros((1, 26), dtype=np.float32)
    dummy_bio = np.zeros((1, 12), dtype=np.float32)
    _ = model({"ppg_input": dummy_ppg, "accel_input": dummy_accel,
               "feature_input": dummy_feat, "biodata_input": dummy_bio})

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(model_path, "r") as z:
            z.extract("model.weights.h5", tmpdir)
        weights_path = Path(tmpdir) / "model.weights.h5"
        logger.info("Loading weights from %s", weights_path)
        model.load_weights(str(weights_path))

    test_out = model({"ppg_input": dummy_ppg, "accel_input": dummy_accel,
                      "feature_input": dummy_feat, "biodata_input": dummy_bio})
    logger.info("Sanity check prediction: %.4f", float(test_out.numpy()[0, 0]))
    return model


def get_predictions(model, data):
    preds = model(
        {"ppg_input": data["X_ppg"], "accel_input": data["X_accel"],
         "feature_input": data["X_feat"], "biodata_input": data["X_bio"]},
        training=False,
    )
    return np.array(preds).flatten()


# ============================================================
# SECTION 1: CALIBRATION ANALYSIS
# ============================================================
def calibration_analysis(y_true, y_prob, n_bins=10):
    from sklearn.metrics import brier_score_loss

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_true_probs, bin_counts = [], [], []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_true_probs.append(y_true[mask].mean())
            bin_counts.append(mask.sum())

    bin_centers = np.array(bin_centers)
    bin_true_probs = np.array(bin_true_probs)
    bin_counts = np.array(bin_counts)
    total = bin_counts.sum()
    ece = np.sum(bin_counts * np.abs(bin_true_probs - bin_centers)) / total
    mce = np.max(np.abs(bin_true_probs - bin_centers))
    brier = brier_score_loss(y_true, y_prob)
    reliability = np.sum(bin_counts * (bin_centers - bin_true_probs) ** 2) / total
    resolution = np.sum(bin_counts * (bin_true_probs - y_true.mean()) ** 2) / total
    uncertainty = y_true.mean() * (1 - y_true.mean())

    return {
        "bin_centers": bin_centers, "bin_true_probs": bin_true_probs,
        "bin_counts": bin_counts, "ece": ece, "mce": mce,
        "brier": brier, "reliability": reliability,
        "resolution": resolution, "uncertainty": uncertainty,
    }


def plot_reliability_diagram(cal_data, label, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.plot(cal_data["bin_centers"], cal_data["bin_true_probs"],
            "o-", color=COLORS["primary"], linewidth=2, markersize=8, label="v16 Model")
    ax.fill_between(cal_data["bin_centers"], cal_data["bin_true_probs"],
                    cal_data["bin_centers"], alpha=0.15, color=COLORS["danger"])
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(f"Reliability Diagram ({label})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(cal_data["bin_centers"], cal_data["bin_counts"], width=0.08,
           color=COLORS["primary"], alpha=0.7, edgecolor="white")
    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Prediction Distribution", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Calibration — ECE={cal_data['ece']:.4f}, MCE={cal_data['mce']:.4f}, Brier={cal_data['brier']:.4f}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / f"reliability_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# SECTION 2: THRESHOLD SWEEP
# ============================================================
def threshold_sweep(y_true, y_prob):
    from sklearn.metrics import confusion_matrix

    thresholds = np.arange(0.05, 0.96, 0.01)
    results = {k: [] for k in ["threshold", "precision", "recall", "f1",
                                 "specificity", "npv", "tp", "fp", "fn", "tn",
                                 "nns", "plr", "nlr"]}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        prev = y_true.mean()
        nns = 1 / (prev * rec) if prev > 0 and rec > 0 else float("inf")
        plr = rec / (1 - spec) if spec < 1 else float("inf")
        nlr = (1 - rec) / spec if spec > 0 else float("inf")

        for k, v in [("threshold", t), ("precision", prec), ("recall", rec),
                      ("f1", f1), ("specificity", spec), ("npv", npv),
                      ("tp", int(tp)), ("fp", int(fp)), ("fn", int(fn)),
                      ("tn", int(tn)), ("nns", nns), ("plr", plr), ("nlr", nlr)]:
            results[k].append(v)

    return results


def plot_threshold_analysis(thresh_data, optimal_t, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t = np.array(thresh_data["threshold"])

    ax = axes[0, 0]
    ax.plot(t, thresh_data["precision"], color=COLORS["primary"], linewidth=2, label="Precision")
    ax.plot(t, thresh_data["recall"], color=COLORS["danger"], linewidth=2, label="Recall")
    ax.plot(t, thresh_data["f1"], color=COLORS["success"], linewidth=2, label="F1", linestyle="--")
    ax.axvline(optimal_t, color=COLORS["grey"], linestyle=":", alpha=0.7, label=f"Optimal ({optimal_t:.3f})")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1 vs Threshold", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    ax = axes[0, 1]
    ax.plot(t, thresh_data["recall"], color=COLORS["danger"], linewidth=2, label="Sensitivity (Recall)")
    ax.plot(t, thresh_data["specificity"], color=COLORS["primary"], linewidth=2, label="Specificity")
    ax.axvline(optimal_t, color=COLORS["grey"], linestyle=":", alpha=0.7, label=f"Optimal ({optimal_t:.3f})")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Sensitivity vs Specificity", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    ax = axes[1, 0]
    ax.plot(t, thresh_data["precision"], color=COLORS["success"], linewidth=2, label="PPV (Precision)")
    ax.plot(t, thresh_data["npv"], color=COLORS["purple"], linewidth=2, label="NPV")
    ax.axvline(optimal_t, color=COLORS["grey"], linestyle=":", alpha=0.7, label=f"Optimal ({optimal_t:.3f})")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Positive/Negative Predictive Value", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    ax = axes[1, 1]
    nns = np.clip(np.array(thresh_data["nns"]), 0, 20)
    ax.plot(t, nns, color=COLORS["warning"], linewidth=2)
    ax.axvline(optimal_t, color=COLORS["grey"], linestyle=":", alpha=0.7, label=f"Optimal ({optimal_t:.3f})")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Number Needed to Screen", fontsize=12)
    ax.set_title("Clinical Utility (NNS, capped at 20)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.suptitle("Threshold Analysis — CVD v16 (Wrist Cardiac Arrest)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# SECTION 3: PRECISION-RECALL CURVE
# ============================================================
def plot_precision_recall_curve(y_true, y_prob, out_dir):
    from sklearn.metrics import precision_recall_curve, auc
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec, prec, color=COLORS["primary"], linewidth=2, label=f"PR AUC = {pr_auc:.4f}")
    ax.axhline(baseline, color=COLORS["grey"], linestyle="--", alpha=0.5,
               label=f"Baseline (prevalence = {baseline:.3f})")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (PPV)", fontsize=12)
    ax.set_title("Precision-Recall Curve — CVD v16", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    return float(pr_auc)


# ============================================================
# SECTION 4: SUBGROUP ANALYSIS
# ============================================================
def subgroup_analysis(y_true, y_prob, metadata, threshold, out_dir):
    from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

    results = {}

    # SQI quartiles
    sqi_q = np.percentile(metadata["sqi"], [25, 50, 75])
    sqi_groups = {
        "Q1 (Lowest SQI)": metadata["sqi"] <= sqi_q[0],
        "Q2": (metadata["sqi"] > sqi_q[0]) & (metadata["sqi"] <= sqi_q[1]),
        "Q3": (metadata["sqi"] > sqi_q[1]) & (metadata["sqi"] <= sqi_q[2]),
        "Q4 (Highest SQI)": metadata["sqi"] > sqi_q[2],
    }
    results["sqi"] = {}
    for name, mask in sqi_groups.items():
        if mask.sum() > 5 and len(np.unique(y_true[mask])) > 1:
            results["sqi"][name] = {
                "n": int(mask.sum()),
                "auroc": float(roc_auc_score(y_true[mask], y_prob[mask])),
                "f1": float(f1_score(y_true[mask], (y_prob[mask] >= threshold).astype(int), zero_division=0)),
                "recall": float(recall_score(y_true[mask], (y_prob[mask] >= threshold).astype(int), zero_division=0)),
                "precision": float(precision_score(y_true[mask], (y_prob[mask] >= threshold).astype(int), zero_division=0)),
            }

    # Heart rate ranges
    hr_groups = {
        "<60 bpm": metadata["hr"] < 60,
        "60-100 bpm": (metadata["hr"] >= 60) & (metadata["hr"] < 100),
        ">100 bpm": metadata["hr"] >= 100,
    }
    results["hr"] = {}
    for name, mask in hr_groups.items():
        if mask.sum() > 5 and len(np.unique(y_true[mask])) > 1:
            results["hr"][name] = {
                "n": int(mask.sum()),
                "auroc": float(roc_auc_score(y_true[mask], y_prob[mask])),
                "f1": float(f1_score(y_true[mask], (y_prob[mask] >= threshold).astype(int), zero_division=0)),
                "recall": float(recall_score(y_true[mask], (y_prob[mask] >= threshold).astype(int), zero_division=0)),
                "precision": float(precision_score(y_true[mask], (y_prob[mask] >= threshold).astype(int), zero_division=0)),
            }

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (group_name, group_data) in enumerate(results.items()):
        if not group_data:
            continue
        ax = axes[idx]
        names = list(group_data.keys())
        aurocs = [group_data[n].get("auroc", 0) for n in names]
        f1s = [group_data[n].get("f1", 0) for n in names]
        recalls = [group_data[n].get("recall", 0) for n in names]
        counts = [group_data[n].get("n", 0) for n in names]

        x = np.arange(len(names))
        width = 0.25
        ax.bar(x - width, aurocs, width, color=COLORS["primary"], label="AUROC")
        ax.bar(x, f1s, width, color=COLORS["success"], label="F1")
        ax.bar(x + width, recalls, width, color=COLORS["danger"], label="Recall")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n(n={c})" for n, c in zip(names, counts)], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(f"By {group_name.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Subgroup Performance — CVD v16 (Wrist Cardiac Arrest)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "subgroup_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    return results


# ============================================================
# SECTION 5: FEATURE IMPORTANCE
# ============================================================
def permutation_importance(model, data, n_repeats=5):
    from sklearn.metrics import roc_auc_score

    y_prob_base = get_predictions(model, data)
    base_auroc = roc_auc_score(data["y"], y_prob_base)

    importances = {}
    n_features = data["X_feat"].shape[1]
    feature_cols = data["feature_cols"]
    rng = np.random.default_rng(42)

    for i in range(n_features):
        scores = []
        for _ in range(n_repeats):
            X_feat_perm = data["X_feat"].copy()
            X_feat_perm[:, i] = rng.permutation(X_feat_perm[:, i])
            perm_data = {**data, "X_feat": X_feat_perm}
            y_prob_perm = get_predictions(model, perm_data)
            perm_auroc = roc_auc_score(data["y"], y_prob_perm)
            scores.append(base_auroc - perm_auroc)
        importances[feature_cols[i]] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }

    return importances, base_auroc


def plot_feature_importance(importances, out_dir, top_n=25):
    sorted_feats = sorted(importances.items(), key=lambda x: x[1]["mean"], reverse=True)[:top_n]
    names = [f[0] for f in sorted_feats]
    means = [f[1]["mean"] for f in sorted_feats]
    stds = [f[1]["std"] for f in sorted_feats]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(names))
    colors = [COLORS["danger"] if m > 0.005 else COLORS["primary"] for m in means]
    ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.8, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("AUROC Drop (mean over repeats)", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances (Permutation) — CVD v16", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# SECTION 6: TRAINING DYNAMICS
# ============================================================
def training_dynamics_analysis(out_dir):
    with open(OUT_DIR / "training_history.json") as f:
        history = json.load(f)

    epochs = list(range(1, len(history["loss"]) + 1))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], color=COLORS["primary"], linewidth=2, label="Train")
    ax.plot(epochs, history["val_loss"], color=COLORS["danger"], linewidth=2, label="Val")
    ax.fill_between(epochs, history["loss"], history["val_loss"], alpha=0.1, color=COLORS["danger"])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history["auc"], color=COLORS["primary"], linewidth=2, label="Train")
    ax.plot(epochs, history["val_auc"], color=COLORS["danger"], linewidth=2, label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUROC")
    ax.set_title("Training vs Validation AUROC", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(epochs, history["precision"], color=COLORS["primary"], linewidth=2, label="Train")
    ax.plot(epochs, history["val_precision"], color=COLORS["danger"], linewidth=2, label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Precision")
    ax.set_title("Training vs Validation Precision", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history["recall"], color=COLORS["primary"], linewidth=2, label="Train")
    ax.plot(epochs, history["val_recall"], color=COLORS["danger"], linewidth=2, label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Recall")
    ax.set_title("Training vs Validation Recall", fontsize=12, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    loss_gap = [v - t for t, v in zip(history["loss"], history["val_loss"])]
    ax.plot(epochs, loss_gap, color=COLORS["warning"], linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val Loss - Train Loss")
    ax.set_title("Overfitting Gap (Loss)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    min_gap_epoch = np.argmin(loss_gap)
    ax.axvline(min_gap_epoch + 1, color=COLORS["success"], linestyle="--",
               label=f"Min gap at epoch {min_gap_epoch + 1}")
    ax.legend()

    ax = axes[1, 2]
    lrs = history.get("learning_rate", [3e-4] * len(epochs))
    ax.plot(epochs, lrs, color=COLORS["purple"], linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.suptitle("Training Dynamics — CVD v16 (Wrist Cardiac Arrest)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "training_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close()

    min_val_loss_epoch = np.argmin(history["val_loss"]) + 1
    max_val_auc_epoch = np.argmax(history["val_auc"]) + 1
    return {
        "epochs_trained": len(epochs),
        "min_val_loss_epoch": min_val_loss_epoch,
        "min_val_loss": float(min(history["val_loss"])),
        "max_val_auc_epoch": max_val_auc_epoch,
        "max_val_auc": float(max(history["val_auc"])),
        "final_loss_gap": float(loss_gap[-1]),
        "train_loss_start": float(history["loss"][0]),
        "train_loss_end": float(history["loss"][-1]),
    }


# ============================================================
# SECTION 7: ROBUSTNESS
# ============================================================
def robustness_analysis(model, data, out_dir):
    from sklearn.metrics import roc_auc_score

    y_prob_clean = get_predictions(model, data)
    auroc_clean = roc_auc_score(data["y"], y_prob_clean)

    noise_levels = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    aurocs = [auroc_clean]
    snrs = ["Clean"]
    rng = np.random.default_rng(42)

    for noise_std in noise_levels:
        noisy_ppg = data["X_ppg"] + rng.normal(0, noise_std, data["X_ppg"].shape).astype(np.float32)
        noisy_data = {**data, "X_ppg": noisy_ppg}
        y_prob_noisy = get_predictions(model, noisy_data)
        auroc_noisy = roc_auc_score(data["y"], y_prob_noisy)
        aurocs.append(auroc_noisy)
        snrs.append(f"σ={noise_std}")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS["success"]] + [COLORS["warning"] if a > auroc_clean - 0.05 else COLORS["danger"] for a in aurocs[1:]]
    bars = ax.bar(snrs, aurocs, color=colors, alpha=0.8, edgecolor="white")
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Model Robustness — Noise Injection Test (v16)", fontsize=14, fontweight="bold")
    ax.set_ylim(0.3, 1.05)
    ax.axhline(auroc_clean, color=COLORS["success"], linestyle="--", alpha=0.5, label=f"Clean: {auroc_clean:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "robustness_noise.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {"clean_auroc": auroc_clean, "noisy_aurocs": dict(zip(snrs[1:], aurocs[1:]))}


# ============================================================
# SECTION 8: CARDIAC ARREST PROFILE BREAKDOWN
# ============================================================
def cardiac_arrest_profile_analysis(model, out_dir):
    from sklearn.metrics import f1_score
    from wristppg import WristPPGSimulator

    profiles = {
        "healthy": ("healthy", 0, "Healthy"),
        "cardiac_arrest_vf": ("cardiac_arrest_vf", 1, "VF Arrest"),
        "cardiac_arrest_asystole": ("cardiac_arrest_asystole", 1, "Asystole"),
        "cardiac_arrest_pea": ("cardiac_arrest_pulseless_electrical", 1, "PEA"),
        "pre_arrest": ("pre_arrest_deterioration", 1, "Pre-Arrest"),
        "resp_failure": ("respiratory_failure_pre_arrest", 1, "Resp Failure"),
    }

    profile_results = {}
    rng = np.random.default_rng(99)

    for name, (prof, label, display) in profiles.items():
        ppgs, accels, labels_arr = [], [], []
        for _ in range(20):
            try:
                sim = WristPPGSimulator(seed=int(rng.integers(0, 2**31)))
                result = sim.generate(profile=prof, duration_s=60.0, activity="rest")
                ppgs.append(result.ppg[:PPG_LENGTH])
                accels.append(result.accel[:PPG_LENGTH])
                labels_arr.append(label)
            except Exception:
                continue

        if len(ppgs) < 5:
            continue

        feat_dicts = []
        for p, a in zip(ppgs, accels):
            feats = {}
            feats.update(extract_ppg_features(p, fs=25))
            if a.ndim == 2:
                feats.update(extract_accel_features(a, fs=25))
            feat_dicts.append(feats)

        with open(OUT_DIR / "feature_columns.json") as f:
            feature_cols = json.load(f)

        features = np.zeros((len(feat_dicts), len(feature_cols)), dtype=np.float32)
        for i, f in enumerate(feat_dicts):
            for j, col in enumerate(feature_cols):
                features[i, j] = f.get(col, 0.0)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        ppgs_arr = np.zeros((len(ppgs), PPG_LENGTH, 1), dtype=np.float32)
        accels_arr = np.zeros((len(accels), PPG_LENGTH, 3), dtype=np.float32)
        for i, (p, a) in enumerate(zip(ppgs, accels)):
            L = min(len(p), PPG_LENGTH)
            ppgs_arr[i, :L, 0] = p[:L]
            accels_arr[i, :L, :] = a[:L]

        biodata = np.zeros((len(ppgs), 12), dtype=np.float32)
        biodata[:, 0] = rng.uniform(20, 80, len(ppgs))     # age
        biodata[:, 1] = rng.choice([0, 1], len(ppgs))      # sex
        biodata[:, 2] = rng.uniform(18, 40, len(ppgs))     # bmi
        biodata[:, 3] = rng.integers(0, 8, len(ppgs))      # comorbidity_count
        biodata[:, 4] = rng.choice([0, 1], len(ppgs), p=[0.95, 0.05])
        biodata[:, 5] = rng.choice([0, 1], len(ppgs), p=[0.97, 0.03])
        biodata[:, 6] = 0  # on_ecmo
        biodata[:, 7] = 0  # on_rrt
        biodata[:, 8] = rng.uniform(1, 5, len(ppgs))      # acuity_score
        if label == 1:
            biodata[:, 9] = rng.uniform(0.5, 0.85, len(ppgs))   # spo2
            biodata[:, 10] = rng.uniform(34, 38, len(ppgs))     # body_temp
            biodata[:, 11] = rng.uniform(28, 33, len(ppgs))     # skin_temp
        else:
            biodata[:, 9] = 0.97
            biodata[:, 10] = 36.5
            biodata[:, 11] = rng.uniform(30, 33, len(ppgs))

        preds = model(
            {"ppg_input": ppgs_arr, "accel_input": accels_arr,
             "feature_input": features, "biodata_input": biodata},
            training=False,
        )
        y_prob = np.array(preds).flatten()
        y_true = np.array(labels_arr)

        profile_results[display] = {
            "n": len(ppgs),
            "label": label,
            "mean_prob": float(y_prob.mean()),
            "std_prob": float(y_prob.std()),
            "min_prob": float(y_prob.min()),
            "max_prob": float(y_prob.max()),
        }

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(profile_results.keys())
    means = [profile_results[n]["mean_prob"] for n in names]
    stds = [profile_results[n]["std_prob"] for n in names]
    labels = [profile_results[n]["label"] for n in names]

    colors = [COLORS["success"] if l == 0 else COLORS["danger"] for l in labels]
    bars = ax.bar(names, means, yerr=stds, color=colors, alpha=0.8, edgecolor="white", capsize=5)

    # Load optimal threshold
    try:
        with open(OUT_DIR / "optimal_threshold.json") as f:
            opt_t = json.load(f)["threshold"]
        ax.axhline(opt_t, color=COLORS["grey"], linestyle="--", alpha=0.7, label=f"Threshold ({opt_t:.3f})")
    except Exception:
        pass

    ax.set_ylabel("Mean Predicted Probability", fontsize=12)
    ax.set_title("Per-Profile Prediction — CVD v16 (Wrist Cardiac Arrest)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "cardiac_arrest_profiles.png", dpi=150, bbox_inches="tight")
    plt.close()

    return profile_results


# ============================================================
# SECTION 9: CONFUSION MATRIX DETAILED
# ============================================================
def plot_detailed_confusion(y_true, y_prob, threshold, out_dir):
    from sklearn.metrics import confusion_matrix
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Detailed breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Standard confusion matrix
    ax = axes[0]
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
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

    # Normalized confusion matrix
    ax = axes[1]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_title("Normalized Confusion Matrix", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_norm[i, j]:.3f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=16)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    plt.suptitle(f"Confusion Matrix — Threshold={threshold:.3f}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_detailed.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


# ============================================================
# MAIN
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("ULTRA-THOROUGH EVALUATION: CVD Watch Model v16")
    logger.info("Wrist-only Cardiac Arrest Detection")
    logger.info("=" * 60)

    model = load_model()
    model.summary(print_fn=logger.info)
    data = generate_test_data(n_healthy=75, n_arrest=75, seed=99)
    y_prob = get_predictions(model, data)
    y_true = data["y"]

    logger.info("Test set: %d samples (%d healthy, %d arrest)",
                len(y_true), int((y_true == 0).sum()), int((y_true == 1).sum()))

    # Load optimal threshold from training
    with open(OUT_DIR / "optimal_threshold.json") as f:
        optimal_threshold = json.load(f)["threshold"]
    logger.info("Optimal threshold from training: %.3f", optimal_threshold)

    all_results = {"optimal_threshold": optimal_threshold}

    # 1. Calibration
    logger.info("\n[1/9] Calibration analysis...")
    cal = calibration_analysis(y_true, y_prob)
    plot_reliability_diagram(cal, "test", EVAL_DIR)
    all_results["calibration"] = {
        "ece": cal["ece"], "mce": cal["mce"],
        "brier": cal["brier"], "reliability": cal["reliability"],
        "resolution": cal["resolution"], "uncertainty": cal["uncertainty"],
    }

    # 2. Threshold sweep
    logger.info("[2/9] Threshold sweep analysis...")
    thresh_data = threshold_sweep(y_true, y_prob)
    plot_threshold_analysis(thresh_data, optimal_threshold, EVAL_DIR)
    best_idx = np.argmax(thresh_data["f1"])
    all_results["threshold_analysis"] = {
        "optimal_threshold": float(thresh_data["threshold"][best_idx]),
        "optimal_f1": float(thresh_data["f1"][best_idx]),
    }

    # 3. Precision-Recall curve
    logger.info("[3/9] Precision-Recall curve...")
    pr_auc = plot_precision_recall_curve(y_true, y_prob, EVAL_DIR)
    all_results["pr_auc"] = pr_auc

    # 4. Subgroup analysis
    logger.info("[4/9] Subgroup analysis...")
    metadata = {"sqi": data["sqi"], "hr": data["hr"], "signal_len": data["signal_len"]}
    subgroup_results = subgroup_analysis(y_true, y_prob, metadata, optimal_threshold, EVAL_DIR)
    all_results["subgroup_analysis"] = subgroup_results

    # 5. Feature importance
    logger.info("[5/9] Feature importance (permutation)...")
    importances, base_auroc = permutation_importance(model, data, n_repeats=5)
    plot_feature_importance(importances, EVAL_DIR, top_n=25)
    sorted_imp = sorted(importances.items(), key=lambda x: x[1]["mean"], reverse=True)
    all_results["feature_importance"] = {
        "base_auroc": float(base_auroc),
        "top_10": {k: v for k, v in sorted_imp[:10]},
    }

    # 6. Training dynamics
    logger.info("[6/9] Training dynamics analysis...")
    dynamics = training_dynamics_analysis(EVAL_DIR)
    all_results["training_dynamics"] = dynamics

    # 7. Robustness
    logger.info("[7/9] Robustness analysis...")
    robustness = robustness_analysis(model, data, EVAL_DIR)
    all_results["robustness"] = robustness

    # 8. Cardiac arrest profile breakdown
    logger.info("[8/9] Cardiac arrest profile analysis...")
    profiles = cardiac_arrest_profile_analysis(model, EVAL_DIR)
    all_results["cardiac_arrest_profiles"] = profiles

    # 9. Detailed confusion matrix
    logger.info("[9/9] Detailed confusion matrix...")
    cm_detail = plot_detailed_confusion(y_true, y_prob, optimal_threshold, EVAL_DIR)
    all_results["confusion_matrix"] = cm_detail

    # Save all results
    results_path = EVAL_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nAll results saved to %s", results_path)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY — CVD v16 (Wrist Cardiac Arrest)")
    logger.info("=" * 60)
    logger.info("  Calibration:  ECE=%.4f  MCE=%.4f  Brier=%.4f", cal["ece"], cal["mce"], cal["brier"])
    logger.info("  Threshold:    Optimal=%.3f (F1=%.4f)", all_results["threshold_analysis"]["optimal_threshold"],
                all_results["threshold_analysis"]["optimal_f1"])
    logger.info("  PR AUC:       %.4f", pr_auc)
    logger.info("  Robustness:   Clean=%.4f  Max noise drop=%.4f", robustness["clean_auroc"],
                robustness["clean_auroc"] - min(robustness["noisy_aurocs"].values()))
    logger.info("  Top features: %s", ", ".join(list(all_results["feature_importance"]["top_10"].keys())[:5]))
    logger.info("  Training:     %d epochs, min val_loss at epoch %d", dynamics["epochs_trained"],
                dynamics["min_val_loss_epoch"])
    logger.info("  Confusion:    TP=%d FP=%d FN=%d TN=%d", cm_detail["tp"], cm_detail["fp"],
                cm_detail["fn"], cm_detail["tn"])
    logger.info("  All graphs saved to %s", EVAL_DIR)


if __name__ == "__main__":
    main()
