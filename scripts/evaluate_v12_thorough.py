#!/usr/bin/env python3
"""Ultra-thorough evaluation of CVD Watch Model v12.

Generates comprehensive analysis including calibration, threshold sweep,
subgroup analysis, feature importance, clinical utility, and robustness.
All results saved to production/cvd_risk_v12_watch/eval_graphs/.
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
from scipy.special import softmax as scipy_softmax

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

OUT_DIR = Path("production/cvd_risk_v12_watch")
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


def load_test_data():
    """Regenerate the exact same test data as training."""
    logger.info("Loading real data...")
    patient_groups = load_real_data_by_patient()
    _, _, test_p = patient_level_split(patient_groups)
    test_sigs, test_accels, test_feats, y_test = flatten_patients(patient_groups, test_p)

    test_feat_dicts = [extract_features_for_signal(s, a) for s, a in zip(test_sigs, test_accels)]

    # Build feature columns from training set (same as train script)
    train_p, val_p, _ = patient_level_split(patient_groups)
    train_sigs, train_accels, train_feats_r, _ = flatten_patients(patient_groups, train_p)
    val_sigs, val_accels, val_feats_r, _ = flatten_patients(patient_groups, val_p)

    train_feat_dicts = [extract_features_for_signal(s, a) for s, a in zip(train_sigs, train_accels)]
    val_feat_dicts = [extract_features_for_signal(s, a) for s, a in zip(val_sigs, val_accels)]

    synth_ppgs, synth_accels, synth_feats, synth_labels, _ = \
        generate_wristppg_synthetic(n_healthy=50, n_at_risk=50, n_borderline=20, seed=42)
    synth_feat_dicts = []
    for i in range(len(synth_ppgs)):
        feats = extract_features_for_signal(synth_ppgs[i], synth_accels[i])
        feats["base_hr"] = synth_feats[i].get("base_hr", 70.0)
        for k, v in synth_feats[i].items():
            if k.startswith("wppg_"):
                feats[k] = v
        synth_feat_dicts.append(feats)

    all_feat_dicts = train_feat_dicts + val_feat_dicts + test_feat_dicts + synth_feat_dicts
    feature_cols = sorted(set().union(*[f.keys() for f in all_feat_dicts]))

    X_test_ppg, X_test_accel, X_test_feat = build_arrays(
        test_sigs, test_accels, test_feat_dicts, feature_cols)
    X_test_bio = build_biodata_array(len(X_test_ppg), seed=44)

    # Extract per-sample signal metadata for subgroup analysis
    test_sqi = []
    test_hr = []
    test_signal_len = []
    for d in test_feat_dicts:
        test_sqi.append(d.get("sqi", 0.5))
        test_hr.append(d.get("base_hr", 70.0))
        test_signal_len.append(d.get("signal_length", PPG_LENGTH))
    test_sqi = np.array(test_sqi)
    test_hr = np.array(test_hr)
    test_signal_len = np.array(test_signal_len)

    return {
        "X_ppg": X_test_ppg, "X_accel": X_test_accel,
        "X_feat": X_test_feat, "X_bio": X_test_bio,
        "y": y_test, "sqi": test_sqi, "hr": test_hr,
        "signal_len": test_signal_len, "feature_cols": feature_cols,
    }


def extract_features_for_signal(ppg, accel=None, fs=25):
    """Extract combined PPG + accel features."""
    feats = extract_ppg_features(ppg, fs=fs)
    if accel is not None and accel.ndim == 2 and accel.shape[-1] == 3:
        accel_feats = extract_accel_features(accel, fs=fs)
        feats.update(accel_feats)
    return feats


def load_model():
    """Build v12 from source and load weights from .keras file."""
    import zipfile
    import tempfile

    model_path = OUT_DIR / "best_model.keras"
    if not model_path.exists():
        model_path = OUT_DIR / "final_model.keras"

    logger.info("Building v12 model from source...")
    from src.model_v12 import build_v12
    model = build_v12(
        ppg_input_shape=(PPG_LENGTH, 1),
        accel_input_shape=(PPG_LENGTH, 3),
        hrv_feature_dim=56,
        biodata_dim=9,
    )

    # Build all layers
    dummy_ppg = np.zeros((1, PPG_LENGTH, 1), dtype=np.float32)
    dummy_accel = np.zeros((1, PPG_LENGTH, 3), dtype=np.float32)
    dummy_feat = np.zeros((1, 56), dtype=np.float32)
    dummy_bio = np.zeros((1, 9), dtype=np.float32)
    _ = model({"ppg_input": dummy_ppg, "accel_input": dummy_accel,
               "feature_input": dummy_feat, "biodata_input": dummy_bio})

    # Extract weights from .keras zip and load
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(model_path, "r") as z:
            z.extract("model.weights.h5", tmpdir)
        weights_path = Path(tmpdir) / "model.weights.h5"
        logger.info("Loading weights from %s", weights_path)
        model.load_weights(str(weights_path))

    # Sanity check
    test_out = model({"ppg_input": dummy_ppg, "accel_input": dummy_accel,
                      "feature_input": dummy_feat, "biodata_input": dummy_bio})
    logger.info("Sanity check prediction: %.4f", float(test_out.numpy()[0, 0]))

    return model


def get_predictions(model, data):
    """Get probability predictions."""
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
    """Compute calibration metrics and reliability diagram data."""
    from sklearn.metrics import brier_score_loss

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_true_probs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_true_probs.append(y_true[mask].mean())
            bin_counts.append(mask.sum())

    # ECE (Expected Calibration Error)
    bin_centers = np.array(bin_centers)
    bin_true_probs = np.array(bin_true_probs)
    bin_counts = np.array(bin_counts)
    total = bin_counts.sum()
    ece = np.sum(bin_counts * np.abs(bin_true_probs - bin_centers)) / total

    # MCE (Maximum Calibration Error)
    mce = np.max(np.abs(bin_true_probs - bin_centers))

    # Brier score decomposition
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
    """Plot reliability diagram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.plot(cal_data["bin_centers"], cal_data["bin_true_probs"],
            "o-", color=COLORS["primary"], linewidth=2, markersize=8, label="v12 Model")
    ax.fill_between(cal_data["bin_centers"], cal_data["bin_true_probs"],
                    cal_data["bin_centers"], alpha=0.15, color=COLORS["danger"])
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title(f"Reliability Diagram ({label})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right: Calibration histogram
    ax = axes[1]
    ax.bar(cal_data["bin_centers"], cal_data["bin_counts"], width=0.08,
           color=COLORS["primary"], alpha=0.7, edgecolor="white")
    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Prediction Distribution", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Calibration Analysis — ECE={cal_data['ece']:.4f}, MCE={cal_data['mce']:.4f}, Brier={cal_data['brier']:.4f}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / f"reliability_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# SECTION 2: THRESHOLD SWEEP ANALYSIS
# ============================================================
def threshold_sweep(y_true, y_prob):
    """Compute metrics at various thresholds."""
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                  confusion_matrix, roc_auc_score)

    thresholds = np.arange(0.05, 0.96, 0.01)
    results = {
        "threshold": [], "precision": [], "recall": [], "f1": [],
        "specificity": [], "npv": [], "tp": [], "fp": [], "fn": [], "tn": [],
        "nns": [], "plr": [], "nlr": [],
    }

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        # Clinical metrics
        n_test = len(y_true)
        prev = y_true.mean()
        nns = 1 / (prev * rec) if prev > 0 and rec > 0 else float("inf")
        plr = rec / (1 - spec) if spec < 1 else float("inf")
        nlr = (1 - rec) / spec if spec > 0 else float("inf")

        results["threshold"].append(t)
        results["precision"].append(prec)
        results["recall"].append(rec)
        results["f1"].append(f1)
        results["specificity"].append(spec)
        results["npv"].append(npv)
        results["tp"].append(int(tp))
        results["fp"].append(int(fp))
        results["fn"].append(int(fn))
        results["tn"].append(int(tn))
        results["nns"].append(nns)
        results["plr"].append(plr)
        results["nlr"].append(nlr)

    return results


def plot_threshold_analysis(thresh_data, out_dir):
    """Plot threshold sweep analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    t = np.array(thresh_data["threshold"])

    # Precision-Recall vs Threshold
    ax = axes[0, 0]
    ax.plot(t, thresh_data["precision"], color=COLORS["primary"], linewidth=2, label="Precision")
    ax.plot(t, thresh_data["recall"], color=COLORS["danger"], linewidth=2, label="Recall")
    ax.plot(t, thresh_data["f1"], color=COLORS["success"], linewidth=2, label="F1", linestyle="--")
    ax.axvline(0.345, color=COLORS["grey"], linestyle=":", alpha=0.7, label="Optimal (0.345)")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1 vs Threshold", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Sensitivity-Specificity crossover
    ax = axes[0, 1]
    ax.plot(t, thresh_data["recall"], color=COLORS["danger"], linewidth=2, label="Sensitivity (Recall)")
    ax.plot(t, thresh_data["specificity"], color=COLORS["primary"], linewidth=2, label="Specificity")
    ax.axvline(0.345, color=COLORS["grey"], linestyle=":", alpha=0.7, label="Optimal (0.345)")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Sensitivity vs Specificity", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # PPV vs NPV
    ax = axes[1, 0]
    ax.plot(t, thresh_data["precision"], color=COLORS["success"], linewidth=2, label="PPV (Precision)")
    ax.plot(t, thresh_data["npv"], color=COLORS["purple"], linewidth=2, label="NPV")
    ax.axvline(0.345, color=COLORS["grey"], linestyle=":", alpha=0.7, label="Optimal (0.345)")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Positive/Negative Predictive Value", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Clinical utility: NNS
    ax = axes[1, 1]
    nns = np.array(thresh_data["nns"])
    nns_capped = np.clip(nns, 0, 20)
    ax.plot(t, nns_capped, color=COLORS["warning"], linewidth=2)
    ax.axvline(0.345, color=COLORS["grey"], linestyle=":", alpha=0.7, label="Optimal (0.345)")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Number Needed to Screen", fontsize=12)
    ax.set_title("Clinical Utility (NNS, capped at 20)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.suptitle("Threshold Analysis — CVD v12", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# SECTION 3: PRECISION-RECALL CURVE
# ============================================================
def plot_precision_recall_curve(y_true, y_prob, out_dir):
    """Plot precision-recall curve with baseline."""
    from sklearn.metrics import precision_recall_curve, auc

    prec, rec, _ = pr_curve = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec, prec, color=COLORS["primary"], linewidth=2,
            label=f"PR AUC = {pr_auc:.4f}")
    ax.axhline(baseline, color=COLORS["grey"], linestyle="--", alpha=0.5,
               label=f"Baseline (prevalence = {baseline:.3f})")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (PPV)", fontsize=12)
    ax.set_title("Precision-Recall Curve — CVD v12 (Real Test)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve_real_test.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# SECTION 4: SUBGROUP ANALYSIS
# ============================================================
def subgroup_analysis(y_true, y_prob, metadata, out_dir):
    """Analyze performance across signal subgroups."""
    from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

    results = {}

    # SQI quartiles
    sqi_quartiles = np.percentile(metadata["sqi"], [25, 50, 75])
    sqi_groups = {
        "Q1 (Lowest SQI)": metadata["sqi"] <= sqi_quartiles[0],
        "Q2": (metadata["sqi"] > sqi_quartiles[0]) & (metadata["sqi"] <= sqi_quartiles[1]),
        "Q3": (metadata["sqi"] > sqi_quartiles[1]) & (metadata["sqi"] <= sqi_quartiles[2]),
        "Q4 (Highest SQI)": metadata["sqi"] > sqi_quartiles[2],
    }
    results["sqi"] = {}
    for name, mask in sqi_groups.items():
        if mask.sum() > 0 and len(np.unique(y_true[mask])) > 1:
            results["sqi"][name] = {
                "n": int(mask.sum()),
                "auroc": float(roc_auc_score(y_true[mask], y_prob[mask])),
                "f1": float(f1_score(y_true[mask], (y_prob[mask] >= 0.345).astype(int), zero_division=0)),
                "recall": float(recall_score(y_true[mask], (y_prob[mask] >= 0.345).astype(int), zero_division=0)),
            }
        elif mask.sum() > 0:
            results["sqi"][name] = {"n": int(mask.sum()), "auroc": float("nan"),
                                    "f1": 0.0, "recall": 0.0}

    # Heart rate ranges
    hr_groups = {
        "<60 bpm": metadata["hr"] < 60,
        "60-100 bpm": (metadata["hr"] >= 60) & (metadata["hr"] < 100),
        ">100 bpm": metadata["hr"] >= 100,
    }
    results["hr"] = {}
    for name, mask in hr_groups.items():
        if mask.sum() > 0 and len(np.unique(y_true[mask])) > 1:
            results["hr"][name] = {
                "n": int(mask.sum()),
                "auroc": float(roc_auc_score(y_true[mask], y_prob[mask])),
                "f1": float(f1_score(y_true[mask], (y_prob[mask] >= 0.345).astype(int), zero_division=0)),
                "recall": float(recall_score(y_true[mask], (y_prob[mask] >= 0.345).astype(int), zero_division=0)),
            }
        elif mask.sum() > 0:
            results["hr"][name] = {"n": int(mask.sum()), "auroc": float("nan"),
                                   "f1": 0.0, "recall": 0.0}

    # Plot subgroup results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (group_name, group_data) in enumerate(results.items()):
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
        for bar_group, vals in zip([aurocs, f1s, recalls], [ax.patches[i::3] for i in range(3)]):
            for bar, val in zip(vals, vals):
                pass  # skip value labels for cleanliness

    plt.suptitle("Subgroup Performance Analysis — CVD v12 (Real Test)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "subgroup_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results


# ============================================================
# SECTION 5: FEATURE IMPORTANCE (Permutation)
# ============================================================
def permutation_importance(model, data, n_repeats=10):
    """Compute permutation importance for each feature."""
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
    """Plot feature importance bar chart."""
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
    ax.set_title(f"Top {top_n} Feature Importances (Permutation) — CVD v12", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# SECTION 6: TRAINING DYNAMICS ANALYSIS
# ============================================================
def training_dynamics_analysis(out_dir):
    """Analyze training curves for overfitting and learning dynamics."""
    with open(OUT_DIR / "training_history.json") as f:
        history = json.load(f)

    epochs = list(range(1, len(history["loss"]) + 1))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], color=COLORS["primary"], linewidth=2, label="Train")
    ax.plot(epochs, history["val_loss"], color=COLORS["danger"], linewidth=2, label="Val")
    ax.fill_between(epochs, history["loss"], history["val_loss"], alpha=0.1, color=COLORS["danger"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AUROC
    ax = axes[0, 1]
    ax.plot(epochs, history["auc"], color=COLORS["primary"], linewidth=2, label="Train")
    ax.plot(epochs, history["val_auc"], color=COLORS["danger"], linewidth=2, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUROC")
    ax.set_title("Training vs Validation AUROC", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision
    ax = axes[0, 2]
    ax.plot(epochs, history["precision"], color=COLORS["primary"], linewidth=2, label="Train")
    ax.plot(epochs, history["val_precision"], color=COLORS["danger"], linewidth=2, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision")
    ax.set_title("Training vs Validation Precision", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Recall
    ax = axes[1, 0]
    ax.plot(epochs, history["recall"], color=COLORS["primary"], linewidth=2, label="Train")
    ax.plot(epochs, history["val_recall"], color=COLORS["danger"], linewidth=2, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall")
    ax.set_title("Training vs Validation Recall", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Overfitting gap
    ax = axes[1, 1]
    loss_gap = [v - t for t, v in zip(history["loss"], history["val_loss"])]
    ax.plot(epochs, loss_gap, color=COLORS["warning"], linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss - Train Loss")
    ax.set_title("Overfitting Gap (Loss)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    min_gap_epoch = np.argmin(loss_gap)
    ax.axvline(min_gap_epoch + 1, color=COLORS["success"], linestyle="--",
               label=f"Min gap at epoch {min_gap_epoch + 1}")
    ax.legend()

    # Learning rate schedule
    ax = axes[1, 2]
    lrs = history.get("learning_rate", [3e-4] * len(epochs))
    ax.plot(epochs, lrs, color=COLORS["purple"], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.suptitle("Training Dynamics — CVD v12", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "training_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary stats
    min_val_loss_epoch = np.argmin(history["val_loss"]) + 1
    max_val_auc_epoch = np.argmax(history["val_auc"]) + 1
    final_loss_gap = loss_gap[-1]

    return {
        "epochs_trained": len(epochs),
        "min_val_loss_epoch": min_val_loss_epoch,
        "min_val_loss": float(min(history["val_loss"])),
        "max_val_auc_epoch": max_val_auc_epoch,
        "max_val_auc": float(max(history["val_auc"])),
        "final_loss_gap": float(final_loss_gap),
        "train_loss_start": float(history["loss"][0]),
        "train_loss_end": float(history["loss"][-1]),
    }


# ============================================================
# SECTION 7: MODEL ROBUSTNESS
# ============================================================
def robustness_analysis(model, data, out_dir):
    """Test model robustness to noise injection."""
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
    ax.set_title("Model Robustness — Noise Injection Test", fontsize=14, fontweight="bold")
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
# SECTION 8: SYNTHETIC PROFILE BREAKDOWN
# ============================================================
def synthetic_profile_analysis(model, out_dir):
    """Analyze performance per synthetic disease profile."""
    from sklearn.metrics import roc_auc_score, f1_score
    from wristppg import WristPPGSimulator

    profiles = {
        "healthy": ("healthy", 0, "Healthy"),
        "shock": ("shock", 1, "Shock"),
        "hfref": ("hfref", 1, "HFrEF"),
        "afib_isolated": ("afib_isolated", 1, "AFib"),
        "hypovolemia": ("hypovolemia", 1, "Hypovolemia"),
        "sepsis_warm": ("sepsis_warm", 1, "Sepsis"),
    }

    profile_results = {}
    rng = np.random.default_rng(99)

    for name, (prof, label, display) in profiles.items():
        ppgs, accels, labels = [], [], []
        n_samples = 20

        for _ in range(n_samples):
            try:
                sim = WristPPGSimulator(seed=int(rng.integers(0, 2**31)))
                result = sim.generate(profile=prof, duration_s=60.0, activity="rest")
                ppgs.append(result.ppg[:PPG_LENGTH])
                accels.append(result.accel[:PPG_LENGTH])
                labels.append(label)
            except Exception:
                continue

        if len(ppgs) < 5:
            continue

        feat_dicts = [extract_features_for_signal(p, a) for p, a in zip(ppgs, accels)]

        # Use the same feature columns
        with open(OUT_DIR / "feature_columns.json") as f:
            feature_cols = json.load(f)

        X_ppg, X_accel, X_feat = build_arrays(ppgs, accels, feat_dicts, feature_cols)
        X_bio = build_biodata_array(len(X_ppg), seed=200)

        preds = model(
            {"ppg_input": X_ppg, "accel_input": X_accel,
             "feature_input": X_feat, "biodata_input": X_bio},
            training=False,
        )
        y_prob = np.array(preds).flatten()
        y_true = np.array(labels)

        mean_prob = float(y_prob.mean())
        profile_results[display] = {
            "n": len(ppgs),
            "label": label,
            "mean_predicted_prob": mean_prob,
            "std_predicted_prob": float(y_prob.std()),
        }

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(profile_results.keys())
    means = [profile_results[n]["mean_predicted_prob"] for n in names]
    stds = [profile_results[n]["std_predicted_prob"] for n in names]
    labels = [profile_results[n]["label"] for n in names]

    colors = [COLORS["success"] if l == 0 else COLORS["danger"] for l in labels]
    bars = ax.bar(names, means, yerr=stds, color=colors, alpha=0.8, edgecolor="white", capsize=5)
    ax.set_ylabel("Mean Predicted Probability", fontsize=12)
    ax.set_title("Per-Profile Prediction — CVD v12", fontsize=14, fontweight="bold")
    ax.axhline(0.345, color=COLORS["grey"], linestyle="--", alpha=0.7, label="Threshold (0.345)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "synthetic_profiles.png", dpi=150, bbox_inches="tight")
    plt.close()

    return profile_results


# ============================================================
# MAIN
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("ULTRA-THOROUGH EVALUATION: CVD Watch Model v12")
    logger.info("=" * 60)

    # Load model and data
    model = load_model()
    model.summary(print_fn=logger.info)
    data = load_test_data()
    y_prob = get_predictions(model, data)
    y_true = data["y"]

    logger.info("Test set: %d samples (%d positive, %d negative)",
                len(y_true), int(y_true.sum()), int((1 - y_true).sum()))

    all_results = {}

    # 1. Calibration
    logger.info("\n[1/8] Calibration analysis...")
    cal_real = calibration_analysis(y_true, y_prob)
    plot_reliability_diagram(cal_real, "real_test", EVAL_DIR)
    all_results["calibration"] = {
        "ece": cal_real["ece"], "mce": cal_real["mce"],
        "brier": cal_real["brier"], "reliability": cal_real["reliability"],
        "resolution": cal_real["resolution"], "uncertainty": cal_real["uncertainty"],
    }

    # 2. Threshold sweep
    logger.info("[2/8] Threshold sweep analysis...")
    thresh_data = threshold_sweep(y_true, y_prob)
    plot_threshold_analysis(thresh_data, EVAL_DIR)
    best_idx = np.argmax(thresh_data["f1"])
    all_results["threshold_analysis"] = {
        "optimal_threshold": float(thresh_data["threshold"][best_idx]),
        "optimal_f1": float(thresh_data["f1"][best_idx]),
        "at_0.345": {
            "precision": float(thresh_data["precision"][np.argmin(np.abs(np.array(thresh_data["threshold"]) - 0.345))]),
            "recall": float(thresh_data["recall"][np.argmin(np.abs(np.array(thresh_data["threshold"]) - 0.345))]),
            "specificity": float(thresh_data["specificity"][np.argmin(np.abs(np.array(thresh_data["threshold"]) - 0.345))]),
            "npv": float(thresh_data["npv"][np.argmin(np.abs(np.array(thresh_data["threshold"]) - 0.345))]),
        },
    }

    # 3. Precision-Recall curve
    logger.info("[3/8] Precision-Recall curve...")
    plot_precision_recall_curve(y_true, y_prob, EVAL_DIR)

    # 4. Subgroup analysis
    logger.info("[4/8] Subgroup analysis...")
    metadata = {"sqi": data["sqi"], "hr": data["hr"], "signal_len": data["signal_len"]}
    subgroup_results = subgroup_analysis(y_true, y_prob, metadata, EVAL_DIR)
    all_results["subgroup_analysis"] = subgroup_results

    # 5. Feature importance
    logger.info("[5/8] Feature importance (permutation, this takes a while)...")
    importances, base_auroc = permutation_importance(model, data, n_repeats=5)
    plot_feature_importance(importances, EVAL_DIR, top_n=25)
    sorted_imp = sorted(importances.items(), key=lambda x: x[1]["mean"], reverse=True)
    all_results["feature_importance"] = {
        "base_auroc": float(base_auroc),
        "top_10": {k: v for k, v in sorted_imp[:10]},
    }

    # 6. Training dynamics
    logger.info("[6/8] Training dynamics analysis...")
    dynamics = training_dynamics_analysis(EVAL_DIR)
    all_results["training_dynamics"] = dynamics

    # 7. Robustness
    logger.info("[7/8] Robustness analysis...")
    robustness = robustness_analysis(model, data, EVAL_DIR)
    all_results["robustness"] = robustness

    # 8. Synthetic profile breakdown
    logger.info("[8/8] Synthetic profile analysis...")
    profiles = synthetic_profile_analysis(model, EVAL_DIR)
    all_results["synthetic_profiles"] = profiles

    # Save all results
    results_path = EVAL_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nAll results saved to %s", results_path)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info("  Calibration:  ECE=%.4f  MCE=%.4f  Brier=%.4f", cal_real["ece"], cal_real["mce"], cal_real["brier"])
    logger.info("  Threshold:    Optimal=%.3f (F1=%.4f)  @0.345: P=%.4f R=%.4f",
                all_results["threshold_analysis"]["optimal_threshold"],
                all_results["threshold_analysis"]["optimal_f1"],
                all_results["threshold_analysis"]["at_0.345"]["precision"],
                all_results["threshold_analysis"]["at_0.345"]["recall"])
    logger.info("  Robustness:   Clean=%.4f  Max noise drop=%.4f",
                robustness["clean_auroc"],
                robustness["clean_auroc"] - min(robustness["noisy_aurocs"].values()))
    logger.info("  Top features: %s", ", ".join(list(all_results["feature_importance"]["top_10"].keys())[:5]))
    logger.info("  Training:     %d epochs, min val_loss at epoch %d",
                dynamics["epochs_trained"], dynamics["min_val_loss_epoch"])
    logger.info("  All graphs saved to %s", EVAL_DIR)


if __name__ == "__main__":
    main()
