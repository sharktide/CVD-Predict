"""Evaluation – AUROC, calibration, subgroup analysis, TensorBoard image logging."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import get_eval_config, get_paths_config, get_training_config
from src.utils import ensure_dir, load_parquet, save_parquet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(0, 1, n_bins + 1)
    indices = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    centres = (bins[:-1] + bins[1:]) / 2
    frac_pos = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = indices == b
        if mask.sum() > 0:
            frac_pos[b] = y_true[mask].mean()
    return centres, frac_pos


def plot_reliability(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_calibrated: Optional[np.ndarray] = None,
    n_bins: int = 10,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Return an RGB image array of the reliability diagram."""
    fig, ax = plt.subplots(figsize=(5, 5))
    centres, frac_pos = reliability_diagram(y_true, y_prob, n_bins)
    valid = ~np.isnan(frac_pos)
    ax.plot(centres[valid], frac_pos[valid], "o-", label="Raw", color="steelblue")
    if y_calibrated is not None:
        centres_c, frac_pos_c = reliability_diagram(y_true, y_calibrated, n_bins)
        valid_c = ~np.isnan(frac_pos_c)
        ax.plot(centres_c[valid_c], frac_pos_c[valid_c], "s-", label="Calibrated", color="darkorange")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "isotonic",
) -> Tuple[np.ndarray, Any]:
    """Fit calibration on validation data and return calibrated predictions + calibrator."""
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        y_cal = calibrator.fit_transform(y_prob, y_true)
    elif method == "platt":
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1.0)
        lr.fit(y_prob.reshape(-1, 1), y_true)
        y_cal = lr.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        calibrator = lr
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    return y_cal, calibrator


# ---------------------------------------------------------------------------
# Subgroup metrics
# ---------------------------------------------------------------------------

def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    subgroups: pd.Series,
) -> pd.DataFrame:
    rows = []
    for name, mask in subgroups.groupby(subgroups):
        idx = mask.index.values
        if len(idx) < 10:
            continue
        yt = y_true[idx]
        yp = y_prob[idx]
        try:
            auc = roc_auc_score(yt, yp)
        except ValueError:
            auc = np.nan
        rows.append({
            "subgroup": name,
            "n": len(idx),
            "auroc": auc,
            "brier": brier_score_loss(yt, yp) if len(np.unique(yt)) > 1 else np.nan,
            "f1": f1_score(yt, (yp >= 0.5).astype(int), zero_division=0),
            "precision": precision_score(yt, (yp >= 0.5).astype(int), zero_division=0),
            "recall": recall_score(yt, (yp >= 0.5).astype(int), zero_division=0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Confusion matrix image
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Neg", "Pos"])
    ax.set_yticklabels(["Neg", "Pos"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_path: str,
    run_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a trained model, evaluate on the held-out test split, and log to TensorBoard."""
    paths = get_paths_config()
    eval_cfg = get_eval_config()
    train_cfg = get_training_config()

    if run_name is None:
        run_name = eval_cfg.get("run_name", train_cfg.get("run_name", "cvd_risk_v1"))

    processed_dir = paths["processed_data_dir"]

    # Load data
    features_df = load_parquet(os.path.join(processed_dir, "features.parquet"))
    signals_df = load_parquet(os.path.join(processed_dir, "signals.parquet"))

    meta_path = os.path.join(processed_dir, "cohort_meta.parquet")
    if os.path.exists(meta_path):
        meta_df = load_parquet(meta_path)
        # Ensure consistent dtypes for merge key
        features_df["patient_id"] = features_df["patient_id"].astype(str)
        meta_df["patient_id"] = meta_df["patient_id"].astype(str)
        features_df = features_df.merge(
            meta_df[["patient_id", "acuity_score", "community_likeness",
                      "importance_weight", "icu_type", "community_likeness_bin",
                      "label_confidence_bin"]],
            on="patient_id", how="left",
        )
        features_df["icu_domain"] = (
            features_df.get("icu_type", pd.Series("unknown", index=features_df.index))
            .isin(["SICU", "MICU", "CCU", "CSRU", "TSICU",
                    "Medical Intensive Care Unit (MICU)",
                    "Surgical Intensive Care Unit (SICU)",
                    "Cardiovascular Intensive Care Unit (CSRU)",
                    "Neuro Stepdown Intensive Care Unit (NSICU)",
                    "Medical/Surgical Intensive Care Unit (MIC/SICU)"]).astype(np.int32)
        )
    else:
        features_df["acuity_score"] = 0
        features_df["community_likeness_bin"] = "medium"
        features_df["label_confidence_bin"] = "medium"
        features_df["icu_domain"] = "unknown"

    # Reconstruct test split (same patient-level split as training)
    from src.utils import train_val_test_split
    patient_ids = features_df["patient_id"].unique()
    _, _, test_ids = train_val_test_split(patient_ids)

    test_feat = features_df[features_df["patient_id"].isin(test_ids)]
    test_sig = signals_df[signals_df["patient_id"].isin(test_ids)]

    # Build arrays (same logic as train.py)
    _sig = test_sig.drop(columns=["window_type", "horizon_hours", "event_type", "device_domain"],
                        errors="ignore")
    df = test_feat.merge(_sig, on=["feature_id", "patient_id"], how="inner")
    y_true = df["event_type"].isin(["MI", "ARREST"]).astype(np.float32).values

    # Load PPG signals from .npy files
    ppg_length = train_cfg.get("ppg_length", 7500)
    wear_col = "wearable_ppg_path" if "wearable_ppg_path" in df.columns else "raw_ppg_path"
    X_ppg = np.zeros((len(df), ppg_length), dtype=np.float32)
    for i, path in enumerate(df[wear_col].values):
        if pd.notna(path) and os.path.exists(str(path)):
            try:
                arr = np.load(str(path), allow_pickle=False).astype(np.float32)
                if len(arr) >= ppg_length:
                    X_ppg[i] = arr[:ppg_length]
                else:
                    X_ppg[i, :len(arr)] = arr
            except Exception:
                pass
    X_ppg = X_ppg[..., np.newaxis]
    X_ppg = np.nan_to_num(X_ppg, nan=0.0, posinf=0.0, neginf=0.0)

    model_dir = os.path.dirname(model_path)
    feat_cols_path = os.path.join(model_dir, "feature_columns.json")
    if os.path.exists(feat_cols_path):
        import json
        with open(feat_cols_path) as f:
            feat_cols = json.load(f)
    else:
        exclude = {"feature_id", "patient_id", "window_type", "event_type",
                    "start_time", "end_time", "raw_ppg_path", "wearable_ppg_path",
                    "raw_ppg", "wearable_ppg"}
        feat_cols = [c for c in df.columns if c not in exclude
                     and pd.api.types.is_numeric_dtype(df[c])]
    X_feat = df[feat_cols].fillna(0.0).values.astype(np.float32)
    X_feat = np.nan_to_num(X_feat)

    # Predict
    from src.losses import GradientReversalLayer
    custom = {"GradientReversalLayer": GradientReversalLayer}
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom)
    preds = model({"ppg_input": X_ppg, "feature_input": X_feat}, training=False)
    y_prob = preds[0].numpy().ravel()

    # Core metrics
    results: Dict[str, Any] = {}
    if len(np.unique(y_true)) > 1:
        results["auroc"] = float(roc_auc_score(y_true, y_prob))
        results["brier"] = float(brier_score_loss(y_true, y_prob))
    else:
        results["auroc"] = np.nan
        results["brier"] = np.nan

    y_pred_bin = (y_prob >= 0.5).astype(int)
    results["accuracy"] = float(accuracy_score(y_true, y_pred_bin))
    results["f1"] = float(f1_score(y_true, y_pred_bin, zero_division=0))
    results["precision"] = float(precision_score(y_true, y_pred_bin, zero_division=0))
    results["recall"] = float(recall_score(y_true, y_pred_bin, zero_division=0))
    results["n_test"] = int(len(y_true))
    results["n_positive"] = int(y_true.sum())

    # Calibration
    cal_method = eval_cfg.get("calibration_method", "isotonic")
    if len(np.unique(y_true)) > 1:
        y_cal, calibrator = calibrate_predictions(y_true, y_prob, method=cal_method)
        results["brier_calibrated"] = float(brier_score_loss(y_true, y_cal))
    else:
        y_cal = y_prob
        calibrator = None

    # TensorBoard logging
    log_dir = os.path.join(paths["logs_dir"], run_name, "eval")
    ensure_dir(log_dir)
    writer = tf.summary.create_file_writer(log_dir)

    with writer.as_default():
        for k, v in results.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                tf.summary.scalar(f"test/{k}", float(v), step=0)

    # Reliability diagram image
    if len(np.unique(y_true)) > 1:
        img = plot_reliability(y_true, y_prob, y_cal)
        with writer.as_default():
            tf.summary.image("test/reliability_diagram", img[np.newaxis], step=0)

        img_cm = plot_confusion_matrix(y_true, y_pred_bin)
        with writer.as_default():
            tf.summary.image("test/confusion_matrix", img_cm[np.newaxis], step=0)

    writer.flush()

    # Subgroup analysis
    if "community_likeness_bin" in df.columns:
        sub_metrics = compute_subgroup_metrics(
            y_true, y_prob, df["community_likeness_bin"],
        )
        results["subgroup_community_likeness"] = sub_metrics.to_dict("records")
        sub_path = os.path.join(log_dir, "subgroup_community.csv")
        ensure_dir(os.path.dirname(sub_path))
        sub_metrics.to_csv(sub_path, index=False)

    if "label_confidence_bin" in df.columns:
        sub_conf = compute_subgroup_metrics(
            y_true, y_prob, df["label_confidence_bin"],
        )
        results["subgroup_label_confidence"] = sub_conf.to_dict("records")
        sub_conf.to_csv(os.path.join(log_dir, "subgroup_confidence.csv"), index=False)

    if "icu_domain" in df.columns:
        sub_icu = compute_subgroup_metrics(
            y_true, y_prob, df["icu_domain"],
        )
        results["subgroup_icu_type"] = sub_icu.to_dict("records")
        sub_icu.to_csv(os.path.join(log_dir, "subgroup_icu.csv"), index=False)

    logger.info("Evaluation results: %s", {k: v for k, v in results.items()
                                             if not isinstance(v, list)})
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    path = sys.argv[1] if len(sys.argv) > 1 else "models/cvd_risk_v1/final_model.keras"
    evaluate_model(path)
