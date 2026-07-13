"""Training script – data pipeline, model compilation, TensorBoard logging, callbacks.

Key improvements over the original:
- Loads signals from .npy files (not from parquet object columns)
- Class balancing via oversampling events / undersampling controls
- Warmup LR schedule
- MMD alignment loss between ICU and wearable domains
- Device domain labels from real wearable data (not hardcoded 0)
- Sensor quality labels derived from SQI
- Per-sample weights properly passed to focal loss via Keras
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import get_model_config, get_paths_config, get_training_config
from src.domain_alignment import compute_alignment_loss
from src.losses import build_combined_loss
from src.model import build_model
from src.utils import ensure_dir, load_numpy, load_parquet, save_parquet, train_val_test_split

# Columns to drop (90% NaN alpha2 HRV features that add noise)
_DROP_COLS = {
    "HRV_DFA_alpha2", "HRV_MFDFA_alpha2_Width", "HRV_MFDFA_alpha2_Peak",
    "HRV_MFDFA_alpha2_Mean", "HRV_MFDFA_alpha2_Max", "HRV_MFDFA_alpha2_Delta",
    "HRV_MFDFA_alpha2_Asymmetry", "HRV_MFDFA_alpha2_Fluctuation",
    "HRV_MFDFA_alpha2_Increment",
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Warmup + cosine decay learning rate schedule
# ---------------------------------------------------------------------------

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay."""

    def __init__(self, base_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)

        # Linear warmup
        warmup_lr = self.base_lr * (step / tf.maximum(warmup, 1.0))

        # Cosine decay
        progress = (step - warmup) / tf.maximum(total - warmup, 1.0)
        cosine_lr = self.base_lr * 0.5 * (1.0 + tf.cos(np.pi * progress))

        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }


# ---------------------------------------------------------------------------
# Class balancing
# ---------------------------------------------------------------------------

def _compute_class_weights(y_event: np.ndarray) -> dict[int, float]:
    """Compute class weights inversely proportional to frequency."""
    n_pos = int(y_event.sum())
    n_neg = len(y_event) - n_pos
    if n_pos == 0:
        return {0: 1.0, 1: 1.0}
    w_neg = len(y_event) / (2.0 * n_neg)
    w_pos = len(y_event) / (2.0 * n_pos)
    return {0: w_neg, 1: w_pos}


def _oversample_undersample(
    X_ppg: np.ndarray,
    X_feat: np.ndarray,
    y_event: np.ndarray,
    y_acuity: np.ndarray,
    y_icu_domain: np.ndarray,
    y_device_domain: np.ndarray,
    y_sensor_quality: np.ndarray,
    sample_weights: np.ndarray,
    max_control_ratio: float = 3.0,
    seed: int = 42,
) -> Tuple[np.ndarray, ...]:
    """Oversample events and undersample controls to achieve a balanced ratio."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_event == 1)[0]
    neg_idx = np.where(y_event == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X_ppg, X_feat, y_event, y_acuity, y_icu_domain, y_device_domain, y_sensor_quality, sample_weights

    # Undersample controls
    max_neg = int(len(pos_idx) * max_control_ratio)
    if len(neg_idx) > max_neg:
        neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)

    # Oversample events to match controls (if controls > events)
    n_target = len(neg_idx)
    if len(pos_idx) < n_target:
        oversample_idx = rng.choice(pos_idx, size=n_target, replace=True)
    else:
        oversample_idx = pos_idx

    idx = np.concatenate([oversample_idx, neg_idx])
    rng.shuffle(idx)

    return (
        X_ppg[idx],
        X_feat[idx],
        y_event[idx],
        y_acuity[idx],
        y_icu_domain[idx],
        y_device_domain[idx],
        y_sensor_quality[idx],
        sample_weights[idx],
    )


# ---------------------------------------------------------------------------
# PPG data augmentation
# ---------------------------------------------------------------------------

def _augment_ppg_batch(
    X_ppg: np.ndarray,
    noise_std: float = 0.02,
    scale_range: tuple = (0.8, 1.2),
    shift_range: int = 150,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Apply random augmentation to a batch of PPG signals (in-place style)."""
    if rng is None:
        rng = np.random.default_rng()
    out = X_ppg.copy()
    n = len(out)

    # 1. Gaussian noise
    noise_mask = rng.random(n) < 0.5
    if noise_mask.any():
        noise = rng.normal(0, noise_std, size=out[noise_mask].shape).astype(np.float32)
        out[noise_mask] += noise

    # 2. Random amplitude scaling
    scale_mask = rng.random(n) < 0.5
    if scale_mask.any():
        scales = rng.uniform(scale_range[0], scale_range[1], size=(scale_mask.sum(), 1, 1)).astype(np.float32)
        out[scale_mask] *= scales

    # 3. Random time shift (circular)
    shift_mask = rng.random(n) < 0.5
    if shift_mask.any():
        shifts = rng.integers(-shift_range, shift_range + 1, size=shift_mask.sum())
        for idx, s in zip(np.where(shift_mask)[0], shifts):
            out[idx] = np.roll(out[idx], int(s), axis=0)

    return out


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _load_signal(path: str, target_length: int) -> np.ndarray:
    """Load a .npy signal file and pad/truncate to target_length."""
    try:
        arr = load_numpy(path)
    except Exception:
        arr = np.zeros(target_length, dtype=np.float32)
    arr = arr.astype(np.float32)
    if len(arr) >= target_length:
        return arr[:target_length]
    out = np.zeros(target_length, dtype=np.float32)
    out[:len(arr)] = arr
    return out


def _build_dataset(
    features_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    train_config: dict,
    is_training: bool = True,
) -> Tuple[tf.data.Dataset, list]:
    """Merge features + signals, build a tf.data.Dataset ready for model.fit()."""
    # Merge on feature_id (drop overlapping non-key cols from signals to avoid suffixes)
    _sig = signals_df.drop(columns=["window_type", "horizon_hours", "event_type", "device_domain"],
                           errors="ignore")
    df = features_df.merge(_sig, on=["feature_id", "patient_id"], how="inner")

    # Binary event target
    y_event = df["event_type"].isin(["MI", "ARREST"]).astype(np.float32).values

    # Acuity target
    y_acuity = df["acuity_score"].fillna(0).astype(np.int32).values if "acuity_score" in df.columns else np.zeros(len(df), dtype=np.int32)

    # Domain labels
    y_icu_domain = df.get("icu_domain", pd.Series(0, index=df.index)).values.astype(np.int32)
    y_device_domain = df.get("device_domain", pd.Series(0, index=df.index)).values.astype(np.int32)

    # Sensor quality labels (0=clean, 1=noisy, 2=dropout) from SQI
    sqi_vals = df.get("sqi", pd.Series(0.5, index=df.index)).values
    y_sensor_quality = np.zeros(len(df), dtype=np.int32)
    y_sensor_quality[sqi_vals < 0.3] = 2  # dropout
    y_sensor_quality[(sqi_vals >= 0.3) & (sqi_vals < 0.6)] = 1  # noisy
    # y_sensor_quality[sqi_vals >= 0.6] = 0  # clean (default)

    # PPG inputs from .npy files
    ppg_length = train_config.get("ppg_length", 7500)
    wear_col = "wearable_ppg_path" if "wearable_ppg_path" in df.columns else "raw_ppg_path"
    X_ppg = np.zeros((len(df), ppg_length), dtype=np.float32)
    for i, path in enumerate(df[wear_col].values):
        if pd.notna(path) and os.path.exists(str(path)):
            X_ppg[i] = _load_signal(str(path), ppg_length)
    X_ppg = X_ppg[..., np.newaxis]  # (N, T, 1)

    # Replace NaN/inf in PPG signals
    X_ppg = np.nan_to_num(X_ppg, nan=0.0, posinf=0.0, neginf=0.0)

    # Feature inputs
    feat_cols = train_config.get("feature_columns", [])
    if not feat_cols:
        exclude = {"feature_id", "patient_id", "window_type", "event_type",
                    "start_time", "end_time", "raw_ppg_path", "wearable_ppg_path",
                    "raw_ppg", "wearable_ppg", "device_domain", "icu_domain",
                    "icu_type", "community_likeness_bin", "label_confidence_bin"}
        feat_cols = [c for c in features_df.columns if c not in exclude
                     and c not in _DROP_COLS
                     and pd.api.types.is_numeric_dtype(features_df[c])]
    else:
        feat_cols = [c for c in feat_cols if c not in _DROP_COLS]
    X_feat = df[feat_cols].fillna(0.0).values.astype(np.float32)

    # Sample weights (label_confidence x importance_weight)
    conf = df.get("label_confidence", pd.Series(1.0, index=df.index)).values
    imp = df.get("importance_weight", pd.Series(1.0, index=df.index)).values
    sample_weights = (conf * imp).astype(np.float32)

    # Clamp NaN/inf
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
    sample_weights = np.nan_to_num(sample_weights, nan=1.0)

    # Class balancing
    if is_training and train_config.get("oversample_events", False):
        (X_ppg, X_feat, y_event, y_acuity, y_icu_domain, y_device_domain,
         y_sensor_quality, sample_weights) = _oversample_undersample(
            X_ppg, X_feat, y_event, y_acuity, y_icu_domain, y_device_domain,
            y_sensor_quality, sample_weights,
            max_control_ratio=train_config.get("max_control_ratio", 3.0),
        )

    # PPG augmentation (training only)
    if is_training and train_config.get("augment_ppg", False):
        X_ppg = _augment_ppg_batch(
            X_ppg,
            noise_std=train_config.get("aug_noise_std", 0.02),
            scale_range=tuple(train_config.get("aug_scale_range", [0.8, 1.2])),
            shift_range=train_config.get("aug_shift_range", 150),
        )

    outputs = {
        "event_output": y_event,
        "acuity_output": y_acuity,
        "icu_domain_output": y_icu_domain,
        "device_domain_output": y_device_domain,
        "sensor_quality_output": y_sensor_quality,
    }

    ds = tf.data.Dataset.from_tensor_slices((
        {"ppg_input": X_ppg, "feature_input": X_feat},
        outputs,
        sample_weights,
    ))

    if is_training:
        ds = ds.shuffle(buffer_size=min(len(df), 10_000), seed=42)
    ds = ds.batch(train_config.get("batch_size", 64))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, feat_cols


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class CalibrationCallback(tf.keras.callbacks.Callback):
    """Logs reliability-diagram scalars every *log_freq* epochs."""

    def __init__(self, val_ds: tf.data.Dataset, log_dir: str, log_freq: int = 5):
        super().__init__()
        self.val_ds = val_ds
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, "calibration"))
        self.log_freq = log_freq

    def on_epoch_end(self, epoch: int, logs=None):
        if (epoch + 1) % self.log_freq != 0:
            return

        y_true, y_pred = [], []
        for batch in self.val_ds:
            x_batch, y_batch, _ = batch
            preds = self.model(x_batch, training=False)
            y_true.append(y_batch["event_output"].numpy())
            y_pred.append(preds[0].numpy())

        y_true = np.concatenate(y_true).ravel()
        y_pred = np.concatenate(y_pred).ravel()

        # Binned calibration
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        frac_pos = np.zeros(10, dtype=np.float32)
        counts = np.zeros(10, dtype=np.float32)
        indices = np.clip(np.digitize(y_pred, bins) - 1, 0, 9)

        for b in range(10):
            mask = indices == b
            counts[b] = mask.sum()
            if counts[b] > 0:
                frac_pos[b] = y_true[mask].mean()

        with self.writer.as_default():
            for i, (bc, fp) in enumerate(zip(bin_centers, frac_pos)):
                tf.summary.scalar(f"calibration/bin_{i}_center", bc, step=epoch)
                tf.summary.scalar(f"calibration/bin_{i}_frac_pos", fp, step=epoch)

            brier = float(np.mean((y_pred - y_true) ** 2))
            tf.summary.scalar("calibration/brier_score", brier, step=epoch)

        self.writer.flush()


class ThresholdTuningCallback(tf.keras.callbacks.Callback):
    """Find optimal classification threshold on validation set each epoch.

    Scans thresholds 0.1–0.9 and picks the one maximising F1, then logs it.
    The optimal threshold is stored in self.best_threshold for later use.
    """

    def __init__(self, val_ds: tf.data.Dataset, log_dir: str, log_freq: int = 5):
        super().__init__()
        self.val_ds = val_ds
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.best_threshold = 0.5

    def on_epoch_end(self, epoch: int, logs=None):
        if (epoch + 1) % self.log_freq != 0:
            return

        y_true, y_pred = [], []
        for batch in self.val_ds:
            x_batch, y_batch, _ = batch
            preds = self.model(x_batch, training=False)
            y_true.append(y_batch["event_output"].numpy())
            y_pred.append(preds[0].numpy())

        y_true = np.concatenate(y_true).ravel()
        y_pred = np.concatenate(y_pred).ravel()

        if len(np.unique(y_true)) < 2:
            return

        # Scan thresholds
        from sklearn.metrics import f1_score, precision_score, recall_score
        best_f1 = 0.0
        best_t = 0.5
        for t in np.arange(0.1, 0.91, 0.05):
            preds_bin = (y_pred >= t).astype(int)
            f1 = f1_score(y_true, preds_bin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        self.best_threshold = float(best_t)

        # Compute metrics at optimal threshold
        preds_bin = (y_pred >= best_t).astype(int)
        prec = precision_score(y_true, preds_bin, zero_division=0)
        rec = recall_score(y_true, preds_bin, zero_division=0)

        log_path = os.path.join(self.log_dir, "threshold_search")
        ensure_dir(log_path)
        writer = tf.summary.create_file_writer(log_path)
        with writer.as_default():
            tf.summary.scalar("threshold/best_threshold", best_t, step=epoch)
            tf.summary.scalar("threshold/best_f1", best_f1, step=epoch)
            tf.summary.scalar("threshold/precision_at_best", prec, step=epoch)
            tf.summary.scalar("threshold/recall_at_best", rec, step=epoch)
        writer.flush()

        logger.info("Epoch %d: best_threshold=%.3f, F1=%.3f, Precision=%.3f, Recall=%.3f",
                     epoch + 1, best_t, best_f1, prec, rec)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    run_name: Optional[str] = None,
    model_ckpt_path: Optional[str] = None,
) -> tf.keras.Model:
    """Full training pipeline: load data -> build model -> compile -> fit -> save.

    Returns the trained Keras Model.
    """
    paths = get_paths_config()
    model_cfg = get_model_config()
    train_cfg = get_training_config()

    if run_name is None:
        run_name = train_cfg.get("run_name", "cvd_risk_v1")

    processed_dir = paths["processed_data_dir"]
    features_df = load_parquet(os.path.join(processed_dir, "features.parquet"))
    signals_df = load_parquet(os.path.join(processed_dir, "signals.parquet"))

    # Merge cohort metadata for acuity / domain labels
    meta_path = os.path.join(processed_dir, "cohort_meta.parquet")
    if os.path.exists(meta_path):
        meta_df = load_parquet(meta_path)
        # Ensure consistent dtypes for merge key
        features_df["patient_id"] = features_df["patient_id"].astype(str)
        meta_df["patient_id"] = meta_df["patient_id"].astype(str)
        features_df = features_df.merge(
            meta_df[["patient_id", "acuity_score", "community_likeness",
                      "importance_weight", "icu_type"]],
            on="patient_id", how="left",
        )
        # Derive ICU domain labels from actual ICU type
        features_df["icu_domain"] = (
            features_df.get("icu_type", pd.Series("unknown", index=features_df.index))
            .isin(["SICU", "MICU", "CCU", "CSRU", "TSICU",
                    "Medical Intensive Care Unit (MICU)",
                    "Surgical Intensive Care Unit (SICU)",
                    "Cardiovascular Intensive Care Unit (CSRU)",
                    "Neuro Stepdown Intensive Care Unit (NSICU)",
                    "Medical/Surgical Intensive Care Unit (MIC/SICU)"]).astype(np.int32)
        )
        # Device domain is already set by preprocess.py (0=ICU, 1=wearable)
        if "device_domain" not in features_df.columns:
            features_df["device_domain"] = 0
    else:
        features_df["acuity_score"] = 0
        features_df["community_likeness"] = 0.5
        features_df["importance_weight"] = 1.0
        features_df["icu_domain"] = 0
        features_df["device_domain"] = 0

    # Fill NaN in numeric feature columns with median (v2: not 0 which destroys signal)
    num_cols = features_df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        median_val = features_df[col].median()
        if np.isfinite(median_val):
            features_df[col] = features_df[col].fillna(median_val)
        else:
            features_df[col] = features_df[col].fillna(0.0)

    # Patient-level split
    patient_ids = features_df["patient_id"].unique()
    train_ids, val_ids, test_ids = train_val_test_split(
        patient_ids,
        train_frac=train_cfg.get("train_split", 0.8),
        val_frac=train_cfg.get("val_split", 0.1),
        seed=42,
    )

    train_feat = features_df[features_df["patient_id"].isin(train_ids)]
    val_feat = features_df[features_df["patient_id"].isin(val_ids)]

    train_sig = signals_df[signals_df["patient_id"].isin(train_ids)]
    val_sig = signals_df[signals_df["patient_id"].isin(val_ids)]

    logger.info("Train patients: %d  |  Val patients: %d", len(train_ids), len(val_ids))

    train_ds, feat_cols = _build_dataset(train_feat, train_sig, train_cfg, is_training=True)
    val_ds, _ = _build_dataset(val_feat, val_sig, train_cfg, is_training=False)

    train_cfg["feature_columns"] = feat_cols

    # Build model
    ppg_length = train_cfg.get("ppg_length", 7500)
    feature_dim = len(feat_cols)
    num_acuity = train_cfg.get("num_acuity_classes", 6)

    model = build_model(
        ppg_input_shape=(ppg_length, 1),
        feature_dim=feature_dim,
        num_event_classes=1,
        num_acuity_classes=num_acuity,
        num_sensor_quality_classes=3,
        model_cfg=model_cfg,
    )

    # Compile
    losses, loss_weights = build_combined_loss(train_cfg)

    # Warmup + cosine decay LR schedule
    n_train_samples = len(train_feat)
    batch_size = train_cfg.get("batch_size", 64)
    steps_per_epoch = max(1, n_train_samples // batch_size)
    total_steps = steps_per_epoch * train_cfg.get("epochs", 200)
    warmup_steps = steps_per_epoch * train_cfg.get("warmup_epochs", 5)

    lr_schedule = WarmupCosineDecay(
        base_lr=train_cfg.get("lr", 1e-4),
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        ),
        loss=losses,
        loss_weights=loss_weights,
        metrics={
            "event_output": [
                tf.keras.metrics.AUC(name="event_auc"),
                tf.keras.metrics.Precision(name="event_precision"),
                tf.keras.metrics.Recall(name="event_recall"),
            ],
            "acuity_output": ["accuracy"],
            "icu_domain_output": ["accuracy"],
            "device_domain_output": ["accuracy"],
            "sensor_quality_output": ["accuracy"],
        },
    )

    model.summary(print_fn=logger.info)

    # Directories
    logs_dir = os.path.join(paths["logs_dir"], run_name)
    models_dir = os.path.join(paths["models_dir"], run_name)
    ensure_dir(logs_dir)
    ensure_dir(models_dir)

    # Callbacks
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(models_dir, "best_model.keras"),
            monitor="val_event_output_event_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_event_output_event_auc",
            patience=train_cfg.get("early_stopping_patience", 15),
            mode="max",
            restore_best_weights=True,
        ),
        CalibrationCallback(val_ds, logs_dir, log_freq=5),
        ThresholdTuningCallback(val_ds, logs_dir, log_freq=5),
    ]

    # Mixed precision (if requested and GPU available)
    if train_cfg.get("mixed_precision", False):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # Fit
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=train_cfg.get("epochs", 200),
        callbacks=callbacks,
    )

    # Retrieve optimal threshold from callback
    threshold_cb = [c for c in callbacks if isinstance(c, ThresholdTuningCallback)]
    optimal_threshold = threshold_cb[0].best_threshold if threshold_cb else 0.5
    logger.info("Optimal threshold from validation: %.3f", optimal_threshold)

    # Save final model
    final_path = os.path.join(models_dir, "final_model.keras")
    model.save(final_path)
    logger.info("Final model saved -> %s", final_path)

    # Save feature column list for inference
    with open(os.path.join(models_dir, "feature_columns.json"), "w") as f:
        json.dump(feat_cols, f)

    # Save optimal threshold
    with open(os.path.join(models_dir, "optimal_threshold.json"), "w") as f:
        json.dump({"threshold": optimal_threshold}, f)

    return model


# ---------------------------------------------------------------------------
# K-fold cross-validation
# ---------------------------------------------------------------------------

def train_kfold(
    n_folds: int = 5,
    run_name_prefix: str = "cvd_risk_v3",
) -> dict:
    """Run k-fold CV and return averaged metrics across folds."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    paths = get_paths_config()
    model_cfg = get_model_config()
    train_cfg = get_training_config()

    processed_dir = paths["processed_data_dir"]
    features_df = load_parquet(os.path.join(processed_dir, "features.parquet"))
    signals_df = load_parquet(os.path.join(processed_dir, "signals.parquet"))

    # Merge cohort metadata
    meta_path = os.path.join(processed_dir, "cohort_meta.parquet")
    if os.path.exists(meta_path):
        meta_df = load_parquet(meta_path)
        features_df["patient_id"] = features_df["patient_id"].astype(str)
        meta_df["patient_id"] = meta_df["patient_id"].astype(str)
        features_df = features_df.merge(
            meta_df[["patient_id", "acuity_score", "community_likeness",
                      "importance_weight", "icu_type"]],
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
        features_df["community_likeness"] = 0.5
        features_df["importance_weight"] = 1.0
        features_df["icu_domain"] = 0
        features_df["device_domain"] = 0

    # Fill NaN in numeric columns with median
    num_cols = features_df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        median_val = features_df[col].median()
        if np.isfinite(median_val):
            features_df[col] = features_df[col].fillna(median_val)
        else:
            features_df[col] = features_df[col].fillna(0.0)

    # Drop alpha2 columns early
    features_df = features_df.drop(columns=[c for c in _DROP_COLS if c in features_df.columns], errors="ignore")

    # Patient-level stratified K-fold
    patient_events = features_df.groupby("patient_id")["event_type"].apply(
        lambda x: int(x.isin(["MI", "ARREST"]).any())
    ).reset_index()
    patient_events.columns = ["patient_id", "has_event"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(patient_events["patient_id"], patient_events["has_event"])
    ):
        logger.info("=" * 60)
        logger.info("FOLD %d/%d", fold + 1, n_folds)
        logger.info("=" * 60)

        train_pids = patient_events.iloc[train_idx]["patient_id"].values
        test_pids = patient_events.iloc[test_idx]["patient_id"].values

        # Further split train into train+val (90/10)
        rng = np.random.default_rng(42 + fold)
        rng.shuffle(train_pids)
        n_val = max(1, int(len(train_pids) * 0.1))
        val_pids = train_pids[:n_val]
        train_pids_final = train_pids[n_val:]

        train_feat = features_df[features_df["patient_id"].isin(train_pids_final)]
        val_feat = features_df[features_df["patient_id"].isin(val_pids)]
        test_feat = features_df[features_df["patient_id"].isin(test_pids)]

        train_sig = signals_df[signals_df["patient_id"].isin(train_pids_final)]
        val_sig = signals_df[signals_df["patient_id"].isin(val_pids)]
        test_sig = signals_df[signals_df["patient_id"].isin(test_pids)]

        logger.info("Fold %d: train=%d, val=%d, test=%d windows",
                     fold + 1, len(train_feat), len(val_feat), len(test_feat))

        fold_run = f"{run_name_prefix}_fold{fold + 1}"

        # Temporarily override config for this fold
        orig_run = train_cfg.get("run_name")
        train_cfg["run_name"] = fold_run

        train_ds, feat_cols = _build_dataset(train_feat, train_sig, train_cfg, is_training=True)
        val_ds, _ = _build_dataset(val_feat, val_sig, train_cfg, is_training=False)
        test_ds, _ = _build_dataset(test_feat, test_sig, train_cfg, is_training=False)
        train_cfg["feature_columns"] = feat_cols

        # Build model
        ppg_length = train_cfg.get("ppg_length", 7500)
        feature_dim = len(feat_cols)
        num_acuity = train_cfg.get("num_acuity_classes", 6)

        model = build_model(
            ppg_input_shape=(ppg_length, 1),
            feature_dim=feature_dim,
            num_event_classes=1,
            num_acuity_classes=num_acuity,
            num_sensor_quality_classes=3,
            model_cfg=model_cfg,
        )

        # Compile
        losses, loss_weights = build_combined_loss(train_cfg)
        n_train_samples = len(train_feat)
        batch_size = train_cfg.get("batch_size", 32)
        steps_per_epoch = max(1, n_train_samples // batch_size)
        total_steps = steps_per_epoch * train_cfg.get("epochs", 100)
        warmup_steps = steps_per_epoch * train_cfg.get("warmup_epochs", 3)

        lr_schedule = WarmupCosineDecay(
            base_lr=train_cfg.get("lr", 3e-4),
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=train_cfg.get("weight_decay", 1e-4),
            ),
            loss=losses,
            loss_weights=loss_weights,
            metrics={
                "event_output": [
                    tf.keras.metrics.AUC(name="event_auc"),
                    tf.keras.metrics.Precision(name="event_precision"),
                    tf.keras.metrics.Recall(name="event_recall"),
                ],
                "acuity_output": ["accuracy"],
                "icu_domain_output": ["accuracy"],
                "device_domain_output": ["accuracy"],
                "sensor_quality_output": ["accuracy"],
            },
        )

        # Callbacks
        logs_dir = os.path.join(paths["logs_dir"], fold_run)
        models_dir = os.path.join(paths["models_dir"], fold_run)
        ensure_dir(logs_dir)
        ensure_dir(models_dir)

        threshold_cb = ThresholdTuningCallback(val_ds, logs_dir, log_freq=3)
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(models_dir, "best_model.keras"),
                monitor="val_event_output_event_auc",
                mode="max",
                save_best_only=True,
                save_weights_only=False,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_event_output_event_auc",
                patience=train_cfg.get("early_stopping_patience", 15),
                mode="max",
                restore_best_weights=True,
            ),
            threshold_cb,
        ]

        # Fit
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=train_cfg.get("epochs", 100),
            callbacks=callbacks,
            verbose=0,
        )

        # Evaluate on test fold
        optimal_t = threshold_cb.best_threshold
        y_true_all, y_prob_all = [], []
        for batch in test_ds:
            x_batch, y_batch, _ = batch
            preds = model(x_batch, training=False)
            y_true_all.append(y_batch["event_output"].numpy())
            y_prob_all.append(preds[0].numpy())

        y_true = np.concatenate(y_true_all).ravel()
        y_prob = np.concatenate(y_prob_all).ravel()

        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = float("nan")

        y_pred_opt = (y_prob >= optimal_t).astype(int)
        y_pred_05 = (y_prob >= 0.5).astype(int)

        f1_opt = f1_score(y_true, y_pred_opt, zero_division=0)
        prec_opt = precision_score(y_true, y_pred_opt, zero_division=0)
        rec_opt = recall_score(y_true, y_pred_opt, zero_division=0)
        acc_opt = float((y_true == y_pred_opt).mean())

        f1_05 = f1_score(y_true, y_pred_05, zero_division=0)
        prec_05 = precision_score(y_true, y_pred_05, zero_division=0)
        rec_05 = recall_score(y_true, y_pred_05, zero_division=0)
        acc_05 = float((y_true == y_pred_05).mean())

        fold_metrics.append({
            "fold": fold + 1,
            "threshold": optimal_t,
            "auroc": auc,
            "accuracy_opt": acc_opt,
            "f1_opt": f1_opt,
            "precision_opt": prec_opt,
            "recall_opt": rec_opt,
            "accuracy_05": acc_05,
            "f1_05": f1_05,
            "precision_05": prec_05,
            "recall_05": rec_05,
            "n_test": len(y_true),
        })

        logger.info("Fold %d: AUROC=%.3f | threshold=%.3f | Acc=%.1f%% F1=%.3f Prec=%.3f Rec=%.3f (optimal)",
                     fold + 1, auc, optimal_t, acc_opt * 100, f1_opt, prec_opt, rec_opt)
        logger.info("Fold %d: Acc@0.5=%.1f%% F1@0.5=%.3f Prec@0.5=%.3f Rec@0.5=%.3f",
                     fold + 1, acc_05 * 100, f1_05, prec_05, rec_05)

        # Cleanup
        if orig_run:
            train_cfg["run_name"] = orig_run

    # Aggregate
    df_metrics = pd.DataFrame(fold_metrics)
    summary = {
        "mean_auroc": df_metrics["auroc"].mean(),
        "std_auroc": df_metrics["auroc"].std(),
        "mean_accuracy_opt": df_metrics["accuracy_opt"].mean(),
        "mean_f1_opt": df_metrics["f1_opt"].mean(),
        "mean_precision_opt": df_metrics["precision_opt"].mean(),
        "mean_recall_opt": df_metrics["recall_opt"].mean(),
        "mean_accuracy_05": df_metrics["accuracy_05"].mean(),
        "mean_f1_05": df_metrics["f1_05"].mean(),
        "mean_precision_05": df_metrics["precision_05"].mean(),
        "mean_recall_05": df_metrics["recall_05"].mean(),
        "mean_threshold": df_metrics["threshold"].mean(),
    }

    logger.info("=" * 60)
    logger.info("K-FOLD CV SUMMARY (%d folds):", n_folds)
    logger.info("  AUROC:           %.3f +/- %.3f", summary["mean_auroc"], summary["std_auroc"])
    logger.info("  Accuracy (opt):  %.1f%%", summary["mean_accuracy_opt"] * 100)
    logger.info("  F1 (optimal t):  %.3f", summary["mean_f1_opt"])
    logger.info("  Precision (opt): %.3f", summary["mean_precision_opt"])
    logger.info("  Recall (opt):    %.3f", summary["mean_recall_opt"])
    logger.info("  Threshold:       %.3f", summary["mean_threshold"])
    logger.info("  Accuracy (@0.5): %.1f%%", summary["mean_accuracy_05"] * 100)
    logger.info("  F1 (@0.5):       %.3f", summary["mean_f1_05"])
    logger.info("  Precision (@0.5):%.3f", summary["mean_precision_05"])
    logger.info("  Recall (@0.5):   %.3f", summary["mean_recall_05"])
    logger.info("=" * 60)

    # Save metrics
    metrics_path = os.path.join(paths["models_dir"], run_name_prefix, "kfold_metrics.csv")
    ensure_dir(os.path.dirname(metrics_path))
    df_metrics.to_csv(metrics_path, index=False)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    if mode == "kfold":
        n_folds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        train_kfold(n_folds=n_folds)
    else:
        train()
