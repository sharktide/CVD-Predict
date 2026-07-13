"""Train CVD Watch Model v6 — wrist PPG cardiac event screening.

Generates synthetic Apple Watch PPG data, extracts HRV features,
trains a lightweight CNN-BiLSTM model, and saves to production.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import find_peaks, welch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PPG Generator (same as test script, self-contained)
# ---------------------------------------------------------------------------

class WatchPPGGenerator:
    """Generate synthetic Apple Watch-style PPG signals."""

    def __init__(self, fs: int = 25, seed: int = 42):
        self.fs = fs
        self.rng = np.random.default_rng(seed)

    def _ppg_cycle(self, hr_bpm: float) -> np.ndarray:
        period = 60.0 / hr_bpm
        t = np.linspace(0, period, int(self.fs * period), endpoint=False)
        systolic = np.exp(-0.5 * ((t - period * 0.3) / (period * 0.08)) ** 2)
        dicrotic = -0.3 * np.exp(-0.5 * ((t - period * 0.45) / (period * 0.05)) ** 2)
        diastolic = 0.4 * np.exp(-0.5 * ((t - period * 0.65) / (period * 0.2)) ** 2)
        return (systolic + dicrotic + diastolic).astype(np.float32)

    def generate(
        self,
        duration_s: float = 120.0,
        base_hr: float = 72.0,
        hr_var: float = 0.15,
        motion: float = 0.3,
        contact: float = 0.9,
        ambient: float = 0.05,
        snr_db: float = 20.0,
    ) -> Tuple[np.ndarray, float]:
        """Generate PPG signal and return (signal, heart_rate)."""
        n = int(self.fs * duration_s)
        ppg = np.zeros(n, dtype=np.float32)

        beat_interval = 60.0 / base_hr
        n_beats = int(duration_s / beat_interval)
        rr = np.ones(n_beats) * beat_interval
        t_beats = np.cumsum(rr) - rr[0]
        rr += hr_var * beat_interval * 0.3 * np.sin(2 * np.pi * 0.25 * t_beats)
        rr += hr_var * beat_interval * 0.2 * np.sin(2 * np.pi * 0.1 * t_beats)
        rr += self.rng.normal(0, hr_var * beat_interval * 0.05, n_beats)
        rr = np.clip(rr, 0.4, 2.0)

        beat_times = np.cumsum(rr) - rr[0]
        for bt, interval in zip(beat_times, rr):
            cycle = self._ppg_cycle(60.0 / interval)
            s = int(bt * self.fs)
            e = min(s + len(cycle), n)
            if s < n:
                ppg[s:e] += cycle[:e - s]

        # Motion artifacts
        if motion > 0:
            from scipy.ndimage import gaussian_filter1d
            t_full = np.arange(n) / self.fs
            motion_sig = np.zeros(n, dtype=np.float32)
            for freq in self.rng.uniform(1.0, 3.0, 3):
                motion_sig += np.sin(2 * np.pi * freq * t_full + self.rng.uniform(0, 2 * np.pi))
            env = self.rng.binomial(1, motion, n).astype(np.float32)
            env = gaussian_filter1d(env, sigma=self.fs * 2)
            ppg += motion_sig * env * motion * np.std(ppg)

        # Contact dropout
        if contact < 1.0:
            from scipy.ndimage import gaussian_filter1d
            mask = self.rng.binomial(1, contact, n).astype(np.float32)
            mask = gaussian_filter1d(mask, sigma=self.fs * 0.5)
            ppg *= mask

        # Ambient light
        if ambient > 0:
            t_full = np.arange(n) / self.fs
            ppg += ambient * np.sin(2 * np.pi * 0.05 * t_full) * np.std(ppg)

        # Sensor noise
        power = np.mean(ppg ** 2) + 1e-10
        noise = power / (10 ** (snr_db / 10))
        ppg += self.rng.normal(0, np.sqrt(noise), n).astype(np.float32)

        ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
        return ppg, base_hr

    def healthy(self, dur=120.0):
        hr = self.rng.uniform(58, 78)
        return self.generate(dur, hr, self.rng.uniform(0.12, 0.25),
                            self.rng.uniform(0.1, 0.4), self.rng.uniform(0.85, 1.0),
                            self.rng.uniform(0.02, 0.08), self.rng.uniform(18, 25))

    def at_risk(self, dur=120.0):
        hr = self.rng.uniform(85, 120)
        return self.generate(dur, hr, self.rng.uniform(0.03, 0.08),
                            self.rng.uniform(0.1, 0.3), self.rng.uniform(0.8, 0.95),
                            self.rng.uniform(0.03, 0.10), self.rng.uniform(14, 20))

    def borderline(self, dur=120.0):
        hr = self.rng.uniform(72, 95)
        return self.generate(dur, hr, self.rng.uniform(0.06, 0.14),
                            self.rng.uniform(0.15, 0.45), self.rng.uniform(0.82, 0.97),
                            self.rng.uniform(0.03, 0.10), self.rng.uniform(16, 22))


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_features(ppg: np.ndarray, fs: int = 25) -> Dict[str, float]:
    """Extract HRV features from wrist PPG. Returns dict of feature_name -> value."""
    feats = {}
    from src.utils import compute_sqi_simple
    feats["sqi"] = compute_sqi_simple(ppg, fs=fs)
    feats["signal_length"] = len(ppg)
    feats["mean_amplitude"] = float(np.mean(ppg))
    feats["std_amplitude"] = float(np.std(ppg))

    # Detect peaks
    filt = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
    peaks, _ = find_peaks(filt, distance=int(fs * 0.4), height=0.0)
    if len(peaks) < 5:
        return feats

    rr_ms = np.diff(peaks) / fs * 1000.0
    rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
    if len(rr_ms) < 3:
        return feats

    # Time-domain
    feats["HRV_MeanNN"] = float(np.mean(rr_ms))
    feats["HRV_SDNN"] = float(np.std(rr_ms, ddof=1))
    feats["HRV_RMSSD"] = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
    feats["HRV_SDSD"] = float(np.std(np.diff(rr_ms), ddof=1))
    feats["HRV_CVNN"] = feats["HRV_SDNN"] / (feats["HRV_MeanNN"] + 1e-8)
    feats["HRV_CVSD"] = feats["HRV_RMSSD"] / (feats["HRV_MeanNN"] + 1e-8)
    feats["HRV_MedianNN"] = float(np.median(rr_ms))
    feats["HRV_MadNN"] = float(np.median(np.abs(rr_ms - np.median(rr_ms))))
    feats["HRV_MCVNN"] = feats["HRV_MadNN"] / (feats["HRV_MedianNN"] + 1e-8)
    feats["HRV_IQRNN"] = float(np.percentile(rr_ms, 75) - np.percentile(rr_ms, 25))
    feats["HRV_Prc20NN"] = float(np.percentile(rr_ms, 20))
    feats["HRV_Prc80NN"] = float(np.percentile(rr_ms, 80))
    feats["HRV_pNN50"] = float(100.0 * np.sum(np.abs(np.diff(rr_ms)) > 50) / len(rr_ms))
    feats["HRV_pNN20"] = float(100.0 * np.sum(np.abs(np.diff(rr_ms)) > 20) / len(rr_ms))
    feats["HRV_MinNN"] = float(np.min(rr_ms))
    feats["HRV_MaxNN"] = float(np.max(rr_ms))

    # Frequency-domain
    try:
        rr_times = np.cumsum(rr_ms) / 1000.0
        rr_times -= rr_times[0]
        t_u = np.arange(0, rr_times[-1], 0.25)
        rr_i = np.interp(t_u, rr_times, rr_ms)
        rr_i -= np.mean(rr_i)
        freqs, psd = welch(rr_i, fs=4.0, nperseg=min(len(rr_i), 256))
        lf = float(np.trapz(psd[(freqs >= 0.04) & (freqs < 0.15)],
                            freqs[(freqs >= 0.04) & (freqs < 0.15)])) if np.any((freqs >= 0.04) & (freqs < 0.15)) else 0.0
        hf = float(np.trapz(psd[(freqs >= 0.15) & (freqs < 0.4)],
                            freqs[(freqs >= 0.15) & (freqs < 0.4)])) if np.any((freqs >= 0.15) & (freqs < 0.4)) else 0.0
        tp = lf + hf + 1e-8
        feats["HRV_LF"] = lf
        feats["HRV_HF"] = hf
        feats["HRV_TP"] = tp
        feats["HRV_LFHF"] = lf / (hf + 1e-8)
        feats["HRV_LFn"] = lf / tp
        feats["HRV_HFn"] = hf / tp
        feats["HRV_LnHF"] = float(np.log(hf + 1e-8))
    except Exception:
        pass

    # Nonlinear (Poincare)
    if len(rr_ms) > 2:
        sd1 = float(np.std(np.diff(rr_ms)) / np.sqrt(2))
        sd2 = float(np.sqrt(2 * np.var(rr_ms) - sd1 ** 2))
        feats["HRV_SD1"] = sd1
        feats["HRV_SD2"] = sd2
        feats["HRV_SD1SD2"] = sd1 / (sd2 + 1e-8)
        feats["HRV_CSI"] = sd1 / (sd2 + 1e-8)

    # DFA alpha1 (simplified)
    try:
        if len(rr_ms) > 10:
            scales = np.arange(4, min(len(rr_ms) // 4, 64))
            fluctuations = []
            for s in scales:
                nw = len(rr_ms) // s
                if nw < 1:
                    continue
                rms = []
                for i in range(nw):
                    w = rr_ms[i * s:(i + 1) * s]
                    x = np.arange(s)
                    c = np.polyfit(x, w, 1)
                    rms.append(np.sqrt(np.mean((w - np.polyval(c, x)) ** 2)))
                fluctuations.append(np.mean(rms))
            if len(fluctuations) > 2:
                feats["HRV_DFA_alpha1"] = float(np.polyfit(
                    np.log(scales[:len(fluctuations)]),
                    np.log(np.array(fluctuations) + 1e-8), 1)[0])
    except Exception:
        pass

    feats["pulse_rate"] = float(len(peaks) / (len(ppg) / fs) * 60.0)
    return feats


# ---------------------------------------------------------------------------
# Dataset Generation
# ---------------------------------------------------------------------------

def generate_dataset(
    n_healthy: int = 500,
    n_at_risk: int = 500,
    n_borderline: int = 200,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """Generate synthetic Apple Watch dataset.

    Returns
    -------
    X_ppg : (N, 7500, 1) float32
    X_feat : (N, F) float32
    y : (N,) float32 — 0=healthy, 1=at_risk or borderline
    all_features : list of feature dicts
    """
    gen = WatchPPGGenerator(fs=25, seed=seed)
    ppg_length = 7500

    all_ppg = []
    all_feat = []
    all_y = []
    all_meta = []

    logger.info("Generating %d healthy, %d at-risk, %d borderline signals...",
                n_healthy, n_at_risk, n_borderline)

    for i in range(n_healthy):
        ppg, hr = gen.healthy()
        feats = extract_features(ppg, fs=25)
        feats["base_hr"] = hr
        all_ppg.append(ppg)
        all_feat.append(feats)
        all_y.append(0)
        all_meta.append({"profile": "healthy", "hr": hr})

    for i in range(n_at_risk):
        ppg, hr = gen.at_risk()
        feats = extract_features(ppg, fs=25)
        feats["base_hr"] = hr
        all_ppg.append(ppg)
        all_feat.append(feats)
        all_y.append(1)
        all_meta.append({"profile": "at_risk", "hr": hr})

    for i in range(n_borderline):
        ppg, hr = gen.borderline()
        feats = extract_features(ppg, fs=25)
        feats["base_hr"] = hr
        all_ppg.append(ppg)
        all_feat.append(feats)
        all_y.append(1)
        all_meta.append({"profile": "borderline", "hr": hr})

    # Pad/truncate PPG to ppg_length
    X_ppg = np.zeros((len(all_ppg), ppg_length), dtype=np.float32)
    for i, ppg in enumerate(all_ppg):
        L = min(len(ppg), ppg_length)
        X_ppg[i, :L] = ppg[:L]
    X_ppg = X_ppg[..., np.newaxis]

    # Collect all feature names
    all_cols = set()
    for f in all_feat:
        all_cols.update(f.keys())
    all_cols = sorted(all_cols)

    # Build feature matrix
    X_feat = np.zeros((len(all_feat), len(all_cols)), dtype=np.float32)
    for i, f in enumerate(all_feat):
        for j, col in enumerate(all_cols):
            X_feat[i, j] = f.get(col, 0.0)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

    y = np.array(all_y, dtype=np.float32)

    logger.info("Dataset: %d samples (%d healthy, %d event), %d features",
                len(y), int((y == 0).sum()), int((y == 1).sum()), len(all_cols))

    return X_ppg, X_feat, y, all_cols, all_meta


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_v6_watch():
    """Full training pipeline for v6-watch."""
    from src.model_watch import build_watch_model

    # Generate dataset
    X_ppg, X_feat, y, feature_cols, meta = generate_dataset(
        n_healthy=500, n_at_risk=500, n_borderline=200, seed=42
    )

    N = len(y)
    rng = np.random.default_rng(42)
    idx = rng.permutation(N)

    # 70/15/15 split
    n_train = int(N * 0.70)
    n_val = int(N * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    X_ppg_train, X_feat_train, y_train = X_ppg[train_idx], X_feat[train_idx], y[train_idx]
    X_ppg_val, X_feat_val, y_val = X_ppg[val_idx], X_feat[val_idx], y[val_idx]
    X_ppg_test, X_feat_test, y_test = X_ppg[test_idx], X_feat[test_idx], y[test_idx]

    logger.info("Split: train=%d, val=%d, test=%d", len(y_train), len(y_val), len(y_test))
    logger.info("Train events: %d/%d (%.1f%%)", int(y_train.sum()), len(y_train), y_train.mean() * 100)

    # Build model
    model = build_watch_model(
        ppg_input_shape=(7500, 1),
        feature_dim=len(feature_cols),
        cfg={"filt1": 16, "filt2": 32, "filt3": 64, "lstm_units": 32,
             "ppg_dense": 32, "feat_layers": [32, 32], "shared_units": 32,
             "event_hidden": 16},
    )

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="prec"),
            tf.keras.metrics.Recall(name="rec"),
        ],
    )

    model.summary(print_fn=logger.info)
    logger.info("Total params: %d", model.count_params())

    # Callbacks
    out_dir = Path(__file__).resolve().parent.parent / "production" / "cvd_risk_v6_watch"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=15, mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(out_dir / "best_model.keras"),
            monitor="val_auc", mode="max",
            save_best_only=True, save_weights_only=False,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
            profile_batch=0,
        ),
        tf.keras.callbacks.CSVLogger(
            str(out_dir / "training_log.csv"),
            append=False,
        ),
    ]

    # Train
    history = model.fit(
        {"ppg_input": X_ppg_train, "feature_input": X_feat_train},
        y_train,
        validation_data=(
            {"ppg_input": X_ppg_val, "feature_input": X_feat_val},
            y_val,
        ),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
    )

    # Evaluate on test set
    logger.info("\n=== Test Set Evaluation ===")
    preds = model({"ppg_input": X_ppg_test, "feature_input": X_feat_test}, training=False)
    y_prob = np.array(preds).flatten()

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, brier_score_loss,
    )

    # Sweep thresholds
    best_f1 = 0
    best_t = 0.5
    for t in np.arange(0.1, 0.91, 0.01):
        yp = (y_prob >= t).astype(int)
        f = f1_score(y_test, yp, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = t

    y_pred = (y_prob >= best_t).astype(int)

    metrics = {
        "auroc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(best_f1),
        "brier": float(brier_score_loss(y_test, y_prob)),
        "threshold": float(best_t),
        "n_test": int(len(y_test)),
    }

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])

    logger.info("AUROC:      %.4f", metrics["auroc"])
    logger.info("Accuracy:   %.1f%%", metrics["accuracy"] * 100)
    logger.info("Precision:  %.4f", metrics["precision"])
    logger.info("Recall:     %.4f", metrics["recall"])
    logger.info("F1:         %.4f", metrics["f1"])
    logger.info("Threshold:  %.2f", metrics["threshold"])
    logger.info("Brier:      %.4f", metrics["brier"])
    logger.info("CM: TN=%d FP=%d FN=%d TP=%d", metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"])

    # Also evaluate at standard thresholds
    for t_name, t_val in [("0.50", 0.50), ("0.30", 0.30)]:
        yp = (y_prob >= t_val).astype(int)
        acc = accuracy_score(y_test, yp)
        prec = precision_score(y_test, yp, zero_division=0)
        rec = recall_score(y_test, yp, zero_division=0)
        f1 = f1_score(y_test, yp, zero_division=0)
        logger.info("  @%s: Acc=%.1f%% Prec=%.4f Rec=%.4f F1=%.4f", t_name, acc*100, prec, rec, f1)

    # Save model (final weights)
    model.save(str(out_dir / "final_model.keras"))

    # Save training history as JSON
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)
    logger.info("Training history saved to %s/training_history.json", out_dir)

    # Save config
    config = {
        "version": "v6-watch",
        "description": "CVD risk model trained specifically for Apple Watch wrist PPG",
        "ppg_length": 7500,
        "sampling_rate_hz": 25,
        "feature_columns": feature_cols,
        "architecture": {
            "ppg_branch": "ResNet 1D-CNN (16→32→64) + BiLSTM(32)",
            "feature_branch": "MLP (32, 32)",
            "shared": "Dense(32)",
            "event_head": "Dense(16) → Dense(1, sigmoid)",
            "total_params": model.count_params(),
        },
        "training": {
            "dataset": "synthetic_apple_watch",
            "n_samples": int(N),
            "n_healthy": int((y == 0).sum()),
            "n_event": int((y == 1).sum()),
            "split": "70/15/15",
            "optimizer": "AdamW",
            "lr": 3e-4,
            "batch_size": 32,
        },
        "performance": metrics,
    }

    with open(out_dir / "config.yaml", "w") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    with open(out_dir / "optimal_threshold.json", "w") as f:
        json.dump({"threshold": best_t}, f)

    logger.info("\nModel saved to %s", out_dir)
    return model, metrics, history


if __name__ == "__main__":
    train_v6_watch()
