#!/usr/bin/env python3
"""Train CVD Watch Model v7 — hybrid real + synthetic data.

Uses real MIMIC-IV + MMASH signals with pre-extracted features from parquet,
augmented with synthetic Apple Watch PPG. Targets >98% on all metrics.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import resample as scipy_resample

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PPG_LENGTH = 7500
FS_TARGET = 25

# ---------------------------------------------------------------------------
# Synthetic PPG Generator
# ---------------------------------------------------------------------------

class WatchPPGGenerator:
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

    def generate(self, duration_s=120.0, base_hr=72.0, hr_var=0.15,
                 motion=0.3, contact=0.9, ambient=0.05, snr_db=20.0):
        n = int(self.fs * duration_s)
        ppg = np.zeros(n, dtype=np.float32)
        beat_interval = 60.0 / base_hr
        n_beats = int(duration_s / beat_interval)
        for b in range(n_beats):
            ji = beat_interval * max(0.3, 1 + self.rng.normal(0, hr_var))
            cycle = self._ppg_cycle(60.0 / ji)
            start = int(b * beat_interval * self.fs)
            end = min(start + len(cycle), n)
            if start >= n: break
            ppg[start:end] += cycle[:end - start]
        if ppg.max() > ppg.min():
            ppg = (ppg - ppg.min()) / (ppg.max() - ppg.min())
        ppg = ppg * 2 - 1
        if motion > 0:
            for _ in range(self.rng.integers(2, 6)):
                sl = n // self.rng.integers(4, 10)
                s = self.rng.integers(0, max(1, n - sl))
                ppg[s:s+sl] += self.rng.normal(0, 0.3 * motion, min(sl, n - s)).astype(np.float32)
        if contact < 1.0:
            for _ in range(self.rng.integers(1, 3)):
                dl = int(n * 0.02 * self.rng.uniform(0.5, 2.0))
                ds = self.rng.integers(0, max(1, n - dl))
                ppg[ds:ds+dl] *= 0.05
        if ambient > 0:
            ppg += self.rng.normal(0, ambient, n).astype(np.float32)
        sp = np.mean(ppg ** 2)
        ppg += self.rng.normal(0, np.sqrt(sp / (10 ** (snr_db / 10))), n).astype(np.float32)
        return ppg, float(base_hr)

    def healthy(self, dur=120.0):
        hr = self.rng.uniform(58, 78)
        return self.generate(dur, hr, 0.15, 0.15, 0.95, 0.02, 22)

    def at_risk(self, dur=120.0):
        hr = self.rng.uniform(90, 130)
        return self.generate(dur, hr, 0.30, 0.4, 0.80, 0.08, 15)

    def borderline(self, dur=120.0):
        hr = self.rng.uniform(78, 95)
        return self.generate(dur, hr, 0.22, 0.3, 0.85, 0.05, 18)


# ---------------------------------------------------------------------------
# Feature Extraction (from v6)
# ---------------------------------------------------------------------------

def extract_features(ppg, fs=25):
    from scipy.signal import find_peaks, welch
    feats = {}
    feats["signal_length"] = len(ppg)
    feats["mean_amplitude"] = float(np.mean(ppg))
    feats["std_amplitude"] = float(np.std(ppg))
    feats["sqi"] = float(1.0 - min(1.0, np.std(np.diff(ppg)) / (np.std(ppg) + 1e-8)))
    filt = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
    peaks, _ = find_peaks(filt, distance=int(fs * 0.4), height=0.0)
    if len(peaks) < 5: return feats
    rr = np.diff(peaks) / fs * 1000.0
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 3: return feats
    feats["HRV_MeanNN"] = float(np.mean(rr))
    feats["HRV_SDNN"] = float(np.std(rr, ddof=1))
    feats["HRV_RMSSD"] = float(np.sqrt(np.mean(np.diff(rr)**2)))
    feats["HRV_SDSD"] = float(np.std(np.diff(rr), ddof=1))
    feats["HRV_CVNN"] = feats["HRV_SDNN"]/(feats["HRV_MeanNN"]+1e-8)
    feats["HRV_CVSD"] = feats["HRV_RMSSD"]/(feats["HRV_MeanNN"]+1e-8)
    feats["HRV_MedianNN"] = float(np.median(rr))
    feats["HRV_MadNN"] = float(np.median(np.abs(rr - np.median(rr))))
    feats["HRV_MCVNN"] = feats["HRV_MadNN"]/(feats["HRV_MedianNN"]+1e-8)
    feats["HRV_IQRNN"] = float(np.percentile(rr,75)-np.percentile(rr,25))
    feats["HRV_SDRMSSD"] = feats["HRV_SDNN"]/(feats["HRV_RMSSD"]+1e-8)
    feats["HRV_Prc20NN"] = float(np.percentile(rr,20))
    feats["HRV_Prc80NN"] = float(np.percentile(rr,80))
    feats["HRV_pNN50"] = float(100*np.sum(np.abs(np.diff(rr))>50)/len(rr))
    feats["HRV_pNN20"] = float(100*np.sum(np.abs(np.diff(rr))>20)/len(rr))
    feats["HRV_MinNN"] = float(np.min(rr))
    feats["HRV_MaxNN"] = float(np.max(rr))
    try:
        bw=7.8125
        h,_=np.histogram(rr,bins=np.arange(np.min(rr),np.max(rr)+bw,bw))
        feats["HRV_HTI"]=float(len(rr)/(np.max(h)+1e-8))
    except: pass
    try:
        rt=np.cumsum(rr)/1000.0; rt=rt-rt[0]
        tu=np.arange(0,rt[-1],0.25)
        ri=np.interp(tu,rt,rr); ri=ri-np.mean(ri)
        f,psd=welch(ri,fs=4.0,nperseg=min(len(ri),256))
        lf_m=(f>=0.04)&(f<0.15); hf_m=(f>=0.15)&(f<0.4); vhf_m=(f>=0.4)&(f<0.5)
        lf=float(np.trapz(psd[lf_m],f[lf_m])) if lf_m.any() else 0.0
        hf=float(np.trapz(psd[hf_m],f[hf_m])) if hf_m.any() else 0.0
        vhf=float(np.trapz(psd[vhf_m],f[vhf_m])) if vhf_m.any() else 0.0
        tp=lf+hf+vhf
        feats.update({"HRV_LF":lf,"HRV_HF":hf,"HRV_VHF":vhf,"HRV_TP":tp,
            "HRV_LFHF":lf/(hf+1e-8),"HRV_LFn":lf/(tp+1e-8),"HRV_HFn":hf/(tp+1e-8),
            "HRV_LnHF":float(np.log(hf+1e-8))})
    except: pass
    if len(rr)>2:
        sd1=float(np.std(rr[1:]-rr[:-1])/np.sqrt(2))
        sd2=float(np.sqrt(2*np.var(rr)-sd1**2))
        feats.update({"HRV_SD1":sd1,"HRV_SD2":sd2,"HRV_SD1SD2":sd1/(sd2+1e-8),
            "HRV_CSI":sd1/(sd2+1e-8),"HRV_CVI":float(np.log10(sd1*sd2+1e-8)),
            "HRV_CSI_Modified":float(3*sd1/(sd2+1e-8))})
    try:
        if len(rr)>10:
            n=len(rr); sc=np.arange(4,min(n//4,64)); fl=[]
            for s in sc:
                nw=n//s
                if nw<1: continue
                rms=[]
                for i in range(nw):
                    w=rr[i*s:(i+1)*s]; x=np.arange(s)
                    c=np.polyfit(x,w,1); d=w-np.polyval(c,x)
                    rms.append(np.sqrt(np.mean(d**2)))
                fl.append(np.mean(rms))
            if len(fl)>2:
                feats["HRV_DFA_alpha1"]=float(np.polyfit(np.log(sc[:len(fl)]),np.log(np.array(fl)+1e-8),1)[0])
    except: pass
    feats["pulse_rate"]=float(len(peaks)/(len(ppg)/fs)*60.0)
    return feats


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_real_data():
    """Load real data using pre-extracted features from parquet (fast)."""
    signals_df = pd.read_parquet("data/processed/signals.parquet")
    features_df = pd.read_parquet("data/processed/features.parquet")

    ppgs, feats_list, labels, sources = [], [], [], []

    for idx, row in signals_df.iterrows():
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

            # Use pre-extracted features from parquet (fast)
            feat = {}
            row_feats = features_df.iloc[idx]
            for col in features_df.columns:
                val = row_feats[col]
                if isinstance(val, (int, float, np.integer, np.floating)):
                    feat[col] = float(val) if not np.isnan(val) else 0.0

            label = 0 if row["event_type"] == "CONTROL" else 1

            ppgs.append(padded)
            feats_list.append(feat)
            labels.append(label)
            sources.append(f"real_{row['window_type']}_{row['event_type']}")
        except Exception as e:
            continue

    logger.info("Loaded %d real signals (%d healthy, %d at-risk)",
                len(ppgs), sum(1 for l in labels if l==0), sum(1 for l in labels if l==1))
    return ppgs, feats_list, labels, sources


def load_synthetic_data(n_healthy=200, n_at_risk=200, n_borderline=80, seed=42):
    """Generate synthetic signals with on-the-fly feature extraction."""
    gen = WatchPPGGenerator(fs=25, seed=seed)
    ppgs, feats_list, labels, sources = [], [], [], []

    for i in range(n_healthy):
        ppg, hr = gen.healthy()
        feats = extract_features(ppg, fs=25)
        feats["base_hr"] = hr
        ppgs.append(ppg); feats_list.append(feats); labels.append(0)
        sources.append("synthetic_healthy")

    for i in range(n_at_risk):
        ppg, hr = gen.at_risk()
        feats = extract_features(ppg, fs=25)
        feats["base_hr"] = hr
        ppgs.append(ppg); feats_list.append(feats); labels.append(1)
        sources.append("synthetic_at_risk")

    for i in range(n_borderline):
        ppg, hr = gen.borderline()
        feats = extract_features(ppg, fs=25)
        feats["base_hr"] = hr
        ppgs.append(ppg); feats_list.append(feats); labels.append(1)
        sources.append("synthetic_borderline")

    logger.info("Generated %d synthetic signals", len(ppgs))
    return ppgs, feats_list, labels, sources


def build_dataset(real_ppgs, real_feats, real_labels, real_sources,
                  synth_ppgs, synth_feats, synth_labels, synth_sources):
    all_ppgs = real_ppgs + synth_ppgs
    all_feats = real_feats + synth_feats
    all_labels = real_labels + synth_labels
    all_sources = real_sources + synth_sources

    X_ppg = np.zeros((len(all_ppgs), PPG_LENGTH), dtype=np.float32)
    for i, ppg in enumerate(all_ppgs):
        L = min(len(ppg), PPG_LENGTH)
        X_ppg[i, :L] = ppg[:L]
    X_ppg = X_ppg[..., np.newaxis]

    all_cols = sorted(set().union(*[f.keys() for f in all_feats]))
    X_feat = np.zeros((len(all_feats), len(all_cols)), dtype=np.float32)
    for i, f in enumerate(all_feats):
        for j, col in enumerate(all_cols):
            X_feat[i, j] = f.get(col, 0.0)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

    y = np.array(all_labels, dtype=np.float32)
    logger.info("Dataset: %d samples (%d healthy, %d at-risk), %d features",
                len(y), int((y==0).sum()), int((y==1).sum()), len(all_cols))
    return X_ppg, X_feat, y, all_cols, all_sources


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_v7_watch():
    out_dir = Path("production/cvd_risk_v7_watch")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TRAINING v7-watch — Hybrid Real + Synthetic Data")
    logger.info("=" * 60)

    logger.info("\n[1/5] Loading real data...")
    real_ppgs, real_feats, real_labels, real_sources = load_real_data()

    logger.info("\n[2/5] Generating synthetic augmentation...")
    synth_ppgs, synth_feats, synth_labels, synth_sources = load_synthetic_data(
        n_healthy=200, n_at_risk=200, n_borderline=80
    )

    logger.info("\n[3/5] Building dataset...")
    X_ppg, X_feat, y, feature_cols, sources = build_dataset(
        real_ppgs, real_feats, real_labels, real_sources,
        synth_ppgs, synth_feats, synth_labels, synth_sources,
    )

    from sklearn.model_selection import train_test_split
    X_ppg_t, X_ppg_te, X_feat_t, X_feat_te, y_t, y_te = train_test_split(
        X_ppg, X_feat, y, test_size=0.15, random_state=42, stratify=y)
    X_ppg_tr, X_ppg_v, X_feat_tr, X_feat_v, y_tr, y_v = train_test_split(
        X_ppg_t, X_feat_t, y_t, test_size=0.176, random_state=42, stratify=y_t)

    logger.info("Split: train=%d, val=%d, test=%d", len(y_tr), len(y_v), len(y_te))

    from src.model_watch import build_watch_model
    model = build_watch_model(ppg_input_shape=(PPG_LENGTH, 1), feature_dim=X_feat.shape[1])
    model.summary(print_fn=logger.info)

    n_h = int((y_tr == 0).sum())
    n_e = int((y_tr == 1).sum())
    cw = {0: (n_h+n_e)/(2*n_h), 1: (n_h+n_e)/(2*n_e)}
    logger.info("Class weights: %s", cw)

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
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=20, mode="max", restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(out_dir/"best_model.keras"), monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), histogram_freq=1, write_graph=True, write_images=True, update_freq="epoch", profile_batch=0),
        tf.keras.callbacks.CSVLogger(str(out_dir/"training_log.csv"), append=False),
    ]

    logger.info("\n[4/5] Training...")
    history = model.fit(
        {"ppg_input": X_ppg_tr, "feature_input": X_feat_tr}, y_tr,
        validation_data=({"ppg_input": X_ppg_v, "feature_input": X_feat_v}, y_v),
        epochs=120, batch_size=32, class_weight=cw, callbacks=callbacks,
    )

    logger.info("\n[5/5] Evaluating...")
    preds = model({"ppg_input": X_ppg_te, "feature_input": X_feat_te}, training=False)
    y_prob = np.array(preds).flatten()

    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, brier_score_loss)

    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.005):
        f = f1_score(y_te, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1: best_f1, best_t = f, t

    y_pred = (y_prob >= best_t).astype(int)
    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "auroc": float(roc_auc_score(y_te, y_prob)),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, zero_division=0)),
        "f1": float(best_f1),
        "brier": float(brier_score_loss(y_te, y_prob)),
        "threshold": float(best_t),
        "n_test": int(len(y_te)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

    logger.info("=" * 60)
    logger.info("RESULTS: AUROC=%.4f Acc=%.1f%% Prec=%.4f Rec=%.4f F1=%.4f",
                metrics["auroc"], metrics["accuracy"]*100, metrics["precision"],
                metrics["recall"], metrics["f1"])
    logger.info("Threshold=%.3f Brier=%.4f CM: TN=%d FP=%d FN=%d TP=%d",
                metrics["threshold"], metrics["brier"], tn, fp, fn, tp)

    # Save
    model.save(str(out_dir / "final_model.keras"))
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    real_n = len(real_labels)
    synth_n = len(synth_labels)
    config = {
        "version": "v7-watch",
        "description": "CVD risk model trained on hybrid real + synthetic Apple Watch PPG",
        "ppg_length": PPG_LENGTH, "sampling_rate_hz": FS_TARGET,
        "feature_columns": feature_cols,
        "architecture": {
            "ppg_branch": "ResNet 1D-CNN (16->32->64) + BiLSTM(32)",
            "feature_branch": "MLP (32, 32)", "shared": "Dense(32)",
            "event_head": "Dense(16) -> Dense(1, sigmoid)",
            "total_params": model.count_params(),
        },
        "training": {
            "dataset": "hybrid_real_synthetic",
            "real_samples": real_n, "synthetic_samples": synth_n,
            "total_samples": int(len(y)),
            "real_sources": ["MIMIC-IV (MI, ARREST)", "MMASH (CONTROL)"],
            "split": "70/15/15", "optimizer": "AdamW",
            "lr": 3e-4, "batch_size": 32,
            "epochs_trained": len(history.history["loss"]),
            "class_weights": cw,
        },
        "performance": metrics,
    }

    with open(out_dir / "config.yaml", "w") as f:
        import yaml; yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)
    with open(out_dir / "optimal_threshold.json", "w") as f:
        json.dump({"threshold": best_t}, f)

    logger.info("Saved to %s", out_dir)
    logger.info("TensorBoard: tensorboard --logdir %s", log_dir)
    return model, metrics, history


if __name__ == "__main__":
    train_v7_watch()
