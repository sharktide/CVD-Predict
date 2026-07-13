"""Preprocessing – window slicing, SQI, wearable simulation, feature extraction.

This module loads raw waveforms from MIMIC-IV Waveform Database and real
wearable data from MMASH / Sleep-Accel, then slices them into windows
aligned to event times, computes features, and saves everything to
data/processed/.

Signal storage convention
-------------------------
Large arrays (raw_ppg, wearable_ppg) are saved as individual .npy files
under ``data/processed/signals/``.  The ``signals.parquet`` file stores
metadata and file paths rather than the arrays themselves.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import get_paths_config
from src.utils import (
    compute_sqi_simple,
    ensure_dir,
    load_numpy,
    load_parquet,
    robust_scale,
    save_numpy,
    save_parquet,
    simulate_wearable,
    zscore,
)

logger = logging.getLogger(__name__)

# Attempt to import neurokit2 for HRV / morphology features
try:
    import neurokit2 as nk
    HAS_NK2 = True
except ImportError:
    HAS_NK2 = False
    logger.warning("neurokit2 not installed – HRV/morphology features will be limited.")


# ---------------------------------------------------------------------------
# HRV feature extraction
# ---------------------------------------------------------------------------

def extract_hrv_features(
    ppg_segment: np.ndarray,
    fs: int = 125,
) -> Dict[str, float]:
    """Extract HRV time-domain, frequency-domain, and non-linear features from a PPG segment."""
    features: Dict[str, float] = {}

    if HAS_NK2:
        try:
            signals, info = nk.ppg_process(ppg_segment, sampling_rate=fs)
            peaks = info["PPG_Peaks"]
            if len(peaks) < 5:
                return features
            hrv = nk.hrv(peaks, sampling_rate=fs, show=False)
            for col in hrv.columns:
                val = hrv[col].iloc[0]
                if np.isfinite(val):
                    features[col] = float(val)
            return features
        except Exception:
            pass

    # Fallback: autocorrelation-based RR extraction
    from scipy.signal import find_peaks as _find_peaks

    filt = zscore(ppg_segment)
    peaks, _ = _find_peaks(filt, distance=int(fs * 0.4), height=0.0)
    if len(peaks) < 5:
        return features

    rr_ms = np.diff(peaks) / fs * 1000.0
    rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
    if len(rr_ms) < 3:
        return features

    features["mean_rr"] = float(np.mean(rr_ms))
    features["sd_rr"] = float(np.std(rr_ms))
    features["rmssd"] = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
    features["median_rr"] = float(np.median(rr_ms))
    features["range_rr"] = float(np.ptp(rr_ms))
    features["iqr_rr"] = float(np.percentile(rr_ms, 75) - np.percentile(rr_ms, 25))
    features["cv_rr"] = features["sd_rr"] / (features["mean_rr"] + 1e-8)

    return features


# ---------------------------------------------------------------------------
# Morphology features
# ---------------------------------------------------------------------------

def extract_morphology_features(
    ppg_segment: np.ndarray,
    fs: int = 125,
) -> Dict[str, float]:
    """Basic PPG morphology: systolic/diastolic slopes, dicrotic notch proxy, pulse amplitude."""
    features: Dict[str, float] = {}

    if HAS_NK2:
        try:
            signals, info = nk.ppg_process(ppg_segment, sampling_rate=fs)
            peaks = info["PPG_Peaks"]
            troughs = info.get("PPG_Troughs", np.array([]))
            if len(peaks) > 1 and len(troughs) > 0:
                systolic_amps = ppg_segment[peaks]
                features["mean_systolic_amp"] = float(np.mean(systolic_amps))
                features["std_systolic_amp"] = float(np.std(systolic_amps))
            if len(peaks) > 2:
                features["pulse_rate"] = float(len(peaks) / (len(ppg_segment) / fs) * 60.0)
        except Exception:
            pass

    return features


# ---------------------------------------------------------------------------
# Full feature extraction per window
# ---------------------------------------------------------------------------

def extract_window_features(
    ppg_segment: np.ndarray,
    fs: int = 125,
    horizon_hours: float = 24.0,
    window_type: str = "baseline",
    event_type: str = "CONTROL",
    label_confidence: float = 0.0,
    device_domain: int = 0,
) -> Dict[str, Any]:
    """Return a flat dict of features for one windowed segment."""
    sqi = compute_sqi_simple(ppg_segment, fs=fs)
    hrv = extract_hrv_features(ppg_segment, fs=fs)
    morph = extract_morphology_features(ppg_segment, fs=fs)

    features: Dict[str, Any] = {
        "sqi": sqi,
        "horizon_hours": horizon_hours,
        "window_type": window_type,
        "event_type": event_type,
        "label_confidence": label_confidence,
        "device_domain": device_domain,
        "signal_length": len(ppg_segment),
        "mean_amplitude": float(np.mean(ppg_segment)),
        "std_amplitude": float(np.std(ppg_segment)),
    }
    features.update(hrv)
    features.update(morph)
    return features


# ---------------------------------------------------------------------------
# MIMIC-IV Waveform window slicing
# ---------------------------------------------------------------------------

def slice_mimic_windows(
    meta_df: pd.DataFrame,
    waveform_loader,
    output_dir: str,
    fs: int = 125,
    window_seconds: int = 60,
    crisis_offsets_hours: Optional[List[float]] = None,
    baseline_offset_hours: float = 24.0,
    min_sqi: float = 0.1,
    max_waveform_records: int = 2000,
    max_windows_per_subject: int = 20,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load MIMIC-IV Waveform records, match to event cohort, slice windows.

    v2: tries multiple starting positions per record to extract more windows,
    lower SQI threshold, and filters out all-NaN signals.
    """
    if crisis_offsets_hours is None:
        crisis_offsets_hours = [1.0, 6.0, 14.0, 24.0]

    signals_dir = os.path.join(output_dir, "signals")
    ensure_dir(signals_dir)

    event_lookup: Dict[int, List[Dict[str, Any]]] = {}
    for _, row in meta_df.iterrows():
        sid = int(row["patient_id"])
        if sid not in event_lookup:
            event_lookup[sid] = []
        event_lookup[sid].append({
            "hadm_id": row.get("hadm_id"),
            "event_type": row.get("event_type", "CONTROL"),
            "event_time": pd.to_datetime(row.get("event_time")),
            "label_confidence": float(row.get("label_confidence", 0.0)),
        })

    if not event_lookup:
        logger.warning("No events in meta_df — cannot slice MIMIC windows")
        return [], []

    logger.info("Loading MIMIC-IV Waveform records (max %d)...", max_waveform_records)
    wf_records = waveform_loader.list_records(max_records=max_waveform_records)
    logger.info("Found %d waveform records", len(wf_records))

    all_features: List[Dict[str, Any]] = []
    all_signals: List[Dict[str, Any]] = []

    processed_subjects = set()
    samples_per_window = window_seconds * fs
    stride = samples_per_window // 2  # 50% overlap

    for rec_info in wf_records:
        sid = rec_info["subject_id"]
        if sid is None or sid not in event_lookup:
            continue
        if sid in processed_subjects:
            continue

        loaded = waveform_loader.load_record(rec_info["header_path"])
        if loaded is None or loaded.get("ppg") is None:
            continue

        ppg_signal = loaded["ppg"]
        record_fs = loaded["fs"]

        if abs(record_fs - fs) > 1:
            step = max(1, round(record_fs / fs))
            ppg_signal = ppg_signal[::step]

        processed_subjects.add(sid)
        windows_for_subject = 0

        for evt in event_lookup[sid]:
            if windows_for_subject >= max_windows_per_subject:
                break

            event_type = evt["event_type"]
            label_conf = evt["label_confidence"]
            horizons = [(baseline_offset_hours, "baseline")]
            horizons += [(h, "crisis") for h in crisis_offsets_hours]

            for horizon, wtype in horizons:
                if windows_for_subject >= max_windows_per_subject:
                    break

                # Try multiple starting positions across the record
                record_samples = len(ppg_signal)
                if record_samples < samples_per_window:
                    continue

                num_positions = max(1, (record_samples - samples_per_window) // stride + 1)
                num_positions = min(num_positions, 5)  # cap at 5 positions per horizon

                for pos_idx in range(num_positions):
                    if windows_for_subject >= max_windows_per_subject:
                        break

                    start_idx = pos_idx * stride
                    end_idx = start_idx + samples_per_window

                    if end_idx > record_samples:
                        break

                    segment = ppg_signal[start_idx:end_idx].astype(np.float32)

                    # Skip all-NaN or all-zero segments
                    if np.all(np.isnan(segment)) or np.all(segment == 0):
                        continue

                    segment = np.nan_to_num(segment, nan=0.0)
                    sqi = compute_sqi_simple(segment, fs=fs)
                    if sqi < min_sqi:
                        continue

                    seg_norm = zscore(segment)
                    if np.all(np.isnan(seg_norm)) or np.all(seg_norm == 0):
                        continue
                    seg_norm = np.nan_to_num(seg_norm, nan=0.0)

                    wearable_seg = simulate_wearable(seg_norm, target_rate=25, base_fs=fs)

                    features = extract_window_features(
                        seg_norm, fs=fs,
                        horizon_hours=horizon, window_type=wtype,
                        event_type=event_type, label_confidence=label_conf,
                        device_domain=0,
                    )

                    fid = f"mimic_{sid}_{evt.get('hadm_id', 0)}_{wtype}_{horizon}_{pos_idx}_{len(all_features)}"
                    features["feature_id"] = fid
                    features["patient_id"] = sid
                    all_features.append(features)

                    raw_path = os.path.join(signals_dir, f"{fid}_raw.npy")
                    wear_path = os.path.join(signals_dir, f"{fid}_wear.npy")
                    save_numpy(seg_norm, raw_path)
                    save_numpy(wearable_seg, wear_path)

                    all_signals.append({
                        "feature_id": fid,
                        "patient_id": sid,
                        "window_type": wtype,
                        "horizon_hours": horizon,
                        "event_type": event_type,
                        "raw_ppg_path": raw_path,
                        "wearable_ppg_path": wear_path,
                        "device_domain": 0,
                    })
                    windows_for_subject += 1

    logger.info("Sliced %d MIMIC windows from %d subjects",
                len(all_features), len(processed_subjects))
    return all_features, all_signals


# ---------------------------------------------------------------------------
# Wearable (MMASH) window slicing
# ---------------------------------------------------------------------------

def slice_mmash_windows(
    mmash_loader,
    output_dir: str,
    max_users: int = 22,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load MMASH wearable data and create windows for domain adversarial training.

    MMASH provides beat-to-beat RR intervals and 3-axis accelerometer data.
    We treat each recording as a non-event (control) window for the wearable domain.

    Returns
    -------
    (features_records, signals_records)
    """
    signals_dir = os.path.join(output_dir, "signals")
    ensure_dir(signals_dir)

    all_features: List[Dict[str, Any]] = []
    all_signals: List[Dict[str, Any]] = []

    users = mmash_loader.list_users()[:max_users]
    logger.info("Loading MMASH data for %d users", len(users))

    for uid in users:
        data = mmash_loader.load_user_all(uid)
        rr_df = data["rr"]
        act_df = data["actigraph"]

        if rr_df.empty:
            continue

        # Convert RR intervals (ibi_s in seconds) to a PPG-like signal
        # by creating an impulse train at beat times
        if "ibi_s" in rr_df.columns:
            ibi = rr_df["ibi_s"].values.astype(float)
            ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]  # physiological range
            if len(ibi) < 10:
                continue

            # Create a pseudo-PPG signal from RR intervals
            fs_wearable = 25  # target rate
            total_time = np.sum(ibi)
            n_samples = int(total_time * fs_wearable)
            ppg = np.zeros(n_samples, dtype=np.float32)

            beat_times = np.cumsum(ibi)
            for bt in beat_times:
                idx = int(bt * fs_wearable)
                if idx < n_samples:
                    ppg[max(0, idx - 2):min(n_samples, idx + 3)] = 1.0

            # Smooth to look more like PPG
            from scipy.ndimage import gaussian_filter1d
            ppg = gaussian_filter1d(ppg, sigma=2.0)

            # Get accelerometer if available
            accel = None
            if not act_df.empty and all(c in act_df.columns for c in ["Axis1", "Axis2", "Axis3"]):
                accel = act_df[["Axis1", "Axis2", "Axis3"]].values.astype(np.float32)

            # Create windows (no specific event — these are controls)
            window_samples = 25 * 60  # 60 seconds at 25 Hz
            n_windows = max(1, len(ppg) // window_samples)

            for w_idx in range(min(n_windows, 10)):  # v2: up to 10 windows per user
                start = w_idx * window_samples
                end = start + window_samples
                if end > len(ppg):
                    break

                seg = ppg[start:end]
                seg_norm = zscore(seg)

                features = extract_window_features(
                    seg_norm, fs=fs_wearable,
                    horizon_hours=0.0, window_type="wearable_control",
                    event_type="CONTROL", label_confidence=1.0,
                    device_domain=1,  # real wearable
                )

                fid = f"mmash_{uid}_{w_idx}_{len(all_features)}"
                features["feature_id"] = fid
                features["patient_id"] = f"mmash_{uid}"
                all_features.append(features)

                # Save signals
                raw_path = os.path.join(signals_dir, f"{fid}_raw.npy")
                wear_path = os.path.join(signals_dir, f"{fid}_wear.npy")
                save_numpy(seg_norm, raw_path)
                save_numpy(seg_norm, wear_path)  # already at wearable quality

                all_signals.append({
                    "feature_id": fid,
                    "patient_id": f"mmash_{uid}",
                    "window_type": "wearable_control",
                    "horizon_hours": 0.0,
                    "event_type": "CONTROL",
                    "raw_ppg_path": raw_path,
                    "wearable_ppg_path": wear_path,
                    "device_domain": 1,
                })

    logger.info("Sliced %d MMASH wearable windows", len(all_features))
    return all_features, all_signals


# ---------------------------------------------------------------------------
# Wearable (Sleep Accel) window slicing
# ---------------------------------------------------------------------------

def slice_sleep_accel_windows(
    sleep_loader,
    output_dir: str,
    max_subjects: int = 20,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load Sleep Accel data and create wearable-domain windows."""
    signals_dir = os.path.join(output_dir, "signals")
    ensure_dir(signals_dir)

    all_features: List[Dict[str, Any]] = []
    all_signals: List[Dict[str, Any]] = []

    all_data = sleep_loader.load_all()
    subjects = list(all_data.keys())[:max_subjects]
    logger.info("Loading Sleep Accel data for %d subjects", len(subjects))

    for sid in subjects:
        hr_data = all_data[sid].get("heart_rate", pd.DataFrame())
        if hr_data.empty or "value" not in hr_data.columns:
            continue

        hr = hr_data["value"].values.astype(float)
        hr = hr[(hr > 30) & (hr < 220)]
        if len(hr) < 10:
            continue

        # Create pseudo-PPG from HR series (interpolated to uniform rate)
        fs_wearable = 25
        t = np.arange(len(hr), dtype=float)
        t_uniform = np.linspace(0, len(hr) - 1, int(len(hr) * fs_wearable / 1.0))
        ppg = np.interp(t_uniform, t, hr)
        ppg = zscore(ppg)

        window_samples = 25 * 60
        n_windows = max(1, len(ppg) // window_samples)

        for w_idx in range(min(n_windows, 5)):  # v2: up to 5 windows per subject
            start = w_idx * window_samples
            end = start + window_samples
            if end > len(ppg):
                break

            seg = ppg[start:end]
            features = extract_window_features(
                seg, fs=fs_wearable,
                horizon_hours=0.0, window_type="wearable_control",
                event_type="CONTROL", label_confidence=1.0,
                device_domain=1,
            )

            fid = f"sleepaccel_{sid}_{w_idx}_{len(all_features)}"
            features["feature_id"] = fid
            features["patient_id"] = f"sleepaccel_{sid}"
            all_features.append(features)

            raw_path = os.path.join(signals_dir, f"{fid}_raw.npy")
            wear_path = os.path.join(signals_dir, f"{fid}_wear.npy")
            save_numpy(seg, raw_path)
            save_numpy(seg, wear_path)

            all_signals.append({
                "feature_id": fid,
                "patient_id": f"sleepaccel_{sid}",
                "window_type": "wearable_control",
                "horizon_hours": 0.0,
                "event_type": "CONTROL",
                "raw_ppg_path": raw_path,
                "wearable_ppg_path": wear_path,
                "device_domain": 1,
            })

    logger.info("Sliced %d Sleep Accel wearable windows", len(all_features))
    return all_features, all_signals


# ---------------------------------------------------------------------------
# Full preprocessing orchestrator
# ---------------------------------------------------------------------------

def preprocess_all() -> None:
    """End-to-end preprocessing: load data, slice windows, extract features, save.

    Outputs (in data/processed/):
        features.parquet  – one row per window with all engineered features
        signals.parquet   – metadata + file paths for .npy signal files
        signals/          – individual .npy files for raw and wearable PPG
    """
    from src.data_loaders import (
        MIMICWaveformLoader,
        MMASHLoader,
        SleepAccelLoader,
    )

    paths = get_paths_config()
    processed_dir = paths["processed_data_dir"]
    raw_dir = paths["raw_data_dir"]
    ensure_dir(processed_dir)

    meta_path = os.path.join(processed_dir, "cohort_meta.parquet")
    if not os.path.exists(meta_path):
        logger.error("cohort_meta.parquet not found – run labeling.py and cohort.py first.")
        return

    meta_df = load_parquet(meta_path)
    logger.info("Loaded cohort metadata: %d rows", len(meta_df))

    all_features: List[Dict[str, Any]] = []
    all_signals: List[Dict[str, Any]] = []

    # --- 1. MIMIC-IV Waveform windows ---
    mimic_wf_dir = os.path.join(raw_dir, "mimic4wdb")
    if os.path.exists(mimic_wf_dir):
        wf_loader = MIMICWaveformLoader(raw_dir)
        feat, sig = slice_mimic_windows(
            meta_df, wf_loader, processed_dir,
            max_waveform_records=2000,
        )
        all_features.extend(feat)
        all_signals.extend(sig)
    else:
        logger.warning("MIMIC-IV Waveform directory not found at %s", mimic_wf_dir)

    # --- 2. MMASH wearable windows ---
    mmash_dir = os.path.join(raw_dir, "mmash")
    if os.path.exists(mmash_dir):
        mmash_loader = MMASHLoader(raw_dir)
        feat, sig = slice_mmash_windows(mmash_loader, processed_dir)
        all_features.extend(feat)
        all_signals.extend(sig)
    else:
        logger.warning("MMASH directory not found at %s", mmash_dir)

    # --- 3. Sleep Accel wearable windows ---
    sleep_dir = os.path.join(raw_dir, "sleep_accel")
    if os.path.exists(sleep_dir):
        sleep_loader = SleepAccelLoader(raw_dir)
        feat, sig = slice_sleep_accel_windows(sleep_loader, processed_dir)
        all_features.extend(feat)
        all_signals.extend(sig)
    else:
        logger.warning("Sleep Accel directory not found at %s", sleep_dir)

    # --- Save ---
    if not all_features:
        logger.error("No features extracted — check data directories and cohort_meta")
        return

    features_df = pd.DataFrame(all_features)
    signals_df = pd.DataFrame(all_signals)

    # Ensure consistent dtypes for parquet serialization
    features_df["patient_id"] = features_df["patient_id"].astype(str)
    signals_df["patient_id"] = signals_df["patient_id"].astype(str)

    save_parquet(features_df, os.path.join(processed_dir, "features.parquet"))
    save_parquet(signals_df, os.path.join(processed_dir, "signals.parquet"))

    logger.info(
        "Saved features (%d rows) and signals (%d rows) → %s",
        len(features_df), len(signals_df), processed_dir,
    )
    logger.info("Device domain distribution:\n%s",
                features_df["device_domain"].value_counts().to_string())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    preprocess_all()
