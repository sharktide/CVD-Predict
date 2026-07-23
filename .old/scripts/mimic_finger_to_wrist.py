#!/usr/bin/env python3
"""Convert MIMIC ICU finger PPG to synthetic wrist-like PPG.

MIMIC PPG (finger pulse oximeter):
  - High SNR (arterial bed, clinical sensor)
  - Sampling rate ~62.5 Hz (variable)
  - Range ~[0, 1], strong pulsatile component
  - AC/DC ratio ~0.24
  - No ambient light, no motion artifacts

Wrist PPG (smartwatch):
  - Low SNR (through skin/tissue/bone)
  - Target sampling rate 25 Hz
  - Much weaker pulsatile component
  - AC/DC ratio ~0.01-0.05
  - Ambient light contamination
  - Motion artifacts, contact variation
  - Temperature-dependent baseline drift

Transformations applied:
  1. Resample to 25 Hz
  2. Reduce pulsatile amplitude (perfusion reduction)
  3. Add DC baseline drift (skin optics)
  4. Add ambient light contamination
  5. Add motion artifacts (filtered noise)
  6. Add sensor noise floor
  7. Apply contact quality variation
  8. Add temperature-dependent baseline drift

Output: PPG waveforms saved as .npy files + metadata CSV.
"""

from __future__ import annotations

import os
import sys
import logging
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, resample_poly, welch
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MIMIC_WAVDB = Path("data/raw/mimic4wdb")
MIMIC_CLINICAL = Path("data/raw/mimiciv-clinical")
OUTPUT_DIR = Path("data/processed/mimic_wristppg")
TARGET_FS = 25  # Target wrist sampling rate
PPG_LENGTH = TARGET_FS * 60  # 60 seconds = 1500 samples


@dataclass
class WristTransformConfig:
    """Parameters controlling finger→wrist PPG transformation."""
    # Perfusion reduction
    perfusion_scale_range: tuple = (0.02, 0.08)  # Wrist AC/DC is ~2-8% of finger
    # Ambient light
    ambient_light_range: tuple = (0.01, 0.05)  # Fraction of DC offset
    ambient_freq_range: tuple = (0.005, 0.03)  # Slow drift frequency (Hz)
    # Motion artifacts
    motion_prob: float = 0.3  # Probability of adding motion
    motion_amplitude_range: tuple = (0.005, 0.03)
    motion_freq_range: tuple = (0.5, 3.0)  # Hz
    # Sensor noise
    noise_snr_range: tuple = (15, 30)  # dB
    # Contact quality
    contact_loss_prob: float = 0.1
    contact_loss_duration_range: tuple = (0.5, 5.0)  # seconds
    contact_gain_range: tuple = (0.3, 0.8)  # Amplitude during partial contact
    # Temperature drift
    temp_drift_amplitude: float = 0.002
    temp_drift_freq_range: tuple = (0.001, 0.005)  # Very slow
    # Baseline wander
    baseline_wander_amplitude: float = 0.01
    baseline_wander_freq_range: tuple = (0.05, 0.15)


def extract_mimic_ppg_segments():
    """Extract all PPG segments from MIMIC waveform database.

    Returns list of dicts with subject_id, hadm_id, path, duration_s, fs.
    """
    import re

    records = []
    for partition_dir in sorted(os.listdir(MIMIC_WAVDB)):
        if not partition_dir.startswith("p") or not os.path.isdir(MIMIC_WAVDB / partition_dir):
            continue
        for patient_dir in os.listdir(MIMIC_WAVDB / partition_dir):
            patient_path = MIMIC_WAVDB / partition_dir / patient_dir
            if not patient_dir.startswith("p") or not patient_path.is_dir():
                continue
            for study_dir in os.listdir(patient_path):
                study_path = patient_path / study_dir
                if not study_path.is_dir():
                    continue

                # Find top-level header
                hea_files = [f for f in os.listdir(study_path) if f.endswith(".hea") and "_" not in f]
                ppg_dat_files = [f for f in os.listdir(study_path) if f.endswith("p.dat")]

                if not hea_files or not ppg_dat_files:
                    continue

                # Parse header for subject_id, hadm_id
                with open(study_path / hea_files[0]) as f:
                    header_text = f.read()
                subj_match = re.search(r"#\s*subject_id\s+(\d+)", header_text)
                hadm_match = re.search(r"#\s*hadm_id\s+(\d+)", header_text)
                if not subj_match:
                    continue

                subject_id = int(subj_match.group(1))
                hadm_id = int(hadm_match.group(1)) if hadm_match else 0

                # Parse segment headers for durations
                total_duration = 0
                for ppg_dat in ppg_dat_files:
                    seg_id = ppg_dat.replace("p.dat", "")
                    seg_hea = study_path / f"{seg_id}.hea"
                    if seg_hea.exists():
                        with open(seg_hea) as f:
                            for line in f:
                                if line.startswith(seg_id) and "Pleth" in line:
                                    parts = line.split()
                                    # Format: file samples fs ... Pleth
                                    try:
                                        samples = int(parts[1].split("x")[0])
                                        fs_parts = parts[2].split("/")
                                        fs = float(fs_parts[0])
                                        total_duration += samples / fs
                                    except (IndexError, ValueError):
                                        pass

                records.append({
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "study_id": int(study_dir),
                    "path": str(study_path),
                    "n_ppg_segments": len(ppg_dat_files),
                    "total_duration_s": total_duration,
                })

    return records


def load_mimic_ppg(study_path: str, max_duration_s: float = 3600) -> tuple:
    """Load and concatenate PPG segments from a MIMIC study.

    Returns (ppg_signal, fs) concatenated up to max_duration_s.
    """
    import wfdb

    ppg_dat_files = sorted([f for f in os.listdir(study_path) if f.endswith("p.dat")])
    all_ppg = []
    fs = None

    for ppg_dat in ppg_dat_files:
        seg_id = ppg_dat.replace("p.dat", "")
        try:
            record = wfdb.rdrecord(os.path.join(study_path, seg_id))
            if "Pleth" not in record.sig_name:
                continue
            ppg_idx = record.sig_name.index("Pleth")
            ppg = record.p_signal[:, ppg_idx]
            fs = record.fs
            all_ppg.append(ppg)

            total_samples = sum(len(p) for p in all_ppg)
            if total_samples / fs >= max_duration_s:
                break
        except Exception as e:
            logger.debug(f"Failed to load {seg_id}: {e}")
            continue

    if not all_ppg or fs is None:
        return None, None

    ppg_concat = np.concatenate(all_ppg)
    max_samples = int(max_duration_s * fs)
    ppg_concat = ppg_concat[:max_samples]

    return ppg_concat, fs


def transform_finger_to_wrist(ppg_finger: np.ndarray, fs_finger: float,
                               rng: np.random.Generator,
                               config: WristTransformConfig = None) -> dict:
    """Transform a finger PPG segment to wrist-like PPG.

    Returns dict with:
      - ppg_wrist: transformed PPG at 25 Hz
      - fs: target sampling rate (25)
      - transform_params: dict of applied parameters
      - quality_score: 0-1 quality metric
    """
    if config is None:
        config = WristTransformConfig()

    # Step 1: Clean the finger PPG
    # Bandpass filter to isolate pulsatile component
    if fs_finger > 2:
        b, a = butter(2, [0.3, min(10, fs_finger/2 - 0.5)], btype='bandpass', fs=fs_finger)
        ppg_clean = filtfilt(b, a, ppg_finger)
    else:
        ppg_clean = ppg_finger - np.mean(ppg_finger)

    # Normalize to zero-mean, unit-variance pulsatile
    ppg_clean = (ppg_clean - np.mean(ppg_clean)) / (np.std(ppg_clean) + 1e-8)

    # Step 2: Resample to 25 Hz
    from math import gcd
    g = gcd(int(fs_finger * 100), int(TARGET_FS * 100))
    up = int(TARGET_FS * 100) // g
    down = int(fs_finger * 100) // g
    ppg_resampled = resample_poly(ppg_clean, up, down)

    # Step 3: Reduce amplitude (perfusion reduction)
    # Finger AC/DC ~0.24, wrist AC/DC ~0.01-0.05
    perfusion_scale = rng.uniform(*config.perfusion_scale_range)
    ppg_ac = ppg_resampled * perfusion_scale

    # Step 4: Add DC baseline (skin tissue absorption)
    dc_baseline = 0.5 + rng.uniform(-0.1, 0.1)  # DC component
    ppg_with_dc = ppg_ac + dc_baseline

    # Step 5: Add ambient light contamination
    ambient_amp = rng.uniform(*config.ambient_light_range)
    ambient_freq = rng.uniform(*config.ambient_freq_range)
    t = np.arange(len(ppg_with_dc)) / TARGET_FS
    ambient = ambient_amp * np.sin(2 * np.pi * ambient_freq * t + rng.uniform(0, 2*np.pi))
    ppg_ambient = ppg_with_dc + ambient

    # Step 6: Add baseline wander (respiration, movement)
    bw_amp = config.baseline_wander_amplitude
    bw_freq = rng.uniform(*config.baseline_wander_freq_range)
    baseline_wander = bw_amp * np.sin(2 * np.pi * bw_freq * t + rng.uniform(0, 2*np.pi))
    ppg_bw = ppg_ambient + baseline_wander

    # Step 7: Add temperature drift
    temp_amp = config.temp_drift_amplitude
    temp_freq = rng.uniform(*config.temp_drift_freq_range)
    temp_drift = temp_amp * np.sin(2 * np.pi * temp_freq * t + rng.uniform(0, 2*np.pi))
    ppg_temp = ppg_bw + temp_drift

    # Step 8: Add motion artifacts (probabilistic)
    has_motion = rng.random() < config.motion_prob
    if has_motion:
        motion_amp = rng.uniform(*config.motion_amplitude_range)
        motion_freq = rng.uniform(*config.motion_freq_range)
        # Multiple motion harmonics
        motion = np.zeros_like(t)
        n_harmonics = rng.integers(1, 4)
        for _ in range(n_harmonics):
            f = motion_freq * rng.uniform(0.5, 2.0)
            motion += motion_amp * rng.uniform(0.3, 1.0) * np.sin(
                2 * np.pi * f * t + rng.uniform(0, 2*np.pi))
        ppg_motion = ppg_temp + motion
    else:
        ppg_motion = ppg_temp

    # Step 9: Add sensor noise
    snr_db = rng.uniform(*config.noise_snr_range)
    signal_power = np.var(ppg_motion - np.mean(ppg_motion))
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(ppg_motion))
    ppg_noisy = ppg_motion + noise

    # Step 10: Apply contact quality variation
    has_contact_loss = rng.random() < config.contact_loss_prob
    quality_score = 1.0
    if has_contact_loss:
        loss_duration = rng.uniform(*config.contact_loss_duration_range)
        loss_samples = int(loss_duration * TARGET_FS)
        loss_start = rng.integers(0, max(1, len(ppg_noisy) - loss_samples))
        loss_end = min(loss_start + loss_samples, len(ppg_noisy))
        gain = rng.uniform(*config.contact_gain_range)
        ppg_noisy[loss_start:loss_end] *= gain
        quality_score = gain
    else:
        # Even without full loss, add slight contact variation
        contact_gain = rng.uniform(0.85, 1.0)
        ppg_noisy *= contact_gain
        quality_score = contact_gain

    # Final normalization to [0, 1] range (mimicking smartwatch output)
    ppg_min, ppg_max = ppg_noisy.min(), ppg_noisy.max()
    if ppg_max - ppg_min > 1e-8:
        ppg_wrist = (ppg_noisy - ppg_min) / (ppg_max - ppg_min)
    else:
        ppg_wrist = np.full_like(ppg_noisy, 0.5)

    transform_params = {
        "perfusion_scale": float(perfusion_scale),
        "ambient_amp": float(ambient_amp),
        "ambient_freq": float(ambient_freq),
        "has_motion": bool(has_motion),
        "snr_db": float(snr_db),
        "has_contact_loss": bool(has_contact_loss),
        "quality_score": float(quality_score),
        "fs_original": float(fs_finger),
        "fs_target": TARGET_FS,
    }

    return {
        "ppg_wrist": ppg_wrist.astype(np.float32),
        "fs": TARGET_FS,
        "transform_params": transform_params,
        "quality_score": quality_score,
    }


def extract_wrist_features(ppg: np.ndarray, fs: int = TARGET_FS) -> dict:
    """Extract signal quality features from wrist PPG for validation."""
    from scipy.signal import find_peaks

    features = {}

    # Basic stats
    features["mean"] = float(np.mean(ppg))
    features["std"] = float(np.std(ppg))
    features["range"] = float(ppg.max() - ppg.min())
    features["skewness"] = float(np.mean(((ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8))**3))

    # AC component (pulsatile)
    b, a = butter(2, [0.5, 8], btype='bandpass', fs=fs)
    ppg_ac = filtfilt(b, a, ppg - np.mean(ppg))
    features["ac_amplitude"] = float(np.std(ppg_ac))
    features["ac_pp"] = float(np.max(ppg_ac) - np.min(ppg_ac))
    features["ac_dc_ratio"] = float(np.std(ppg_ac) / (np.mean(ppg) + 1e-8))

    # Peak detection
    peaks, props = find_peaks(ppg_ac, distance=int(fs * 0.4), prominence=0.001)
    if len(peaks) > 2:
        ibi = np.diff(peaks) / fs
        features["heart_rate_est"] = float(60.0 / np.mean(ibi))
        features["hrv_sdnn"] = float(np.std(ibi) * 1000)  # ms
        features["n_peaks"] = len(peaks)
    else:
        features["heart_rate_est"] = 0.0
        features["hrv_sdnn"] = 0.0
        features["n_peaks"] = 0

    # Spectral features
    freqs, psd = welch(ppg, fs=fs, nperseg=min(256, len(ppg)))
    total_power = np.sum(psd) + 1e-12
    pulse_mask = (freqs >= 0.5) & (freqs <= 3.0)
    features["spectral_snr"] = float(np.sum(psd[pulse_mask]) / total_power)

    # Noise floor estimation
    noise_mask = (freqs >= 10) & (freqs <= fs/2 - 1)
    if noise_mask.any():
        features["noise_floor_db"] = float(10 * np.log10(np.mean(psd[noise_mask]) + 1e-12))
    else:
        features["noise_floor_db"] = -60.0

    # Flatness (noise vs signal)
    eps = 1e-12
    log_psd = np.log(psd + eps)
    geo_mean = np.exp(np.mean(log_psd))
    ari_mean = np.mean(psd)
    features["spectral_flatness"] = float(geo_mean / (ari_mean + eps))

    return features


def process_all(output_dir: Path = OUTPUT_DIR, max_studies: int = None):
    """Process all MIMIC PPG segments → wrist-like PPG."""

    output_dir.mkdir(parents=True, exist_ok=True)
    ppg_dir = output_dir / "ppg"
    ppg_dir.mkdir(exist_ok=True)

    logger.info("Extracting MIMIC PPG records...")
    records = extract_mimic_ppg_segments()
    logger.info(f"Found {len(records)} studies with PPG")

    if max_studies:
        records = records[:max_studies]

    rng = np.random.default_rng(42)
    metadata = []
    total_segments = 0

    for i, rec in enumerate(records):
        logger.info(f"  [{i+1}/{len(records)}] Subject {rec['subject_id']} "
                     f"Study {rec['study_id']} ({rec['total_duration_s']/3600:.1f}h)")

        ppg_raw, fs = load_mimic_ppg(rec["path"], max_duration_s=3600)
        if ppg_raw is None or fs is None:
            logger.debug(f"    No PPG loaded, skipping")
            continue

        logger.info(f"    Loaded {len(ppg_raw)/fs:.1f}s at {fs:.1f} Hz")

        # Segment into 60-second windows with 50% overlap
        seg_samples = int(60 * fs)
        hop = seg_samples // 2
        n_segs = max(1, (len(ppg_raw) - seg_samples) // hop + 1)

        for seg_i in range(n_segs):
            start = seg_i * hop
            end = start + seg_samples
            if end > len(ppg_raw):
                break
            segment = ppg_raw[start:end]

            # Skip segments with too many NaN or constant values
            if np.any(np.isnan(segment)) or np.std(segment) < 1e-6:
                continue

            # Transform to wrist
            result = transform_finger_to_wrist(segment, fs, rng)
            ppg_wrist = result["ppg_wrist"]

            # Validate: check it looks reasonable
            if np.std(ppg_wrist) < 0.01 or np.any(np.isnan(ppg_wrist)):
                continue

            # Save
            seg_id = f"{rec['subject_id']}_{rec['study_id']}_{seg_i:04d}"
            np.save(ppg_dir / f"{seg_id}.npy", ppg_wrist)

            # Extract features for validation
            features = extract_wrist_features(ppg_wrist)

            metadata.append({
                "segment_id": seg_id,
                "subject_id": rec["subject_id"],
                "hadm_id": rec["hadm_id"],
                "study_id": rec["study_id"],
                "segment_index": seg_i,
                "duration_s": 60.0,
                "fs": TARGET_FS,
                "n_samples": len(ppg_wrist),
                "quality_score": result["quality_score"],
                **result["transform_params"],
                **features,
            })
            total_segments += 1

        if (i + 1) % 10 == 0:
            logger.info(f"    Processed {total_segments} segments so far")

    # Save metadata
    df = pd.DataFrame(metadata)
    df.to_csv(output_dir / "metadata.csv", index=False)

    # Save summary
    summary = {
        "total_studies": len(records),
        "total_segments": total_segments,
        "target_fs": TARGET_FS,
        "segment_duration_s": 60,
        "output_dir": str(output_dir),
        "config": asdict(WristTransformConfig()),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"DONE: {total_segments} wrist-like PPG segments saved")
    logger.info(f"Output: {output_dir}")
    logger.info(f"  PPG files: {ppg_dir}/")
    logger.info(f"  Metadata:  {output_dir}/metadata.csv")
    logger.info(f"  Summary:   {output_dir}/summary.json")
    logger.info(f"{'='*60}")

    return df


if __name__ == "__main__":
    process_all()
