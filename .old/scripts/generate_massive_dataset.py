#!/usr/bin/env python3
"""
Generate massive synthetic PPG dataset using wristppg simulator.
Creates 10,000+ segments across all physiological profiles with realistic variation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import find_peaks, butter, filtfilt
import json
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from wristppg import WristPPGSimulator
from wristppg.disease import PROFILES


# ── Feature extraction (single segment) ──────────────────────────────────────

def bandpass_filter(sig, fs=25, low=0.5, high=4.0, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    try:
        return filtfilt(b, a, sig)
    except Exception:
        return sig


def extract_features(seg, fs=25):
    """Extract features from a single PPG segment."""
    feats = {}
    feats["mean"] = float(np.mean(seg))
    feats["std"] = float(np.std(seg))
    feats["skewness"] = float(pd.Series(seg).skew())
    feats["kurtosis"] = float(pd.Series(seg).kurtosis())
    feats["range"] = float(np.ptp(seg))
    feats["iqr"] = float(np.percentile(seg, 75) - np.percentile(seg, 25))
    feats["energy"] = float(np.sum(seg**2))
    feats["zero_crossings"] = int(np.sum(np.diff(np.sign(seg)) != 0))

    # Peak detection
    try:
        filt = bandpass_filter(seg, fs)
    except Exception:
        filt = seg
    
    threshold = np.mean(filt) + 0.1 * np.std(filt)
    peaks, props = find_peaks(filt, height=threshold, distance=8, prominence=0.01)
    feats["n_peaks"] = len(peaks)
    
    if len(peaks) >= 3:
        intervals = np.diff(peaks) / fs * 1000  # ms
        feats["heart_rate_bpm"] = float(60000 / np.mean(intervals))
        diffs = np.diff(intervals)
        feats["rmssd"] = float(np.sqrt(np.mean(diffs**2)))
        feats["sdnn"] = float(np.std(intervals, ddof=1))
        feats["pnn50"] = float(np.sum(np.abs(diffs) > 50) / len(diffs) * 100)
        feats["mean_nn"] = float(np.mean(intervals))
        feats["sdnn_ratio"] = feats["sdnn"] / (feats["mean_nn"] + 1e-6)
        feats["hr_std"] = float(np.std(np.diff(peaks) / fs * 60))
        
        # Peak morphology
        peak_heights = filt[peaks]
        feats["peak_mean"] = float(np.mean(peak_heights))
        feats["peak_std"] = float(np.std(peak_heights))
        feats["peak_min"] = float(np.min(peak_heights))
        feats["peak_max"] = float(np.max(peak_heights))
        feats["peak_cv"] = float(np.std(peak_heights) / (np.mean(peak_heights) + 1e-6))
        
        # Trough-to-peak ratios
        troughs, _ = find_peaks(-filt, distance=8, prominence=0.01)
        if len(troughs) >= 2 and len(peaks) >= 2:
            feats["pulse_width_mean"] = float(np.mean(np.abs(peaks[:len(troughs)] - troughs[:len(peaks)])) / fs)
        else:
            feats["pulse_width_mean"] = 0.0
    else:
        for k in ["heart_rate_bpm", "rmssd", "sdnn", "pnn50", "mean_nn",
                   "sdnn_ratio", "hr_std", "peak_mean", "peak_std", "peak_min",
                   "peak_max", "peak_cv", "pulse_width_mean"]:
            feats[k] = 0.0

    # Frequency domain
    if len(peaks) >= 5:
        try:
            rr_intervals = np.diff(peaks) / fs
            if len(rr_intervals) > 10:
                from scipy.interpolate import interp1d
                t_rr = np.cumsum(rr_intervals)
                t_interp = np.arange(t_rr[0], t_rr[-1], 1.0/fs)
                f_interp = interp1d(t_rr, rr_intervals, kind='linear', fill_value='extrapolate')
                rr_interp = f_interp(t_interp)
                rr_interp = rr_interp - np.mean(rr_interp)
                
                freqs = np.fft.rfftfreq(len(rr_interp), d=1.0/fs)
                fft_power = np.abs(np.fft.rfft(rr_interp))**2
                
                vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
                lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                hf_mask = (freqs >= 0.15) & (freqs < 0.4)
                
                feats["vlf_power"] = float(np.sum(fft_power[vlf_mask])) if np.any(vlf_mask) else 0.0
                feats["lf_power"] = float(np.sum(fft_power[lf_mask])) if np.any(lf_mask) else 0.0
                feats["hf_power"] = float(np.sum(fft_power[hf_mask])) if np.any(hf_mask) else 0.0
                total_power = feats["vlf_power"] + feats["lf_power"] + feats["hf_power"]
                feats["lf_hf_ratio"] = feats["lf_power"] / (feats["hf_power"] + 1e-6)
                feats["lf_nu"] = feats["lf_power"] / (feats["lf_power"] + feats["hf_power"] + 1e-6)
                feats["hf_nu"] = feats["hf_power"] / (feats["lf_power"] + feats["hf_power"] + 1e-6)
                feats["total_power"] = total_power
            else:
                for k in ["vlf_power", "lf_power", "hf_power", "lf_hf_ratio", "lf_nu", "hf_nu", "total_power"]:
                    feats[k] = 0.0
        except Exception:
            for k in ["vlf_power", "lf_power", "hf_power", "lf_hf_ratio", "lf_nu", "hf_nu", "total_power"]:
                feats[k] = 0.0
    else:
        for k in ["vlf_power", "lf_power", "hf_power", "lf_hf_ratio", "lf_nu", "hf_nu", "total_power"]:
            feats[k] = 0.0

    # Signal quality
    snr = np.var(filt) / (np.var(seg - filt) + 1e-10)
    feats["snr_db"] = float(10 * np.log10(snr))
    feats["signal_quality"] = float(min(1.0, snr / 100))
    
    # Nonlinear dynamics
    diffs_sig = np.diff(seg)
    feats["d1_mean"] = float(np.mean(np.abs(diffs_sig)))
    feats["d1_std"] = float(np.std(diffs_sig))
    feats["d2_mean"] = float(np.mean(np.abs(np.diff(diffs_sig)))) if len(diffs_sig) > 1 else 0.0
    
    return feats


# ── Profile configurations for generation ─────────────────────────────────────

# Map disease profiles to 3-class labels:
# 0 = healthy, 1 = general_deterioration, 2 = cardiac_arrest
PROFILE_LABELS = {
    # Cardiac arrest continuum (class 2)
    "cardiac_arrest_vf": 2,
    "cardiac_arrest_asystole": 2,
    "cardiac_arrest_pulseless_electrical": 2,
    "pre_arrest_deterioration": 2,
    "post_resuscitation": 2,
    "respiratory_failure_pre_arrest": 2,
    "electrocution_arrest": 2,
    "drowning_arrest": 2,
    "shock": 2,
    
    # General deterioration (class 1)
    "hypovolemia": 1,
    "sepsis_warm": 1,
    "hfref": 1,
    "hfpef": 1,
    "respiratory_failure_pre_arrest": 1,  # Will be overridden above
    
    # Healthy / chronic conditions (class 0)
    "healthy": 0,
    "aging": 0,
    "hypertension": 0,
    "diabetes": 0,
    "afib_isolated": 0,
    "arterial_stiffness_isolated": 0,
    "pad": 0,
    "exercise": 0,
    "recovery": 0,
    "sleep": 0,
}

# How many segments per profile
SEGMENTS_PER_PROFILE = {
    # Cardiac arrest - generate MORE
    "cardiac_arrest_vf": 800,
    "cardiac_arrest_asystole": 800,
    "cardiac_arrest_pulseless_electrical": 800,
    "pre_arrest_deterioration": 600,
    "post_resuscitation": 400,
    "respiratory_failure_pre_arrest": 500,
    "electrocution_arrest": 300,
    "drowning_arrest": 300,
    "shock": 600,
    
    # General deterioration
    "hypovolemia": 400,
    "sepsis_warm": 400,
    "hfref": 300,
    "hfpef": 300,
    
    # Healthy / chronic
    "healthy": 600,
    "aging": 200,
    "hypertension": 200,
    "diabetes": 200,
    "afib_isolated": 200,
    "arterial_stiffness_isolated": 150,
    "pad": 150,
    "exercise": 200,
    "recovery": 200,
    "sleep": 200,
}


def generate_one_segment(args):
    """Generate a single PPG segment with features. Runs in a worker process."""
    profile_name, segment_idx, base_seed, duration_s = args
    
    try:
        sim = WristPPGSimulator(seed=base_seed + segment_idx)
        
        # Vary parameters for realism
        rng = np.random.RandomState(base_seed + segment_idx)
        activity = rng.choice(["rest", "rest", "rest", "sleep"], p=[0.7, 0.1, 0.1, 0.1])
        contact = rng.choice(["good", "good", "loose", "partial_lift"], p=[0.6, 0.15, 0.15, 0.1])
        
        # Vary duration slightly
        dur = duration_s + rng.uniform(-5, 5)
        dur = max(30, min(90, dur))
        
        result = sim.generate(
            profile=profile_name,
            duration_s=dur,
            activity=activity,
            contact_mode=contact,
            wavelength="green",
            fs_output_hz=25.0,
        )
        
        ppg = result.ppg
        if ppg is None or len(ppg) < 250:
            return None
        
        # Normalize PPG
        ppg = ppg - np.mean(ppg)
        ppg_std = np.std(ppg)
        if ppg_std > 1e-8:
            ppg = ppg / ppg_std
        
        # Pad/truncate to fixed length (1500 samples = 60s at 25Hz)
        target_len = 1500
        if len(ppg) < target_len:
            ppg = np.pad(ppg, (0, target_len - len(ppg)), mode='edge')
        elif len(ppg) > target_len:
            ppg = ppg[:target_len]
        
        # Extract features
        features = extract_features(ppg, fs=25)
        
        # Get latent physiology
        latent = result.latent_physiology
        features["latent_hr_bpm"] = latent.get("hr_bpm", 0)
        features["latent_spo2"] = latent.get("spo2", 0)
        features["latent_body_temp"] = latent.get("body_temp_c", 37)
        features["latent_perfusion_index"] = latent.get("perfusion_index", 0.1)
        
        return {
            "ppg": ppg,
            "features": features,
            "profile": profile_name,
            "label": PROFILE_LABELS.get(profile_name, 0),
            "seed": base_seed + segment_idx,
            "latent_hr": latent.get("hr_bpm", 0),
            "latent_spo2": latent.get("spo2", 0),
        }
    except Exception as e:
        return None


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "processed" / "synthetic_v2"
    ppg_dir = output_dir / "ppg_segments"
    output_dir.mkdir(parents=True, exist_ok=True)
    ppg_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MASSIVE SYNTHETIC PPG DATASET GENERATION")
    print("=" * 70)
    
    # Build task list
    tasks = []
    base_seed = 42
    for profile_name, n_segments in SEGMENTS_PER_PROFILE.items():
        for i in range(n_segments):
            tasks.append((profile_name, i, base_seed, 60.0))
        base_seed += 100000
    
    total = len(tasks)
    print(f"Total segments to generate: {total}")
    print(f"Profiles: {len(SEGMENTS_PER_PROFILE)}")
    print()
    
    for profile, count in SEGMENTS_PER_PROFILE.items():
        label_name = ["healthy", "general_det", "cardiac_arrest"][PROFILE_LABELS[profile]]
        print(f"  {profile:40s}: {count:5d} segments -> {label_name}")
    print()
    
    # Generate in parallel
    results = []
    n_workers = 8
    start_time = time.time()
    
    print(f"Generating with {n_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(generate_one_segment, task): task for task in tasks}
        completed = 0
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
            completed += 1
            
            if completed % 500 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  [{completed:5d}/{total}] {rate:.0f} seg/s, "
                      f"{len(results)} successful, {elapsed:.1f}s elapsed")
    
    elapsed = time.time() - start_time
    print(f"\nGeneration complete: {len(results)}/{total} successful in {elapsed:.1f}s")
    
    # Save PPG segments
    print("\nSaving PPG segments...")
    ppg_arrays = []
    labels = []
    profiles = []
    metadata = []
    
    for i, r in enumerate(results):
        # Ensure PPG is fixed length
        ppg = r["ppg"]
        target_len = 1500
        if len(ppg) < target_len:
            ppg = np.pad(ppg, (0, target_len - len(ppg)), mode='edge')
        elif len(ppg) > target_len:
            ppg = ppg[:target_len]
        r["ppg"] = ppg
        
        # Save PPG segment
        seg_path = ppg_dir / f"synth_{i:05d}.npy"
        np.save(seg_path, ppg.astype(np.float32))
        
        ppg_arrays.append(r["ppg"])
        labels.append(r["label"])
        profiles.append(r["profile"])
        metadata.append({
            "segment_id": f"synth_{i:05d}",
            "profile": r["profile"],
            "label": r["label"],
            "seed": r["seed"],
            "latent_hr": r["latent_hr"],
            "latent_spo2": r["latent_spo2"],
            "ppg_length": len(r["ppg"]),
        })
    
    # Save features CSV
    features_df = pd.DataFrame([r["features"] for r in results])
    features_df["segment_id"] = [f"synth_{i:05d}" for i in range(len(results))]
    features_df["profile"] = profiles
    features_df["label"] = labels
    
    features_csv = output_dir / "features.csv"
    features_df.to_csv(features_csv, index=False)
    print(f"  Saved features: {features_csv} ({len(features_df)} rows, {len(features_df.columns)} cols)")
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_csv = output_dir / "metadata.csv"
    metadata_df.to_csv(metadata_csv, index=False)
    print(f"  Saved metadata: {metadata_csv}")
    
    # Save PPG as single large array (for faster loading)
    ppg_array = np.stack(ppg_arrays)
    ppg_npy = output_dir / "ppg_segments.npy"
    np.save(ppg_npy, ppg_array)
    print(f"  Saved PPG array: {ppg_npy} ({ppg_array.shape})")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    label_counts = pd.Series(labels).value_counts().sort_index()
    label_names = {0: "Healthy", 1: "General Deterioration", 2: "Cardiac Arrest"}
    for label, count in label_counts.items():
        print(f"  {label_names[label]:30s}: {count:5d} segments")
    print(f"  {'TOTAL':30s}: {len(labels):5d} segments")
    
    profile_counts = pd.Series(profiles).value_counts()
    print("\nBy profile:")
    for profile, count in profile_counts.items():
        label_name = label_names[PROFILE_LABELS[profile]]
        print(f"  {profile:40s}: {count:5d} ({label_name})")
    
    # Save config
    config = {
        "total_segments": len(results),
        "n_profiles": len(SEGMENTS_PER_PROFILE),
        "segments_per_profile": SEGMENTS_PER_PROFILE,
        "profile_labels": {k: v for k, v in PROFILE_LABELS.items()},
        "label_names": label_names,
        "feature_columns": [c for c in features_df.columns if c not in ["segment_id", "profile", "label"]],
        "n_features": len([c for c in features_df.columns if c not in ["segment_id", "profile", "label"]]),
        "ppg_length": int(ppg_array.shape[1]),
        "fs_hz": 25.0,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {output_dir / 'config.json'}")
    
    print(f"\nDone! Dataset ready at {output_dir}")


if __name__ == "__main__":
    main()
