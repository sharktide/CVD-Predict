"""
Cardiac Arrest Data Augmentation Pipeline

Augments the 7 cardiac arrest patients' PPG data to create balanced training:
1. Signal-level augmentation (time warping, scaling, noise, baseline wander)
2. Synthetic PPG generation using wristppg simulator with CA-like parameters
3. Feature-level augmentation (SMOTE-inspired interpolation)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, resample
from scipy.interpolate import interp1d

# ── Signal-Level Augmentation ─────────────────────────────────────────────────

def augment_time_warp(signal, max_warp=0.1):
    """Slightly stretch/compress time axis to simulate HR variability."""
    length = len(signal)
    warp_factor = 1.0 + np.random.uniform(-max_warp, max_warp)
    new_length = int(length * warp_factor)
    warped = resample(signal, new_length)
    if len(warped) > length:
        warped = warped[:length]
    else:
        warped = np.pad(warped, (0, length - len(warped)), mode='edge')
    return warped


def augment_amplitude_scale(signal, scale_range=(0.7, 1.3)):
    """Scale PPG amplitude to simulate different perfusion levels."""
    scale = np.random.uniform(*scale_range)
    return signal * scale


def augment_gaussian_noise(signal, snr_db_range=(15, 30)):
    """Add Gaussian noise with controlled SNR."""
    snr_db = np.random.uniform(*snr_db_range)
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def augment_baseline_wander(signal, fs=25, freq_range=(0.05, 0.3), amp_range=(0.02, 0.08)):
    """Add low-frequency baseline wander to simulate motion artifacts."""
    duration = len(signal) / fs
    freq = np.random.uniform(*freq_range)
    amp = np.random.uniform(*amp_range)
    t = np.arange(len(signal)) / fs
    wander = amp * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))
    return signal + wander


def augment_peak_suppression(signal, fs=25, suppress_prob=0.1):
    """Randomly suppress some peaks to simulate irregular rhythms."""
    from scipy.signal import find_peaks
    filtered = signal - np.convolve(signal, np.ones(25)/25, mode='same')
    threshold = np.mean(filtered) + 0.1 * np.std(filtered)
    peaks, _ = find_peaks(filtered, height=threshold, distance=8, prominence=0.01)
    
    if len(peaks) == 0:
        return signal
    
    result = signal.copy()
    for peak in peaks:
        if np.random.random() < suppress_prob:
            # Smooth around the peak
            start = max(0, peak - 5)
            end = min(len(signal), peak + 5)
            result[start:end] = np.mean(signal[start:end])
    
    return result


def augment_segment_swap(signal, n_swaps=3, swap_len=50):
    """Randomly swap small segments to simulate rhythm irregularity."""
    result = signal.copy()
    for _ in range(n_swaps):
        pos1 = np.random.randint(0, len(signal) - swap_len)
        pos2 = np.random.randint(0, len(signal) - swap_len)
        chunk = result[pos1:pos1 + swap_len].copy()
        result[pos1:pos1 + swap_len] = result[pos2:pos2 + swap_len]
        result[pos2:pos2 + swap_len] = chunk
    return result


def augment_phase_shift(signal, max_shift=50):
    """Shift the signal phase to simulate different sensor positions."""
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(signal, shift)


def augment_all(signal, fs=25, augmentations_per_sample=3):
    """Apply random combination of augmentations."""
    augmented = []
    
    for _ in range(augmentations_per_sample):
        result = signal.copy()
        
        # Always apply amplitude scaling
        result = augment_amplitude_scale(result)
        
        # Randomly apply other augmentations
        if np.random.random() < 0.7:
            result = augment_gaussian_noise(result)
        if np.random.random() < 0.5:
            result = augment_baseline_wander(result, fs)
        if np.random.random() < 0.3:
            result = augment_time_warp(result)
        if np.random.random() < 0.2:
            result = augment_peak_suppression(result, fs)
        if np.random.random() < 0.2:
            result = augment_phase_shift(result)
        
        augmented.append(result)
    
    return augmented


# ── Synthetic PPG Generation ─────────────────────────────────────────────────

def generate_synthetic_ca_ppg(duration_s=60, fs=25, n_samples=1):
    """
    Generate synthetic cardiac arrest-like PPG using the wristppg simulator.
    
    Based on real CA patient statistics:
    - HR: 86.6 ± 8.9 bpm
    - RMSSD: 174.6 ± 67.1 ms
    - SDNN: 128.6 ± 49.8 ms
    - PPG std: 0.02 (lower perfusion)
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from wristppg import WristPPGSimulator
        
        simulator = WristPPGSimulator(seed=None)
        synthetic_segments = []
        
        for _ in range(n_samples):
            # Generate with sepsis/shock profile (similar to CA physiology)
            result = simulator.generate(
                profile="sepsis",
                duration_s=duration_s,
                activity="rest",
                contact_mode="good",
            )
            
            # Resample to target fs
            ppg = result.ppg
            if len(ppg) != int(duration_s * fs):
                ppg = resample(ppg, int(duration_s * fs))
            
            # Reduce perfusion to match CA characteristics
            ppg = ppg * 0.3  # Lower amplitude
            
            # Add slight baseline wander
            t = np.arange(len(ppg)) / fs
            ppg = ppg + 0.02 * np.sin(2 * np.pi * 0.1 * t)
            
            synthetic_segments.append(ppg.astype(np.float32))
        
        return synthetic_segments
    except Exception as e:
        print(f"  Synthetic generation failed: {e}")
        import traceback
        traceback.print_exc()
        return []


# ── Feature-Level Augmentation (SMOTE-inspired) ──────────────────────────────

def augment_features_smote(features_df, target_class="cardiac_arrest", n_synthetic=100, k=5):
    """
    SMOTE-inspired feature augmentation.
    Interpolates between existing CA feature vectors to create new ones.
    """
    ca_features = features_df[features_df["primary_event"] == target_class].copy()
    if len(ca_features) < k:
        return pd.DataFrame()
    
    meta_cols = ["window_id", "subject_id", "primary_event", "is_healthy", "time_to_event_hours", "n_segments"]
    feat_cols = [c for c in ca_features.columns if c not in meta_cols]
    
    synthetic_rows = []
    for _ in range(n_synthetic):
        # Pick a random sample
        idx = np.random.randint(0, len(ca_features))
        base = ca_features.iloc[idx][feat_cols].values.astype(float)
        
        # Pick k nearest neighbors
        distances = []
        for j in range(len(ca_features)):
            if j != idx:
                other = ca_features.iloc[j][feat_cols].values.astype(float)
                diff = np.nan_to_num(base - other)
                distances.append((j, np.sqrt(np.sum(diff ** 2))))
        distances.sort(key=lambda x: x[1])
        neighbor_idx = distances[np.random.randint(0, min(k, len(distances)))][0]
        neighbor = ca_features.iloc[neighbor_idx][feat_cols].values.astype(float)
        
        # Interpolate
        lam = np.random.uniform(0, 1)
        synthetic = base * (1 - lam) + neighbor * lam
        
        # Add small noise
        noise = np.random.normal(0, 0.01, len(synthetic))
        synthetic = synthetic + noise
        
        synthetic = np.nan_to_num(synthetic, nan=0.0, posinf=0.0, neginf=0.0)
        
        row = {col: synthetic[i] for i, col in enumerate(feat_cols)}
        row["window_id"] = f"synthetic_ca_{_}"
        row["subject_id"] = -1
        row["primary_event"] = target_class
        row["is_healthy"] = False
        row["time_to_event_hours"] = np.random.uniform(0, 12)
        row["n_segments"] = 24
        synthetic_rows.append(row)
    
    return pd.DataFrame(synthetic_rows)


# ── Main Augmentation Pipeline ────────────────────────────────────────────────

def main():
    project_root = Path(__file__).parent.parent
    cohort_dir = project_root / "data" / "processed" / "cohort_v1"
    output_dir = cohort_dir / "augmented"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ppg_segments").mkdir(exist_ok=True)
    
    print("=" * 70)
    print("CARDIAC ARREST DATA AUGMENTATION")
    print("=" * 70)
    
    # Load CA statistics
    with open(cohort_dir / "ca_patient_stats.json") as f:
        ca_stats = json.load(f)
    
    print(f"\nCA Patient Statistics:")
    print(f"  HR: {ca_stats['heart_rate_bpm']['mean']:.1f} ± {ca_stats['heart_rate_bpm']['std']:.1f} bpm")
    print(f"  RMSSD: {ca_stats['rmssd']['mean']:.1f} ± {ca_stats['rmssd']['std']:.1f}")
    print(f"  SDNN: {ca_stats['sdnn']['mean']:.1f} ± {ca_stats['sdnn']['std']:.1f}")
    print(f"  PPG std: {ca_stats['raw_ppg']['mean_std']:.4f}")
    
    # ── 1. Signal-level augmentation ──────────────────────────────────────
    print(f"\n{'─'*70}")
    print("1. SIGNAL-LEVEL AUGMENTATION")
    print(f"{'─'*70}")
    
    windows = pd.read_csv(cohort_dir / "windows.csv")
    ca_windows = windows[windows["primary_event"] == "cardiac_arrest"]
    
    n_augmented_segments = 0
    augmentations_per_sample = 10  # Generate 10 augmented versions per original
    
    for _, row in ca_windows.iterrows():
        window_id = row["window_id"]
        n_segments = int(row["n_segments"])
        
        # Load first segment as template
        seg_path = cohort_dir / "ppg_segments" / f"{window_id}_s00.npy"
        if not seg_path.exists():
            continue
        
        template = np.load(seg_path)
        if np.any(np.isnan(template)):
            continue
        
        # Generate augmented versions
        augmented = augment_all(template, fs=25, augmentations_per_sample=augmentations_per_sample)
        
        for aug_idx, aug_seg in enumerate(augmented):
            aug_id = f"{window_id}_aug{aug_idx:03d}"
            seg_out_path = output_dir / "ppg_segments" / f"{aug_id}.npy"
            np.save(seg_out_path, aug_seg.astype(np.float32))
            n_augmented_segments += 1
    
    print(f"  Generated {n_augmented_segments} augmented segments from {len(ca_windows)} CA windows")
    
    # ── 2. Synthetic PPG generation ──────────────────────────────────────
    print(f"\n{'─'*70}")
    print("2. SYNTHETIC PPG GENERATION (wristppg simulator)")
    print(f"{'─'*70}")
    
    n_synthetic = 200  # Generate 200 synthetic CA-like PPG segments
    synthetic_segments = generate_synthetic_ca_ppg(
        duration_s=60, fs=25, n_samples=n_synthetic
    )
    
    for i, seg in enumerate(synthetic_segments):
        seg_path = output_dir / "ppg_segments" / f"synthetic_ca_{i:04d}.npy"
        np.save(seg_path, seg)
    
    print(f"  Generated {len(synthetic_segments)} synthetic CA-like PPG segments")
    
    # ── 3. Feature-level augmentation (SMOTE) ────────────────────────────
    print(f"\n{'─'*70}")
    print("3. FEATURE-LEVEL AUGMENTATION (SMOTE)")
    print(f"{'─'*70}")
    
    features_df = pd.read_csv(cohort_dir / "features" / "features_fixed.csv")
    n_smote = 200
    
    synthetic_features = augment_features_smote(
        features_df, target_class="cardiac_arrest",
        n_synthetic=n_smote, k=5
    )
    
    print(f"  Generated {len(synthetic_features)} synthetic feature vectors")
    
    # ── 4. Build augmented dataset ──────────────────────────────────────
    print(f"\n{'─'*70}")
    print("4. BUILDING AUGMENTED DATASET")
    print(f"{'─'*70}")
    
    # Inline feature extraction (avoid import issues)
    def extract_features_from_segment_fixed(ppg_segment, fs=25):
        from scipy.signal import find_peaks, welch
        features = {}
        features["mean"] = float(np.mean(ppg_segment))
        features["std"] = float(np.std(ppg_segment))
        features["skewness"] = float(pd.Series(ppg_segment).skew())
        features["range"] = float(np.ptp(ppg_segment))
        
        try:
            nyq = fs / 2
            b, a = butter(4, [0.5 / nyq, 4.0 / nyq], btype='band')
            ppg_filtered = filtfilt(b, a, ppg_segment)
        except:
            ppg_filtered = ppg_segment
        
        threshold = np.mean(ppg_filtered) + 0.1 * np.std(ppg_filtered)
        peaks, _ = find_peaks(ppg_filtered, height=threshold, distance=8, prominence=0.01)
        features["n_peaks"] = len(peaks)
        
        if len(peaks) < 3:
            for k in ["heart_rate_bpm", "rmssd", "sdnn", "pnn50", "mean_nn", "lf_hf_ratio", "hf_power", "lf_power", "edr_rate"]:
                features[k] = 0.0
            return features
        
        peak_intervals_ms = np.diff(peaks) / fs * 1000
        features["heart_rate_bpm"] = float(60000.0 / np.mean(peak_intervals_ms))
        successive_diffs = np.diff(peak_intervals_ms)
        features["rmssd"] = float(np.sqrt(np.mean(successive_diffs ** 2)))
        features["sdnn"] = float(np.std(peak_intervals_ms, ddof=1))
        features["pnn50"] = float(np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100)
        features["mean_nn"] = float(np.mean(peak_intervals_ms))
        features["lf_power"] = 0.0
        features["hf_power"] = 0.0
        features["lf_hf_ratio"] = 0.0
        features["edr_rate"] = 0.0
        return features
    
    # Combine all features
    augmented_rows = []
    
    # Original CA windows (all segments)
    for _, row in ca_windows.iterrows():
        window_id = row["window_id"]
        # Find corresponding features
        feat_row = features_df[features_df["window_id"] == window_id]
        if len(feat_row) > 0:
            augmented_rows.append(feat_row.iloc[0].to_dict())
    
    # Augmented signal features (extract from augmented segments)
    print("  Extracting features from augmented segments...")
    aug_seg_dir = output_dir / "ppg_segments"
    aug_features_list = []
    for seg_file in sorted(aug_seg_dir.glob("*.npy")):
        if seg_file.name.startswith("synthetic"):
            continue
        seg = np.load(seg_file)
        if not np.any(np.isnan(seg)):
            feats = extract_features_from_segment_fixed(seg, fs=25)
            feats["window_id"] = seg_file.stem
            feats["subject_id"] = -1
            feats["primary_event"] = "cardiac_arrest"
            feats["is_healthy"] = False
            feats["time_to_event_hours"] = np.random.uniform(0, 12)
            feats["n_segments"] = 1
            aug_features_list.append(feats)
    
    # Synthetic feature vectors
    for _, row in synthetic_features.iterrows():
        augmented_rows.append(row.to_dict())
    
    # Augmented signal features
    for feats in aug_features_list:
        augmented_rows.append(feats)
    
    augmented_df = pd.DataFrame(augmented_rows)
    
    # Save augmented features
    augmented_df.to_csv(output_dir / "augmented_features.csv", index=False)
    
    print(f"\n{'='*70}")
    print("AUGMENTATION SUMMARY")
    print(f"{'='*70}")
    print(f"Original CA windows: {len(ca_windows)}")
    print(f"Signal-augmented segments: {n_augmented_segments}")
    print(f"Synthetic PPG segments: {len(synthetic_segments)}")
    print(f"SMOTE feature vectors: {len(synthetic_features)}")
    print(f"Total augmented CA rows: {len(augmented_df)}")
    
    # Distribution
    print(f"\nEvent distribution in augmented dataset:")
    all_features = pd.concat([features_df, augmented_df], ignore_index=True)
    print(all_features["primary_event"].value_counts())
    
    # Save combined dataset
    all_features.to_csv(output_dir / "full_augmented_features.csv", index=False)
    print(f"\n✓ Full augmented dataset saved to: {output_dir / 'full_augmented_features.csv'}")


if __name__ == "__main__":
    main()
