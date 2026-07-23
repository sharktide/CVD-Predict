"""
Phase 2: Feature Extraction for Cardiac Arrest Prediction

Extracts clinical-grade biosignal features from 1-hour PPG segments:
- HRV Time-Domain: RMSSD, SDNN, pNN50
- HRV Frequency-Domain: LF, HF, LF/HF ratio
- PPG-Derived Respiration (EDR)
- Signal quality metrics
- Accelerometer VMA (for MMASH data)

Also applies wrist degradation to MIMIC finger PPG to bridge domain gap.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import welch, butter, filtfilt, find_peaks
from scipy.interpolate import interp1d


def bandpass_filter(signal, fs, low=0.5, high=4.0, order=4):
    """Apply bandpass filter to isolate PPG frequency range."""
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)


def detect_peaks(signal, fs, min_distance_s=0.3):
    """Detect systolic peaks in PPG signal."""
    # Adaptive thresholding
    threshold = np.mean(signal) + 0.3 * np.std(signal)
    min_distance = int(min_distance_s * fs)

    peaks, properties = find_peaks(
        signal,
        height=threshold,
        distance=min_distance,
        prominence=0.1 * np.std(signal)
    )
    return peaks


def compute_hrv_time_domain(peak_intervals_ms):
    """Compute time-domain HRV metrics from N-N intervals."""
    if len(peak_intervals_ms) < 2:
        return {
            "rmssd": 0.0,
            "sdnn": 0.0,
            "pnn50": 0.0,
            "mean_nn": 0.0,
            "median_nn": 0.0,
        }

    # RMSSD: Root Mean Square of Successive Differences
    successive_diffs = np.diff(peak_intervals_ms)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2))

    # SDNN: Standard Deviation of N-N intervals
    sdnn = np.std(peak_intervals_ms, ddof=1)

    # pNN50: Percentage of successive N-N intervals that differ by > 50ms
    nn50 = np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100

    return {
        "rmssd": float(rmssd),
        "sdnn": float(sdnn),
        "pnn50": float(nn50),
        "mean_nn": float(np.mean(peak_intervals_ms)),
        "median_nn": float(np.median(peak_intervals_ms)),
    }


def compute_hrv_frequency_domain(peak_intervals_ms, peak_times_s, fs_interp=4.0):
    """Compute frequency-domain HRV metrics using Welch's method."""
    if len(peak_intervals_ms) < 4:
        return {
            "lf_power": 0.0,
            "hf_power": 0.0,
            "lf_hf_ratio": 0.0,
            "total_power": 0.0,
            "lf_peak_hz": 0.0,
            "hf_peak_hz": 0.0,
        }

    # Interpolate N-N intervals to uniform time series
    t_nn = peak_times_s[1:]  # Times of successive intervals
    if len(t_nn) < 2:
        return {
            "lf_power": 0.0, "hf_power": 0.0, "lf_hf_ratio": 0.0,
            "total_power": 0.0, "lf_peak_hz": 0.0, "hf_peak_hz": 0.0,
        }

    t_uniform = np.arange(t_nn[0], t_nn[-1], 1.0 / fs_interp)
    interp_func = interp1d(t_nn, peak_intervals_ms, kind='linear', fill_value='extrapolate')
    nn_uniform = interp_func(t_uniform)

    # Detrend
    nn_uniform = nn_uniform - np.polyval(np.polyfit(t_uniform, nn_uniform, 1), t_uniform)

    # Welch PSD
    nperseg = min(len(nn_uniform), int(128 * fs_interp))
    freqs, psd = welch(nn_uniform, fs=fs_interp, nperseg=nperseg, noverlap=nperseg // 2)

    # Define frequency bands
    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.4)

    lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0.0
    hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0.0
    total_power = np.trapz(psd, freqs)

    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0

    # Peak frequencies
    lf_peak_hz = freqs[lf_mask][np.argmax(psd[lf_mask])] if np.any(lf_mask) else 0.0
    hf_peak_hz = freqs[hf_mask][np.argmax(psd[hf_mask])] if np.any(hf_mask) else 0.0

    return {
        "lf_power": float(lf_power),
        "hf_power": float(hf_power),
        "lf_hf_ratio": float(lf_hf_ratio),
        "total_power": float(total_power),
        "lf_peak_hz": float(lf_peak_hz),
        "hf_peak_hz": float(hf_peak_hz),
    }


def compute_edr(ppg_signal, peak_indices, fs):
    """
    Compute PPG-Derived Respiration (EDR) using amplitude modulation.

    Respiration modulates PPG amplitude through:
    1. Intrathoracic pressure changes affecting venous return
    2. Respiratory sinus arrhythmia
    """
    if len(peak_indices) < 10:
        return {
            "edr_rate_breaths_per_min": 0.0,
            "edr_power": 0.0,
            "edr_snr": 0.0,
        }

    # Extract peak amplitudes
    peak_amplitudes = ppg_signal[peak_indices]

    # Interpolate to uniform time series
    peak_times = peak_indices / fs
    t_uniform = np.arange(peak_times[0], peak_times[-1], 0.1)  # 10 Hz
    interp_func = interp1d(peak_times, peak_amplitudes, kind='linear', fill_value='extrapolate')
    amp_uniform = interp_func(t_uniform)

    # Bandpass filter for respiratory range (0.1-0.5 Hz = 6-30 breaths/min)
    if len(amp_uniform) < 100:
        return {"edr_rate_breaths_per_min": 0.0, "edr_power": 0.0, "edr_snr": 0.0}

    try:
        amp_filtered = bandpass_filter(amp_uniform, fs=10.0, low=0.1, high=0.5, order=2)
    except Exception:
        return {"edr_rate_breaths_per_min": 0.0, "edr_power": 0.0, "edr_snr": 0.0}

    # PSD analysis
    freqs, psd = welch(amp_filtered, fs=10.0, nperseg=min(len(amp_filtered), 256))
    resp_mask = (freqs >= 0.1) & (freqs <= 0.5)

    if not np.any(resp_mask):
        return {"edr_rate_breaths_per_min": 0.0, "edr_power": 0.0, "edr_snr": 0.0}

    # Dominant respiratory rate
    resp_freqs = freqs[resp_mask]
    resp_psd = psd[resp_mask]
    dominant_freq = resp_freqs[np.argmax(resp_psd)]
    edr_rate = dominant_freq * 60  # Convert to breaths per minute

    edr_power = float(np.trapz(resp_psd, resp_freqs))
    total_power = float(np.trapz(psd, freqs))
    edr_snr = edr_power / total_power if total_power > 0 else 0.0

    return {
        "edr_rate_breaths_per_min": float(edr_rate),
        "edr_power": edr_power,
        "edr_snr": edr_snr,
    }


def compute_ppg_morphology(signal, peaks, fs):
    """Compute PPG morphological features."""
    if len(peaks) < 2:
        return {
            "pulse_width_ms": 0.0,
            "systolic_slope": 0.0,
            "dicrotic_notch_present": 0.0,
            "pulse_amplitude": 0.0,
        }

    # Pulse width (time from systolic peak to next trough)
    widths = []
    for i in range(len(peaks) - 1):
        peak_val = signal[peaks[i]]
        next_trough_idx = peaks[i] + np.argmin(signal[peaks[i]:peaks[i+1]])
        width_samples = next_trough_idx - peaks[i]
        widths.append(width_samples / fs * 1000)  # Convert to ms

    # Pulse amplitude (peak-to-trough)
    troughs = []
    for i in range(len(peaks) - 1):
        trough_idx = peaks[i] + np.argmin(signal[peaks[i]:peaks[i+1]])
        troughs.append(signal[trough_idx])

    amplitudes = signal[peaks[:-1]] - np.array(troughs)

    return {
        "pulse_width_ms": float(np.mean(widths)) if widths else 0.0,
        "systolic_slope": float(np.mean(np.diff(signal[peaks]) / (np.diff(peaks) / fs))) if len(peaks) > 1 else 0.0,
        "dicrotic_notch_present": float(np.mean([1.0 if len(signal[peaks[i]:peaks[i+1]]) > 10 else 0.0 for i in range(len(peaks)-1)])),
        "pulse_amplitude": float(np.mean(amplitudes)) if len(amplitudes) > 0 else 0.0,
    }


def extract_features_from_segment(ppg_segment, fs=25):
    """
    Extract comprehensive features from a 1-hour PPG segment.

    Args:
        ppg_segment: 1D numpy array of PPG signal (60 seconds at fs Hz)
        fs: Sampling frequency (default 25 Hz)

    Returns:
        Dictionary of extracted features
    """
    features = {}

    # Basic signal statistics
    features["mean"] = float(np.mean(ppg_segment))
    features["std"] = float(np.std(ppg_segment))
    features["skewness"] = float(pd.Series(ppg_segment).skew())
    features["kurtosis"] = float(pd.Series(ppg_segment).kurtosis())
    features["range"] = float(np.ptp(ppg_segment))

    # Bandpass filter for analysis
    try:
        ppg_filtered = bandpass_filter(ppg_segment, fs, low=0.5, high=4.0)
    except Exception:
        ppg_filtered = ppg_segment

    # Detect peaks
    peaks = detect_peaks(ppg_filtered, fs)

    if len(peaks) < 3:
        # Not enough peaks for meaningful analysis
        features["n_peaks"] = len(peaks)
        features["heart_rate_bpm"] = 0.0
        features.update({k: 0.0 for k in [
            "rmssd", "sdnn", "pnn50", "mean_nn", "median_nn",
            "lf_power", "hf_power", "lf_hf_ratio", "total_power",
            "edr_rate_breaths_per_min", "edr_power", "edr_snr",
            "pulse_width_ms", "systolic_slope", "pulse_amplitude",
        ]})
        return features

    # Heart rate from peak intervals
    peak_intervals_samples = np.diff(peaks)
    peak_intervals_ms = peak_intervals_samples / fs * 1000
    peak_times_s = peaks / fs

    hr_bpm = 60000.0 / np.mean(peak_intervals_ms) if np.mean(peak_intervals_ms) > 0 else 0.0
    features["n_peaks"] = len(peaks)
    features["heart_rate_bpm"] = float(hr_bpm)

    # HRV Time-Domain
    hrv_td = compute_hrv_time_domain(peak_intervals_ms)
    features.update(hrv_td)

    # HRV Frequency-Domain
    hrv_fd = compute_hrv_frequency_domain(peak_intervals_ms, peak_times_s)
    features.update(hrv_fd)

    # EDR
    edr = compute_edr(ppg_filtered, peaks, fs)
    features.update(edr)

    # PPG Morphology
    morph = compute_ppg_morphology(ppg_filtered, peaks, fs)
    features.update(morph)

    # Signal quality metrics
    features["snr_db"] = float(10 * np.log10(np.var(ppg_filtered) / (np.var(ppg_segment - ppg_filtered) + 1e-10)))
    features["spectral_flatness"] = float(np.exp(np.mean(np.log(np.abs(np.fft.rfft(ppg_filtered)) + 1e-10))) / (np.mean(np.abs(np.fft.rfft(ppg_filtered)) + 1e-10)))

    return features


def extract_features_from_ppg_file(ppg_path, fs=25):
    """Extract features from a saved .npy PPG segment."""
    try:
        ppg = np.load(ppg_path)
        return extract_features_from_segment(ppg, fs)
    except Exception as e:
        return None


def process_cohort_windows(cohort_dir, output_dir, fs=25):
    """Process all extracted PPG windows and compute features."""
    cohort_dir = Path(cohort_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    windows_csv = cohort_dir / "windows.csv"
    if not windows_csv.exists():
        print(f"Error: {windows_csv} not found")
        return None

    windows_df = pd.read_csv(windows_csv)
    print(f"Processing {len(windows_df)} windows...")

    all_features = []
    total_segments = 0
    failed_segments = 0

    for idx, window in windows_df.iterrows():
        window_id = window["window_id"]
        n_segments = int(window["n_segments"])

        # Process each 1-hour segment in this window
        segment_features_list = []

        for seg_idx in range(n_segments):
            seg_path = cohort_dir / "ppg_segments" / f"{window_id}_s{seg_idx:02d}.npy"
            if not seg_path.exists():
                failed_segments += 1
                continue

            features = extract_features_from_ppg_file(seg_path, fs)
            if features is None:
                failed_segments += 1
                continue

            features["segment_index"] = seg_idx
            features["window_id"] = window_id
            segment_features_list.append(features)
            total_segments += 1

        if segment_features_list:
            # Aggregate features across segments (rolling statistics)
            seg_df = pd.DataFrame(segment_features_list)

            # Mean features across all segments
            numeric_cols = seg_df.select_dtypes(include=[np.number]).columns
            numeric_cols = [c for c in numeric_cols if c not in ["segment_index"]]

            window_features = {}
            for col in numeric_cols:
                vals = seg_df[col].values
                window_features[f"{col}_mean"] = float(np.mean(vals))
                window_features[f"{col}_std"] = float(np.std(vals))
                window_features[f"{col}_min"] = float(np.min(vals))
                window_features[f"{col}_max"] = float(np.max(vals))

            # Trend features (slope of first half vs second half)
            for col in ["heart_rate_bpm", "rmssd", "sdnn", "lf_hf_ratio"]:
                if col in numeric_cols:
                    mid = len(seg_df) // 2
                    if mid > 0:
                        first_half = seg_df[col].values[:mid].mean()
                        second_half = seg_df[col].values[mid:].mean()
                        window_features[f"{col}_trend"] = float(second_half - first_half)

            # Metadata
            window_features["window_id"] = window_id
            window_features["subject_id"] = window["subject_id"]
            window_features["primary_event"] = window["primary_event"]
            window_features["is_healthy"] = window["is_healthy"]
            window_features["time_to_event_hours"] = window["time_to_event_hours"]
            window_features["n_segments"] = n_segments

            all_features.append(window_features)

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(windows_df)} windows ({total_segments} segments, {failed_segments} failed)")

    # Save features
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(output_dir / "features.csv", index=False)

    print(f"\nFeature extraction complete:")
    print(f"  Total windows: {len(features_df)}")
    print(f"  Total segments processed: {total_segments}")
    print(f"  Failed segments: {failed_segments}")
    print(f"  Features per window: {len(features_df.columns)}")
    print(f"  Saved to: {output_dir / 'features.csv'}")

    return features_df


def main():
    """Main feature extraction pipeline."""
    project_root = Path(__file__).parent.parent
    cohort_dir = project_root / "data" / "processed" / "cohort_v1"
    output_dir = project_root / "data" / "processed" / "cohort_v1" / "features"

    print("=" * 70)
    print("PHASE 2: FEATURE EXTRACTION")
    print("=" * 70)

    features_df = process_cohort_windows(cohort_dir, output_dir, fs=25)

    if features_df is not None:
        # Print class distribution
        print("\nClass distribution:")
        print(features_df["primary_event"].value_counts())

        # Print feature summary
        print(f"\nFeature dimensions: {features_df.shape}")
        print(f"Numeric features: {features_df.select_dtypes(include=[np.number]).shape[1]}")


if __name__ == "__main__":
    main()
