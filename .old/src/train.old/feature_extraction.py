"""Minute-level feature assembly from raw physiological signals.

This module takes :class:`WaveformRecord` objects and produces a pandas
DataFrame with one row per minute containing:

- HRV features (14 metrics, computed on 5-minute rolling windows)
- HR statistics (mean, std, slope)
- Blood pressure features (systolic/diastolic mean, pulse pressure)
- Activity features (accelerometer-derived)
- Wear-time and missingness metrics

Architectural rationale
-----------------------
The previous pipeline produced a feature DataFrame directly from 1-minute
averaged data, which meant HRV was computed on meaningless "RR intervals"
derived from 1 Hz sample indices.  This module:

1. Keeps waveforms at native sampling rate for beat detection.
2. Computes HRV on 5-minute windows of genuine RR intervals.
3. Interpolates/forward-fills HRV features onto a 1-minute timeline.
4. Preserves temporal alignment for downstream windowing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .config import HRV_FEATURES
from .hrv import BeatDetectionResult, clean_rr_intervals, compute_hrv_features, detect_beats
from .signal_processing import (
    bandpass_abp,
    bandpass_ecg,
    bandpass_ppg,
    clean_signal,
    compute_snr,
    detect_motion_artifacts,
)
from .waveform_loader import WaveformRecord

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HRV feature timeline
# ---------------------------------------------------------------------------


def _compute_hrv_on_rolling_windows(
    rr_ms: np.ndarray,
    rr_times_sec: np.ndarray,
    total_duration_sec: float,
    window_sec: float = 300.0,
    step_sec: float = 60.0,
) -> dict[str, np.ndarray]:
    """Compute HRV features on rolling windows and return per-minute values.

    Parameters
    ----------
    rr_ms : cleaned RR intervals in milliseconds.
    rr_times_sec : time of each RR interval midpoint in seconds.
    total_duration_sec : total recording duration in seconds.
    window_sec : HRV analysis window (default 300 s = 5 minutes).
    step_sec : step between windows (default 60 s = 1 minute).

    Returns
    -------
    Dict mapping feature name to array of length ``n_minutes``.
    """
    n_minutes = int(np.ceil(total_duration_sec / 60.0))
    features: dict[str, list[float]] = {k: [0.0] for k in HRV_FEATURES}

    if len(rr_ms) < 5:
        # Not enough data -- fill with zeros
        for k in HRV_FEATURES:
            features[k] = [0.0] * n_minutes
        return {k: np.array(v) for k, v in features.items()}

    # Rolling window HRV computation
    for minute_idx in range(n_minutes):
        center_sec = (minute_idx + 0.5) * 60.0
        window_start = center_sec - window_sec / 2.0
        window_end = center_sec + window_sec / 2.0

        # Select RR intervals within this window
        mask = (rr_times_sec >= window_start) & (rr_times_sec < window_end)
        window_rr = rr_ms[mask]

        if len(window_rr) >= 5:
            hrv = compute_hrv_features(window_rr)
            for k in HRV_FEATURES:
                features[k].append(hrv[k])
        else:
            # Not enough RR intervals -- forward-fill last known value
            for k in HRV_FEATURES:
                last_val = features[k][-1] if features[k] else 0.0
                features[k].append(last_val)

    # Remove the initial sentinel values
    for k in HRV_FEATURES:
        features[k] = features[k][1:]  # drop the [0.0] we started with
        # Trim or pad to exact n_minutes
        if len(features[k]) > n_minutes:
            features[k] = features[k][:n_minutes]
        elif len(features[k]) < n_minutes:
            features[k].extend([features[k][-1]] * (n_minutes - len(features[k])))

    return {k: np.array(v) for k, v in features.items()}


# ---------------------------------------------------------------------------
# HR statistics
# ---------------------------------------------------------------------------


def _compute_hr_statistics(
    rr_ms: np.ndarray,
    rr_times_sec: np.ndarray,
    total_duration_sec: float,
) -> pd.DataFrame:
    """Compute per-minute HR statistics from RR intervals.

    Returns DataFrame with columns: hr_mean, hr_std, hr_slope.
    """
    n_minutes = int(np.ceil(total_duration_sec / 60.0))
    hr_mean = np.zeros(n_minutes)
    hr_std = np.zeros(n_minutes)
    hr_slope = np.zeros(n_minutes)

    if len(rr_ms) < 2:
        return pd.DataFrame({"hr_mean": hr_mean, "hr_std": hr_std, "hr_slope": hr_slope})

    # Instantaneous HR from each RR interval
    hr_inst = 60000.0 / rr_ms  # bpm

    for minute_idx in range(n_minutes):
        # 5-minute rolling window for HR stats
        center_sec = (minute_idx + 0.5) * 60.0
        window_start = center_sec - 150.0  # 2.5 min before
        window_end = center_sec + 150.0    # 2.5 min after
        mask = (rr_times_sec >= window_start) & (rr_times_sec < window_end)
        window_hr = hr_inst[mask]

        if len(window_hr) >= 2:
            hr_mean[minute_idx] = np.mean(window_hr)
            hr_std[minute_idx] = np.std(window_hr, ddof=1)
            # Linear slope over the window
            if len(window_hr) >= 3:
                x = np.arange(len(window_hr), dtype=float)
                slope = np.polyfit(x, window_hr, 1)[0]
                hr_slope[minute_idx] = slope
            else:
                hr_slope[minute_idx] = 0.0
        elif len(window_hr) == 1:
            hr_mean[minute_idx] = window_hr[0]
            hr_std[minute_idx] = 0.0
            hr_slope[minute_idx] = 0.0

    return pd.DataFrame({"hr_mean": hr_mean, "hr_std": hr_std, "hr_slope": hr_slope})


# ---------------------------------------------------------------------------
# Blood pressure features
# ---------------------------------------------------------------------------


def _compute_bp_features(abp: np.ndarray, fs: float) -> pd.DataFrame:
    """Compute per-minute BP features from raw ABP waveform.

    Returns DataFrame with columns: bp_sys_mean, bp_dia_mean, bp_pulse_pressure,
    bp_mean_arterial.
    """
    total_duration_sec = len(abp) / fs
    n_minutes = int(np.ceil(total_duration_sec / 60.0))
    samples_per_minute = int(60.0 * fs)

    bp_sys = np.zeros(n_minutes)
    bp_dia = np.zeros(n_minutes)
    bp_pulse = np.zeros(n_minutes)
    bp_map = np.zeros(n_minutes)

    for minute_idx in range(n_minutes):
        start = minute_idx * samples_per_minute
        end = min(start + samples_per_minute, len(abp))
        if end - start < fs:  # less than 1 second of data
            continue

        segment = abp[start:end]
        segment = segment[~np.isnan(segment)]
        if len(segment) < 10:
            continue

        # Systolic = max of segment, Diastolic = min
        bp_sys[minute_idx] = np.percentile(segment, 95)  # 95th percentile for systolic
        bp_dia[minute_idx] = np.percentile(segment, 5)   # 5th percentile for diastolic
        bp_pulse[minute_idx] = bp_sys[minute_idx] - bp_dia[minute_idx]
        bp_map[minute_idx] = bp_dia[minute_idx] + bp_pulse[minute_idx] / 3.0

    return pd.DataFrame({
        "bp_sys_mean": bp_sys,
        "bp_dia_mean": bp_dia,
        "bp_pulse_pressure": bp_pulse,
        "bp_mean_arterial": bp_map,
    })


# ---------------------------------------------------------------------------
# Activity features (per-minute)
# ---------------------------------------------------------------------------


def _compute_activity_per_minute(
    accel: np.ndarray,
    fs: float,
    total_duration_sec: float,
) -> pd.DataFrame:
    """Compute per-minute accelerometer features.

    Returns DataFrame with columns: total_activity, mean_activity, std_activity.
    """
    n_minutes = int(np.ceil(total_duration_sec / 60.0))
    samples_per_minute = int(60.0 * fs)

    total_act = np.zeros(n_minutes)
    mean_act = np.zeros(n_minutes)
    std_act = np.zeros(n_minutes)

    for minute_idx in range(n_minutes):
        start = minute_idx * samples_per_minute
        end = min(start + samples_per_minute, len(accel))
        if end - start < 10:
            continue

        segment = accel[start:end]
        if segment.ndim == 1:
            mag = np.abs(segment)
        else:
            mag = np.sqrt(np.sum(segment ** 2, axis=1))

        total_act[minute_idx] = np.sum(mag)
        mean_act[minute_idx] = np.mean(mag)
        std_act[minute_idx] = np.std(mag)

    return pd.DataFrame({
        "total_activity": total_act,
        "mean_activity": mean_act,
        "std_activity": std_act,
    })


def _compute_circadian_activity(
    accel: np.ndarray,
    fs: float,
    start_time: pd.Timestamp | None = None,
) -> dict[str, float]:
    """Compute aggregate circadian activity features from accelerometer data.

    These are scalar features broadcast across the entire recording.
    """
    if accel is None or len(accel) == 0:
        return {k: 0.0 for k in [
            "day_mean_activity", "night_mean_activity", "day_night_ratio",
            "activity_entropy", "peak_activity_hour", "activity_slope",
        ]}

    if accel.ndim == 1:
        mag = np.abs(accel)
    else:
        mag = np.sqrt(np.sum(accel ** 2, axis=1))

    total_duration = len(mag) / fs
    n_minutes = int(np.ceil(total_duration / 60.0))

    # Create minute-level time index
    if start_time is not None:
        minutes = pd.date_range(start_time, periods=n_minutes, freq="1min")
    else:
        minutes = pd.date_range("2000-01-01", periods=n_minutes, freq="1min")

    hour = minutes.hour
    is_nocturnal = (hour >= 22) | (hour < 6)

    # Resample mag to minute-level
    mag_minute = np.zeros(n_minutes)
    samples_per_minute = int(60.0 * fs)
    for i in range(n_minutes):
        s = i * samples_per_minute
        e = min(s + samples_per_minute, len(mag))
        if e > s:
            mag_minute[i] = np.mean(mag[s:e])

    day_mag = mag_minute[~is_nocturnal]
    night_mag = mag_minute[is_nocturnal]

    features = {
        "day_mean_activity": float(np.mean(day_mag)) if len(day_mag) > 0 else 0.0,
        "night_mean_activity": float(np.mean(night_mag)) if len(night_mag) > 0 else 0.0,
        "day_night_ratio": (
            float(np.mean(day_mag) / np.mean(night_mag))
            if len(night_mag) > 0 and np.mean(night_mag) > 0 else 0.0
        ),
        "activity_entropy": 0.0,
        "peak_activity_hour": float(hour[np.argmax(mag_minute)]) if len(mag_minute) > 0 else 0.0,
        "activity_slope": float(np.polyfit(range(len(mag_minute)), mag_minute, 1)[0]) if len(mag_minute) > 1 else 0.0,
    }

    # Shannon entropy of magnitude distribution
    if np.sum(mag_minute) > 0:
        probs = mag_minute / np.sum(mag_minute) + 1e-10
        features["activity_entropy"] = float(-np.sum(probs * np.log2(probs)))

    return features


# ---------------------------------------------------------------------------
# Wear-time / missingness metrics
# ---------------------------------------------------------------------------


def _compute_wear_metrics(record: WaveformRecord) -> pd.DataFrame:
    """Compute per-minute wear-time and missingness metrics.

    Returns DataFrame with columns: pct_missing, wear_time, signal_quality.
    """
    total_duration_sec = record.duration_seconds
    n_minutes = int(np.ceil(total_duration_sec / 60.0))

    pct_missing = np.zeros(n_minutes)
    wear_time = np.zeros(n_minutes)
    signal_quality = np.zeros(n_minutes)

    signals = [s for s in [record.ecg, record.ppg, record.abp] if s is not None]

    if not signals:
        return pd.DataFrame({
            "pct_missing": np.ones(n_minutes),
            "wear_time": np.zeros(n_minutes),
            "signal_quality": np.zeros(n_minutes),
        })

    samples_per_minute = int(60.0 * record.fs)

    for minute_idx in range(n_minutes):
        start = minute_idx * samples_per_minute
        total_present = 0
        total_samples = 0

        for sig in signals:
            end = min(start + samples_per_minute, len(sig))
            if end > start:
                segment = sig[start:end]
                n_nan = np.sum(np.isnan(segment))
                total_present += len(segment) - n_nan
                total_samples += len(segment)

        if total_samples > 0:
            pct_missing[minute_idx] = 1.0 - (total_present / total_samples)
            wear_time[minute_idx] = total_present / total_samples
            # Simple quality: fraction of non-NaN, non-zero samples
            signal_quality[minute_idx] = total_present / total_samples

    return pd.DataFrame({
        "pct_missing": pct_missing,
        "wear_time": wear_time,
        "signal_quality": signal_quality,
    })


# ---------------------------------------------------------------------------
# Main feature extraction pipeline
# ---------------------------------------------------------------------------


def extract_features_from_record(
    record: WaveformRecord,
    start_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Extract a complete minute-level feature DataFrame from a WaveformRecord.

    This is the main entry point for feature extraction.  It orchestrates:

    1. Signal cleaning (bandpass filtering at native fs)
    2. Beat detection (R-peak or pulse peak)
    3. RR interval extraction and cleaning
    4. HRV feature computation on rolling windows
    5. HR statistics, BP features, activity features
    6. Assembly into a minute-level DataFrame

    Parameters
    ----------
    record : raw waveform data.
    start_time : optional reference timestamp for circadian features.

    Returns
    -------
    DataFrame with one row per minute and columns for all features.
    """
    fs = record.fs
    total_duration_sec = record.duration_seconds
    n_minutes = int(np.ceil(total_duration_sec / 60.0))

    if n_minutes == 0:
        log.warning("Empty record: %s", record.metadata.get("path", "unknown"))
        return pd.DataFrame()

    # Create minute-level index
    if start_time is None:
        start_time = pd.Timestamp("2000-01-01")
    minute_index = pd.date_range(start_time, periods=n_minutes, freq="1min")

    # --- Step 1: Clean signals ---
    ecg_clean = None
    ppg_clean = None
    abp_clean = None

    if record.has_ecg():
        ecg_clean = clean_signal(record.ecg, fs, signal_type="ecg")
        log.debug("ECG cleaned: %d samples, fs=%.1f Hz", len(record.ecg), fs)

    if record.has_ppg():
        ppg_clean = clean_signal(record.ppg, fs, signal_type="ppg")
        log.debug("PPG cleaned: %d samples, fs=%.1f Hz", len(record.ppg), fs)

    if record.has_abp():
        abp_clean = clean_signal(record.abp, fs, signal_type="abp")
        log.debug("ABP cleaned: %d samples, fs=%.1f Hz", len(record.abp), fs)

    # --- Step 2 & 3: Beat detection + RR extraction ---
    beat_result: BeatDetectionResult | None = None
    if ecg_clean is not None:
        beat_result = detect_beats(ecg_clean, fs, signal_type="ecg")
        if beat_result.n_beats > 0:
            log.info("ECG: detected %d beats (quality=%.2f, method=%s)",
                     beat_result.n_beats, beat_result.quality, beat_result.method)
    elif ppg_clean is not None:
        beat_result = detect_beats(ppg_clean, fs, signal_type="ppg")
        if beat_result.n_beats > 0:
            log.info("PPG: detected %d beats (quality=%.2f, method=%s)",
                     beat_result.n_beats, beat_result.quality, beat_result.method)

    # Clean RR intervals
    rr_ms = np.array([])
    rr_times = np.array([])
    if beat_result is not None and beat_result.n_beats >= 3:
        rr_ms, rr_times = clean_rr_intervals(beat_result.rr_ms, beat_result.rr_times_sec)

    # --- Step 4: HRV features on rolling windows ---
    hrv_features = _compute_hrv_on_rolling_windows(
        rr_ms, rr_times, total_duration_sec,
        window_sec=300.0,  # 5-minute window
        step_sec=60.0,    # 1-minute step
    )

    out = pd.DataFrame(index=minute_index)
    for k in HRV_FEATURES:
        out[k] = hrv_features[k]

    # --- Step 5: HR statistics ---
    hr_stats = _compute_hr_statistics(rr_ms, rr_times, total_duration_sec)
    out["hr_mean"] = hr_stats["hr_mean"].values
    out["hr_std"] = hr_stats["hr_std"].values
    out["hr_slope"] = hr_stats["hr_slope"].values

    # --- Step 6: BP features ---
    if abp_clean is not None:
        bp = _compute_bp_features(abp_clean, fs)
        out["bp_sys_mean"] = bp["bp_sys_mean"].values
        out["bp_dia_mean"] = bp["bp_dia_mean"].values
        out["bp_pulse_pressure"] = bp["bp_pulse_pressure"].values
        out["bp_mean_arterial"] = bp["bp_mean_arterial"].values
    else:
        out["bp_sys_mean"] = 0.0
        out["bp_dia_mean"] = 0.0
        out["bp_pulse_pressure"] = 0.0
        out["bp_mean_arterial"] = 0.0

    # --- Step 7: Activity features ---
    if record.has_accel():
        act_per_min = _compute_activity_per_minute(record.accel, fs, total_duration_sec)
        out["total_activity"] = act_per_min["total_activity"].values
        out["mean_activity"] = act_per_min["mean_activity"].values
        out["std_activity"] = act_per_min["std_activity"].values

        circadian = _compute_circadian_activity(record.accel, fs, start_time)
        for k, v in circadian.items():
            out[k] = v
    else:
        for k in ["total_activity", "mean_activity", "std_activity",
                   "day_mean_activity", "night_mean_activity", "day_night_ratio",
                   "activity_entropy", "peak_activity_hour", "activity_slope"]:
            out[k] = 0.0

    # --- Step 8: Wear-time metrics ---
    wear = _compute_wear_metrics(record)
    out["pct_missing"] = wear["pct_missing"].values
    out["wear_time"] = wear["wear_time"].values

    # --- Step 9: Signal quality ---
    quality_scores = []
    if ecg_clean is not None:
        quality_scores.append(compute_snr(ecg_clean, fs, peak_freq=1.0))
    if ppg_clean is not None:
        quality_scores.append(compute_snr(ppg_clean, fs, peak_freq=1.0))
    out["signal_quality"] = np.mean(quality_scores) if quality_scores else 0.0

    # Replace infinities and NaN
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    log.info("Extracted %d features over %d minutes", len(out.columns), len(out))
    return out


def extract_features_from_records(
    records: list[WaveformRecord],
) -> pd.DataFrame:
    """Extract features from multiple WaveformRecords and concatenate.

    Each record produces its own minute-level feature DataFrame.
    Records are concatenated along the time axis.

    Returns
    -------
    Combined DataFrame with one row per minute across all records.
    """
    frames: list[pd.DataFrame] = []

    for i, rec in enumerate(records):
        log.info("Processing record %d / %d (%.1f min, fs=%.1f Hz)",
                 i + 1, len(records), rec.duration_minutes, rec.fs)

        features = extract_features_from_record(rec)
        if not features.empty:
            frames.append(features)

    if not frames:
        log.warning("No features extracted from any record")
        return pd.DataFrame()

    combined = pd.concat(frames, axis=0)
    log.info("Combined features: %d rows x %d columns", len(combined), len(combined.columns))
    return combined
