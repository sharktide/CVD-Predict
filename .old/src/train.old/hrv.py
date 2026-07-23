"""Heart-rate variability (HRV) feature extraction.

This module provides:

- **Beat detection**: R-peak (ECG) or pulse peak (PPG) detection using
  robust algorithms with graceful fallbacks.
- **RR interval extraction**: conversion of peak indices to RR intervals
  in milliseconds, with outlier removal and interpolation.
- **HRV features**: time-domain, frequency-domain, and non-linear metrics
  computed from RR intervals.

Architectural rationale
-----------------------
HRV features must be computed on *RR intervals* (the time between successive
heartbeats), not on raw ECG/PPG samples.  The previous implementation detected
peaks on 1 Hz downsampled data, producing "RR intervals" measured in minutes
rather than milliseconds.  This module operates entirely at the native
sampling rate and produces clinically meaningful RR intervals.

Frequency-domain HRV requires an evenly-sampled tachogram.  We cubic-spline
interpolate the RR interval time series to 4 Hz (the standard for short-
duration HRV analysis) before applying Welch's method.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import signal as scipy_signal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Beat detection (R-peak / pulse peak)
# ---------------------------------------------------------------------------


@dataclass
class BeatDetectionResult:
    """Result of heartbeat detection.

    Attributes
    ----------
    peak_indices : np.ndarray
        Sample indices of detected peaks.
    peak_times_sec : np.ndarray
        Peak times in seconds (peak_indices / fs).
    rr_ms : np.ndarray
        RR intervals in milliseconds.
    rr_times_sec : np.ndarray
        Midpoints of successive RR intervals in seconds.
    n_beats : int
        Total number of detected beats.
    quality : float
        Detection quality score 0--1 (heuristic).
    method : str
        Name of the algorithm that was used.
    """

    peak_indices: np.ndarray
    peak_times_sec: np.ndarray
    rr_ms: np.ndarray
    rr_times_sec: np.ndarray
    n_beats: int
    quality: float
    method: str


def _detect_peaks_neurokit2(
    signal: np.ndarray,
    fs: float,
    signal_type: str,
) -> np.ndarray | None:
    """Detect peaks using NeuroKit2 (preferred for research-grade processing).

    Returns peak indices or None if detection fails.
    """
    try:
        import neurokit2 as nk  # type: ignore[import-untyped]

        if signal_type == "ecg":
            _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)
            return rpeaks["ECG_R_Peaks"]
        elif signal_type == "ppg":
            _, peaks = nk.ppg_peaks(signal, sampling_rate=fs)
            return peaks["PPG_Peaks"]
    except ImportError:
        log.debug("NeuroKit2 not installed -- falling back to wfdb/scipy")
    except Exception as e:
        log.debug("NeuroKit2 peak detection failed: %s", e)

    return None


def _detect_peaks_wfdb(
    signal: np.ndarray,
    fs: float,
    signal_type: str,
) -> np.ndarray | None:
    """Detect peaks using wfdb.processing (PhysioNet standard).

    Returns peak indices or None if detection fails.
    """
    try:
        import wfdb  # type: ignore[import-untyped]
        from wfdb.processing import xqrs_detect  # type: ignore[import-untyped]

        if signal_type == "ecg":
            rpeaks = xqrs_detect(sig=signal, fs=fs)
            if len(rpeaks) > 0:
                return rpeaks
    except ImportError:
        log.debug("wfdb.processing not available")
    except Exception as e:
        log.debug("wfdb peak detection failed: %s", e)

    return None


def _detect_peaks_scipy(
    signal: np.ndarray,
    fs: float,
    signal_type: str,
) -> np.ndarray | None:
    """Fallback peak detection using scipy.signal.find_peaks.

    This uses physiological constraints for minimum heart rate (~30 bpm)
    and maximum heart rate (~220 bpm) to set distance and height thresholds.
    """
    min_rr_sec = 60.0 / 220.0  # ~0.27 s (max HR 220 bpm)
    max_rr_sec = 60.0 / 30.0   # ~2.0 s (min HR 30 bpm)
    min_distance = int(min_rr_sec * fs)

    # Adaptive height threshold: median-based
    sig_std = np.std(signal)
    sig_median = np.median(signal)
    height_thresh = sig_median + 0.5 * sig_std

    # For PPG, peaks are positive deflections
    if signal_type == "ppg":
        height_thresh = sig_median + 0.3 * sig_std

    peaks, properties = scipy_signal.find_peaks(
        signal,
        distance=min_distance,
        height=height_thresh,
    )

    if len(peaks) < 3:
        # Relax constraints and retry
        peaks, _ = scipy_signal.find_peaks(
            signal,
            distance=max(1, int(0.3 * fs)),
            prominence=0.3 * sig_std,
        )

    return peaks if len(peaks) >= 3 else None


def detect_beats(
    signal: np.ndarray,
    fs: float,
    signal_type: str = "ecg",
) -> BeatDetectionResult:
    """Detect heartbeats in a filtered ECG or PPG signal.

    Tries algorithms in order of preference:
    1. NeuroKit2 (most robust, handles noisy signals well)
    2. wfdb.processing (PhysioNet standard, ECG only)
    3. scipy.signal.find_peaks (fallback with physiological constraints)

    Parameters
    ----------
    signal : bandpass-filtered 1-D signal.
    fs : sampling frequency in Hz.
    signal_type : ``"ecg"`` or ``"ppg"``.

    Returns
    -------
    :class:`BeatDetectionResult` with peak indices, RR intervals, and metadata.
    """
    # Try each detector in order
    for detector, name in [
        (_detect_peaks_neurokit2, "neurokit2"),
        (_detect_peaks_wfdb, "wfdb"),
        (_detect_peaks_scipy, "scipy"),
    ]:
        peaks = detector(signal, fs, signal_type)
        if peaks is not None and len(peaks) >= 3:
            break
    else:
        log.warning("No peaks detected with any algorithm (signal_type=%s, fs=%.1f)", signal_type, fs)
        empty = np.array([])
        return BeatDetectionResult(
            peak_indices=empty,
            peak_times_sec=empty,
            rr_ms=empty,
            rr_times_sec=empty,
            n_beats=0,
            quality=0.0,
            method="none",
        )

    peak_times = peaks.astype(float) / fs
    rr_ms = np.diff(peaks).astype(float) / fs * 1000.0
    rr_times = (peak_times[:-1] + peak_times[1:]) / 2.0

    # Quality heuristic: fraction of "normal" RR intervals (300--2000 ms)
    normal_frac = np.mean((rr_ms > 300) & (rr_ms < 2000)) if len(rr_ms) > 0 else 0.0
    # Penalise if too few beats for reliable HRV
    beat_score = min(1.0, len(peaks) / (60.0 * fs / 1000.0))  # normalise by expected beats
    quality = float(0.6 * normal_frac + 0.4 * beat_score)

    return BeatDetectionResult(
        peak_indices=peaks,
        peak_times_sec=peak_times,
        rr_ms=rr_ms,
        rr_times_sec=rr_times,
        n_beats=len(peaks),
        quality=quality,
        method=name,
    )


# ---------------------------------------------------------------------------
# RR interval cleaning
# ---------------------------------------------------------------------------


def clean_rr_intervals(
    rr_ms: np.ndarray,
    rr_times_sec: np.ndarray,
    min_rr_ms: float = 300.0,
    max_rr_ms: float = 2000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove outliers and interpolate RR intervals.

    Physiological bounds: 300 ms (200 bpm) to 2000 ms (30 bpm).
    Outliers are replaced by linear interpolation.

    Returns
    -------
    (cleaned_rr_ms, cleaned_rr_times_sec)
    """
    if len(rr_ms) < 3:
        return rr_ms, rr_times_sec

    # Flag out-of-range intervals
    outlier = (rr_ms < min_rr_ms) | (rr_ms > max_rr_ms)
    n_outliers = int(outlier.sum())

    if n_outliers == 0:
        return rr_ms, rr_times_sec

    log.debug("Removed %d / %d RR outliers (%.1f%%)", n_outliers, len(rr_ms), 100 * n_outliers / len(rr_ms))

    # Interpolate outliers
    cleaned = rr_ms.copy()
    valid = ~outlier
    if valid.any():
        cleaned[outlier] = np.interp(
            rr_times_sec[outlier],
            rr_times_sec[valid],
            rr_ms[valid],
        )

    return cleaned, rr_times_sec


# ---------------------------------------------------------------------------
# HRV feature computation
# ---------------------------------------------------------------------------


def compute_hrv_features(
    rr_ms: np.ndarray,
    fs_tachogram: float = 4.0,
) -> dict[str, float]:
    """Compute comprehensive HRV features from RR intervals.

    Parameters
    ----------
    rr_ms : RR intervals in milliseconds.
    fs_tachogram : effective sampling rate for the interpolated tachogram
        (used for frequency-domain analysis).  Default 4 Hz is standard.

    Returns
    -------
    Dictionary of 14 HRV features.
    """
    from .config import HRV_FEATURES

    if len(rr_ms) < 5:
        return {k: 0.0 for k in HRV_FEATURES}

    rr_s = rr_ms / 1000.0
    diff_rr = np.diff(rr_s)

    # --- Time domain ---
    mean_rr = float(np.mean(rr_ms))
    sd_rr = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2))) * 1000.0
    sdsd = float(np.std(diff_rr, ddof=1)) * 1000.0 if len(diff_rr) > 1 else 0.0
    pnn50 = float(np.mean(np.abs(diff_rr) > 0.050)) * 100.0
    cv_rr = sd_rr / mean_rr if mean_rr > 0 else 0.0
    median_rr = float(np.median(rr_ms))
    range_rr = float(np.ptp(rr_ms))
    iqr_rr = float(np.percentile(rr_ms, 75) - np.percentile(rr_ms, 25))

    # --- Frequency domain ---
    # Interpolate RR intervals to evenly-spaced tachogram
    tachogram, tachogram_fs = _interpolate_tachogram(rr_ms, fs=fs_tachogram)
    lf_power = hf_power = lf_hf_ratio = 0.0

    if len(tachogram) >= 16:
        nperseg = min(len(tachogram), 256)
        freqs, psd = scipy_signal.welch(tachogram, fs=tachogram_fs, nperseg=nperseg)
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.40)
        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        lf_power = float(_trapz(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
        hf_power = float(_trapz(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0

    # --- Non-linear complexity ---
    sample_ent = _sample_entropy(rr_s, m=2, r=0.2 * np.std(rr_s)) if len(rr_s) > 20 else 0.0
    approx_ent = _approx_entropy(rr_s, m=2, r=0.2 * np.std(rr_s)) if len(rr_s) > 20 else 0.0

    return {
        "mean_rr": mean_rr,
        "sd_rr": sd_rr,
        "rmssd": rmssd,
        "sdsd": sdsd,
        "pnn50": pnn50,
        "cv_rr": cv_rr,
        "median_rr": median_rr,
        "range_rr": range_rr,
        "iqr_rr": iqr_rr,
        "lf_power": lf_power,
        "hf_power": hf_power,
        "lf_hf_ratio": lf_hf_ratio,
        "sample_entropy": sample_ent,
        "approximate_entropy": approx_ent,
    }


def _interpolate_tachogram(
    rr_ms: np.ndarray,
    fs: float = 4.0,
) -> tuple[np.ndarray, float]:
    """Cubic-spline interpolate unevenly-spaced RR intervals to uniform rate.

    This is the standard approach for frequency-domain HRV analysis:
    the RR interval series is resampled to a uniform rate (typically 4 Hz)
    using cubic spline interpolation.

    Returns
    -------
    (interpolated_tachogram, effective_fs)
    """
    if len(rr_ms) < 4:
        return np.array([]), fs

    # Cumulative time of each RR interval endpoint
    cum_time = np.cumsum(rr_ms) / 1000.0  # seconds
    # Time at each RR interval midpoint
    mid_times = (cum_time[:-1] + cum_time[1:]) / 2.0 if len(cum_time) > 1 else cum_time

    # Remove the mean to get detrended tachogram
    tachogram = rr_ms - np.mean(rr_ms)

    # Create uniform time grid
    total_duration = cum_time[-1] - cum_time[0]
    if total_duration <= 0:
        return np.array([]), fs

    n_points = max(int(total_duration * fs), 4)
    uniform_time = np.linspace(cum_time[0], cum_time[-1], n_points)

    # Cubic spline interpolation
    try:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(cum_time, tachogram)
        interpolated = cs(uniform_time)
    except Exception:
        # Fallback to linear interpolation
        interpolated = np.interp(uniform_time, cum_time, tachogram)

    return interpolated, fs


# ---------------------------------------------------------------------------
# Non-linear complexity measures
# ---------------------------------------------------------------------------


def _sample_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Compute sample entropy (SampEn).

    SampEn(m, r, N) measures the logarithmic probability that m consecutive
    data points within tolerance r are similar to the next point.

    Parameters
    ----------
    data : 1-D signal.
    m : embedding dimension.
    r : tolerance (typically 0.2 * std of data).
    """
    n = len(data)
    if n < m + 2:
        return 0.0

    def _count_matches(template_len: int) -> int:
        count = 0
        templates = np.array([data[i : i + template_len] for i in range(n - template_len)])
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count

    b = _count_matches(m)
    a = _count_matches(m + 1)
    if b == 0 or a == 0:
        return 0.0
    return float(-np.log(a / b))


def _approx_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Compute approximate entropy (ApEn).

    ApEn measures the regularity of a time series.  Lower values indicate
    more regularity (less complexity).

    Parameters
    ----------
    data : 1-D signal.
    m : embedding dimension.
    r : tolerance.
    """
    n = len(data)
    if n < m + 2:
        return 0.0

    def _phi(template_len: int) -> float:
        templates = np.array([data[i : i + template_len] for i in range(n - template_len + 1)])
        counts = np.zeros(len(templates))
        for i in range(len(templates)):
            diffs = np.max(np.abs(templates - templates[i]), axis=1)
            counts[i] = np.mean(diffs < r)
        return float(np.mean(np.log(counts + 1e-10)))

    return float(_phi(m) - _phi(m + 1))
