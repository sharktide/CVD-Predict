"""Signal cleaning: bandpass filtering, motion/artifact detection, interpolation.

All functions in this module operate on raw waveforms *before* peak detection.
They accept and return numpy arrays at the native sampling rate.

Architectural rationale
-----------------------
Bandpass filtering must happen at the original sampling rate (e.g. 125 Hz)
to properly remove baseline wander (< 0.5 Hz) and high-frequency noise
(> 40 Hz for ECG, > 8 Hz for PPG).  Applying a 40 Hz high-pass cutoff
to 1 Hz data is above Nyquist and produces mathematically invalid results.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import signal as scipy_signal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bandpass filtering
# ---------------------------------------------------------------------------


def bandpass_filter(
    raw: np.ndarray,
    fs: float,
    low: float = 0.5,
    high: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    raw : 1-D signal array.
    fs : sampling frequency in Hz.  **Must** be provided (no default) to
        prevent the old bug of filtering at the wrong rate.
    low : lower cutoff frequency in Hz.
    high : upper cutoff frequency in Hz (must be < fs/2).
    order : filter order.

    Returns
    -------
    Filtered signal, same length as *raw*.

    Raises
    ------
    ValueError if *high* >= *fs / 2*.
    """
    nyq = fs / 2.0
    if high >= nyq:
        raise ValueError(
            f"High cutoff {high} Hz >= Nyquist {nyq} Hz (fs={fs} Hz). "
            "Cannot filter above Nyquist."
        )
    b, a = scipy_signal.butter(order, [low / nyq, high / nyq], btype="band")
    return scipy_signal.filtfilt(b, a, raw)


def bandpass_ecg(raw: np.ndarray, fs: float) -> np.ndarray:
    """Bandpass filter optimised for ECG (0.5--40 Hz)."""
    return bandpass_filter(raw, fs=fs, low=0.5, high=min(40.0, fs / 2 - 0.1))


def bandpass_ppg(raw: np.ndarray, fs: float) -> np.ndarray:
    """Bandpass filter optimised for PPG (0.5--8 Hz)."""
    return bandpass_filter(raw, fs=fs, low=0.5, high=min(8.0, fs / 2 - 0.1))


def bandpass_abp(raw: np.ndarray, fs: float) -> np.ndarray:
    """Bandpass filter optimised for ABP (0.5--20 Hz)."""
    return bandpass_filter(raw, fs=fs, low=0.5, high=min(20.0, fs / 2 - 0.1))


# ---------------------------------------------------------------------------
# Motion / artifact detection
# ---------------------------------------------------------------------------


def detect_motion_artifacts(
    accel: np.ndarray,
    fs: float,
    window_sec: float = 30.0,
    var_thresh: float = 2.0,
) -> np.ndarray:
    """Flag windows contaminated by motion artifacts using accelerometer variance.

    Parameters
    ----------
    accel : accelerometer data, shape ``(n_samples,)`` or ``(n_samples, 3)``.
    fs : sampling frequency.
    window_sec : analysis window length in seconds.
    var_thresh : variance threshold above which a window is flagged.

    Returns
    -------
    Boolean array, one entry per window, ``True`` = artifact present.
    """
    if accel.ndim == 1:
        accel = accel[:, np.newaxis]

    mag = np.sqrt(np.sum(accel ** 2, axis=1))
    win_samples = int(window_sec * fs)
    n_windows = len(mag) // win_samples

    if n_windows == 0:
        return np.array([], dtype=bool)

    artifacts = np.zeros(n_windows, dtype=bool)
    for i in range(n_windows):
        start = i * win_samples
        end = start + win_samples
        artifacts[i] = np.var(mag[start:end]) > var_thresh

    n_flagged = int(artifacts.sum())
    if n_flagged > 0:
        log.debug("Motion artifacts detected in %d / %d windows", n_flagged, n_windows)

    return artifacts


def compute_snr(
    signal: np.ndarray,
    fs: float,
    peak_freq: float = 1.0,
    bandwidth: float = 0.5,
) -> float:
    """Estimate SNR by comparing power at peak frequency vs. noise floor.

    Useful for quality assessment of ECG/PPG segments.
    """
    nperseg = min(len(signal), 1024)
    if nperseg < 16:
        return 0.0

    freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=nperseg)

    # Signal power: peak_freq +/- bandwidth
    sig_mask = (freqs >= peak_freq - bandwidth) & (freqs <= peak_freq + bandwidth)
    noise_mask = (freqs >= 0.5) & (freqs <= 50.0) & ~sig_mask

    sig_power = np.mean(psd[sig_mask]) if sig_mask.any() else 0.0
    noise_power = np.mean(psd[noise_mask]) if noise_mask.any() else 1e-10

    return float(10 * np.log10(sig_power / noise_power)) if noise_power > 0 else 0.0


# ---------------------------------------------------------------------------
# Gap interpolation and signal repair
# ---------------------------------------------------------------------------


def interpolate_gaps(
    signal: np.ndarray,
    fs: float,
    max_gap_sec: float = 5.0,
) -> np.ndarray:
    """Forward-fill short gaps, then linearly interpolate medium gaps.

    Parameters
    ----------
    signal : 1-D raw signal with possible NaN gaps.
    fs : sampling frequency.
    max_gap_sec : maximum gap length (seconds) to interpolate.

    Returns
    -------
    Repaired signal with NaN gaps filled.
    """
    max_gap_samples = int(max_gap_sec * fs)

    # Forward-fill up to max_gap_samples consecutive NaNs
    filled = signal.copy()
    nan_mask = np.isnan(filled)
    consecutive = 0
    for i in range(len(filled)):
        if nan_mask[i]:
            consecutive += 1
            if consecutive > max_gap_samples:
                # Too long a gap -- leave as NaN for now
                consecutive = 0
        else:
            if 0 < consecutive <= max_gap_samples:
                # Fill the gap with the last valid value
                fill_val = filled[i]
                for j in range(i - consecutive, i):
                    filled[j] = fill_val
            consecutive = 0

    # Linear interpolation for any remaining NaNs
    nans = np.isnan(filled)
    if nans.any():
        valid = ~nans
        if valid.any():
            xp = np.where(valid)[0]
            fp = filled[valid]
            x = np.where(nans)[0]
            filled[nans] = np.interp(x, xp, fp)

    return filled


def clean_signal(
    raw: np.ndarray,
    fs: float,
    signal_type: str = "ecg",
    interpolate: bool = True,
) -> np.ndarray:
    """End-to-end signal cleaning: interpolation -> bandpass filter.

    Parameters
    ----------
    raw : raw signal with possible NaNs and noise.
    fs : sampling frequency.
    signal_type : ``"ecg"``, ``"ppg"``, or ``"abp"``.
    interpolate : whether to fill NaN gaps before filtering.

    Returns
    -------
    Cleaned signal.
    """
    sig = raw.astype(float).copy()

    if interpolate:
        nan_frac = np.isnan(sig).mean()
        if nan_frac > 0.0 and nan_frac < 0.3:
            sig = interpolate_gaps(sig, fs)
        elif nan_frac >= 0.3:
            log.warning("Signal has %.1f%% NaNs -- gaps too large to interpolate", nan_frac * 100)

    # Replace any remaining NaNs with zeros for filtering
    sig = np.nan_to_num(sig, nan=0.0)

    filter_fn = {
        "ecg": bandpass_ecg,
        "ppg": bandpass_ppg,
        "abp": bandpass_abp,
    }.get(signal_type, lambda s, f: bandpass_filter(s, f))

    return filter_fn(sig, fs)
