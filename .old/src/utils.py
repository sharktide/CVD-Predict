"""Shared utilities: I/O, signal helpers, noise simulation, normalisation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist; return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_numpy(arr: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, arr, allow_pickle=False)


def load_numpy(path: str | Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


# ---------------------------------------------------------------------------
# Signal normalisation
# ---------------------------------------------------------------------------

def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalise a 1-D signal."""
    mu = np.mean(x)
    sigma = np.std(x) + eps
    return (x - mu) / sigma


def robust_scale(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale by median / IQR (robust to outliers)."""
    med = np.median(x)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25 + eps
    return (x - med) / iqr


# ---------------------------------------------------------------------------
# Wearable noise simulation
# ---------------------------------------------------------------------------

def simulate_wearable(
    ppg: np.ndarray,
    accel: Optional[np.ndarray] = None,
    target_rate: int = 25,
    base_fs: int = 125,
    contact_drop_prob: float = 0.05,
    noise_std: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate a wrist-wearable PPG from a high-fidelity ICU signal.

    Steps:
        1. Downsample to *target_rate* Hz.
        2. Add motion-driven noise when accelerometer data is available.
        3. Simulate random contact dropout.
    """
    rng = rng or np.random.default_rng()

    step = max(1, int(base_fs / target_rate))
    downsampled = ppg[::step].copy()
    n = len(downsampled)

    # Motion-driven noise
    if accel is not None:
        accel_2d = accel.reshape(-1, 3) if accel.ndim == 1 else accel
        accel_mag = np.linalg.norm(accel_2d, axis=1)
        accel_mag = accel_mag[:n]
        # Normalise magnitude to [0, 1] range for stable scaling
        amax = accel_mag.max() + 1e-8
        accel_mag = accel_mag / amax
        motion_noise = rng.normal(0, noise_std * (1.0 + accel_mag), size=n)
    else:
        motion_noise = rng.normal(0, noise_std, size=n)

    # Contact dropout
    mask = rng.binomial(1, 1.0 - contact_drop_prob, size=n).astype(np.float32)

    return (downsampled + motion_noise) * mask


# ---------------------------------------------------------------------------
# Signal quality index (simple peak-based proxy)
# ---------------------------------------------------------------------------

def compute_sqi_simple(
    ppg: np.ndarray,
    fs: int = 125,
    min_bpm: float = 40.0,
    max_bpm: float = 200.0,
) -> float:
    """Return a 0-1 signal quality estimate based on peak regularity.

    Uses autocorrelation of the bandpassed signal; higher values indicate
    more periodic (cleaner) PPG.
    """
    from scipy.signal import butter, filtfilt

    lo, hi = min_bpm / 60.0, max_bpm / 60.0
    b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    filt = filtfilt(b, a, ppg)

    # Normalise
    filt = (filt - filt.mean()) / (filt.std() + 1e-8)

    # Autocorrelation at plausible heart-rate lags
    min_lag = int(fs * 60.0 / max_bpm)
    max_lag = int(fs * 60.0 / min_bpm)
    if max_lag >= len(filt):
        return 0.0

    ac = np.correlate(filt, filt, mode="full")
    ac = ac[len(ac) // 2:]  # one-sided
    peak = ac[min_lag:max_lag].max() if max_lag > min_lag else 0.0
    normaliser = ac[0] + 1e-8
    return float(np.clip(peak / normaliser, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def train_val_test_split(
    patient_ids: np.ndarray,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic patient-level split."""
    rng = np.random.default_rng(seed)
    ids = patient_ids.copy()
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]
