"""Signal processing, windowing, and feature engineering for wearable / ICU data.

This module converts raw downloaded datasets into model-ready numpy arrays:
  1. Load raw waveforms (ECG, PPG, accel, HR) from WFDB or CSV.
  2. Band-pass filter to remove noise and motion artifacts.
  3. Resample to a common 1-minute time base.
  4. Compute HRV features (time + frequency domain).
  5. Compute accelerometer / activity features.
  6. Assemble rolling windows of 24 / 48 / 72 hours.
  7. Normalise and produce train / val / test splits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

from .config import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    BUTTERWORTH_ORDER,
    DATASETS,
    HRV_FEATURES,
    N_STATIC_FEATURES,
    N_TIME_FEATURES,
    PROCESSED_DIR,
    RAW_DIR,
    RESAMPLE_RATE_HZ,
    SAMPLING_RATE_HZ,
    SLIDE_HOURS,
    WINDOW_HOURS,
    WINDOW_LENGTH_MINUTES,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal cleaning
# ---------------------------------------------------------------------------


def bandpass_filter(
    raw: np.ndarray,
    fs: float = SAMPLING_RATE_HZ,
    low: float = BANDPASS_LOW_HZ,
    high: float = BANDPASS_HIGH_HZ,
    order: int = BUTTERWORTH_ORDER,
) -> np.ndarray:
    """Apply a Butterworth band-pass filter to a 1-D signal."""
    nyq = fs / 2.0
    b, a = scipy_signal.butter(order, [low / nyq, high / nyq], btype="band")
    return scipy_signal.filtfilt(b, a, raw)


def detect_artifacts(hr: np.ndarray, accel_var: np.ndarray, var_thresh: float = 2.0) -> np.ndarray:
    """Return a boolean mask where ``True`` marks corrupted windows.

    Uses high accelerometer variance as a proxy for motion artifacts.
    """
    return accel_var > var_thresh


def interpolate_gaps(series: pd.Series, max_gap_minutes: int = 5) -> pd.Series:
    """Forward-fill short gaps, then linearly interpolate medium gaps."""
    filled = series.ffill(limit=max_gap_minutes)
    return filled.interpolate(limit_direction="forward")


# ---------------------------------------------------------------------------
# HRV feature extraction
# ---------------------------------------------------------------------------


def compute_hrv_features(rr_ms: np.ndarray, fs: float = 4.0) -> dict[str, float]:
    """Compute time-domain and frequency-domain HRV features from R-R intervals.

    Parameters
    ----------
    rr_ms : array of R-R intervals in milliseconds.
    fs    : effective sampling rate of the RR tachogram (Hz).

    Returns
    -------
    Dictionary of feature name → value.
    """
    if len(rr_ms) < 10:
        return {k: 0.0 for k in HRV_FEATURES}

    rr_s = rr_ms / 1000.0
    diff_rr = np.diff(rr_s)

    # Time domain
    mean_rr = float(np.mean(rr_ms))
    sd_rr = float(np.std(rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2))) * 1000.0
    sdsd = float(np.std(diff_rr, ddof=1)) * 1000.0 if len(diff_rr) > 1 else 0.0
    pnn50 = float(np.mean(np.abs(diff_rr) > 0.050)) * 100.0
    cv_rr = sd_rr / mean_rr if mean_rr else 0.0
    median_rr = float(np.median(rr_ms))
    range_rr = float(np.ptp(rr_ms))
    iqr_rr = float(np.percentile(rr_ms, 75) - np.percentile(rr_ms, 25))

    # Frequency domain (Welch PSD)
    tachogram = rr_ms - np.mean(rr_ms)
    nperseg = min(len(tachogram), 256)
    if nperseg >= 16:
        freqs, psd = scipy_signal.welch(tachogram, fs=fs, nperseg=nperseg)
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.40)
        lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0.0
        hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0.0
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0.0
    else:
        lf_power = hf_power = lf_hf_ratio = 0.0

    # Complexity
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


def _sample_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Compute sample entropy of a 1-D signal."""
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
    return -np.log(a / b)


def _approx_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Compute approximate entropy of a 1-D signal."""
    n = len(data)
    if n < m + 2:
        return 0.0

    def _phi(template_len: int) -> float:
        templates = np.array([data[i : i + template_len] for i in range(n - template_len + 1)])
        counts = np.zeros(len(templates))
        for i in range(len(templates)):
            diffs = np.max(np.abs(templates - templates[i]), axis=1)
            counts[i] = np.mean(diffs < r)
        return np.mean(np.log(counts + 1e-10))

    return _phi(m) - _phi(m + 1)


# ---------------------------------------------------------------------------
# Activity / accelerometer features
# ---------------------------------------------------------------------------


def compute_activity_features(
    accel: pd.DataFrame,
    hr: pd.Series | None = None,
) -> dict[str, float]:
    """Derive circadian and aggregate activity features from 3-axis accelerometer.

    Parameters
    ----------
    accel : DataFrame with columns ``x``, ``y``, ``z`` indexed by datetime.
    hr    : optional heart-rate series aligned to the same index.
    """
    magnitude = np.sqrt(accel["x"] ** 2 + accel["y"] ** 2 + accel["z"] ** 2)
    hour = accel.index.hour

    is_nocturnal = (hour >= 22) | (hour < 6)
    day_mag = magnitude[~is_nocturnal]
    night_mag = magnitude[is_nocturnal]

    features: dict[str, float] = {
        "total_activity": float(magnitude.sum()),
        "mean_activity": float(magnitude.mean()),
        "std_activity": float(magnitude.std()),
        "day_mean_activity": float(day_mag.mean()) if len(day_mag) else 0.0,
        "night_mean_activity": float(night_mag.mean()) if len(night_mag) else 0.0,
        "day_night_ratio": (
            float(day_mag.mean() / night_mag.mean()) if len(night_mag) and night_mag.mean() > 0 else 0.0
        ),
        "activity_entropy": float(-np.sum((magnitude.value_counts(normalize=True) + 1e-10)
                                          * np.log2(magnitude.value_counts(normalize=True) + 1e-10))),
        "peak_activity_hour": float(hour.value_counts().idxmax()) if len(hour) else 0.0,
        "activity_slope": float(np.polyfit(range(len(magnitude)), magnitude, 1)[0]) if len(magnitude) > 1 else 0.0,
    }

    # HR-derived nocturnal spike feature
    if hr is not None and len(hr) > 0:
        night_hr = hr[is_nocturnal]
        day_hr = hr[~is_nocturnal]
        features["night_hr_mean"] = float(night_hr.mean()) if len(night_hr) else 0.0
        features["night_hr_spike"] = (
            float(night_hr.max() - day_hr.mean()) if len(night_hr) and len(day_hr) else 0.0
        )
        features["hr_slope"] = float(np.polyfit(range(len(hr)), hr, 1)[0]) if len(hr) > 1 else 0.0
    else:
        features["night_hr_mean"] = 0.0
        features["night_hr_spike"] = 0.0
        features["hr_slope"] = 0.0

    return features


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def build_windows(
    feature_df: pd.DataFrame,
    window_hours: list[int] = WINDOW_HOURS,
    slide_hours: int = SLIDE_HOURS,
) -> dict[int, list[np.ndarray]]:
    """Slice a continuous feature DataFrame into sliding windows.

    Returns ``{window_hours_value: [np.ndarray, ...]}`` where each array
    has shape ``(window_minutes, n_features)``.
    """
    results: dict[int, list[np.ndarray]] = {}
    for wh in window_hours:
        win_len = wh * 60  # minutes
        step = slide_hours * 60
        windows: list[np.ndarray] = []
        for start in range(0, len(feature_df) - win_len + 1, step):
            chunk = feature_df.iloc[start : start + win_len].values
            if np.isnan(chunk).sum() / chunk.size < 0.3:  # skip if >30 % NaN
                windows.append(chunk)
        results[wh] = windows
        log.info("Window %dh: %d windows from %d rows", wh, len(windows), len(feature_df))
    return results


# ---------------------------------------------------------------------------
# Per-dataset loaders
# ---------------------------------------------------------------------------


def _load_wfdb_record(record_path: Path) -> dict[str, Any]:
    """Load a WFDB record (`.hea` / `.dat`) and return signals + metadata."""
    try:
        import wfdb  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("Install `wfdb` to load PhysioNet waveform data: pip install wfdb")

    record_name = str(record_path.with_suffix(""))
    record = wfdb.rdrecord(record_name)
    sig_names = record.sig_name
    fs = record.fs
    signals = record.p_signal  # shape (n_samples, n_signals)
    return {"signals": signals, "sig_names": sig_names, "fs": fs}


def load_mimic_waveform(local_dir: Path) -> pd.DataFrame:
    """Load MIMIC-III/IV waveform records into a single DataFrame.

    Selects ECG (MLII or II) and PPG (PLETH) channels when available.
    """
    all_records: list[pd.DataFrame] = []
    header_files = list(local_dir.rglob("*.hea"))[:200]  # cap for initial training

    for hf in header_files:
        try:
            rec = _load_wfdb_record(hf)
        except Exception:
            log.warning("Skipping unreadable record: %s", hf)
            continue

        fs = rec["fs"]
        names = rec["sig_names"]
        sigs = rec["signals"]
        time_index = pd.date_range("2000-01-01", periods=len(sigs), freq=f"{1_000 / fs:.0f}ms")

        df = pd.DataFrame(index=time_index)
        for i, name in enumerate(names):
            lower = name.lower()
            if "ecg" in lower or lower in ("mlii", "ii"):
                df["ecg"] = sigs[:, i]
            elif "pleth" in lower or "ppg" in lower:
                df["ppg"] = sigs[:, i]
            elif "abp" in lower or "arterial" in lower:
                df["abp"] = sigs[:, i]

        if not df.empty:
            df = df.resample("1min").mean()
            all_records.append(df)

    if not all_records:
        log.warning("No MIMIC waveform records loaded from %s", local_dir)
        return pd.DataFrame()

    combined = pd.concat(all_records, axis=0)
    log.info("Loaded %d MIMIC waveform rows", len(combined))
    return combined


def load_cves(local_dir: Path) -> pd.DataFrame:
    """Load CVES dataset (ECG + accel + BP)."""
    try:
        import wfdb
    except ImportError:
        raise ImportError("Install `wfdb`: pip install wfdb")

    all_dfs: list[pd.DataFrame] = []
    for hea in list(local_dir.rglob("*.hea"))[:120]:
        try:
            rec = wfdb.rdrecord(str(hea.with_suffix("")))
        except Exception:
            continue
        fs = rec.fs
        names = rec.sig_name
        sigs = rec.p_signal
        idx = pd.date_range("2000-01-01", periods=len(sigs), freq=f"{1_000 / fs:.0f}ms")
        df = pd.DataFrame(index=idx)
        for i, n in enumerate(names):
            ln = n.lower()
            if "ecg" in ln:
                df["ecg"] = sigs[:, i]
            elif "acc" in ln or "accel" in ln:
                if "x" not in df:
                    df["accel_x"] = sigs[:, i]
                elif "y" not in df:
                    df["accel_y"] = sigs[:, i]
                else:
                    df["accel_z"] = sigs[:, i]
            elif "abp" in ln or "bp" in ln:
                df["abp"] = sigs[:, i]
        if not df.empty:
            all_dfs.append(df.resample("1min").mean())

    return pd.concat(all_dfs, axis=0) if all_dfs else pd.DataFrame()


def load_sleep_accel(local_dir: Path) -> pd.DataFrame:
    """Load Apple Watch sleep-accel dataset (accel + HR)."""
    csvs = list(local_dir.rglob("*.csv"))
    if not csvs:
        log.warning("No CSV files in %s", local_dir)
        return pd.DataFrame()

    frames = []
    for csv in csvs:
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        # Normalise column names
        df.columns = [c.strip().lower() for c in df.columns]
        # Attempt to find time, HR, accel columns
        time_col = next((c for c in df.columns if "time" in c or "date" in c), None)
        hr_col = next((c for c in df.columns if "hr" in c or "heart" in c or "ppg" in c), None)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.set_index(time_col).sort_index()
        if hr_col:
            df = df.rename(columns={hr_col: "hr"})
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=0).resample("1min").mean()
    return combined


def load_mmash(local_dir: Path) -> pd.DataFrame:
    """Load MMASH dataset (RR intervals + accel)."""
    all_dfs: list[pd.DataFrame] = []
    for csv in local_dir.rglob("*.csv"):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        df.columns = [c.strip().lower() for c in df.columns]
        time_col = next((c for c in df.columns if "time" in c), None)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.set_index(time_col).sort_index()
        all_dfs.append(df.resample("1min").mean())

    return pd.concat(all_dfs, axis=0) if all_dfs else pd.DataFrame()


def load_generic_physionet(local_dir: Path) -> pd.DataFrame:
    """Generic loader: try WFDB first, fall back to CSV."""
    if list(local_dir.rglob("*.hea")):
        try:
            import wfdb
        except ImportError:
            pass
        else:
            all_dfs: list[pd.DataFrame] = []
            for hea in list(local_dir.rglob("*.hea"))[:100]:
                try:
                    rec = wfdb.rdrecord(str(hea.with_suffix("")))
                except Exception:
                    continue
                fs = rec.fs
                idx = pd.date_range("2000-01-01", periods=len(rec.p_signal), freq=f"{1_000 / fs:.0f}ms")
                df = pd.DataFrame(rec.p_signal, index=idx, columns=rec.sig_name)
                all_dfs.append(df.resample("1min").mean())
            if all_dfs:
                return pd.concat(all_dfs, axis=0)

    # Fallback: CSVs
    csvs = list(local_dir.rglob("*.csv"))
    if csvs:
        frames = [pd.read_csv(c) for c in csvs[:50]]
        combined = pd.concat(frames, axis=0)
        combined.columns = [c.strip().lower() for c in combined.columns]
        return combined.resample("1min").mean() if "time" in combined.columns else combined

    return pd.DataFrame()


LOADERS = {
    "mimic3_waveform": load_mimic_waveform,
    "mimic4_waveform": load_mimic_waveform,
    "cves": load_cves,
    "sleep_accel": load_sleep_accel,
    "mmash": load_mmash,
}


def load_dataset(name: str) -> pd.DataFrame:
    """Dispatch to the appropriate loader for *name*."""
    local_dir = DATASETS[name]["local_dir"]
    if not local_dir.exists():
        raise FileNotFoundError(f"Dataset '{name}' not downloaded yet: {local_dir}")
    loader = LOADERS.get(name, load_generic_physionet)
    return loader(local_dir)


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------


def extract_features_per_window(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-minute feature vectors from raw signal DataFrame.

    Expected columns (any subset): ``ecg``, ``ppg``, ``abp``, ``hr``,
    ``accel_x``, ``accel_y``, ``accel_z``, ``step_count``.
    """
    out = pd.DataFrame(index=df.index)

    # --- Derive RR intervals from ECG or PPG ---
    rr_col: np.ndarray | None = None
    if "ecg" in df.columns:
        ecg_clean = bandpass_filter(df["ecg"].fillna(0).values, fs=1.0, low=0.5, high=40.0)
        peaks, _ = scipy_signal.find_peaks(ecg_clean, distance=30, height=np.std(ecg_clean) * 0.5)
        if len(peaks) > 2:
            rr_col = np.diff(peaks).astype(float) * 1000.0  # ms
    elif "ppg" in df.columns:
        ppg_clean = bandpass_filter(df["ppg"].fillna(0).values, fs=1.0, low=0.5, high=8.0)
        peaks, _ = scipy_signal.find_peaks(ppg_clean, distance=20, height=np.std(ppg_clean) * 0.3)
        if len(peaks) > 2:
            rr_col = np.diff(peaks).astype(float) * 1000.0

    # --- HRV features (computed per 5-min sub-window, then aggregated) ---
    hrv_prefixes = {k: [] for k in HRV_FEATURES}
    sub_win = 5  # minutes
    for start in range(0, len(df) - sub_win, sub_win):
        if rr_col is not None and len(rr_col) > start and len(rr_col) < start + sub_win * 4:
            chunk_rr = rr_col[start : start + sub_win * 4]
        elif rr_col is not None and len(rr_col) > sub_win:
            chunk_rr = rr_col[start : start + min(sub_win * 4, len(rr_col) - start)]
        else:
            chunk_rr = np.array([800.0])  # placeholder normal RR

        hrv = compute_hrv_features(chunk_rr)
        for k, v in hrv.items():
            hrv_prefixes[k].append(v)

    # Pad/trim to match df length
    for k in hrv_prefixes:
        vals = hrv_prefixes[k]
        if len(vals) < len(df):
            vals.extend([0.0] * (len(df) - len(vals)))
        else:
            vals = vals[: len(df)]
        out[k] = vals

    # --- Activity features ---
    accel_cols = [c for c in df.columns if "accel" in c]
    if len(accel_cols) >= 3:
        accel_df = df[accel_cols].rename(columns=lambda c: c.split("_")[-1])
        for col in ["x", "y", "z"]:
            if col not in accel_df.columns:
                accel_df[col] = 0.0
        act_feats = compute_activity_features(accel_df, hr=df.get("hr"))
        for k, v in act_feats.items():
            out[k] = v
    else:
        for k in ("total_activity", "mean_activity", "std_activity", "day_mean_activity",
                   "night_mean_activity", "day_night_ratio", "activity_entropy",
                   "peak_activity_hour", "activity_slope", "night_hr_mean",
                   "night_hr_spike", "hr_slope"):
            out[k] = 0.0

    # --- Simple aggregate features ---
    if "hr" in df.columns:
        out["hr_mean"] = df["hr"].rolling(5, min_periods=1).mean()
        out["hr_std"] = df["hr"].rolling(5, min_periods=1).std().fillna(0)
        out["hr_slope"] = df["hr"].rolling(10, min_periods=2).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0.0, raw=True
        )
    elif "ppg" in df.columns:
        out["hr_mean"] = df["ppg"].rolling(5, min_periods=1).mean()
        out["hr_std"] = df["ppg"].rolling(5, min_periods=1).std().fillna(0)
        out["hr_slope"] = 0.0
    else:
        out["hr_mean"] = 0.0
        out["hr_std"] = 0.0
        out["hr_slope"] = 0.0

    # --- Missing-data flags ---
    out["pct_missing"] = df.isna().mean(axis=1).values
    out["wear_time"] = (1.0 - out["pct_missing"]).values

    return out


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalise_features(
    train_df: pd.DataFrame,
    *other_dfs: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    """Z-score normalise using training-set statistics. Returns all dfs."""
    means = train_df.mean()
    stds = train_df.std().replace(0, 1)

    normed = [((d - means) / stds).fillna(0) for d in (train_df, *other_dfs)]
    return tuple(normed)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def prepare_dataset(
    dataset_names: list[str] | None = None,
    window_hours: list[int] = WINDOW_HOURS,
) -> dict[str, Any]:
    """Full pipeline: load → clean → feature-engineer → window → split.

    Returns a dict with keys:
        ``"windows"`` : ``{wh: [np.ndarray, ...]}``
        ``"labels"``  : ``{wh: np.ndarray}``  (placeholder zeros until real labels exist)
        ``"feature_names"`` : list[str]
        ``"n_time_features"`` : int
    """
    if dataset_names is None:
        dataset_names = list(DATASETS.keys())

    # 1. Load and concatenate
    frames: list[pd.DataFrame] = []
    for name in dataset_names:
        log.info("Loading dataset: %s", name)
        try:
            df = load_dataset(name)
            if not df.empty:
                frames.append(df)
        except Exception:
            log.exception("Failed to load %s", name)

    if not frames:
        raise RuntimeError("No data loaded from any dataset.")

    combined = pd.concat(frames, axis=0).sort_index()
    log.info("Combined raw data: %d rows, %d columns", len(combined), len(combined.columns))

    # 2. Feature engineering
    features = extract_features_per_window(combined)
    features = features.replace([np.inf, -np.inf], 0).fillna(0)
    log.info("Feature matrix: %d rows × %d cols", len(features), len(features.columns))

    # 3. Windowing
    windows = build_windows(features, window_hours=window_hours)

    # 4. Pad / truncate windows to fixed length and stack
    result_windows: dict[int, np.ndarray] = {}
    result_labels: dict[int, np.ndarray] = {}
    for wh, wins in windows.items():
        padded = []
        for w in wins:
            if len(w) < WINDOW_LENGTH_MINUTES:
                pad = np.zeros((WINDOW_LENGTH_MINUTES - len(w), w.shape[1]))
                w = np.vstack([w, pad])
            elif len(w) > WINDOW_LENGTH_MINUTES:
                w = w[:WINDOW_LENGTH_MINUTES]
            padded.append(w)
        arr = np.stack(padded)  # (n_windows, WINDOW_LENGTH_MINUTES, n_features)
        result_windows[wh] = arr
        result_labels[wh] = np.zeros(arr.shape[0], dtype=np.float32)  # placeholder
        log.info("Window %dh tensor: %s", wh, arr.shape)

    # 5. Save to disk
    for wh, arr in result_windows.items():
        out_path = PROCESSED_DIR / f"windows_{wh}h.npz"
        np.savez_compressed(out_path, features=arr, labels=result_labels[wh])
        log.info("Saved %s", out_path)

    return {
        "windows": result_windows,
        "labels": result_labels,
        "feature_names": list(features.columns),
        "n_time_features": features.shape[1],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess downloaded datasets into model-ready windows.")
    parser.add_argument("--datasets", nargs="*", help="Specific dataset keys (default: all).")
    parser.add_argument("--window-hours", nargs="+", type=int, default=WINDOW_HOURS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    prepare_dataset(args.datasets, args.window_hours)
