"""Preprocessing orchestrator — connects loading, processing, and windowing.

This module is a thin orchestrator that:

1. Dispatches to :mod:`waveform_loader` for raw waveform loading.
2. Dispatches to :mod:`feature_extraction` for feature computation.
3. Slices the minute-level feature DataFrame into training windows.
4. Saves windowed ``.npz`` files for the training pipeline.

Architecture
------------
The previous ``preprocess.py`` was a 662-line monolith that mixed waveform
loading, signal processing, HRV computation, activity features, windowing,
and normalisation in a single file.  This rewrite delegates each concern to
dedicated modules and preserves only the orchestration logic here.

The public API (``prepare_dataset``) is backward-compatible with ``train.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DATASETS,
    HRV_FEATURES,
    N_TIME_FEATURES,
    PROCESSED_DIR,
    SLIDE_HOURS,
    WINDOW_HOURS,
    WINDOW_LENGTH_MINUTES,
)
from .feature_extraction import extract_features_from_records
from .waveform_loader import WaveformRecord, load_dataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def build_windows(
    feature_df: pd.DataFrame,
    window_hours: list[int] = WINDOW_HOURS,
    slide_hours: int = SLIDE_HOURS,
) -> dict[int, list[np.ndarray]]:
    """Slice a minute-level feature DataFrame into sliding windows.

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
            if np.isnan(chunk).sum() / chunk.size < 0.3:  # skip if >30% NaN
                windows.append(chunk)
        results[wh] = windows
        log.info("Window %dh: %d windows from %d rows", wh, len(windows), len(feature_df))
    return results


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalise_features(
    train_df: pd.DataFrame,
    *other_dfs: pd.DataFrame,
) -> tuple[pd.DataFrame, ...]:
    """Z-score normalise using training-set statistics.  Returns all dfs."""
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
    """Full pipeline: load → filter → detect beats → HRV → features → window.

    Returns a dict with keys:
        ``"windows"`` : ``{wh: [np.ndarray, ...]}``
        ``"labels"``  : ``{wh: np.ndarray}``  (placeholder zeros)
        ``"feature_names"`` : list[str]
        ``"n_time_features"`` : int
    """
    if dataset_names is None:
        dataset_names = list(DATASETS.keys())

    # 1. Load raw waveforms (preserves native sampling rate)
    all_records: list[WaveformRecord] = []
    for name in dataset_names:
        log.info("Loading dataset: %s", name)
        try:
            info = DATASETS[name]
            local_dir = info["local_dir"]
            if not local_dir.exists():
                log.warning("Dataset '%s' not downloaded yet: %s", name, local_dir)
                continue
            records = load_dataset(name, local_dir)
            all_records.extend(records)
        except Exception:
            log.exception("Failed to load %s", name)

    if not all_records:
        raise RuntimeError("No data loaded from any dataset.")

    log.info("Loaded %d total records across %d datasets", len(all_records), len(dataset_names))

    # 2. Extract minute-level features from raw waveforms
    features = extract_features_from_records(all_records)

    if features.empty:
        raise RuntimeError("Feature extraction produced an empty DataFrame.")

    features = features.replace([np.inf, -np.inf], 0).fillna(0)
    log.info("Feature matrix: %d rows x %d cols", len(features), len(features.columns))

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

        if not padded:
            log.warning("No valid windows for %dh", wh)
            continue

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
# Backward-compatible re-exports
#
# The old preprocess.py exposed these names at module level.  Existing code
# (including tests and notebooks) may import them directly.
# ---------------------------------------------------------------------------

from .signal_processing import bandpass_filter as bandpass_filter  # noqa: F811, E402
from .hrv import compute_hrv_features as compute_hrv_features  # noqa: F811, E402
from .feature_extraction import (
    extract_features_from_record as extract_features_from_window,  # noqa: F811, E402
)


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
