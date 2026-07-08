"""Raw waveform loaders that preserve native sampling rates.

Each loader returns a :class:`WaveformRecord` containing raw numpy arrays
at the device's native sampling frequency.  No resampling or averaging is
performed -- that responsibility belongs to the downstream signal-processing
and feature-extraction modules.

Architectural rationale
-----------------------
The previous implementation resampled ECG/PPG to 1-minute averages *before*
heartbeat detection, which destroyed waveform morphology and produced
meaningless RR intervals.  By returning raw waveforms, the pipeline can
apply bandpass filtering and R-peak detection at the correct temporal
resolution (typically 125--500 Hz for ICU monitors, 256 Hz for Holter).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------


@dataclass
class WaveformRecord:
    """Container for raw physiological waveforms from a single recording.

    Attributes
    ----------
    ecg : np.ndarray | None
        Raw ECG signal (voltage or arbitrary units).
    ppg : np.ndarray | None
        Raw PPG / photoplethysmogram signal.
    abp : np.ndarray | None
        Raw arterial blood pressure waveform.
    accel : np.ndarray | None
        Accelerometer data.  Shape ``(n_samples, 3)`` for 3-axis,
        or ``(n_samples,)`` for single-axis.
    hr : np.ndarray | None
        Heart-rate series if provided directly (e.g. from pulse ox).
    fs : float
        Sampling frequency in Hz.  All signals share this rate.
    metadata : dict
        Arbitrary metadata (patient ID, dataset name, record path, etc.).
    """

    ecg: np.ndarray | None = None
    ppg: np.ndarray | None = None
    abp: np.ndarray | None = None
    accel: np.ndarray | None = None
    hr: np.ndarray | None = None
    fs: float = 125.0
    metadata: dict = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Total recording duration in seconds."""
        for sig in (self.ecg, self.ppg, self.abp, self.hr):
            if sig is not None:
                return len(sig) / self.fs
        return 0.0

    @property
    def duration_minutes(self) -> float:
        return self.duration_seconds / 60.0

    def has_ecg(self) -> bool:
        return self.ecg is not None and len(self.ecg) > 0

    def has_ppg(self) -> bool:
        return self.ppg is not None and len(self.ppg) > 0

    def has_abp(self) -> bool:
        return self.abp is not None and len(self.abp) > 0

    def has_accel(self) -> bool:
        return self.accel is not None and len(self.accel) > 0


# ---------------------------------------------------------------------------
# WFDB loader (MIMIC-III, MIMIC-IV, CVES, PAF, AFDB)
# ---------------------------------------------------------------------------


def _load_wfdb_record(record_path: Path) -> dict[str, Any]:
    """Load a WFDB record (``.hea`` / ``.dat``) and return raw signals + metadata.

    Returns
    -------
    dict with keys: ``signals`` (ndarray), ``sig_names`` (list[str]),
    ``fs`` (float), ``signal_units`` (list[str]).
    """
    try:
        import wfdb  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("Install `wfdb` to load PhysioNet waveform data: pip install wfdb")

    record_name = str(record_path.with_suffix(""))
    record = wfdb.rdrecord(record_name)
    return {
        "signals": record.p_signal,
        "sig_names": record.sig_name,
        "fs": float(record.fs),
        "signal_units": [u.strip() for u in record.units] if hasattr(record, "units") else [],
    }


def _select_channel(names: list[str], signals: np.ndarray, target: str) -> tuple[np.ndarray | None, str]:
    """Select a signal channel by name pattern matching.

    Parameters
    ----------
    names : list of channel names from the WFDB header.
    signals : full signal array ``(n_samples, n_channels)``.
    target : one of ``"ecg"``, ``"ppg"``, ``"abp"``.

    Returns
    -------
    (channel_data, channel_name) or (None, "") if not found.
    """
    lower_names = [n.lower() for n in names]

    patterns: dict[str, list[str]] = {
        "ecg": ["ecg", "mlii", "ii", "i ", "v1", "v2", "v3", "v4", "v5", "v6"],
        "ppg": ["pleth", "ppg", "plethysmograph"],
        "abp": ["abp", "arterial", "blood pressure", "art"],
    }

    for pattern in patterns[target]:
        for i, ln in enumerate(lower_names):
            if pattern in ln:
                return signals[:, i], names[i]

    return None, ""


def _extract_accel_channels(names: list[str], signals: np.ndarray) -> np.ndarray | None:
    """Extract 3-axis accelerometer data if available."""
    accel_indices = []
    for i, n in enumerate(names):
        ln = n.lower()
        if any(k in ln for k in ("acc", "accel", "x", "y", "z")):
            accel_indices.append(i)

    if len(accel_indices) >= 3:
        return signals[:, accel_indices[:3]]
    elif len(accel_indices) == 1:
        return signals[:, accel_indices[0]]
    return None


def load_mimic_waveform(local_dir: Path) -> list[WaveformRecord]:
    """Load MIMIC-III/IV waveform records, preserving raw sampling rate.

    Selects ECG (MLII or II), PPG (PLETH), and ABP channels when available.
    Returns one :class:`WaveformRecord` per WFDB record.
    """
    try:
        import wfdb  # noqa: F811
    except ImportError:
        raise ImportError("Install `wfdb`: pip install wfdb")

    records: list[WaveformRecord] = []
    header_files = list(local_dir.rglob("*.hea"))[:200]

    for hf in header_files:
        try:
            rec = _load_wfdb_record(hf)
        except Exception:
            log.warning("Skipping unreadable record: %s", hf)
            continue

        fs = rec["fs"]
        names = rec["sig_names"]
        sigs = rec["signals"]

        ecg, _ = _select_channel(names, sigs, "ecg")
        ppg, _ = _select_channel(names, sigs, "ppg")
        abp, _ = _select_channel(names, sigs, "abp")
        accel = _extract_accel_channels(names, sigs)

        if ecg is None and ppg is None:
            log.debug("No ECG/PPG in %s, skipping", hf.name)
            continue

        records.append(WaveformRecord(
            ecg=ecg,
            ppg=ppg,
            abp=abp,
            accel=accel,
            fs=fs,
            metadata={"source": "mimic", "path": str(hf), "channels": names},
        ))

    log.info("Loaded %d MIMIC waveform records from %s", len(records), local_dir)
    return records


def load_cves(local_dir: Path) -> list[WaveformRecord]:
    """Load CVES dataset (ECG + 3-axis accel + ABP), raw sampling rate."""
    try:
        import wfdb  # noqa: F811
    except ImportError:
        raise ImportError("Install `wfdb`: pip install wfdb")

    records: list[WaveformRecord] = []
    for hea in list(local_dir.rglob("*.hea"))[:120]:
        try:
            rec = wfdb.rdrecord(str(hea.with_suffix("")))
        except Exception:
            continue

        fs = float(rec.fs)
        names = rec.sig_name
        sigs = rec.p_signal

        ecg, _ = _select_channel(names, sigs, "ecg")
        abp, _ = _select_channel(names, sigs, "abp")
        accel = _extract_accel_channels(names, sigs)

        if ecg is None:
            continue

        records.append(WaveformRecord(
            ecg=ecg,
            abp=abp,
            accel=accel,
            fs=fs,
            metadata={"source": "cves", "path": str(hea), "channels": names},
        ))

    log.info("Loaded %d CVES records from %s", len(records), local_dir)
    return records


def load_sleep_accel(local_dir: Path) -> list[WaveformRecord]:
    """Load Apple Watch sleep-accel dataset (HR + accel).

    These datasets typically provide pre-computed HR and accelerometer at
    lower sampling rates.  We preserve whatever rate is in the CSV.
    """
    csvs = list(local_dir.rglob("*.csv"))
    if not csvs:
        log.warning("No CSV files in %s", local_dir)
        return []

    records: list[WaveformRecord] = []
    for csv in csvs:
        try:
            import pandas as pd
            df = pd.read_csv(csv)
        except Exception:
            continue

        df.columns = [c.strip().lower() for c in df.columns]

        time_col = next((c for c in df.columns if "time" in c or "date" in c), None)
        hr_col = next((c for c in df.columns if "hr" in c or "heart" in c or "ppg" in c), None)
        accel_cols = [c for c in df.columns if "accel" in c or c in ("x", "y", "z")]

        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.set_index(time_col).sort_index()

        # Infer sampling rate from time index
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            median_diff = pd.Series(df.index).diff().median()
            fs = 1.0 / median_diff.total_seconds() if median_diff.total_seconds() > 0 else 1.0
        else:
            fs = 1.0

        hr = df[hr_col].values.astype(float) if hr_col else None
        accel = df[accel_cols].values.astype(float) if len(accel_cols) >= 3 else None

        if hr is None and accel is None:
            continue

        records.append(WaveformRecord(
            hr=hr,
            accel=accel,
            fs=fs,
            metadata={"source": "sleep_accel", "path": str(csv)},
        ))

    log.info("Loaded %d sleep-accel records from %s", len(records), local_dir)
    return records


def load_mmash(local_dir: Path) -> list[WaveformRecord]:
    """Load MMASH dataset (beat-to-beat RR intervals + 3-axis accel).

    MMASH provides pre-computed RR intervals rather than raw ECG.
    We store them in the ``hr`` field and note the format in metadata.
    """
    import pandas as pd

    records: list[WaveformRecord] = []
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

        rr_col = next((c for c in df.columns if "rr" in c or "interval" in c), None)
        accel_cols = [c for c in df.columns if "accel" in c or c in ("x", "y", "z")]

        hr = df[rr_col].values.astype(float) if rr_col else None
        accel = df[accel_cols].values.astype(float) if len(accel_cols) >= 3 else None

        if hr is None and accel is None:
            continue

        records.append(WaveformRecord(
            hr=hr,
            accel=accel,
            fs=1.0,
            metadata={"source": "mmash", "path": str(csv), "rr_intervals": rr_col is not None},
        ))

    log.info("Loaded %d MMASH records from %s", len(records), local_dir)
    return records


def load_generic_physionet(local_dir: Path) -> list[WaveformRecord]:
    """Generic loader: WFDB first, CSV fallback.  Preserves raw sampling rate."""
    # Try WFDB first
    hea_files = list(local_dir.rglob("*.hea"))
    if hea_files:
        try:
            import wfdb  # noqa: F811
        except ImportError:
            pass
        else:
            records: list[WaveformRecord] = []
            for hea in hea_files[:100]:
                try:
                    rec = wfdb.rdrecord(str(hea.with_suffix("")))
                except Exception:
                    continue

                fs = float(rec.fs)
                names = rec.sig_name
                sigs = rec.p_signal

                ecg, _ = _select_channel(names, sigs, "ecg")
                ppg, _ = _select_channel(names, sigs, "ppg")
                abp, _ = _select_channel(names, sigs, "abp")
                accel = _extract_accel_channels(names, sigs)

                records.append(WaveformRecord(
                    ecg=ecg,
                    ppg=ppg,
                    abp=abp,
                    accel=accel,
                    fs=fs,
                    metadata={"source": "generic_physionet", "path": str(hea), "channels": names},
                ))

            if records:
                log.info("Loaded %d generic PhysioNet records from %s", len(records), local_dir)
                return records

    # Fallback: CSVs
    csvs = list(local_dir.rglob("*.csv"))
    if csvs:
        import pandas as pd

        records = []
        for csv in csvs[:50]:
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue

            df.columns = [c.strip().lower() for c in df.columns]
            hr_col = next((c for c in df.columns if "hr" in c or "heart" in c), None)

            hr = df[hr_col].values.astype(float) if hr_col else None
            if hr is not None:
                records.append(WaveformRecord(
                    hr=hr,
                    fs=1.0,
                    metadata={"source": "generic_csv", "path": str(csv)},
                ))

        log.info("Loaded %d generic CSV records from %s", len(records), local_dir)
        return records

    return []


# ---------------------------------------------------------------------------
# Loader dispatch
# ---------------------------------------------------------------------------

LOADERS: dict[str, callable] = {
    "mimic3_waveform": load_mimic_waveform,
    "mimic4_waveform": load_mimic_waveform,
    "cves": load_cves,
    "sleep_accel": load_sleep_accel,
    "mmash": load_mmash,
}


def load_dataset(name: str, local_dir: Path) -> list[WaveformRecord]:
    """Dispatch to the appropriate loader for dataset *name*.

    Returns a list of :class:`WaveformRecord` objects with raw waveforms.
    """
    loader = LOADERS.get(name, load_generic_physionet)
    return loader(local_dir)
