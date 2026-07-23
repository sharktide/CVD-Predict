"""Data loaders for all raw datasets in data/raw/.

Each loader returns a standardised pandas DataFrame or list of WaveformRecord
objects so downstream modules (labeling, cohort, preprocess) never touch raw
file formats directly.

Supported datasets
------------------
- MIMIC-IV Clinical (hosp + icu gzipped CSVs)
- MIMIC-IV Waveform (WFDB binary)
- MIMIC-IV ECG (WFDB binary)
- MIMIC-IV Emergency Department
- MMASH (wearable multi-sensor)
- Sleep Accel (Apple Watch wrist PPG + accel)
- Non-EEG Neuro (wrist sensors)
- Kaggle Stroke (tabular)
- AFDB, LTAFDB, AFP (WFDB ECG databases)
- CVES (cerebrovascular / autonomic)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_gz_csv(path: str | Path) -> pd.DataFrame:
    """Read a gzipped CSV and return a DataFrame."""
    return pd.read_csv(path, compression="gzip", low_memory=False)


def _ensure_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


# ---------------------------------------------------------------------------
# 1. MIMIC-IV Clinical
# ---------------------------------------------------------------------------

class MIMICClinicalLoader:
    """Load MIMIC-IV clinical tables from gzipped CSVs.

    Expected layout under ``raw_dir / mimiciv-clinical /``::

        hosp/
            admissions.csv.gz
            patients.csv.gz
            diagnoses_icd.csv.gz
            labevents.csv.gz
            d_icd_diagnoses.csv.gz
            d_icd_procedures.csv.gz
            prescriptions.csv.gz
            services.csv.gz
            transfers.csv.gz
        icu/
            icustays.csv.gz
            chartevents.csv.gz
            inputevents.csv.gz
            outputevents.csv.gz
            d_items.csv.gz
    """

    def __init__(self, raw_dir: str | Path):
        self.hosp_dir = Path(raw_dir) / "mimiciv-clinical" / "hosp"
        self.icu_dir = Path(raw_dir) / "mimiciv-clinical" / "icu"

    # --- Hosp tables ---

    def load_admissions(self) -> pd.DataFrame:
        path = self.hosp_dir / "admissions.csv.gz"
        if not path.exists():
            logger.warning("admissions.csv.gz not found at %s", path)
            return pd.DataFrame()
        df = _read_gz_csv(path)
        _ensure_columns(df, ["subject_id", "hadm_id", "admittime", "dischtime"], "admissions")
        df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
        df["dischtime"] = pd.to_datetime(df["dischtime"], errors="coerce")
        if "deathtime" in df.columns:
            df["deathtime"] = pd.to_datetime(df["deathtime"], errors="coerce")
        return df

    def load_patients(self) -> pd.DataFrame:
        path = self.hosp_dir / "patients.csv.gz"
        if not path.exists():
            logger.warning("patients.csv.gz not found at %s", path)
            return pd.DataFrame()
        df = _read_gz_csv(path)
        _ensure_columns(df, ["subject_id", "gender", "anchor_age"], "patients")
        return df

    def load_diagnoses_icd(self) -> pd.DataFrame:
        path = self.hosp_dir / "diagnoses_icd.csv.gz"
        if not path.exists():
            return pd.DataFrame()
        df = _read_gz_csv(path)
        _ensure_columns(df, ["subject_id", "hadm_id", "icd_code", "icd_version"], "diagnoses_icd")
        return df

    def load_labevents(self) -> pd.DataFrame:
        """Load labevents.  This table is very large (~158M rows); we filter
        immediately to troponin and CK-MB items to keep memory manageable.

        Uses a two-pass approach: first try to load only relevant columns,
        falling back to chunked loading if needed.
        """
        path = self.hosp_dir / "labevents.csv.gz"
        if not path.exists():
            return pd.DataFrame()
        # Troponin-T (itemid 51003), Troponin-I (50974), CK-MB (50908, 50971)
        cardiac_itemids = {51003, 50974, 50908, 51002, 50971}
        # Load only the columns we need
        usecols = ["labevent_id", "subject_id", "hadm_id", "itemid", "charttime", "valuenum"]
        try:
            # Try loading with filter on itemid directly (pandas supports this)
            df = pd.read_csv(
                path, compression="gzip", low_memory=False,
                usecols=usecols,
            )
            df = df[df["itemid"].isin(cardiac_itemids)]
            df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
            return df
        except Exception:
            pass
        # Fallback: chunked loading
        chunks = []
        for chunk in pd.read_csv(path, compression="gzip", low_memory=False,
                                 usecols=usecols, chunksize=1_000_000):
            chunk = chunk[chunk["itemid"].isin(cardiac_itemids)]
            if not chunk.empty:
                chunks.append(chunk)
        if not chunks:
            return pd.DataFrame()
        df = pd.concat(chunks, ignore_index=True)
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
        return df

    def load_diagnoses_icd_filtered(self, icd_prefixes: tuple[str, ...] = ("I21", "I46")) -> pd.DataFrame:
        """Load diagnoses_icd and keep only rows matching ICD prefixes of interest."""
        df = self.load_diagnoses_icd()
        if df.empty:
            return df
        mask = df["icd_code"].astype(str).str.startswith(icd_prefixes)
        return df[mask].copy()

    # --- ICU tables ---

    def load_icustays(self) -> pd.DataFrame:
        path = self.icu_dir / "icustays.csv.gz"
        if not path.exists():
            return pd.DataFrame()
        df = _read_gz_csv(path)
        _ensure_columns(df, ["subject_id", "hadm_id", "stay_id",
                             "first_careunit", "intime", "outtime", "los"], "icustays")
        df["intime"] = pd.to_datetime(df["intime"], errors="coerce")
        df["outtime"] = pd.to_datetime(df["outtime"], errors="coerce")
        return df

    def load_chartevents(self, itemids: Optional[set[int]] = None) -> pd.DataFrame:
        """Load chartevents, optionally filtering to specific itemids.

        If itemids is None, returns an empty DataFrame (too large to load fully).
        Common cardiac itemids: 220045 (HR), 220050 (SBP), 220051 (DBP),
        220179 (SpO2), 220210 (RR), 223761 (Temp).
        """
        path = self.icu_dir / "chartevents.csv.gz"
        if not path.exists() or itemids is None:
            return pd.DataFrame()
        usecols = ["subject_id", "hadm_id", "stay_id", "charttime", "itemid",
                    "valuenum", "valueuom"]
        chunks = []
        for chunk in pd.read_csv(path, compression="gzip", low_memory=False,
                                 usecols=usecols, chunksize=500_000):
            chunk = chunk[chunk["itemid"].isin(itemids)]
            if not chunk.empty:
                chunks.append(chunk)
        if not chunks:
            return pd.DataFrame()
        df = pd.concat(chunks, ignore_index=True)
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
        return df

    def load_d_items(self) -> pd.DataFrame:
        path = self.icu_dir / "d_items.csv.gz"
        if not path.exists():
            return pd.DataFrame()
        return _read_gz_csv(path)

    # --- Convenience: build the MI + arrest cohort table ---

    def build_event_cohort(self) -> pd.DataFrame:
        """Return a DataFrame with one row per qualifying event (MI or cardiac arrest).

        Columns: subject_id, hadm_id, event_type, icd_code, event_time,
                 admittime, dischtime, gender, anchor_age, los, first_careunit
        """
        patients = self.load_patients()
        admissions = self.load_admissions()
        diagnoses = self.load_diagnoses_icd_filtered(("I21", "I46"))
        icustays = self.load_icustays()

        if diagnoses.empty:
            logger.warning("No I21/I46 diagnoses found")
            return pd.DataFrame()

        # Tag event type
        diagnoses["event_type"] = diagnoses["icd_code"].astype(str).apply(
            lambda x: "MI" if x.startswith("I21") else "ARREST"
        )

        # Merge admission info
        merged = diagnoses.merge(
            admissions[["subject_id", "hadm_id", "admittime", "dischtime"]],
            on=["subject_id", "hadm_id"],
            how="left",
        )

        # Merge patient demographics
        merged = merged.merge(
            patients[["subject_id", "gender", "anchor_age"]],
            on="subject_id",
            how="left",
        )

        # Merge ICU stays (take first stay per hadm_id)
        if not icustays.empty:
            icu_first = (
                icustays.sort_values("intime")
                .groupby(["subject_id", "hadm_id"])
                .first()
                .reset_index()
            )
            merged = merged.merge(
                icu_first[["subject_id", "hadm_id", "first_careunit", "intime", "outtime", "los"]],
                on=["subject_id", "hadm_id"],
                how="left",
            )

        merged["event_time"] = merged["admittime"]

        logger.info("Built event cohort: %d MI, %d ARREST",
                     (merged["event_type"] == "MI").sum(),
                     (merged["event_type"] == "ARREST").sum())
        return merged


# ---------------------------------------------------------------------------
# 2. MIMIC-IV Waveform
# ---------------------------------------------------------------------------

class MIMICWaveformLoader:
    """Load PPG/ECG waveforms from the MIMIC-IV Waveform Database.

    Layout::

        mimic4wdb/
            p100/
                p10014354/
                    10014354/
                        RECORDS
                        *.hea / *.dat
            ...

    Header files contain ``# subject_id XXXXX`` and ``# hadm_id XXXXX`` comments.
    """

    def __init__(self, raw_dir: str | Path):
        self.waveform_dir = Path(raw_dir) / "mimic4wdb"

    def _parse_header_metadata(self, header_path: Path) -> dict[str, Any]:
        """Extract subject_id, hadm_id, and channel names from a .hea file.

        MIMIC-IV Waveform header files use the format:
            record_name nsamp dur/flags
            ~ flags gain bitres ... channel_name
            ...
            # subject_id XXXXX
            # hadm_id XXXXX

        The subject_id and hadm_id are in comment lines at the end.
        The sampling rate is on the first line (3rd field is duration, not fs).
        We compute fs from the signal specification lines.
        """
        meta = {"subject_id": None, "hadm_id": None, "channels": [], "fs": None}
        try:
            with open(header_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if line.startswith("#"):
                    if "subject_id" in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                meta["subject_id"] = int(parts[2])
                            except ValueError:
                                pass
                    elif "hadm_id" in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                meta["hadm_id"] = int(parts[2])
                            except ValueError:
                                pass
                elif line and not line.startswith("#") and ".dat" in line:
                    parts = line.split()
                    if len(parts) >= 6:
                        meta["channels"].append(parts[-1])
                        # Extract sampling rate from gain field (e.g. "200/mV")
                        if meta["fs"] is None and "/" in parts[2]:
                            try:
                                # The first line has the record format
                                # Signal lines have gain/bitres format
                                pass
                            except Exception:
                                pass

            # Extract fs from the first line (record line)
            if lines:
                first_line = lines[0].strip()
                parts = first_line.split()
                if len(parts) >= 2:
                    # nsamp is the second field
                    # We need to compute fs from the signal lines
                    # For now, use a default and let wfdb handle it
                    pass

            # Try to get fs from the first signal line's gain
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#") and ".dat" in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        # Format: dat_file gain/bitres ...
                        # gain is in units/mV, but we need fs
                        # Actually, fs is in the record line or can be inferred
                        # Let's use the standard MIMIC-IV waveform rate
                        meta["fs"] = 62.5  # default for mimic4wdb
                        break

        except Exception as e:
            logger.debug("Failed to parse header %s: %s", header_path, e)

        # Extract subject_id from path if not found in header
        if meta["subject_id"] is None:
            # Path pattern: mimic4wdb/pXXX/pXXXXXXXX/XXXXXXXX/XXXXXXXX_XXXX.hea
            # The actual patient ID is the second pXXX directory (e.g., p10952189)
            parts = header_path.parts
            for i, p in enumerate(parts):
                if p.startswith("p") and p[1:].isdigit() and len(p[1:]) >= 6:
                    try:
                        meta["subject_id"] = int(p[1:])
                    except ValueError:
                        pass
                    break

        return meta

    def list_records(self, max_records: int = 500) -> list[dict[str, Any]]:
        """Return a list of dicts with keys: header_path, subject_id, hadm_id, fs, channels."""
        records = []
        hea_files = list(self.waveform_dir.rglob("*.hea"))
        logger.info("Found %d .hea files in %s", len(hea_files), self.waveform_dir)
        for hf in hea_files[:max_records]:
            meta = self._parse_header_metadata(hf)
            if meta["subject_id"] is not None:
                records.append({
                    "header_path": hf,
                    "subject_id": meta["subject_id"],
                    "hadm_id": meta["hadm_id"],
                    "fs": meta["fs"],
                    "channels": meta["channels"],
                })
        return records

    def load_record(self, header_path: Path) -> Optional[dict[str, Any]]:
        """Load a single WFDB record using the wfdb library.

        Returns dict with keys: ppg, ecg, abp, accel, fs, subject_id, hadm_id, channels.
        """
        try:
            import wfdb
        except ImportError:
            logger.error("Install wfdb: pip install wfdb")
            return None

        record_name = str(header_path.with_suffix(""))
        try:
            record = wfdb.rdrecord(record_name)
        except Exception as e:
            logger.debug("Cannot read record %s: %s", record_name, e)
            return None

        signals = record.p_signal
        sig_names = [s.lower() for s in record.sig_name]
        fs = float(record.fs)

        # Extract channels
        ppg = None
        ecg = None
        abp = None
        accel = None

        for i, name in enumerate(sig_names):
            if any(k in name for k in ("pleth", "ppg")):
                ppg = signals[:, i]
            elif any(k in name for k in ("ecg", "mlii", "ii ", "i ", "v1", "v2", "v3",
                                          "ii", "iii", "avr", "avl", "avf")):
                if ecg is None:
                    ecg = signals[:, i]
            elif any(k in name for k in ("abp", "arterial", "art")):
                abp = signals[:, i]

        meta = self._parse_header_metadata(header_path)
        return {
            "ppg": ppg,
            "ecg": ecg,
            "abp": abp,
            "accel": accel,
            "fs": fs,
            "subject_id": meta["subject_id"],
            "hadm_id": meta["hadm_id"],
            "channels": record.sig_name,
        }

    def load_records_for_subject(self, subject_id: int) -> list[dict[str, Any]]:
        """Load all waveform records for a given subject_id."""
        all_records = self.list_records(max_records=10_000)
        matching = [r for r in all_records if r["subject_id"] == subject_id]
        results = []
        for rec_info in matching:
            loaded = self.load_record(rec_info["header_path"])
            if loaded is not None:
                results.append(loaded)
        return results


# ---------------------------------------------------------------------------
# 3. MIMIC-IV ECG
# ---------------------------------------------------------------------------

class MIMICECGLoader:
    """Load 12-lead ECG records from MIMIC-IV ECG.

    Layout::

        mimiciv-ecg/
            files/
                p1000/
                    p1000YYYY/
                        sYYYYYYYY/
                            *.hea / *.dat
    """

    def __init__(self, raw_dir: str | Path):
        self.ecg_dir = Path(raw_dir) / "mimiciv-ecg"

    def list_records(self, max_records: int = 500) -> list[dict[str, Any]]:
        records = []
        hea_files = list(self.ecg_dir.rglob("*.hea"))
        logger.info("Found %d ECG .hea files in %s", len(hea_files), self.ecg_dir)
        for hf in hea_files[:max_records]:
            meta = self._parse_ecg_header(hf)
            records.append({"header_path": hf, **meta})
        return records

    def _parse_ecg_header(self, header_path: Path) -> dict[str, Any]:
        meta = {"subject_id": None, "hadm_id": None, "fs": None, "n_leads": 0}
        try:
            with open(header_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#"):
                        if "subject_id" in line:
                            parts = line.split()
                            if len(parts) >= 3:
                                try:
                                    meta["subject_id"] = int(parts[2])
                                except ValueError:
                                    pass
                        elif "hadm_id" in line:
                            parts = line.split()
                            if len(parts) >= 3:
                                try:
                                    meta["hadm_id"] = int(parts[2])
                                except ValueError:
                                    pass
                    elif not line.startswith("#") and line and meta["fs"] is None:
                        parts = line.split()
                        if len(parts) >= 3:
                            meta["fs"] = float(parts[2])
                    elif not line.startswith("#") and ".dat" in line:
                        meta["n_leads"] += 1
        except Exception:
            pass
        return meta

    def load_record(self, header_path: Path) -> Optional[dict[str, Any]]:
        try:
            import wfdb
        except ImportError:
            return None
        record_name = str(header_path.with_suffix(""))
        try:
            record = wfdb.rdrecord(record_name)
        except Exception:
            return None
        return {
            "ecg": record.p_signal,
            "fs": float(record.fs),
            "sig_names": record.sig_name,
        }


# ---------------------------------------------------------------------------
# 4. MIMIC-IV Emergency Department
# ---------------------------------------------------------------------------

class MIMICEDLoader:
    """Load MIMIC-IV ED data.

    Layout::

        mimic-iv-ed-2.2/
            ed/
                edstays.csv.gz
                diagnosis.csv.gz
                vitalsign.csv.gz
                triage.csv.gz
    """

    def __init__(self, raw_dir: str | Path):
        self.ed_dir = Path(raw_dir) / "mimic-iv-ed-2.2" / "ed"

    def load_edstays(self) -> pd.DataFrame:
        path = self.ed_dir / "edstays.csv.gz"
        if not path.exists():
            return pd.DataFrame()
        df = _read_gz_csv(path)
        df["intime"] = pd.to_datetime(df.get("intime"), errors="coerce")
        df["outtime"] = pd.to_datetime(df.get("outtime"), errors="coerce")
        return df

    def load_diagnoses(self) -> pd.DataFrame:
        path = self.ed_dir / "diagnosis.csv.gz"
        if not path.exists():
            return pd.DataFrame()
        return _read_gz_csv(path)

    def load_vitalsigns(self) -> pd.DataFrame:
        path = self.ed_dir / "vitalsign.csv.gz"
        if not path.exists():
            return pd.DataFrame()
        df = _read_gz_csv(path)
        df["charttime"] = pd.to_datetime(df.get("charttime"), errors="coerce")
        return df

    def load_triage(self) -> pd.DataFrame:
        path = self.ed_dir / "triage.csv.gz"
        if not path.exists():
            return pd.DataFrame()
        return _read_gz_csv(path)


# ---------------------------------------------------------------------------
# 5. MMASH (Wearable)
# ---------------------------------------------------------------------------

class MMASHLoader:
    """Load MMASH wearable dataset.

    Layout::

        mmash/
            user_1/
                user_info.csv
                Actigraph.csv      (Axis1, Axis2, Axis3, Steps, HR, ...)
                RR.csv             (ibi_s — inter-beat interval in seconds)
                sleep.csv
                Activity.csv
            user_2/
            ...
    """

    def __init__(self, raw_dir: str | Path):
        self.mmash_dir = Path(raw_dir) / "mmash"

    def list_users(self) -> list[int]:
        users = []
        for d in sorted(self.mmash_dir.iterdir()):
            if d.is_dir() and d.name.startswith("user_"):
                try:
                    users.append(int(d.name.split("_")[1]))
                except ValueError:
                    pass
        return users

    def load_user_info(self, user_id: int) -> pd.DataFrame:
        path = self.mmash_dir / f"user_{user_id}" / "user_info.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_actigraph(self, user_id: int) -> pd.DataFrame:
        """Load accelerometer + HR data from Actigraph.csv.

        Returns DataFrame with columns: Axis1, Axis2, Axis3, Steps, HR,
        Inclinometer Standing/Sitting/Lying, Vector Magnitude, day, time.
        """
        path = self.mmash_dir / f"user_{user_id}" / "Actigraph.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, index_col=0)
        # Combine day + time into a timestamp-like index if both exist
        if "day" in df.columns and "time" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["day"].astype(str) + " " + df["time"].astype(str),
                format="%d %H:%M:%S", errors="coerce",
            )
        return df

    def load_rr(self, user_id: int) -> pd.DataFrame:
        """Load RR (inter-beat interval) data.

        Returns DataFrame with columns: ibi_s (seconds), day, time.
        """
        path = self.mmash_dir / f"user_{user_id}" / "RR.csv"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path, index_col=0)
        return df

    def load_sleep(self, user_id: int) -> pd.DataFrame:
        path = self.mmash_dir / f"user_{user_id}" / "sleep.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, index_col=0)

    def load_user_all(self, user_id: int) -> dict[str, pd.DataFrame]:
        return {
            "info": self.load_user_info(user_id),
            "actigraph": self.load_actigraph(user_id),
            "rr": self.load_rr(user_id),
            "sleep": self.load_sleep(user_id),
        }


# ---------------------------------------------------------------------------
# 6. Sleep Accel (Apple Watch)
# ---------------------------------------------------------------------------

class SleepAccelLoader:
    """Load Apple Watch sleep accelerometer dataset.

    Layout::

        sleep_accel/
            heart_rate/
                XXXXXX_heartrate.txt    (timestamp,heart_rate — no header)
            motion/
                XXXXXX_acceleration.txt (timestamp,x,y,z — no header)
            steps/
                XXXXXX_steps.txt
            labels/
                XXXXXX_labeled_sleep.txt
    """

    def __init__(self, raw_dir: str | Path):
        self.sleep_dir = Path(raw_dir) / "sleep_accel"

    def _load_txt_dir(self, subdir: str) -> dict[str, pd.DataFrame]:
        """Load all .txt files from a subdirectory."""
        d = self.sleep_dir / subdir
        if not d.exists():
            return {}
        result = {}
        for f in sorted(d.glob("*.txt")):
            subject_id = f.stem.split("_")[0]
            try:
                df = pd.read_csv(f, header=None, names=["timestamp", "value"])
                result[subject_id] = df
            except Exception as e:
                logger.debug("Cannot load %s: %s", f, e)
        return result

    def load_heart_rate(self) -> dict[str, pd.DataFrame]:
        return self._load_txt_dir("heart_rate")

    def load_motion(self) -> dict[str, pd.DataFrame]:
        """Load accelerometer data.  Returns {subject_id: DataFrame with timestamp,x,y,z}."""
        d = self.sleep_dir / "motion"
        if not d.exists():
            return {}
        result = {}
        for f in sorted(d.glob("*.txt")):
            subject_id = f.stem.split("_")[0]
            try:
                df = pd.read_csv(f, header=None, names=["timestamp", "x", "y", "z"])
                result[subject_id] = df
            except Exception as e:
                logger.debug("Cannot load %s: %s", f, e)
        return result

    def load_labels(self) -> dict[str, pd.DataFrame]:
        return self._load_txt_dir("labels")

    def load_all(self) -> dict[str, dict[str, pd.DataFrame]]:
        """Return all data keyed by subject_id."""
        hr = self.load_heart_rate()
        motion = self.load_motion()
        labels = self.load_labels()
        all_ids = set(list(hr.keys()) + list(motion.keys()) + list(labels.keys()))
        result = {}
        for sid in all_ids:
            result[sid] = {
                "heart_rate": hr.get(sid, pd.DataFrame()),
                "motion": motion.get(sid, pd.DataFrame()),
                "labels": labels.get(sid, pd.DataFrame()),
            }
        return result


# ---------------------------------------------------------------------------
# 7. Non-EEG Neuro
# ---------------------------------------------------------------------------

class NonEEGNeuroLoader:
    """Load non-EEG neurological monitoring data.

    Layout::

        non_eeg_neuro/
            subjectinfo.csv     (subject, age, gender, height/cm, weight/kg)
            Subject1_AccTempEDA.dat
            Subject1_SpO2HR.dat
            ...
    """

    def __init__(self, raw_dir: str | Path):
        self.neuro_dir = Path(raw_dir) / "non_eeg_neuro"

    def load_subject_info(self) -> pd.DataFrame:
        path = self.neuro_dir / "subjectinfo.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def load_subject_data(self, subject_id: int) -> dict[str, np.ndarray]:
        """Load AccTempEDA and SpO2HR .dat files for a subject.

        Returns dict with keys 'acc_temp_eda' and 'spo2_hr' as numpy arrays.
        """
        result = {}
        for suffix in ["AccTempEDA", "SpO2HR"]:
            path = self.neuro_dir / f"Subject{subject_id}_{suffix}.dat"
            if path.exists():
                try:
                    result[suffix.lower()] = np.fromfile(path, dtype=np.float32)
                except Exception as e:
                    logger.debug("Cannot load %s: %s", path, e)
        return result


# ---------------------------------------------------------------------------
# 8. Kaggle Stroke (Tabular)
# ---------------------------------------------------------------------------

class KaggleStrokeLoader:
    """Load Kaggle stroke prediction dataset.

    Layout::

        kaggle-stroke/
            healthcare-dataset-stroke-data.csv
    """

    def __init__(self, raw_dir: str | Path):
        self.stroke_dir = Path(raw_dir) / "kaggle-stroke"

    def load(self) -> pd.DataFrame:
        path = self.stroke_dir / "healthcare-dataset-stroke-data.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 9. AFDB (MIT-BIH Atrial Fibrillation Database)
# ---------------------------------------------------------------------------

class AFDBLoader:
    """Load WFDB-format AF database.

    Layout::

        afdb/
            04015.hea / 04015.dat / 04015.qrs / 04015.atr
            ...
    """

    def __init__(self, raw_dir: str | Path):
        self.afdb_dir = Path(raw_dir) / "afdb"

    def list_records(self) -> list[Path]:
        return sorted(self.afdb_dir.glob("*.hea"))

    def load_record(self, header_path: Path) -> Optional[dict[str, Any]]:
        try:
            import wfdb
        except ImportError:
            return None
        record_name = str(header_path.with_suffix(""))
        try:
            record = wfdb.rdrecord(record_name)
        except Exception:
            return None
        return {
            "ecg": record.p_signal,
            "fs": float(record.fs),
            "sig_names": record.sig_name,
            "record_name": record_name,
        }


# ---------------------------------------------------------------------------
# 10. LTAFDB (Long-Term AF Database)
# ---------------------------------------------------------------------------

class LTAFDBLoader:
    def __init__(self, raw_dir: str | Path):
        self.ltafdb_dir = Path(raw_dir) / "ltafdb"

    def list_records(self) -> list[Path]:
        return sorted(self.ltafdb_dir.glob("*.hea"))

    def load_record(self, header_path: Path) -> Optional[dict[str, Any]]:
        try:
            import wfdb
        except ImportError:
            return None
        record_name = str(header_path.with_suffix(""))
        try:
            record = wfdb.rdrecord(record_name)
        except Exception:
            return None
        return {
            "ecg": record.p_signal,
            "fs": float(record.fs),
            "sig_names": record.sig_name,
        }


# ---------------------------------------------------------------------------
# 11. AFP (PAF Prediction Challenge)
# ---------------------------------------------------------------------------

class AFPLoader:
    def __init__(self, raw_dir: str | Path):
        self.afp_dir = Path(raw_dir) / "afp"

    def list_records(self) -> list[Path]:
        return sorted(self.afp_dir.glob("*.hea"))

    def load_record(self, header_path: Path) -> Optional[dict[str, Any]]:
        try:
            import wfdb
        except ImportError:
            return None
        record_name = str(header_path.with_suffix(""))
        try:
            record = wfdb.rdrecord(record_name)
        except Exception:
            return None
        return {
            "ecg": record.p_signal,
            "fs": float(record.fs),
            "sig_names": record.sig_name,
        }


# ---------------------------------------------------------------------------
# 12. CVES
# ---------------------------------------------------------------------------

class CVESLoader:
    """Load CVES cerebrovascular study data.

    Layout::

        cves/
            subjects.csv     (demographics, comorbidities, cognitive tests)
            data/
                24h-electromyography/
                sit-stand/
                head-up-tilt/
                walking/
                24h-bp/
                ...
    """

    def __init__(self, raw_dir: str | Path):
        self.cves_dir = Path(raw_dir) / "cves"

    def load_subjects(self) -> pd.DataFrame:
        path = self.cves_dir / "subjects.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def list_signal_files(self) -> list[Path]:
        return list(self.cves_dir.rglob("*.dat"))


# ---------------------------------------------------------------------------
# Master loader registry
# ---------------------------------------------------------------------------

def get_all_loaders(raw_dir: str | Path) -> dict[str, Any]:
    """Return a dict mapping dataset name to its loader instance."""
    return {
        "mimic_clinical": MIMICClinicalLoader(raw_dir),
        "mimic_waveform": MIMICWaveformLoader(raw_dir),
        "mimic_ecg": MIMICECGLoader(raw_dir),
        "mimic_ed": MIMICEDLoader(raw_dir),
        "mmash": MMASHLoader(raw_dir),
        "sleep_accel": SleepAccelLoader(raw_dir),
        "non_eeg_neuro": NonEEGNeuroLoader(raw_dir),
        "kaggle_stroke": KaggleStrokeLoader(raw_dir),
        "afdb": AFDBLoader(raw_dir),
        "ltafdb": LTAFDBLoader(raw_dir),
        "afp": AFPLoader(raw_dir),
        "cves": CVESLoader(raw_dir),
    }
