"""Event labelling – MI onset refinement, cardiac-arrest labelling, confidence scores.

Works directly with the actual MIMIC-IV clinical CSV column names:
    - diagnoses_icd: subject_id, hadm_id, icd_code, icd_version
    - labevents: subject_id, hadm_id, itemid, charttime, valuenum
    - admissions: subject_id, hadm_id, admittime, dischtime
    - icustays: subject_id, hadm_id, stay_id, intime, outtime, los
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils import save_parquet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MIMIC-IV lab item IDs for cardiac markers
# ---------------------------------------------------------------------------
# Troponin-T: 51003, Troponin-I: 50974
# CK-MB (mass): 50908, CK-MB (activity): 50971
TROPONIN_ITEMIDS = {51003, 50974}
CKMB_ITEMIDS = {50908, 50971}
ALL_CARDIAC_LAB_ITEMIDS = TROPONIN_ITEMIDS | CKMB_ITEMIDS


# ---------------------------------------------------------------------------
# Lab helpers
# ---------------------------------------------------------------------------

def _extract_troponin_series(
    labs: pd.DataFrame,
    time_col: str = "charttime",
) -> Optional[pd.DataFrame]:
    """Extract troponin time-series from the labevents DataFrame.

    Returns DataFrame with columns [time_col, 'valuenum'] sorted by time,
    or None if no troponin data found.
    """
    if labs.empty:
        return None
    mask = labs["itemid"].isin(TROPONIN_ITEMIDS)
    troponin = labs.loc[mask, [time_col, "valuenum"]].dropna()
    if troponin.empty:
        return None
    troponin = troponin.sort_values(time_col)
    return troponin


def _extract_ckmb_series(
    labs: pd.DataFrame,
    time_col: str = "charttime",
) -> Optional[pd.DataFrame]:
    """Extract CK-MB time-series from the labevents DataFrame."""
    if labs.empty:
        return None
    mask = labs["itemid"].isin(CKMB_ITEMIDS)
    ckmb = labs.loc[mask, [time_col, "valuenum"]].dropna()
    if ckmb.empty:
        return None
    ckmb = ckmb.sort_values(time_col)
    return ckmb


def _first_troponin_elevation(
    labs: pd.DataFrame,
    time_col: str = "charttime",
    relative_threshold: float = 2.0,
) -> Optional[pd.Timestamp]:
    """Return timestamp of first troponin >= threshold * baseline.

    Baseline is the patient's minimum troponin value.
    """
    ts = _extract_troponin_series(labs, time_col=time_col)
    if ts is None or ts.empty:
        return None
    baseline = ts["valuenum"].min()
    elevated = ts[ts["valuenum"] >= baseline * relative_threshold]
    if elevated.empty:
        return None
    return elevated[time_col].iloc[0]


def _first_ckmb_elevation(
    labs: pd.DataFrame,
    time_col: str = "charttime",
    relative_threshold: float = 2.0,
) -> Optional[pd.Timestamp]:
    """Return timestamp of first CK-MB >= threshold * baseline."""
    ts = _extract_ckmb_series(labs, time_col=time_col)
    if ts is None or ts.empty:
        return None
    baseline = ts["valuenum"].min()
    elevated = ts[ts["valuenum"] >= baseline * relative_threshold]
    if elevated.empty:
        return None
    return elevated[time_col].iloc[0]


# ---------------------------------------------------------------------------
# MI onset estimation
# ---------------------------------------------------------------------------

def compute_mi_onset(
    subject_id: int,
    hadm_id: int,
    labs: pd.DataFrame,
    icd_code: str = "I21",
    admittime: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """Estimate a consensus MI onset and confidence score.

    Parameters
    ----------
    subject_id : MIMIC subject ID.
    hadm_id : MIMIC hadm_id for this admission.
    labs : full labevents DataFrame (will be filtered by subject_id/hadm_id).
    icd_code : the ICD code that triggered inclusion.
    admittime : admission time (used as fallback anchor).

    Returns
    -------
    dict with keys ``onset_time`` (pd.Timestamp | None) and ``confidence`` (float 0-1).
    """
    signals_found: List[pd.Timestamp] = []

    # Filter labs to this patient/admission
    if labs.empty or "subject_id" not in labs.columns:
        p_labs = pd.DataFrame()
    else:
        p_labs = labs[(labs["subject_id"] == subject_id) &
                      (labs["hadm_id"] == hadm_id)]

    # 1. Troponin elevation
    t_trop = _first_troponin_elevation(p_labs)
    if t_trop is not None:
        signals_found.append(("troponin", t_trop))

    # 2. CK-MB elevation
    t_ckmb = _first_ckmb_elevation(p_labs)
    if t_ckmb is not None:
        signals_found.append(("ckmb", t_ckmb))

    # 3. Admission time as fallback
    if admittime is not None and pd.notna(admittime):
        signals_found.append(("admittime", pd.Timestamp(admittime)))

    if not signals_found:
        return {"onset_time": None, "confidence": 0.0}

    # Consensus onset: earliest timestamp among all signals
    onset = min(t for _, t in signals_found)

    # Confidence based on number and agreement of signals
    if len(signals_found) == 1:
        confidence = 0.3  # only one signal (e.g. just ICD code)
    elif len(signals_found) == 2:
        confidence = 0.6
    else:
        # Tighter temporal clustering -> higher confidence
        timestamps = [t for _, t in signals_found]
        time_span = max(timestamps) - min(timestamps)
        hours_span = time_span.total_seconds() / 3600.0
        confidence = float(np.clip(1.0 - hours_span / 24.0, 0.6, 0.95))

    return {"onset_time": onset, "confidence": confidence}


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def build_event_labels(
    diagnoses_df: pd.DataFrame,
    admissions_df: pd.DataFrame,
    labs_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Build a DataFrame of refined event labels from raw MIMIC-IV tables.

    Parameters
    ----------
    diagnoses_df : diagnoses_icd.csv.gz content (subject_id, hadm_id, icd_code, icd_version).
    admissions_df : admissions.csv.gz content (subject_id, hadm_id, admittime, dischtime).
    labs_df : labevents.csv.gz content (subject_id, hadm_id, itemid, charttime, valuenum).
        Can be pre-filtered to cardiac lab itemids for efficiency.

    Returns
    -------
    DataFrame with columns: patient_id (subject_id), hadm_id, event_type, event_time,
                            label_confidence, icd_code
    """
    rows: List[Dict[str, Any]] = []

    # Filter to MI (I21) and cardiac arrest (I46)
    mi_mask = diagnoses_df["icd_code"].astype(str).str.startswith("I21")
    arrest_mask = diagnoses_df["icd_code"].astype(str).str.startswith("I46")
    events = diagnoses_df[mi_mask | arrest_mask].copy()

    if events.empty:
        logger.warning("No MI or cardiac arrest diagnoses found")
        return pd.DataFrame(columns=["patient_id", "hadm_id", "event_type",
                                     "event_time", "label_confidence", "icd_code"])

    logger.info("Found %d MI + %d ARREST diagnosis rows",
                mi_mask.sum(), arrest_mask.sum())

    # Merge admission times
    events = events.merge(
        admissions_df[["subject_id", "hadm_id", "admittime"]].drop_duplicates(
            subset=["subject_id", "hadm_id"]
        ),
        on=["subject_id", "hadm_id"],
        how="left",
    )

    # Process each (subject, hadm, icd_code) group
    seen = set()
    for _, row in events.iterrows():
        sid = int(row["subject_id"])
        hid = int(row["hadm_id"])
        icd = str(row["icd_code"])
        key = (sid, hid, icd[:3])  # group by I21.x or I46.x
        if key in seen:
            continue
        seen.add(key)

        admittime = row.get("admittime")
        if pd.notna(admittime):
            admittime = pd.Timestamp(admittime)
        else:
            admittime = None

        if icd.startswith("I21"):
            onset_info = compute_mi_onset(
                subject_id=sid,
                hadm_id=hid,
                labs=labs_df,
                icd_code=icd,
                admittime=admittime,
            )
            if onset_info["onset_time"] is not None:
                rows.append({
                    "patient_id": sid,
                    "hadm_id": hid,
                    "event_type": "MI",
                    "event_time": onset_info["onset_time"],
                    "label_confidence": onset_info["confidence"],
                    "icd_code": icd,
                })
        elif icd.startswith("I46"):
            # Cardiac arrest: use admission time as proxy for event time
            # (the ICD code is assigned at discharge; the actual arrest time
            # is harder to pinpoint without chartevents data)
            event_time = admittime
            if event_time is not None:
                rows.append({
                    "patient_id": sid,
                    "hadm_id": hid,
                    "event_type": "ARREST",
                    "event_time": event_time,
                    "label_confidence": 0.7,  # slightly lower than before since we use admittime
                    "icd_code": icd,
                })

    labels_df = pd.DataFrame(rows)
    if output_path is not None:
        save_parquet(labels_df, output_path)
        logger.info("Saved %d event labels → %s", len(labels_df), output_path)

    return labels_df


def build_event_labels_from_loader(
    clinical_loader,
    output_path: Optional[str] = None,
    skip_labs: bool = False,
) -> pd.DataFrame:
    """High-level helper: build event labels using a MIMICClinicalLoader.

    Parameters
    ----------
    clinical_loader : instance of data_loaders.MIMICClinicalLoader.
    output_path : optional path to save the labels parquet.
    skip_labs : if True, skip labevents loading (uses admittime as MI event time).
        Set this to True when labevents is too slow to load.
    """
    logger.info("Loading MIMIC-IV clinical tables for labeling...")
    diagnoses = clinical_loader.load_diagnoses_icd_filtered(("I21", "I46"))
    admissions = clinical_loader.load_admissions()

    labs = pd.DataFrame()
    if not skip_labs:
        try:
            labs = clinical_loader.load_labevents()
        except Exception as e:
            logger.warning("Could not load labevents (will use admittime for MI onset): %s", e)

    if diagnoses.empty:
        logger.warning("No MI/ARREST diagnoses found — cannot build labels")
        return pd.DataFrame()

    return build_event_labels(
        diagnoses_df=diagnoses,
        admissions_df=admissions,
        labs_df=labs,
        output_path=output_path,
    )
