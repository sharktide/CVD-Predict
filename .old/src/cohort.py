"""Cohort metadata – acuity scoring, community-likeness, sampling weights.

Works directly with the actual MIMIC-IV clinical CSV column names:
    - patients: subject_id, gender, anchor_age
    - admissions: subject_id, hadm_id, admittime, dischtime, insurance, race
    - icustays: subject_id, hadm_id, stay_id, first_careunit, intime, outtime, los
    - chartevents: subject_id, hadm_id, stay_id, charttime, itemid, valuenum
    - labels: patient_id, hadm_id, event_type, event_time, label_confidence
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.utils import save_parquet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MIMIC-IV chart event item IDs for organ-support detection
# ---------------------------------------------------------------------------
# Vasopressors: norepinephrine (221906), epinephrine (221289), dopamine (221662),
#               phenylephrine (221744), vasopressin (222315), milrinone (221986)
VASOPRESSOR_ITEMIDS = {221906, 221289, 221662, 221744, 222315, 221986}
# Mechanical ventilation: various itemids
VENTILATION_ITEMIDS = {
    225792, 225794, 225796, 225798,  # ET tube
    224685, 224686, 224687, 224695,  # Trach
    220339, 224700, 226873,  # Ventilation settings
}
# RRT (renal replacement therapy): 225916, 225898, 225897
RRT_ITEMIDS = {225916, 225898, 225897}
# ECMO: 229268
ECMO_ITEMIDS = {229268}


# ---------------------------------------------------------------------------
# Acuity scoring
# ---------------------------------------------------------------------------

def compute_acuity_score(row: pd.Series) -> int:
    """Return an integer acuity score (0-5) derived from organ-support and comorbidity data."""
    score = 0
    score += int(row.get("on_vasopressors", 0))
    score += int(row.get("on_ventilation", 0))
    score += int(row.get("on_ecmo", 0))
    score += int(row.get("on_rrt", 0))
    cc = int(row.get("comorbidity_count", 0))
    score += min(cc // 3, 2)  # at most +2 from comorbidities
    return min(score, 5)


# ---------------------------------------------------------------------------
# Community-likeness
# ---------------------------------------------------------------------------

def compute_community_likeness(row: pd.Series) -> float:
    """Higher = more community-like (less acute, fewer comorbidities, shorter stay)."""
    acuity = row.get("acuity_score", 0)
    los = row.get("length_of_stay_hours", 48)
    comorb = row.get("comorbidity_count", 0)

    score = 1.0 / (1.0 + 0.3 * acuity + 0.01 * float(los) + 0.2 * float(comorb))
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Community-likeness binning (for subgroup eval)
# ---------------------------------------------------------------------------

def bin_community_likeness(score: float) -> str:
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "medium"
    return "low"


def bin_label_confidence(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Comorbidity counting
# ---------------------------------------------------------------------------

def count_comorbidities(diagnoses_df: pd.DataFrame) -> pd.DataFrame:
    """Count distinct ICD-9/10 comorbidity codes per (subject_id, hadm_id).

    Returns a DataFrame with columns: subject_id, hadm_id, comorbidity_count.
    """
    if diagnoses_df.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "comorbidity_count"])

    counts = (
        diagnoses_df
        .groupby(["subject_id", "hadm_id"])["icd_code"]
        .nunique()
        .reset_index()
        .rename(columns={"icd_code": "comorbidity_count"})
    )
    return counts


# ---------------------------------------------------------------------------
# Organ support detection from chartevents
# ---------------------------------------------------------------------------

def detect_organ_support(
    chartevents_df: pd.DataFrame,
    icustays_df: pd.DataFrame,
) -> pd.DataFrame:
    """Detect vasopressor, ventilation, RRT, and ECMO usage per ICU stay.

    Returns DataFrame with columns:
        subject_id, hadm_id, stay_id,
        on_vasopressors, on_ventilation, on_rrt, on_ecmo
    """
    if chartevents_df.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id",
                                     "on_vasopressors", "on_ventilation",
                                     "on_rrt", "on_ecmo"])

    stays = icustays_df[["subject_id", "hadm_id", "stay_id"]].drop_duplicates()

    # Check each organ support type
    def _has_item(df, itemids):
        present = df["itemid"].isin(itemids)
        return df.loc[present, ["subject_id", "hadm_id", "stay_id"]].drop_duplicates()

    vp = _has_item(chartevents_df, VASOPRESSOR_ITEMIDS).assign(on_vasopressors=1)
    vent = _has_item(chartevents_df, VENTILATION_ITEMIDS).assign(on_ventilation=1)
    rrt = _has_item(chartevents_df, RRT_ITEMIDS).assign(on_rrt=1)
    ecmo = _has_item(chartevents_df, ECMO_ITEMIDS).assign(on_ecmo=1)

    # Merge all onto stays
    result = stays.copy()
    for support_df, col in [(vp, "on_vasopressors"), (vent, "on_ventilation"),
                             (rrt, "on_rrt"), (ecmo, "on_ecmo")]:
        if not support_df.empty:
            result = result.merge(
                support_df[["subject_id", "hadm_id", "stay_id", col]],
                on=["subject_id", "hadm_id", "stay_id"],
                how="left",
            )
        else:
            result[col] = 0

    result = result.fillna(0)
    return result


# ---------------------------------------------------------------------------
# Cohort metadata builder
# ---------------------------------------------------------------------------

def build_cohort_metadata(
    labels_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    admissions_df: pd.DataFrame,
    icustays_df: pd.DataFrame,
    diagnoses_df: Optional[pd.DataFrame] = None,
    chartevents_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Merge clinical + label data and compute acuity, community-likeness, and
    community-likeness importance weights.

    Parameters
    ----------
    labels_df : from labeling.build_event_labels (patient_id, hadm_id, event_type, ...).
    patients_df : from MIMICClinicalLoader.load_patients (subject_id, gender, anchor_age).
    admissions_df : from MIMICClinicalLoader.load_admissions.
    icustays_df : from MIMICClinicalLoader.load_icustays.
    diagnoses_df : optional full diagnoses_icd for comorbidity counting.
    chartevents_df : optional chartevents for organ-support detection.
    output_path : optional parquet output path.

    Returns
    -------
    DataFrame with one row per (patient, hadm_id, event_type) combination.
    """
    # --- Comorbidity counts ---
    comorbidity_counts = pd.DataFrame(
        columns=["subject_id", "hadm_id", "comorbidity_count"]
    )
    if diagnoses_df is not None and not diagnoses_df.empty:
        comorbidity_counts = count_comorbidities(diagnoses_df)

    # --- Organ support ---
    organ_support = pd.DataFrame(
        columns=["subject_id", "hadm_id", "stay_id",
                 "on_vasopressors", "on_ventilation", "on_rrt", "on_ecmo"]
    )
    if chartevents_df is not None and not chartevents_df.empty:
        organ_support = detect_organ_support(chartevents_df, icustays_df)

    # --- Merge labels with demographics ---
    # labels_df has 'patient_id' (which is subject_id), 'hadm_id', 'event_type'
    merged = labels_df.merge(
        patients_df[["subject_id", "gender", "anchor_age"]],
        left_on="patient_id",
        right_on="subject_id",
        how="left",
    )
    # Drop the redundant subject_id column from the merge
    if "subject_id" in merged.columns and "patient_id" in merged.columns:
        merged = merged.drop(columns=["subject_id"])

    # Merge admissions
    adm = admissions_df[["subject_id", "hadm_id", "admittime", "dischtime"]].drop_duplicates(
        subset=["subject_id", "hadm_id"]
    )
    merged = merged.merge(
        adm,
        left_on=["patient_id", "hadm_id"],
        right_on=["subject_id", "hadm_id"],
        how="left",
    )
    if "subject_id" in merged.columns:
        merged = merged.drop(columns=["subject_id"])

    # Compute length of stay
    merged["length_of_stay_hours"] = (
        (pd.to_datetime(merged["dischtime"]) - pd.to_datetime(merged["admittime"]))
        .dt.total_seconds() / 3600.0
    ).fillna(48.0)

    # Merge ICU stays
    if not icustays_df.empty:
        icu_agg = (
            icustays_df
            .groupby(["subject_id", "hadm_id"])
            .agg(
                first_careunit=("first_careunit", "first"),
                stay_id=("stay_id", "first"),
            )
            .reset_index()
        )
        merged = merged.merge(
            icu_agg,
            left_on=["patient_id", "hadm_id"],
            right_on=["subject_id", "hadm_id"],
            how="left",
        )
        if "subject_id" in merged.columns:
            merged = merged.drop(columns=["subject_id"])
    else:
        merged["first_careunit"] = "unknown"
        merged["stay_id"] = None

    # Merge comorbidity counts
    merged = merged.merge(
        comorbidity_counts,
        left_on=["patient_id", "hadm_id"],
        right_on=["subject_id", "hadm_id"],
        how="left",
    )
    if "subject_id" in merged.columns:
        merged = merged.drop(columns=["subject_id"])
    merged["comorbidity_count"] = merged["comorbidity_count"].fillna(0).astype(int)

    # Merge organ support (aggregate to hadm_id level)
    if not organ_support.empty:
        os_hadm = (
            organ_support
            .groupby(["subject_id", "hadm_id"])
            .agg({
                "on_vasopressors": "max",
                "on_ventilation": "max",
                "on_rrt": "max",
                "on_ecmo": "max",
            })
            .reset_index()
        )
        merged = merged.merge(
            os_hadm,
            left_on=["patient_id", "hadm_id"],
            right_on=["subject_id", "hadm_id"],
            how="left",
        )
        if "subject_id" in merged.columns:
            merged = merged.drop(columns=["subject_id"])
    else:
        for col in ["on_vasopressors", "on_ventilation", "on_rrt", "on_ecmo"]:
            merged[col] = 0

    merged = merged.fillna(0)

    # --- Compute acuity and community-likeness ---
    rows = []
    for _, row in merged.iterrows():
        acuity = compute_acuity_score(row)
        row_with_acuity = row.copy()
        row_with_acuity["acuity_score"] = acuity
        community_score = compute_community_likeness(row_with_acuity)
        importance_weight = community_score

        rows.append({
            "patient_id": row["patient_id"],
            "hadm_id": row["hadm_id"],
            "icu_type": row.get("first_careunit", "unknown"),
            "gender": row.get("gender", "unknown"),
            "anchor_age": row.get("anchor_age", 0),
            "acuity_score": acuity,
            "comorbidity_count": int(row.get("comorbidity_count", 0)),
            "length_of_stay_hours": float(row.get("length_of_stay_hours", 0)),
            "community_likeness": community_score,
            "importance_weight": importance_weight,
            "community_likeness_bin": bin_community_likeness(community_score),
            "event_type": row.get("event_type", "CONTROL"),
            "event_time": row.get("event_time"),
            "label_confidence": float(row.get("label_confidence", 0.0)),
            "label_confidence_bin": bin_label_confidence(float(row.get("label_confidence", 0.0))),
            "on_vasopressors": int(row.get("on_vasopressors", 0)),
            "on_ventilation": int(row.get("on_ventilation", 0)),
            "on_ecmo": int(row.get("on_ecmo", 0)),
            "on_rrt": int(row.get("on_rrt", 0)),
        })

    meta_df = pd.DataFrame(rows).drop_duplicates(subset=["patient_id", "hadm_id", "event_type"])
    # Ensure icu_type is always a string
    meta_df["icu_type"] = meta_df["icu_type"].astype(str).fillna("unknown")

    if output_path is not None:
        save_parquet(meta_df, output_path)
        logger.info("Saved cohort metadata (%d rows) → %s", len(meta_df), output_path)

    return meta_df


def build_cohort_from_loader(
    clinical_loader,
    labels_df: pd.DataFrame,
    output_path: Optional[str] = None,
    skip_chartevents: bool = True,
) -> pd.DataFrame:
    """High-level helper: build cohort metadata using a MIMICClinicalLoader.

    Parameters
    ----------
    clinical_loader : instance of data_loaders.MIMICClinicalLoader.
    labels_df : from labeling.build_event_labels.
    output_path : optional parquet output path.
    skip_chartevents : if True, skip chartevents loading (sets organ support to 0).
        Chartevents is very large and slow to load.
    """
    logger.info("Loading MIMIC-IV clinical tables for cohort...")
    patients = clinical_loader.load_patients()
    admissions = clinical_loader.load_admissions()
    icustays = clinical_loader.load_icustays()

    # For comorbidity counting, load diagnoses (all, not just MI/ARREST)
    diagnoses = clinical_loader.load_diagnoses_icd()

    # For organ support detection, load a subset of chartevents
    chartevents = pd.DataFrame()
    if not skip_chartevents:
        try:
            chartevents = clinical_loader.load_chartevents(
                itemids=VASOPRESSOR_ITEMIDS | VENTILATION_ITEMIDS | RRT_ITEMIDS | ECMO_ITEMIDS
            )
        except Exception as e:
            logger.warning("Could not load chartevents for organ support: %s", e)

    return build_cohort_metadata(
        labels_df=labels_df,
        patients_df=patients,
        admissions_df=admissions,
        icustays_df=icustays,
        diagnoses_df=diagnoses,
        chartevents_df=chartevents,
        output_path=output_path,
    )
