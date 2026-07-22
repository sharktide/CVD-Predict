"""
Phase 1: Cohort Building for Cardiac Arrest Prediction

Builds two cohorts from MIMIC-IV:
1. Clinical Domain (Dc): Patients with deterioration events (cardiac arrest, AKI, HF, sepsis, RF)
   - Extracts 24h PPG windows preceding each event
   - Labels with time-to-event (1-24 hours)

2. Healthy Domain (Dh): Patients with NO deterioration events
   - Extracts 24h continuous PPG windows
   - Labels as healthy (time-to-event = inf)

Also links with MMASH data for the healthy wearable domain.
"""

import os
import sys
import gzip
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

# ── ICD Code Definitions ──────────────────────────────────────────────────────

ICD_CODES = {
    "cardiac_arrest": {
        "icd10": ["I460", "I461", "I462", "I468", "I469", "I4901", "I4902",
                   "I472", "I471", "I97710", "I97711"],
        "icd9": ["42741", "42742", "4275", "4271", "V1253", "Z8674"],
        "description": "Cardiac arrest / Ventricular fibrillation / Ventricular tachycardia"
    },
    "aki": {
        "icd10": ["N170", "N171", "N172", "N178", "N179", "N990"],
        "icd9": ["5845", "5846", "5847", "5848", "5849"],
        "description": "Acute Kidney Injury"
    },
    "respiratory_failure": {
        "icd10": ["J9600", "J9601", "J9602", "J9610", "J9611", "J9612",
                   "J9620", "J9621", "J9622", "J9690", "J9691", "J9692"],
        "icd9": ["51881", "51882", "51883", "51884", "51885"],
        "description": "Respiratory Failure"
    },
    "heart_failure": {
        "icd10": ["I501", "I5020", "I5021", "I5022", "I5023", "I5030",
                   "I5031", "I5032", "I5033", "I5040", "I5041", "I5042",
                   "I5043", "I509", "I110", "I130", "I132"],
        "icd9": ["4280", "4281", "42820", "42821", "42822", "42823",
                 "42830", "42831", "42832", "42833", "42840", "42841",
                 "42842", "42843", "4289"],
        "description": "Heart Failure"
    },
    "sepsis": {
        "icd10": ["A4101", "A4102", "A411", "A412", "A413", "A414", "A4150",
                   "A4151", "A4152", "A4153", "A4159", "A4181", "A4189",
                   "A419", "R6520", "R6521"],
        "icd9": ["99591", "99592", "7907"],
        "description": "Sepsis"
    }
}

# Priority: cardiac arrest is primary, others are secondary
EVENT_PRIORITY = {
    "cardiac_arrest": 0,
    "aki": 1,
    "respiratory_failure": 2,
    "heart_failure": 3,
    "sepsis": 4,
}


def load_compressed_csv(path):
    """Load a gzipped CSV file."""
    with gzip.open(path, 'rt') as f:
        return pd.read_csv(f)


def load_mimic_databases(mimic_clinical_dir):
    """Load all MIMIC clinical tables needed for cohort building."""
    hosp_dir = mimic_clinical_dir / "hosp"
    icu_dir = mimic_clinical_dir / "icu"

    print("Loading MIMIC clinical tables...")
    diagnoses = load_compressed_csv(hosp_dir / "diagnoses_icd.csv.gz")
    admissions = load_compressed_csv(hosp_dir / "admissions.csv.gz")
    icustays = load_compressed_csv(icu_dir / "icustays.csv.gz")

    print(f"  Diagnoses: {len(diagnoses):,} records, {diagnoses['subject_id'].nunique():,} patients")
    print(f"  Admissions: {len(admissions):,} records, {admissions['subject_id'].nunique():,} patients")
    print(f"  ICU stays: {len(icustays):,} records, {icustays['subject_id'].nunique():,} patients")

    return diagnoses, admissions, icustays


def identify_events(diagnoses, admissions):
    """Identify deterioration events for each patient admission."""
    print("\nIdentifying deterioration events...")

    # Build ICD code lookup
    icd_to_event = {}
    for event_type, codes in ICD_CODES.items():
        all_codes = codes["icd10"] + codes["icd9"]
        for code in all_codes:
            icd_to_event[code] = event_type

    # Map each diagnosis to an event type
    diagnoses["event_type"] = diagnoses["icd_code"].map(icd_to_event)
    event_diags = diagnoses[diagnoses["event_type"].notna()].copy()

    print(f"  Found {len(event_diags):,} event diagnoses across {event_diags['subject_id'].nunique():,} patients")

    # For each patient+admission, find the first occurrence of each event
    event_diags = event_diags.sort_values(["subject_id", "hadm_id", "event_type", "seq_num"])
    first_events = event_diags.groupby(["subject_id", "hadm_id", "event_type"]).first().reset_index()

    # Pivot to get event columns per admission
    event_summary = first_events.groupby(["subject_id", "hadm_id"]).agg(
        events=("event_type", list),
        event_seq_nums=("seq_num", list),
    ).reset_index()

    # Determine primary event (lowest priority number = most severe)
    def get_primary_event(events):
        if not events:
            return None
        return min(events, key=lambda e: EVENT_PRIORITY.get(e, 99))

    event_summary["primary_event"] = event_summary["events"].apply(get_primary_event)
    event_summary["n_events"] = event_summary["events"].apply(len)

    # Count by event type
    for event_type in EVENT_PRIORITY:
        count = (event_summary["primary_event"] == event_type).sum()
        print(f"  {event_type}: {count} admissions (primary)")

    return event_summary, event_diags


def find_waveform_patients(mimic_wavedb_dir):
    """Find all patients with PPG waveform data.

    Directory structure: mimic4wdb/pXXX/pYYYY/STUDY_ID/
    where pXXX is hundreds group, pYYYY is patient ID.
    """
    print("\nScanning MIMIC waveform database...")
    patients = []

    # Scan two levels deep: pXXX/pYYYY/STUDY_ID/
    for group_dir in sorted(mimic_wavedb_dir.iterdir()):
        if not group_dir.is_dir() or not group_dir.name.startswith("p"):
            continue

        for patient_dir in sorted(group_dir.iterdir()):
            if not patient_dir.is_dir() or not patient_dir.name.startswith("p"):
                continue
            try:
                patient_id = int(patient_dir.name[1:])
            except ValueError:
                continue

            for study_dir in sorted(patient_dir.iterdir()):
                if not study_dir.is_dir():
                    continue
                # Study directories are named like "81739927" (no 's' prefix)
                study_id = study_dir.name

                # Check for PPG segments
                ppg_segments = []
                for hea_file in sorted(study_dir.glob("*.hea")):
                    try:
                        with open(hea_file) as f:
                            header_lines = f.readlines()

                        # Parse WFDB header format:
                        # Line 0: record_name n_signals fs/... ...
                        # Lines starting with ~: signal descriptions
                        signal_names = []
                        for line in header_lines:
                            if line.startswith("#"):
                                continue
                            if line.startswith("~"):
                                # Signal line: ~ 0x2 4096(-2048)/NU 12 0 0 0 0 Pleth
                                parts = line.strip().split()
                                if len(parts) >= 2:
                                    signal_names.append(parts[-1])

                        if "Pleth" in signal_names:
                            segment_name = hea_file.stem
                            ppg_segments.append(segment_name)
                    except Exception:
                        continue

                if ppg_segments:
                    patients.append({
                        "subject_id": patient_id,
                        "study_id": study_id,
                        "path": str(study_dir),
                        "n_ppg_segments": len(ppg_segments),
                        "ppg_segments": ppg_segments,
                    })

    print(f"  Found {len(patients)} patients with PPG data")
    print(f"  Total PPG segments: {sum(p['n_ppg_segments'] for p in patients):,}")

    return patients


def link_waveform_to_clinical(waveform_patients, event_summary, admissions):
    """Link waveform patients to clinical events."""
    print("\nLinking waveform patients to clinical events...")

    waveform_ids = set(p["subject_id"] for p in waveform_patients)
    waveform_map = {p["subject_id"]: p for p in waveform_patients}

    # Find which waveform patients have events
    patient_events = {}
    for _, row in event_summary.iterrows():
        sid = row["subject_id"]
        if sid in waveform_ids:
            if sid not in patient_events:
                patient_events[sid] = []
            patient_events[sid].append({
                "hadm_id": row["hadm_id"],
                "primary_event": row["primary_event"],
                "all_events": row["events"],
            })

    # Build final cohort
    cohort = []
    for pid in sorted(waveform_ids):
        wp = waveform_map[pid]
        events = patient_events.get(pid, [])

        # Get admission times
        adm = admissions[admissions["subject_id"] == pid]
        adm_times = {}
        for _, row in adm.iterrows():
            adm_times[row["hadm_id"]] = {
                "admittime": pd.to_datetime(row["admittime"]),
                "dischtime": pd.to_datetime(row["dischtime"]),
            }

        cohort.append({
            "subject_id": pid,
            "study_id": wp["study_id"],
            "path": wp["path"],
            "n_ppg_segments": wp["n_ppg_segments"],
            "ppg_segments": wp["ppg_segments"],
            "events": events,
            "has_event": len(events) > 0,
            "primary_event": events[0]["primary_event"] if events else None,
            "admission_times": adm_times,
        })

    n_with_events = sum(1 for c in cohort if c["has_event"])
    n_healthy = sum(1 for c in cohort if not c["has_event"])

    print(f"  Waveform patients with events: {n_with_events}")
    print(f"  Waveform patients without events (healthy): {n_healthy}")

    # Event type breakdown
    event_counts = defaultdict(int)
    for c in cohort:
        if c["primary_event"]:
            event_counts[c["primary_event"]] += 1
    for event_type, count in sorted(event_counts.items()):
        print(f"    {event_type}: {count}")

    return cohort


def extract_ppg_windows_from_wfdb(cohort, mimic_wavedb_dir, output_dir, window_hours=24, segment_duration_s=60, fs=25):
    """
    Extract 24-hour PPG windows from MIMIC waveform data.

    For patients with events:
        - Extract 24h of PPG preceding the event
        - Fragment into 1-hour rolling windows
        - Label with time-to-event (1-24 hours)

    For healthy patients:
        - Extract any 24h continuous PPG window
        - Label as healthy (time-to-event = inf)
    """
    import wfdb
    from scipy.signal import resample

    print("\nExtracting PPG windows from WFDB...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ppg_segments").mkdir(exist_ok=True)

    all_windows = []
    segment_duration_samples = int(segment_duration_s * fs)

    for patient in cohort:
        pid = patient["subject_id"]
        study_id = patient["study_id"]
        patient_path = Path(patient["path"])

        # Load full PPG via master header (multi-segment WFDB record)
        master_header = patient_path / f"{study_id}.hea"
        if not master_header.exists():
            continue

        try:
            # Probe first 1000 samples to discover signals and sampling rate
            sig_probe, fields_probe = wfdb.rdsamp(
                record_name=str(master_header).replace('.hea', ''),
                sampto=1000
            )
            fs_orig = fields_probe['fs']
            sig_names = fields_probe.get('sig_name', [])
            pleth_idx = next((i for i, n in enumerate(sig_names) if 'Pleth' in n), None)
            if pleth_idx is None:
                continue

            # Get total length from header
            rec = wfdb.rdheader(str(master_header).replace('.hea', ''))
            total_samples = rec.sig_len

            # Load in chunks to avoid memory issues
            chunk_size = int(3600 * fs_orig)  # 1 hour chunks
            all_ppg_data = []
            max_hours = 48  # Cap at 48 hours to avoid huge files

            for start in range(0, min(total_samples, int(max_hours * 3600 * fs_orig)), chunk_size):
                end = min(start + chunk_size, total_samples)
                sig, _ = wfdb.rdsamp(
                    record_name=str(master_header).replace('.hea', ''),
                    sampfrom=start, sampto=end,
                    channels=[pleth_idx]
                )
                ppg_chunk = sig[:, 0]

                # Resample to target fs
                target_samples = int(len(ppg_chunk) * fs / fs_orig)
                ppg_resampled = resample(ppg_chunk, target_samples)
                # Skip NaN/inf segments
                if np.any(np.isnan(ppg_resampled)) or np.any(np.isinf(ppg_resampled)):
                    continue
                all_ppg_data.append(ppg_resampled)

            if not all_ppg_data:
                continue

            full_ppg = np.concatenate(all_ppg_data)
            total_hours = len(full_ppg) / (fs * 3600)

            if total_hours < 1:
                continue

        except Exception as e:
            continue

        # Determine label
        has_event = patient["has_event"]
        primary_event = patient["primary_event"]

        if has_event:
            # For event patients, we extract windows ending before the event
            # Since we don't have exact event timestamps in waveform data,
            # we use the last segment as the "closest to event" point
            n_windows = max(1, int(total_hours) - window_hours + 1)
            n_windows = min(n_windows, 24)  # Cap at 24 windows

            for w in range(n_windows):
                # Calculate time-to-event (hours from start of window to end of recording)
                time_to_event = total_hours - w - window_hours
                if time_to_event < 0:
                    time_to_event = 0

                # Extract window
                start_sample = int(w * 3600 * fs)
                end_sample = start_sample + int(window_hours * 3600 * fs)
                if end_sample > total_samples:
                    break

                window_ppg = full_ppg[start_sample:end_sample]

                # Fragment into 1-hour segments
                segments = []
                for s in range(window_hours):
                    seg_start = s * 3600 * fs
                    seg_end = seg_start + segment_duration_samples
                    if seg_end <= len(window_ppg):
                        segments.append(window_ppg[seg_start:seg_end])

                if len(segments) < window_hours:
                    continue

                # Save segments
                window_id = f"{pid}_{study_id}_w{w:03d}"
                for s_idx, seg in enumerate(segments):
                    seg_path = output_dir / "ppg_segments" / f"{window_id}_s{s_idx:02d}.npy"
                    np.save(seg_path, seg.astype(np.float32))

                all_windows.append({
                    "window_id": window_id,
                    "subject_id": pid,
                    "study_id": study_id,
                    "primary_event": primary_event,
                    "time_to_event_hours": float(time_to_event),
                    "is_healthy": False,
                    "n_segments": len(segments),
                    "total_hours_available": float(total_hours),
                })
        else:
            # Healthy patient: extract any 24h window
            if total_hours >= window_hours:
                # Take the first 24h window
                window_ppg = full_ppg[:int(window_hours * 3600 * fs)]
                segments = []
                for s in range(window_hours):
                    seg_start = s * 3600 * fs
                    seg_end = seg_start + segment_duration_samples
                    if seg_end <= len(window_ppg):
                        segments.append(window_ppg[seg_start:seg_end])

                if len(segments) >= window_hours:
                    window_id = f"{pid}_{study_id}_healthy"
                    for s_idx, seg in enumerate(segments):
                        seg_path = output_dir / "ppg_segments" / f"{window_id}_s{s_idx:02d}.npy"
                        np.save(seg_path, seg.astype(np.float32))

                    all_windows.append({
                        "window_id": window_id,
                        "subject_id": pid,
                        "study_id": study_id,
                        "primary_event": None,
                        "time_to_event_hours": float('inf'),
                        "is_healthy": True,
                        "n_segments": len(segments),
                        "total_hours_available": float(total_hours),
                    })
            else:
                # Less than 24h but still usable for healthy
                segments = []
                n_segs = min(int(total_hours * 60 / (segment_duration_s / 60)), window_hours)
                for s in range(n_segs):
                    seg_start = s * segment_duration_samples
                    seg_end = seg_start + segment_duration_samples
                    if seg_end <= len(full_ppg):
                        segments.append(full_ppg[seg_start:seg_end])

                if len(segments) >= 4:  # At least 4 hours
                    window_id = f"{pid}_{study_id}_healthy"
                    for s_idx, seg in enumerate(segments):
                        seg_path = output_dir / "ppg_segments" / f"{window_id}_s{s_idx:02d}.npy"
                        np.save(seg_path, seg.astype(np.float32))

                    all_windows.append({
                        "window_id": window_id,
                        "subject_id": pid,
                        "study_id": study_id,
                        "primary_event": None,
                        "time_to_event_hours": float('inf'),
                        "is_healthy": True,
                        "n_segments": len(segments),
                        "total_hours_available": float(total_hours),
                    })

    # Save window metadata
    windows_df = pd.DataFrame(all_windows)
    windows_df.to_csv(output_dir / "windows.csv", index=False)

    print(f"\n  Total windows extracted: {len(all_windows)}")
    print(f"  Event windows: {sum(1 for w in all_windows if not w['is_healthy'])}")
    print(f"  Healthy windows: {sum(1 for w in all_windows if w['is_healthy'])}")

    # Summary statistics
    if all_windows:
        event_types = defaultdict(int)
        for w in all_windows:
            if w["primary_event"]:
                event_types[w["primary_event"]] += 1
            else:
                event_types["healthy"] += 1
        for et, count in sorted(event_types.items()):
            print(f"    {et}: {count}")

    return all_windows


def main():
    """Main cohort building pipeline."""
    project_root = Path(__file__).parent.parent
    mimic_clinical_dir = project_root / "data" / "raw" / "mimiciv-clinical"
    mimic_wavedb_dir = project_root / "data" / "raw" / "mimic4wdb"
    output_dir = project_root / "data" / "processed" / "cohort_v1"

    print("=" * 70)
    print("PHASE 1: COHORT BUILDING")
    print("=" * 70)

    # Step 1: Load MIMIC clinical data
    diagnoses, admissions, icustays = load_mimic_databases(mimic_clinical_dir)

    # Step 2: Identify deterioration events
    event_summary, event_diags = identify_events(diagnoses, admissions)

    # Step 3: Find waveform patients
    waveform_patients = find_waveform_patients(mimic_wavedb_dir)

    # Step 4: Link waveform to clinical
    cohort = link_waveform_to_clinical(waveform_patients, event_summary, admissions)

    # Step 5: Extract PPG windows
    windows = extract_ppg_windows_from_wfdb(
        cohort, mimic_wavedb_dir, output_dir,
        window_hours=24, segment_duration_s=60, fs=25
    )

    # Save cohort summary
    summary = {
        "n_patients": len(cohort),
        "n_with_events": sum(1 for c in cohort if c["has_event"]),
        "n_healthy": sum(1 for c in cohort if not c["has_event"]),
        "n_windows": len(windows),
        "n_event_windows": sum(1 for w in windows if not w["is_healthy"]),
        "n_healthy_windows": sum(1 for w in windows if w["is_healthy"]),
        "event_types": {},
    }
    for w in windows:
        et = w["primary_event"] if w["primary_event"] else "healthy"
        summary["event_types"][et] = summary["event_types"].get(et, 0) + 1

    with open(output_dir / "cohort_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"COHORT SUMMARY")
    print(f"{'=' * 70}")
    print(json.dumps(summary, indent=2))
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
