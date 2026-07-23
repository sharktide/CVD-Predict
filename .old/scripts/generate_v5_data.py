#!/usr/bin/env python3
"""
Generate v5 training data: PPG + 3-axis Accelerometer + Biodata.
Uses wristppg simulator with realistic wrist motion profiles.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.signal import resample
import json
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from wristppg import WristPPGSimulator


# ── Profile configs ──────────────────────────────────────────────────────────

PROFILE_LABELS = {
    # Cardiac arrest (label=1)
    "cardiac_arrest_vf": 1,
    "cardiac_arrest_asystole": 1,
    "cardiac_arrest_pulseless_electrical": 1,
    "pre_arrest_deterioration": 1,
    "post_resuscitation": 1,
    "shock": 1,
    "electrocution_arrest": 1,
    "drowning_arrest": 1,
    # General deterioration (label=0 - not arrest, but sick)
    "respiratory_failure_pre_arrest": 0,
    "hypovolemia": 0,
    "sepsis_warm": 0,
    "hfref": 0,
    "hfpef": 0,
    # Healthy / chronic (label=0)
    "healthy": 0,
    "aging": 0,
    "hypertension": 0,
    "diabetes": 0,
    "afib_isolated": 0,
    "arterial_stiffness_isolated": 0,
    "pad": 0,
    "exercise": 0,
    "recovery": 0,
    "sleep": 0,
}

SEGMENTS_PER_PROFILE = {
    # Cardiac arrest - generate MORE for balance
    "cardiac_arrest_vf": 600,
    "cardiac_arrest_asystole": 600,
    "cardiac_arrest_pulseless_electrical": 600,
    "pre_arrest_deterioration": 500,
    "post_resuscitation": 300,
    "shock": 500,
    "electrocution_arrest": 200,
    "drowning_arrest": 200,
    # General deterioration
    "respiratory_failure_pre_arrest": 400,
    "hypovolemia": 300,
    "sepsis_warm": 300,
    "hfref": 250,
    "hfpef": 250,
    # Healthy / chronic
    "healthy": 500,
    "aging": 200,
    "hypertension": 200,
    "diabetes": 200,
    "afib_isolated": 150,
    "arterial_stiffness_isolated": 150,
    "pad": 150,
    "exercise": 200,
    "recovery": 200,
    "sleep": 200,
}

# Realistic activity distributions per profile
ACTIVITY_WEIGHTS = {
    "cardiac_arrest_vf": {"rest": 0.3, "seizure": 0.4, "cpr_chest_compressions": 0.3},
    "cardiac_arrest_asystole": {"rest": 0.8, "cpr_chest_compressions": 0.2},
    "cardiac_arrest_pulseless_electrical": {"rest": 0.6, "cpr_chest_compressions": 0.4},
    "pre_arrest_deterioration": {"rest": 0.5, "walking": 0.3, "sleep": 0.2},
    "post_resuscitation": {"rest": 0.8, "sleep": 0.2},
    "shock": {"rest": 0.7, "sleep": 0.3},
    "healthy": {"rest": 0.3, "walking": 0.3, "sleep": 0.2, "exercise": 0.1, "typing": 0.1},
    "sleep": {"rest": 0.1, "sleep": 0.9},
    "exercise": {"rest": 0.1, "exercise": 0.7, "running": 0.2},
}

DEFAULT_ACTIVITY = {"rest": 0.4, "walking": 0.3, "sleep": 0.2, "typing": 0.1}


def generate_one_segment(args):
    """Generate a single PPG + ACC + biodata segment."""
    profile_name, segment_idx, base_seed = args
    
    try:
        sim = WristPPGSimulator(seed=base_seed + segment_idx)
        rng = np.random.RandomState(base_seed + segment_idx)
        
        # Select activity based on profile
        activity_weights = ACTIVITY_WEIGHTS.get(profile_name, DEFAULT_ACTIVITY)
        activities = list(activity_weights.keys())
        weights = list(activity_weights.values())
        activity = rng.choice(activities, p=weights)
        
        # Vary contact quality
        contact = rng.choice(
            ["good", "good", "good", "loose", "partial_lift", "sweat"],
            p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05]
        )
        
        # Vary duration slightly for realism
        duration = rng.uniform(45, 75)
        
        result = sim.generate(
            profile=profile_name,
            duration_s=duration,
            activity=activity,
            contact_mode=contact,
            wavelength="green",
            fs_output_hz=25.0,
        )
        
        ppg = result.ppg
        accel = result.accel  # (N, 3)
        
        if ppg is None or accel is None:
            return None
        if len(ppg) < 250 or len(accel) < 250:
            return None
        
        # Resample to fixed length (1500 samples = 60s at 25Hz)
        target_len = 1500
        ppg = resample(ppg, target_len).astype(np.float32)
        accel = resample(accel, target_len, axis=0).astype(np.float32)
        
        # Normalize PPG
        ppg = ppg - np.mean(ppg)
        ppg_std = np.std(ppg)
        if ppg_std > 1e-8:
            ppg = ppg / ppg_std
        
        # Normalize ACC (remove gravity, scale)
        accel[:, 2] = accel[:, 2] - np.mean(accel[:, 2])  # Remove gravity from Z
        accel = accel / (np.std(accel) + 1e-8)
        
        # Extract biodata from latent physiology
        latent = result.latent_physiology
        meta = result.meta
        
        biodata = {
            "age": float(latent.get("age_years", 50)),
            "sex": float(rng.choice([0, 1])),  # 0=F, 1=M
            "bmi": float(rng.uniform(18, 40)),
            "spo2": float(meta.get("spo2", latent.get("spo2", 0.97))),
            "body_temp": float(meta.get("body_temp_c", latent.get("body_temp_c", 36.8))),
            "perfusion_index": float(latent.get("perfusion_index", 0.1)),
            "melanin": float(latent.get("melanin_fraction", 0.3)),
            # Comorbidities (synthetic but realistic)
            "has_hypertension": float(rng.random() < 0.3),
            "has_diabetes": float(rng.random() < 0.15),
            "has_hf": float(profile_name in ["hfref", "hfpef", "shock"]),
            "has_ckd": float(rng.random() < 0.1),
            "has_afib": float(profile_name == "afib_isolated" or rng.random() < 0.05),
            "has_copd": float(rng.random() < 0.08),
            # Medications
            "on_betas": float(rng.random() < 0.2),
            "on_anticoag": float(rng.random() < 0.1),
            # Signal quality
            "snr_db": float(meta.get("estimated_snr_db_after_motion", 20)),
        }
        
        # Label
        label = PROFILE_LABELS.get(profile_name, 0)
        
        return {
            "ppg": ppg,
            "accel": accel,
            "biodata": biodata,
            "label": label,
            "profile": profile_name,
            "activity": activity,
            "seed": base_seed + segment_idx,
            "hr_bpm": float(latent.get("hr_bpm", 70)),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def main():
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "processed" / "synthetic_v5"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("V5 TRAINING DATA GENERATION (PPG + ACC + Biodata)")
    print("=" * 70)

    # Build tasks
    tasks = []
    base_seed = 42
    for profile_name, n_segments in SEGMENTS_PER_PROFILE.items():
        for i in range(n_segments):
            tasks.append((profile_name, i, base_seed))
        base_seed += 100000

    total = len(tasks)
    print(f"Total segments: {total}")
    for profile, count in SEGMENTS_PER_PROFILE.items():
        label_name = "CA" if PROFILE_LABELS[profile] == 1 else "Normal"
        print(f"  {profile:40s}: {count:5d} ({label_name})")
    print()

    # Generate in parallel
    results = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(generate_one_segment, task): task for task in tasks}
        completed = 0
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
            completed += 1
            
            if completed % 500 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                print(f"  [{completed:5d}/{total}] {rate:.0f} seg/s, {len(results)} ok")

    elapsed = time.time() - start_time
    print(f"\nGenerated: {len(results)}/{total} in {elapsed:.1f}s")

    # Stack arrays
    ppg_array = np.stack([r["ppg"] for r in results])
    accel_array = np.stack([r["accel"] for r in results])
    labels = np.array([r["label"] for r in results])
    biodata_df = pd.DataFrame([r["biodata"] for r in results])
    profiles = [r["profile"] for r in results]
    activities = [r["activity"] for r in results]
    hr_bpm = np.array([r["hr_bpm"] for r in results])

    # Save
    np.save(output_dir / "ppg.npy", ppg_array)
    np.save(output_dir / "accel.npy", accel_array)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "hr_bpm.npy", hr_bpm)
    biodata_df.to_csv(output_dir / "biodata.csv", index=False)
    pd.DataFrame({"profile": profiles, "activity": activities}).to_csv(
        output_dir / "metadata.csv", index=False
    )

    # Config
    config = {
        "total_segments": len(results),
        "ppg_length": 1500,
        "accel_axes": 3,
        "fs_hz": 25.0,
        "n_biodata_features": len(biodata_df.columns),
        "biodata_columns": list(biodata_df.columns),
        "profile_labels": {k: v for k, v in PROFILE_LABELS.items()},
        "segments_per_profile": SEGMENTS_PER_PROFILE,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Summary
    print(f"\nSaved to {output_dir}:")
    print(f"  ppg.npy:     {ppg_array.shape}")
    print(f"  accel.npy:   {accel_array.shape}")
    print(f"  labels.npy:  {labels.shape}")
    print(f"  biodata.csv: {biodata_df.shape}")
    print(f"  hr_bpm.npy:  {hr_bpm.shape}")
    print(f"\nLabel distribution:")
    print(f"  Normal (0): {(labels == 0).sum()}")
    print(f"  CA (1):     {(labels == 1).sum()}")
    print(f"\nBiodata columns: {list(biodata_df.columns)}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
