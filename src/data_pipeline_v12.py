"""Unified data pipeline for v12-v15 training.

Generates synthetic data from wristppg/ and loads real MIMIC/MMASH data.
Handles PPG, 3-axis accelerometer, HRV features, and biodata.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import resample as scipy_resample

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

PPG_LENGTH = 7500
FS_TARGET = 25


def extract_accel_features(accel: np.ndarray, fs: int = 25) -> Dict[str, float]:
    """Extract features from 3-axis accelerometer signal.

    Parameters
    ----------
    accel : (N, 3) array
    """
    feats = {}
    if accel.ndim != 2 or accel.shape[1] != 3:
        return feats

    mag = np.sqrt(np.sum(accel ** 2, axis=-1))
    feats["accel_mean_magnitude"] = float(np.mean(mag))
    feats["accel_std_magnitude"] = float(np.std(mag))
    feats["accel_max_magnitude"] = float(np.max(mag))
    feats["accel_mean_x"] = float(np.mean(accel[:, 0]))
    feats["accel_mean_y"] = float(np.mean(accel[:, 1]))
    feats["accel_mean_z"] = float(np.mean(accel[:, 2]))
    feats["accel_std_x"] = float(np.std(accel[:, 0]))
    feats["accel_std_y"] = float(np.std(accel[:, 1]))
    feats["accel_std_z"] = float(np.std(accel[:, 2]))

    # Dominant frequency via FFT
    fft_mag = np.abs(np.fft.rfft(mag))
    freqs = np.fft.rfftfreq(len(mag), d=1.0/fs)
    if len(fft_mag) > 1:
        dominant_idx = np.argmax(fft_mag[1:]) + 1
        feats["accel_dominant_freq"] = float(freqs[dominant_idx])
    else:
        feats["accel_dominant_freq"] = 0.0

    # Motion energy (band 0.5-5 Hz, typical human motion)
    mask = (freqs >= 0.5) & (freqs <= 5.0)
    feats["accel_motion_energy"] = float(np.sum(fft_mag[mask] ** 2) / (np.sum(fft_mag ** 2) + 1e-8))

    return feats


def extract_wristppg_features(result) -> Dict[str, float]:
    """Extract features from wristppg SimulationResult metadata."""
    feats = {}

    # Beat-level hemodynamic features
    if len(result.stroke_volume_ml) > 0:
        feats["wppg_mean_sv"] = float(np.mean(result.stroke_volume_ml))
        feats["wppg_std_sv"] = float(np.std(result.stroke_volume_ml))
    if len(result.ejection_fraction) > 0:
        feats["wppg_mean_ef"] = float(np.mean(result.ejection_fraction))
        feats["wppg_std_ef"] = float(np.std(result.ejection_fraction))
    if len(result.ptt_s) > 0:
        feats["wppg_mean_ptt"] = float(np.mean(result.ptt_s))
        feats["wppg_std_ptt"] = float(np.std(result.ptt_s))
    if len(result.augmentation_index) > 0:
        feats["wppg_mean_aix"] = float(np.mean(result.augmentation_index))
        feats["wppg_std_aix"] = float(np.std(result.augmentation_index))
    if len(result.pwv_m_s) > 0:
        feats["wppg_mean_pwv"] = float(np.mean(result.pwv_m_s))
        feats["wppg_std_pwv"] = float(np.std(result.pwv_m_s))
    if len(result.hr_instantaneous_bpm) > 0:
        feats["wppg_mean_hr_beats"] = float(np.mean(result.hr_instantaneous_bpm))
        feats["wppg_std_hr_beats"] = float(np.std(result.hr_instantaneous_bpm))

    # Rhythm distribution
    if result.rhythm_labels:
        total = len(result.rhythm_labels)
        feats["wppg_frac_afib"] = sum(1 for r in result.rhythm_labels if r == "afib") / total
        feats["wppg_frac_pvc"] = sum(1 for r in result.rhythm_labels if "pvc" in r) / total
        feats["wppg_frac_sinus"] = sum(1 for r in result.rhythm_labels if r == "sinus") / total

    # Latent physiology
    lp = result.latent_physiology
    feats["wppg_latent_stiffness"] = float(lp.get("stiffness", 1.0))
    feats["wppg_latent_resistance"] = float(lp.get("resistance", 1.0))
    feats["wppg_latent_contractility"] = float(lp.get("contractility", 1.0))
    feats["wppg_latent_vascular_tone"] = float(lp.get("vascular_tone", 0.5))

    return feats


def generate_wristppg_synthetic(
    n_healthy: int = 300,
    n_at_risk: int = 300,
    n_borderline: int = 100,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict], np.ndarray, np.ndarray]:
    """Generate synthetic data using wristppg/ simulator.

    Returns
    -------
    ppgs : list of (7500,) arrays
    accels : list of (7500, 3) arrays
    feats_list : list of feature dicts
    labels : (N,) binary labels
    severities : (N,) severity labels [0=none, 1=mild, 2=moderate, 3=severe]
    """
    from wristppg import WristPPGSimulator, PROFILES

    ppgs, accels, feats_list, labels, severities = [], [], [], [], []
    rng = np.random.default_rng(seed)

    # At-risk profiles and their severity mappings
    at_risk_profiles = [
        ("shock", 3),
        ("hfref", 3),
        ("afib_isolated", 2),
        ("hypovolemia", 2),
        ("sepsis_warm", 2),
        ("hypertension", 1),
        ("diabetes", 1),
        ("hfpef", 1),
        ("aging", 1),
        ("pad", 1),
        ("arterial_stiffness_isolated", 1),
    ]

    activities = ["rest", "rest", "rest", "walking", "sleep"]
    contact_modes = ["good", "good", "good", "loose", "tight"]

    logger.info("Generating %d healthy + %d at-risk + %d borderline wristppg signals...",
                n_healthy, n_at_risk, n_borderline)

    for i in range(n_healthy):
        try:
            sim = WristPPGSimulator(seed=int(rng.integers(0, 2**31)))
            act = rng.choice(activities)
            result = sim.generate(
                profile="healthy",
                duration_s=60.0,
                activity=act,
                contact_mode=rng.choice(contact_modes),
            )
            ppgs.append(result.ppg[:PPG_LENGTH])
            accels.append(result.accel[:PPG_LENGTH])
            feats = {}
            feats.update(extract_wristppg_features(result))
            # For real data, HRV features are extracted at inference; for synthetic we add base_hr
            feats["base_hr"] = float(np.mean(result.hr_instantaneous_bpm)) if len(result.hr_instantaneous_bpm) > 0 else 70.0
            feats_list.append(feats)
            labels.append(0)
            severities.append(0)
        except Exception as e:
            logger.debug("Healthy sample %d failed: %s", i, e)
            continue

    for i in range(n_at_risk):
        try:
            sim = WristPPGSimulator(seed=int(rng.integers(0, 2**31)))
            prof_name, sev = rng.choice(at_risk_profiles)
            act = rng.choice(activities)
            result = sim.generate(
                profile=prof_name,
                duration_s=60.0,
                activity=act,
                contact_mode=rng.choice(contact_modes),
            )
            ppgs.append(result.ppg[:PPG_LENGTH])
            accels.append(result.accel[:PPG_LENGTH])
            feats = {}
            feats.update(extract_wristppg_features(result))
            feats["base_hr"] = float(np.mean(result.hr_instantaneous_bpm)) if len(result.hr_instantaneous_bpm) > 0 else 100.0
            feats_list.append(feats)
            labels.append(1)
            severities.append(int(sev))
        except Exception as e:
            logger.debug("At-risk sample %d failed: %s", i, e)
            continue

    for i in range(n_borderline):
        try:
            sim = WristPPGSimulator(seed=int(rng.integers(0, 2**31)))
            prof_name = rng.choice(["hypertension", "diabetes", "aging", "hfpef"])
            result = sim.generate(
                profile=prof_name,
                duration_s=60.0,
                activity=rng.choice(activities),
                contact_mode=rng.choice(contact_modes),
            )
            ppgs.append(result.ppg[:PPG_LENGTH])
            accels.append(result.accel[:PPG_LENGTH])
            feats = {}
            feats.update(extract_wristppg_features(result))
            feats["base_hr"] = float(np.mean(result.hr_instantaneous_bpm)) if len(result.hr_instantaneous_bpm) > 0 else 85.0
            feats_list.append(feats)
            labels.append(1)
            severities.append(1)
        except Exception as e:
            logger.debug("Borderline sample %d failed: %s", i, e)
            continue

    labels = np.array(labels, dtype=np.float32)
    severities = np.array(severities, dtype=np.int32)
    logger.info("Generated %d wristppg signals (%d healthy, %d at-risk, %d borderline)",
                len(ppgs), int((labels == 0).sum()), int((labels == 1).sum()), n_borderline)
    return ppgs, accels, feats_list, labels, severities


def extract_ppg_features(ppg: np.ndarray, fs: int = 25) -> Dict[str, float]:
    """Extract HRV and morphological features from a PPG signal."""
    from scipy.signal import find_peaks, welch

    feats = {}
    feats["signal_length"] = len(ppg)
    feats["mean_amplitude"] = float(np.mean(ppg))
    feats["std_amplitude"] = float(np.std(ppg))
    feats["sqi"] = float(1.0 - min(1.0, np.std(np.diff(ppg)) / (np.std(ppg) + 1e-8)))

    filt = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
    peaks, _ = find_peaks(filt, distance=int(fs * 0.4), height=0.0)
    if len(peaks) < 5:
        return feats

    rr = np.diff(peaks) / fs * 1000.0
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 3:
        return feats

    # Time-domain HRV
    feats["HRV_MeanNN"] = float(np.mean(rr))
    feats["HRV_SDNN"] = float(np.std(rr, ddof=1))
    feats["HRV_RMSSD"] = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    feats["HRV_SDSD"] = float(np.std(np.diff(rr), ddof=1))
    feats["HRV_CVNN"] = feats["HRV_SDNN"] / (feats["HRV_MeanNN"] + 1e-8)
    feats["HRV_CVSD"] = feats["HRV_RMSSD"] / (feats["HRV_MeanNN"] + 1e-8)
    feats["HRV_MedianNN"] = float(np.median(rr))
    feats["HRV_MadNN"] = float(np.median(np.abs(rr - np.median(rr))))
    feats["HRV_MCVNN"] = feats["HRV_MadNN"] / (feats["HRV_MedianNN"] + 1e-8)
    feats["HRV_IQRNN"] = float(np.percentile(rr, 75) - np.percentile(rr, 25))
    feats["HRV_SDRMSSD"] = feats["HRV_SDNN"] / (feats["HRV_RMSSD"] + 1e-8)
    feats["HRV_Prc20NN"] = float(np.percentile(rr, 20))
    feats["HRV_Prc80NN"] = float(np.percentile(rr, 80))
    feats["HRV_pNN50"] = float(100 * np.sum(np.abs(np.diff(rr)) > 50) / len(rr))
    feats["HRV_pNN20"] = float(100 * np.sum(np.abs(np.diff(rr)) > 20) / len(rr))
    feats["HRV_MinNN"] = float(np.min(rr))
    feats["HRV_MaxNN"] = float(np.max(rr))

    # Frequency-domain HRV
    try:
        rt = np.cumsum(rr) / 1000.0
        rt = rt - rt[0]
        tu = np.arange(0, rt[-1], 0.25)
        ri = np.interp(tu, rt, rr)
        ri = ri - np.mean(ri)
        f, psd = welch(ri, fs=4.0, nperseg=min(len(ri), 256))
        lf_m = (f >= 0.04) & (f < 0.15)
        hf_m = (f >= 0.15) & (f < 0.4)
        lf = float(np.trapz(psd[lf_m], f[lf_m])) if lf_m.any() else 0.0
        hf = float(np.trapz(psd[hf_m], f[hf_m])) if hf_m.any() else 0.0
        tp = lf + hf
        feats.update({"HRV_LF": lf, "HRV_HF": hf, "HRV_TP": tp,
                       "HRV_LFHF": lf / (hf + 1e-8), "HRV_LFn": lf / (tp + 1e-8),
                       "HRV_HFn": hf / (tp + 1e-8), "HRV_LnHF": float(np.log(hf + 1e-8))})
    except Exception:
        pass

    # Poincare
    if len(rr) > 2:
        sd1 = float(np.std(rr[1:] - rr[:-1]) / np.sqrt(2))
        sd2 = float(np.sqrt(2 * np.var(rr) - sd1 ** 2))
        feats.update({"HRV_SD1": sd1, "HRV_SD2": sd2, "HRV_SD1SD2": sd1 / (sd2 + 1e-8)})

    feats["pulse_rate"] = float(len(peaks) / (len(ppg) / fs) * 60.0)

    # --- Signal quality features (distinguish pulsatile from flat+noise) ---
    try:
        # Spectral flatness (Wiener entropy): 1=flat/noise, 0=tonal/pulsatile
        fft_mag = np.abs(np.fft.rfft(ppg - np.mean(ppg)))
        fft_mag = fft_mag[fft_mag > 0]
        feats["spectral_flatness"] = float(
            np.exp(np.mean(np.log(fft_mag))) / (np.mean(fft_mag) + 1e-12)
        )

        # Autocorrelation peak height at expected HR (0.8-2.0 Hz)
        # High = strong periodicity (pulsatile), low = noise
        ac = np.correlate(ppg - np.mean(ppg), ppg - np.mean(ppg), mode='full')
        ac = ac[len(ac)//2:]
        ac = ac / (ac[0] + 1e-12)
        # Find peak in 0.8-2.0 Hz range (lag 12-31 samples at 25Hz)
        lag_lo, lag_hi = int(fs / 2.0), int(fs / 0.8)
        if lag_hi < len(ac):
            ac_pulse_band = ac[lag_lo:lag_hi]
            feats["autocorr_pulse_peak"] = float(np.max(ac_pulse_band)) if len(ac_pulse_band) > 0 else 0.0
        else:
            feats["autocorr_pulse_peak"] = 0.0

        # Peak prominence: real PPG peaks have consistent prominence,
        # noise peaks have random prominence
        from scipy.signal import find_peaks as _find_peaks
        ppg_norm = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-8)
        pks, props = _find_peaks(ppg_norm, distance=int(fs * 0.3), height=0.0, prominence=0.0)
        if len(pks) > 3:
            prominences = props.get("prominences", np.array([]))
            feats["peak_prominence_mean"] = float(np.mean(prominences)) if len(prominences) > 0 else 0.0
            feats["peak_prominence_std"] = float(np.std(prominences)) if len(prominences) > 1 else 0.0
            feats["peak_prominence_cv"] = float(np.std(prominences) / (np.mean(prominences) + 1e-8)) if len(prominences) > 1 else 0.0
        else:
            feats["peak_prominence_mean"] = 0.0
            feats["peak_prominence_std"] = 0.0
            feats["peak_prominence_cv"] = 1.0

        # Kurtosis: Gaussian noise ~3.0, pulsatile PPG ~higher (sharper peaks)
        from scipy.stats import kurtosis as _kurtosis
        feats["ppg_kurtosis"] = float(_kurtosis(ppg - np.mean(ppg)))

        # Spectral concentration: fraction of power in pulse band vs total
        fft_full = np.abs(np.fft.rfft(ppg - np.mean(ppg))) ** 2
        freqs_full = np.fft.rfftfreq(len(ppg), d=1.0/fs)
        pulse_mask = (freqs_full >= 0.8) & (freqs_full <= 2.5)
        total_power = np.sum(fft_full) + 1e-12
        pulse_power = np.sum(fft_full[pulse_mask])
        feats["spectral_concentration"] = float(pulse_power / total_power)

        # Spectral entropy: noise has high entropy, pulsatile has low entropy
        psd_norm = fft_full / (total_power + 1e-12)
        psd_norm = psd_norm[psd_norm > 0]
        feats["spectral_entropy"] = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))

    except Exception:
        feats.setdefault("spectral_flatness", 0.5)
        feats.setdefault("autocorr_pulse_peak", 0.0)
        feats.setdefault("peak_prominence_mean", 0.0)
        feats.setdefault("peak_prominence_std", 0.0)
        feats.setdefault("peak_prominence_cv", 1.0)
        feats.setdefault("ppg_kurtosis", 0.0)
        feats.setdefault("spectral_concentration", 0.0)
        feats.setdefault("spectral_entropy", 5.0)

    return feats


def load_real_data_by_patient():
    """Load real MIMIC/MMASH data organized by patient."""
    signals_df = pd.read_parquet("data/processed/signals.parquet")
    features_df = pd.read_parquet("data/processed/features.parquet")
    patient_groups = {}

    for patient_id, group in signals_df.groupby("patient_id"):
        label = 0 if group.iloc[0]["event_type"] == "CONTROL" else 1
        patient_groups[patient_id] = {
            "label": label, "signals": [], "accels": [], "features": [],
            "event_type": group.iloc[0]["event_type"],
        }
        for idx, row in group.iterrows():
            try:
                if row["window_type"] == "wearable_control":
                    sig = np.load(row["wearable_ppg_path"])
                    fs = 25
                else:
                    sig = np.load(row["raw_ppg_path"])
                    fs = 125
                sig = sig.astype(np.float32)
                if fs != FS_TARGET:
                    sig = scipy_resample(sig, int(len(sig) * FS_TARGET / fs)).astype(np.float32)
                padded = np.zeros(PPG_LENGTH, dtype=np.float32)
                L = min(len(sig), PPG_LENGTH)
                padded[:L] = sig[:L]

                # For real data, no accel available — use zeros
                accel_zeros = np.zeros((PPG_LENGTH, 3), dtype=np.float32)

                feat = {}
                feat_row = features_df.loc[idx] if idx in features_df.index else features_df.iloc[signals_df.index.get_loc(idx)]
                for col in features_df.columns:
                    val = feat_row[col]
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        feat[col] = float(val) if not np.isnan(val) else 0.0

                patient_groups[patient_id]["signals"].append(padded)
                patient_groups[patient_id]["accels"].append(accel_zeros)
                patient_groups[patient_id]["features"].append(feat)
            except Exception:
                continue

    logger.info("Loaded %d real patients (%d healthy, %d at-risk)",
                len(patient_groups),
                sum(1 for p in patient_groups.values() if p["label"] == 0),
                sum(1 for p in patient_groups.values() if p["label"] == 1))
    return patient_groups


def patient_level_split(patient_groups, test_ratio=0.15, val_ratio=0.15, seed=42):
    from sklearn.model_selection import train_test_split
    patients = list(patient_groups.keys())
    labels = [patient_groups[p]["label"] for p in patients]
    pv_train, pv_test = train_test_split(
        list(range(len(patients))), test_size=test_ratio, random_state=seed, stratify=labels)
    pv_train_inner, pv_val = train_test_split(
        pv_train, test_size=val_ratio / (1 - test_ratio), random_state=seed,
        stratify=[labels[i] for i in pv_train])
    train_patients = [patients[i] for i in pv_train_inner]
    val_patients = [patients[i] for i in pv_val]
    test_patients = [patients[i] for i in pv_test]
    logger.info("Patient-level split: Train=%d, Val=%d, Test=%d patients",
                len(train_patients), len(val_patients), len(test_patients))
    assert len(set(train_patients) & set(val_patients)) == 0
    assert len(set(train_patients) & set(test_patients)) == 0
    assert len(set(val_patients) & set(test_patients)) == 0
    return train_patients, val_patients, test_patients


def flatten_patients(patient_groups, patient_list):
    signals, accels, feats, labels = [], [], [], []
    for p in patient_list:
        for sig, accel, feat in zip(
            patient_groups[p]["signals"],
            patient_groups[p]["accels"],
            patient_groups[p]["features"],
        ):
            signals.append(sig)
            accels.append(accel)
            feats.append(feat)
            labels.append(patient_groups[p]["label"])
    return signals, accels, feats, np.array(labels, dtype=np.float32)


def build_arrays(
    signals_list: List[np.ndarray],
    accels_list: List[np.ndarray],
    features_list: List[Dict],
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build numpy arrays for model training."""
    X_ppg = np.zeros((len(signals_list), PPG_LENGTH, 1), dtype=np.float32)
    X_accel = np.zeros((len(accels_list), PPG_LENGTH, 3), dtype=np.float32)
    X_feat = np.zeros((len(features_list), len(feature_cols)), dtype=np.float32)

    for i, sig in enumerate(signals_list):
        L = min(len(sig), PPG_LENGTH)
        X_ppg[i, :L, 0] = sig[:L]

    for i, accel in enumerate(accels_list):
        L = min(len(accel), PPG_LENGTH)
        if accel.ndim == 2:
            X_accel[i, :L, :] = accel[:L]
        else:
            X_accel[i, :L, 0] = accel[:L]

    for i, f in enumerate(features_list):
        for j, col in enumerate(feature_cols):
            X_feat[i, j] = f.get(col, 0.0)

    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
    return X_ppg, X_accel, X_feat


def build_biodata_array(n: int, seed: int = 42) -> np.ndarray:
    """Build biodata array. For synthetic data, sample realistic values."""
    rng = np.random.default_rng(seed)
    biodata = np.zeros((n, 9), dtype=np.float32)
    biodata[:, 0] = rng.uniform(20, 80, n)     # age
    biodata[:, 1] = rng.choice([0, 1], n)       # sex (0=male, 1=female)
    biodata[:, 2] = rng.uniform(18, 40, n)      # bmi
    biodata[:, 3] = rng.integers(0, 8, n)       # comorbidity_count
    biodata[:, 4] = rng.choice([0, 1], n, p=[0.95, 0.05])  # on_vasopressors
    biodata[:, 5] = rng.choice([0, 1], n, p=[0.97, 0.03])  # on_ventilation
    biodata[:, 6] = rng.choice([0, 1], n, p=[0.99, 0.01])  # on_ecmo
    biodata[:, 7] = rng.choice([0, 1], n, p=[0.98, 0.02])  # on_rrt
    biodata[:, 8] = rng.uniform(1, 5, n)        # acuity_score
    return biodata
