#!/usr/bin/env python3
"""
OHCA Data Generator v3 — Production-Grade
==========================================
Key insight: Pre-arrest patients DO show pathological features consistently.
The challenge isn't absence — it's variability and noise. This version makes
features learnable while keeping realistic overlap and noise.
"""

import numpy as np
from pathlib import Path

np.random.seed(42)

ECG_FS = 130
ACC_FS = 25
WINDOW_SEC = 10
ECG_LEN = ECG_FS * WINDOW_SEC
ACC_LEN = ACC_FS * WINDOW_SEC
N_SAMPLES = 4000  # More data for better generalization
N_POS = N_SAMPLES // 2
N_NEG = N_SAMPLES // 2

BEAT_TEMPLATE = np.array([
    [0.12,  -0.18,  0.025],
    [-0.10, -0.04,  0.010],
    [1.00,   0.00,  0.012],
    [-0.15,  0.04,  0.010],
    [0.25,   0.00,  0.060],
], dtype=np.float64)


def generate_beat_timestamps(duration_sec, hr_bpm, hrv_std):
    mean_rr = 60.0 / hr_bpm
    n_beats = int(duration_sec / mean_rr) + 4
    rr_intervals = np.random.normal(mean_rr, hrv_std, n_beats)
    rr_intervals = np.clip(rr_intervals, mean_rr * 0.75, mean_rr * 1.25)
    beat_times = np.cumsum(rr_intervals) - rr_intervals[0]
    beat_times = beat_times[beat_times >= -0.05]
    return beat_times, rr_intervals


def make_ecg_waveform(duration_sec, hr_bpm, hrv_std, pathology_type='none',
                      apply_artifact=False):
    """
    pathology_type:
      - 'none': Normal ECG
      - 'early': Subtle changes (early ischemia)
      - 'moderate': Clear ST changes, mild QT prolongation
      - 'severe': Wide QRS, significant ST elevation/depression, prolonged QT
    """
    t = np.linspace(0, duration_sec, ECG_LEN, endpoint=False)
    ecg = np.zeros(ECG_LEN, dtype=np.float64)

    beat_times, rr_intervals = generate_beat_timestamps(duration_sec, hr_bpm, hrv_std)

    for idx, bt in enumerate(beat_times):
        if idx >= len(rr_intervals):
            break
        rr = rr_intervals[idx]
        template = BEAT_TEMPLATE.copy()

        qt_corrected = 0.41
        actual_qt = qt_corrected * np.sqrt(rr)
        t_center = actual_qt * 0.65

        # Pathological modifications based on severity
        if pathology_type == 'early':
            if np.random.random() < 0.6:
                t_center += np.random.uniform(0.02, 0.05)
            if np.random.random() < 0.4:
                st_off = np.random.uniform(-0.12, 0.18)
                st_start = bt + 0.06
                st_end = bt + t_center * 0.45
                st_mask = (t >= st_start) & (t <= st_end)
                ecg[st_mask] += st_off

        elif pathology_type == 'moderate':
            t_center += np.random.uniform(0.03, 0.08)
            if np.random.random() < 0.5:
                template[2, 2] = np.random.uniform(0.018, 0.030)
            st_off = np.random.choice([
                np.random.uniform(0.15, 0.30),
                np.random.uniform(-0.25, -0.10),
                0.0
            ], p=[0.4, 0.4, 0.2])
            if st_off != 0.0:
                st_start = bt + 0.06
                st_end = bt + t_center * 0.45
                st_mask = (t >= st_start) & (t <= st_end)
                ecg[st_mask] += st_off

        elif pathology_type == 'severe':
            t_center += np.random.uniform(0.05, 0.12)
            template[1, 2] = np.random.uniform(0.020, 0.035)
            template[2, 2] = np.random.uniform(0.025, 0.045)
            template[3, 2] = np.random.uniform(0.020, 0.035)
            st_off = np.random.choice([
                np.random.uniform(0.20, 0.40),
                np.random.uniform(-0.35, -0.15),
                0.0
            ], p=[0.45, 0.45, 0.10])
            if st_off != 0.0:
                st_start = bt + 0.06
                st_end = bt + t_center * 0.45
                st_mask = (t >= st_start) & (t <= st_end)
                ecg[st_mask] += st_off

        template[4, 1] = t_center - bt

        for peak_amp, peak_center, peak_width in template:
            tc = bt + peak_center
            ecg += peak_amp * np.exp(-((t - tc) ** 2) / (2 * peak_width ** 2))

    # Baseline wander (respiration) — applied to ALL signals equally
    n_wander = np.random.randint(1, 4)
    for _ in range(n_wander):
        freq = np.random.uniform(0.1, 0.5)
        amp = np.random.uniform(0.05, 0.20)
        phase = np.random.uniform(0, 2 * np.pi)
        ecg += amp * np.sin(2 * np.pi * freq * t + phase)

    # Powerline interference (50/60 Hz)
    if np.random.random() < 0.3:
        pl_freq = np.random.choice([50, 60])
        ecg += np.random.uniform(0.01, 0.04) * np.sin(2 * np.pi * pl_freq * t + np.random.uniform(0, 2*np.pi))

    # Muscle artifact (EMG noise)
    if np.random.random() < 0.25:
        n_bursts = np.random.randint(1, 4)
        for _ in range(n_bursts):
            start = np.random.randint(0, ECG_LEN - 30)
            length = np.random.randint(5, 30)
            ecg[start:start+length] += np.random.normal(0, np.random.uniform(0.1, 0.25), length)

    # Electrode contact noise
    if apply_artifact and np.random.random() < 0.3:
        start = np.random.randint(0, ECG_LEN - 100)
        length = np.random.randint(20, 100)
        ecg[start:start+length] += np.random.normal(0, 0.4, length)

    # Additive Gaussian noise floor
    ecg += np.random.normal(0, 0.02, ecg.shape)

    return ecg.astype(np.float32).reshape(-1, 1)


def make_acc_signal(duration_sec, is_pre_arrest):
    t = np.linspace(0, duration_sec, ACC_LEN, endpoint=False)
    acc = np.zeros((ACC_LEN, 3), dtype=np.float64)

    # Base gravity
    acc[:, 2] += 1.0

    if not is_pre_arrest:
        # Normal activity: walking, sitting, light movement
        activity_type = np.random.choice(['sedentary', 'walking', 'standing'], p=[0.4, 0.4, 0.2])

        if activity_type == 'sedentary':
            acc += np.random.normal(0, 0.03, acc.shape)
            acc[:, 2] += np.random.uniform(-0.05, 0.05)
        elif activity_type == 'walking':
            freq = np.random.uniform(1.5, 2.5)
            for axis in range(3):
                amp = np.random.uniform(0.05, 0.20)
                phase = np.random.uniform(0, 2 * np.pi)
                acc[:, axis] += amp * np.sin(2 * np.pi * freq * t + phase)
            acc += np.random.normal(0, 0.05, acc.shape)
        else:  # standing
            for axis in range(3):
                freq = np.random.uniform(0.3, 1.5)
                amp = np.random.uniform(0.02, 0.10)
                phase = np.random.uniform(0, 2 * np.pi)
                acc[:, axis] += amp * np.sin(2 * np.pi * freq * t + phase)
            acc += np.random.normal(0, 0.04, acc.shape)
    else:
        # Pre-arrest: various scenarios
        scenario = np.random.random()

        if scenario < 0.25:
            # Syncope/fall event
            fall_idx = np.random.randint(20, 200)
            # Pre-fall: normal or slightly unsteady
            tremor = np.random.uniform(0.02, 0.06)
            for axis in range(3):
                freq = np.random.uniform(0.5, 2.0)
                amp = np.random.uniform(0.03, 0.12)
                phase = np.random.uniform(0, 2 * np.pi)
                acc[:, axis] += amp * np.sin(2 * np.pi * freq * t + phase)
            acc += np.random.normal(0, tremor, acc.shape)

            # Impact
            impact_len = min(np.random.randint(3, 8), ACC_LEN - fall_idx)
            if impact_len > 0:
                impact_amp = np.random.uniform(1.5, 3.0)
                acc[fall_idx:fall_idx+impact_len, :] += np.random.normal(0, impact_amp, (impact_len, 3))

            # Post-fall: lying down with micro-movements
            post_start = fall_idx + impact_len
            if post_start < ACC_LEN:
                orientation = np.random.uniform(-0.8, 0.8, 3)
                orientation[2] = np.random.uniform(0.1, 0.5)  # Not perfectly flat
                acc[post_start:, :] = orientation
                # Micro-tremors (agonal breathing, muscle fasciculations)
                for axis in range(3):
                    tremor_freq = np.random.uniform(3, 8)
                    acc[post_start:, axis] += 0.015 * np.sin(2 * np.pi * tremor_freq * t[post_start:])
                acc[post_start:, :] += np.random.normal(0, 0.02, (ACC_LEN - post_start, 3))

        elif scenario < 0.50:
            # Bradycardia/weak pulse — very low amplitude, irregular
            for axis in range(3):
                freq = np.random.uniform(0.3, 1.0)
                amp = np.random.uniform(0.01, 0.06)
                phase = np.random.uniform(0, 2 * np.pi)
                acc[:, axis] += amp * np.sin(2 * np.pi * freq * t + phase)
            # Add irregular tremor
            tremor_freq = np.random.uniform(4, 7)
            for axis in range(3):
                acc[:, axis] += 0.02 * np.sin(2 * np.pi * tremor_freq * t + np.random.uniform(0, 2*np.pi))
            acc += np.random.normal(0, 0.04, acc.shape)

        elif scenario < 0.70:
            # Agonal breathing pattern — slow, deep respirations
            resp_freq = np.random.uniform(0.1, 0.3)  # Very slow
            for axis in range(3):
                acc[:, axis] += 0.08 * np.sin(2 * np.pi * resp_freq * t + np.random.uniform(0, 2*np.pi))
            acc += np.random.normal(0, 0.05, acc.shape)

        else:
            # Patient still conscious but deteriorating — slight tremor, elevated HR
            for axis in range(3):
                freq = np.random.uniform(0.5, 2.0)
                amp = np.random.uniform(0.03, 0.10)
                phase = np.random.uniform(0, 2 * np.pi)
                acc[:, axis] += amp * np.sin(2 * np.pi * freq * t + phase)
            # Fine tremor
            tremor_freq = np.random.uniform(6, 10)
            for axis in range(3):
                acc[:, axis] += 0.03 * np.sin(2 * np.pi * tremor_freq * t + np.random.uniform(0, 2*np.pi))
            acc += np.random.normal(0, 0.05, acc.shape)

    return acc.astype(np.float32)


def make_biodata(age, sex, ischemia, bp_sys, bp_dia, spo2):
    norm_age = np.clip((age - 18) / (95 - 18), 0, 1).astype(np.float32)
    norm_bp_sys = np.clip((bp_sys - 80) / (200 - 80), 0, 1).astype(np.float32)
    norm_bp_dia = np.clip((bp_dia - 40) / (120 - 40), 0, 1).astype(np.float32)
    norm_spo2 = np.clip((spo2 - 80) / (100 - 80), 0, 1).astype(np.float32)
    return np.array([norm_age, float(sex), float(ischemia),
                     norm_bp_sys, norm_bp_dia, norm_spo2], dtype=np.float32)


def generate_dataset():
    ecg_list, acc_list, bio_list, labels = [], [], [], []

    for i in range(N_SAMPLES):
        is_pre = i < N_POS

        # Age — significant overlap
        if is_pre:
            age = np.clip(np.random.normal(62, 13), 25, 90)
        else:
            age = np.clip(np.random.normal(52, 15), 25, 90)

        sex = np.random.randint(0, 2)
        ischemia = int(np.random.random() < (0.50 if is_pre else 0.20))

        # Vital signs with overlap
        if is_pre:
            hr = np.random.uniform(45, 135)  # Can be bradycardic OR tachycardic
            hrv = np.random.uniform(0.008, 0.035)
            bp_sys = np.random.normal(95, 20)  # Often hypotensive
            bp_dia = np.random.normal(55, 12)
            spo2 = np.random.normal(88, 6)  # Often hypoxic
        else:
            hr = np.random.uniform(55, 110)
            hrv = np.random.uniform(0.018, 0.050)
            bp_sys = np.random.normal(120, 18)
            bp_dia = np.random.normal(78, 10)
            spo2 = np.random.normal(97, 2)

        bp_sys = np.clip(bp_sys, 70, 200)
        bp_dia = np.clip(bp_dia, 35, 120)
        spo2 = np.clip(spo2, 75, 100)

        # Pathology assignment — high probability for pre-arrest
        if is_pre:
            path_roll = np.random.random()
            if path_roll < 0.20:
                pathology = 'severe'
            elif path_roll < 0.55:
                pathology = 'moderate'
            elif path_roll < 0.80:
                pathology = 'early'
            else:
                pathology = 'none'  # 20% with no visible ECG changes yet
        else:
            path_roll = np.random.random()
            if path_roll < 0.03:
                pathology = 'early'  # 3% of healthy have incidental findings
            else:
                pathology = 'none'

        apply_artifact = np.random.random() < 0.4

        ecg = make_ecg_waveform(10.0, hr, hrv, pathology, apply_artifact)
        acc = make_acc_signal(10.0, is_pre)
        bio = make_biodata(age, sex, ischemia, bp_sys, bp_dia, spo2)

        ecg_list.append(ecg)
        acc_list.append(acc)
        bio_list.append(bio)
        labels.append(1 if is_pre else 0)

    return {
        'ecg': np.array(ecg_list),
        'acc': np.array(acc_list),
        'bio': np.array(bio_list),
        'labels': np.array(labels, dtype=np.float32)
    }


def main():
    print("Generating OHCA dataset v3 (4000 samples, 6-dim biodata)...")
    data = generate_dataset()

    indices = np.random.permutation(N_SAMPLES)
    ecg = data['ecg'][indices]
    acc = data['acc'][indices]
    bio = data['bio'][indices]
    labels = data['labels'][indices]

    split = int(0.8 * N_SAMPLES)
    X_ecg_train, X_ecg_test = ecg[:split], ecg[split:]
    X_acc_train, X_acc_test = acc[:split], acc[split:]
    X_bio_train, X_bio_test = bio[:split], bio[split:]
    y_train, y_test = labels[:split], labels[split:]

    save_dir = Path(__file__).parent
    np.save(save_dir / 'X_ecg_train.npy', X_ecg_train)
    np.save(save_dir / 'X_ecg_test.npy', X_ecg_test)
    np.save(save_dir / 'X_acc_train.npy', X_acc_train)
    np.save(save_dir / 'X_acc_test.npy', X_acc_test)
    np.save(save_dir / 'X_bio_train.npy', X_bio_train)
    np.save(save_dir / 'X_bio_test.npy', X_bio_test)
    np.save(save_dir / 'y_train.npy', y_train)
    np.save(save_dir / 'y_test.npy', y_test)

    print(f"Saved to {save_dir}/")
    print(f"  ECG train: {X_ecg_train.shape}, test: {X_ecg_test.shape}")
    print(f"  ACC train: {X_acc_train.shape}, test: {X_acc_test.shape}")
    print(f"  BIO train: {X_bio_train.shape}, test: {X_bio_test.shape}")
    print(f"  Labels: train pos={int(y_train.sum())}, neg={int(len(y_train)-y_train.sum())}")
    print(f"  Labels: test  pos={int(y_test.sum())}, neg={int(len(y_test)-y_test.sum())}")
    print(f"  Biodata features: [age, sex, ischemia, bp_sys, bp_dia, spo2]")


if __name__ == '__main__':
    main()
