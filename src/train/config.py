"""Central configuration for dataset paths, PhysioNet credentials, and training hyperparameters."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

for d in (RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# PhysioNet credentials (set via env vars or .env file)
# ---------------------------------------------------------------------------
import os

PHYSIONET_USER = os.getenv("PHYSIONET_USER", "")
PHYSIONET_PASS = os.getenv("PHYSIONET_PASS", "")

# ---------------------------------------------------------------------------
# Dataset registry — PhysioNet URLs and local target dirs
# ---------------------------------------------------------------------------
DATASETS: dict[str, dict] = {
    "mimic3_waveform": {
        "physionet_name": "mimic3wdb",
        "version": "1.0",
        "url": "https://physionet.org/files/mimic3wdb/1.0/",
        "local_dir": RAW_DIR / "mimic3_waveform",
        "description": "MIMIC-III Waveform Database — ICU continuous signals (ECG, ABP, PPG, SpO2).",
        "size_gb": 50,
    },
    "mimic4_waveform": {
        "physionet_name": "mimic4wdb",
        "version": "0.1.0",
        "url": "https://physionet.org/files/mimic4wdb/0.1.0/",
        "local_dir": RAW_DIR / "mimic4_waveform",
        "description": "MIMIC-IV Waveform Database — newer ICU PPG/ECG waveforms.",
        "size_gb": 80,
    },
    "cves": {
        "physionet_name": "cves",
        "version": "1.0.0",
        "url": "https://physionet.org/files/cves/1.0.0/",
        "local_dir": RAW_DIR / "cves",
        "description": "Cerebral Vasoregulation in Elderly with Stroke — ECG, accel, BP from 60 stroke patients + 60 controls.",
    },
    "sleep_accel": {
        "physionet_name": "sleep-accel",
        "version": "1.0.0",
        "url": "https://physionet.org/files/sleep-accel/1.0.0/",
        "local_dir": RAW_DIR / "sleep_accel",
        "description": "Apple Watch wrist accelerometry + PPG HR for 31 subjects with sleep labels.",
    },
    "mmash": {
        "physionet_name": "mmash",
        "version": "1.0.0",
        "url": "https://physionet.org/files/mmash/1.0.0/",
        "local_dir": RAW_DIR / "mmash",
        "description": "Multilevel Monitoring of Activity and Sleep — beat-to-beat RR intervals + 3-axis accel, 22 subjects.",
    },
    "paf_prediction": {
        "physionet_name": "afp",
        "version": "1.0.0",
        "url": "https://physionet.org/files/afp/1.0.0/",
        "local_dir": RAW_DIR / "paf_prediction",
        "description": "PAF Prediction Challenge — 100 long-term ECG segments, half preceding paroxysmal AF.",
    },
    "afdb": {
        "physionet_name": "afdb",
        "version": "1.0.0",
        "url": "https://physionet.org/files/afdb/1.0.0/",
        "local_dir": RAW_DIR / "afdb",
        "description": "MIT-BIH Atrial Fibrillation Database — 25 long-duration AF ECG recordings.",
    },
    "non_eeg_neuro": {
        "physionet_name": "nneuro",
        "version": "1.0.0",
        "url": "https://physionet.org/files/nneuro/1.0.0/",
        "local_dir": RAW_DIR / "non_eeg_neuro",
        "description": "Non-EEG Neurological Status — EDA, temp, accel, HR, SpO2 from 20 adults under stress.",
    },
}

# ---------------------------------------------------------------------------
# Signal processing constants
# ---------------------------------------------------------------------------
SAMPLING_RATE_HZ: int = 125  # defaultFs for PhysioNet waveform databases
RESAMPLE_RATE_HZ: int = 1     # 1-minute aggregates → resample to 1 Hz
BANDPASS_LOW_HZ: float = 0.5
BANDPASS_HIGH_HZ: float = 8.0
BUTTERWORTH_ORDER: int = 4

# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
WINDOW_HOURS: list[int] = [24, 48, 72]
SLIDE_HOURS: int = 1

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
HRV_FEATURES: list[str] = [
    "mean_rr", "sd_rr", "rmssd", "sdsd", "pnn50",
    "cv_rr", "median_rr", "range_rr", "iqr_rr",
    "lf_power", "hf_power", "lf_hf_ratio",
    "sample_entropy", "approximate_entropy",
]

STATIC_FEATURES: list[str] = [
    "age", "sex", "cha2ds2_vasc", "has_af", "has_hypertension",
    "has_diabetes", "has_chf", "has_vascular_disease",
]

# ---------------------------------------------------------------------------
# Model / training hyperparameters
# ---------------------------------------------------------------------------
WINDOW_LENGTH_MINUTES: int = 1440  # 24 h at 1-min resolution
N_TIME_FEATURES: int = 32         # HRV + activity + derived
N_STATIC_FEATURES: int = len(STATIC_FEATURES)

LSTM_UNITS: int = 64
DENSE_UNITS: int = 64
DROPOUT: float = 0.3
LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 32
EPOCHS: int = 100
EARLY_STOP_PATIENCE: int = 10
VAL_SPLIT: float = 0.15
TEST_SPLIT: float = 0.15
CLASS_WEIGHT_STROKE: float = 10.0  # upweight rare stroke events

RANDOM_SEED: int = 42
