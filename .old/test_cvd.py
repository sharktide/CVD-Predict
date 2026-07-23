import numpy as np
from src.dann.inference import CardiacArrestPredictor

# 1. Initialize your production model path
MODEL_PATH = "models/cardiac_arrest_v4/"
predictor = CardiacArrestPredictor(MODEL_PATH)


# 2. Setup your structural testing dimensions (1500 samples @ 25Hz = 60s)
fs = 25
duration_sec = 60
n_samples = fs * duration_sec  # 1500
t = np.linspace(0, duration_sec, n_samples, endpoint=False)


# --- TEST CASE A: Normal Stable Heart Rate (~72 BPM) ---
# Emulating standard systolic/diastolic curves + typical background sensor jitter
fundamental_hr_hz = 72 / 60  
clean_signal = 0.5 * np.sin(2 * np.pi * fundamental_hr_hz * t) + 0.2 * np.sin(2 * np.pi * (2 * fundamental_hr_hz) * t)
normal_ppg_signal = clean_signal + np.random.normal(0, 0.03, n_samples)

print("--- Running Inference on Normal Signal ---")
result_normal = predictor.predict_ppg(normal_ppg_signal)
print(f"Result: {result_normal}\n")


# --- TEST CASE B: Flatline / Micro-voltage Ventricular Fibrillation ---
# Emulating a cardiac arrest event where pulse pressure vanishes, leaving only line noise
arrest_ppg_signal = np.random.normal(0, 0.002, n_samples)

print("--- Running Inference on Cardiac Arrest Signal ---")
result_arrest = predictor.predict_ppg(arrest_ppg_signal)
print(f"Result: {result_arrest}")
