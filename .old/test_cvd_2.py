import numpy as np
from src.dann.inference import CardiacArrestPredictor, EarlyWarningPredictor

MODEL_PATH = "models/cardiac_arrest_v4/"
predictor = EarlyWarningPredictor(MODEL_PATH)

fs = 25
duration_sec = 60
n_samples = fs * duration_sec  
t = np.linspace(0, duration_sec, n_samples, endpoint=False)

# --- SCENARIO 1: Severe Progressive Bradycardia (Heart Crashing) ---
# Simulates a heart dropping from a weak 50 BPM down to an agonal 25 BPM.
# This represents a patient losing cardiac output minutes before flatlining.
start_bpm = 50
end_bpm = 25
bpm_trend = np.linspace(start_bpm, end_bpm, n_samples)
phase = 2 * np.pi * np.cumsum(bpm_trend / 60) / fs

# Signal amplitude also fades out as blood pressure drops
amplitude_decay = np.linspace(0.6, 0.15, n_samples)
bradycardia_signal = amplitude_decay * np.sin(phase) + np.random.normal(0, 0.02, n_samples)

print("--- Scenario 1: Severe Progressive Bradycardia ---")
print("Physiology: Heart rate drops 50 -> 25 BPM; pulse pressure fades.")
print(f"Result: {predictor.process_stream_step(bradycardia_signal)}\n")


# --- SCENARIO 2: Autonomic Breakdown (Anatomy of a Sudden Collapse) ---
# High Heart Rate Variability (HRV) is usually healthy, but chaotic, fragmented 
# variability with erratic amplitude indicates impending ventricular fibrillation.
base_hr_hz = 110 / 60  # Tachycardia baseline (110 BPM)
chaotic_frequency_modulation = 0.4 * np.sin(2 * np.pi * 0.15 * t) * np.random.normal(1, 0.3, n_samples)
chaotic_phase = 2 * np.pi * (base_hr_hz * t + chaotic_frequency_modulation)

# Erratic pulse volumes (pulsus alternans/erratic amplitude)
erratic_amplitude = 0.4 + 0.2 * np.sin(2 * np.pi * 0.05 * t) + np.random.normal(0, 0.05, n_samples)
chaotic_signal = erratic_amplitude * np.sin(chaotic_phase) + np.random.normal(0, 0.03, n_samples)

print("--- Scenario 2: Chaotic Autonomic Breakdown ---")
print("Physiology: Unstable tachycardia with severe amplitude fluctuations.")
print(f"Result: {predictor.process_stream_step(chaotic_signal)}\n")


# --- SCENARIO 3: Sudden Structural Collapse (The 'Golden Minute') ---
# The first 30 seconds are normal (75 BPM), followed by a sudden plunge 
# into agonal rhythm and micro-voltage baseline within the same window.
normal_part = 0.5 * np.sin(2 * np.pi * (75/60) * t[:n_samples//2])
arrest_part = np.random.normal(0, 0.005, n_samples//2) # Flatline sets in halfway
mixed_signal = np.concatenate([normal_part, arrest_part]) + np.random.normal(0, 0.02, n_samples)

print("--- Scenario 3: Mixed Mid-Window Collapse ---")
print("Physiology: First 30s stable, final 30s transitions to flatline.")
print(f"Result: {predictor.process_stream_step(mixed_signal)}\n")
