import numpy as np
import pandas as pd
from src.dann.inference_v5 import EarlyWarningPredictor

def generate_wrist_noise(n_samples, fs=25):
    """Simulates wrist-PPG specific artifacts: baseline wander and motion spikes."""
    t = np.linspace(0, n_samples / fs, n_samples, endpoint=False)
    # Slow baseline wander (breathing and watch slippage)
    wander = 0.25 * np.sin(2 * np.pi * 0.12 * t) + 0.1 * np.sin(2 * np.pi * 0.03 * t)
    # Micro-volt motion jitter/sensor noise
    jitter = np.random.normal(0, 0.03, n_samples)
    return wander + jitter

def run_simulation(scenario_name, signal, time_axis, pipeline, log_interval=10, arrest_time_min=None):
    """Streams a continuous signal window by window through the predictor wrapper."""
    print(f"\n🚀 Running Evaluation: {scenario_name}")
    print(f"{'Time (HH:MM)':<15}{'Risk Level':<15}{'Probability':<15}{'Trigger Status / Message':<35}")
    print("-" * 80)
    
    window_size = 1500  # 60s @ 25Hz
    step_size = 1500    # Advance by 1 minute increments for readable logs
    
    alert_triggered_at_min = None
    
    for start_idx in range(0, len(signal) - window_size, step_size):
        end_idx = start_idx + window_size
        current_minute = time_axis[end_idx] / 60.0
        
        segment = signal[start_idx:end_idx]
        output = pipeline.process_stream_step(segment)
        
        hh = int(current_minute // 60)
        mm = int(current_minute % 60)
        time_str = f"{hh:02d}:{mm:02d}"
        
        trigger_msg = output.get("early_warning_trigger", output.get("status_message", "Stable Normal Rhythm"))
        
        # Log status matches or alerts
        if mm % log_interval == 0 or output['alert'] or "UNSTABLE" in output['risk_level']:
            print(f"{time_str:<15}{output['risk_level']:<15}{output['probability']:<15.4f}{trigger_msg:<35}")
            
        if output['alert'] and alert_triggered_at_min is None:
            alert_triggered_at_min = current_minute
            
    if arrest_time_min is not None:
        print("\n--- PERFORMANCE METRICS ---")
        if alert_triggered_at_min is not None:
            lead_time = arrest_time_min - alert_triggered_at_min
            print(f"✨ SUCCESS: Advanced Wrist-PPG Warning Provided = {lead_time:.1f} Minutes in advance!")
        else:
            print("❌ FAILURE: The model failed to alert before the clinical flatline.")

# --- SCENARIO EMULATORS ---

def build_healthy_day_stream(fs=25):
    """Scenario 1: 45 minutes of healthy daily life (typing, walking, resting)."""
    duration_min = 45
    n_samples = fs * duration_min * 60
    t = np.linspace(0, duration_min * 60, n_samples, endpoint=False)
    
    # Baseline stable pulse (72 BPM)
    phase = 2 * np.pi * (72 / 60) * t
    clean_pulse = 0.15 * np.sin(phase) # Wrist capillary amplitude is weak (0.15)
    
    signal = clean_pulse + generate_wrist_noise(n_samples, fs)
    
    # Inject heavy walking motion spikes between minutes 15 and 25
    motion_start, motion_end = fs * 15 * 60, fs * 25 * 60
    signal[motion_start:motion_end] += 0.8 * np.sin(2 * np.pi * 1.8 * t[motion_start:motion_end]) # 1.8Hz step cadence
    
    return signal, t

def build_workout_recovery_stream(fs=25):
    """Scenario 2: 30 minutes of a healthy heart recovering from a run (140 down to 72 BPM)."""
    duration_min = 30
    n_samples = fs * duration_min * 60
    t = np.linspace(0, duration_min * 60, n_samples, endpoint=False)
    
    bpm_profile = np.zeros(n_samples)
    for i, ts in enumerate(t):
        current_min = ts / 60.0
        # Exponential heart rate decline down from workout peak
        bpm_profile[i] = 72 + 68 * np.exp(-current_min / 8.0)
        
    phase = 2 * np.pi * np.cumsum(bpm_profile / 60.0) / fs
    signal = 0.18 * np.sin(phase) + generate_wrist_noise(n_samples, fs)
    return signal, t

def build_hours_ahead_arrest_stream(fs=25):
    """Scenario 3: 3-hour resting cardiac decline culminating in an arrest at minute 170."""
    duration_min = 180
    n_samples = fs * duration_min * 60
    t = np.linspace(0, duration_min * 60, n_samples, endpoint=False)
    
    bpm_profile = np.zeros(n_samples)
    amplitude_profile = np.zeros(n_samples)
    
    for i, ts in enumerate(t):
        current_min = ts / 60.0
        if current_min < 60.0:        # Hour 1: Healthy rest
            bpm_profile[i] = 70
            amplitude_profile[i] = 0.15
        elif current_min < 150.0:     # Mins 60-150: Silent trend drop
            pct = (current_min - 60.0) / 90.0
            bpm_profile[i] = 70 - (16 * pct)
            amplitude_profile[i] = 0.15 - (0.05 * pct)
        elif current_min < 170.0:     # Mins 150-170: Severe progressive collapse
            pct = (current_min - 150.0) / 20.0
            bpm_profile[i] = 54 - (16 * pct)
            amplitude_profile[i] = 0.10 - (0.08 * pct)
        else:                         # Mins 170-180: Death flatline
            bpm_profile[i] = 0.0
            amplitude_profile[i] = 0.001
            
    phase = 2 * np.pi * np.cumsum(bpm_profile / 60.0) / fs
    signal = amplitude_profile * np.sin(phase) + generate_wrist_noise(n_samples, fs)
    return signal, t

# --- EXECUTION ENTRANCE ---

if __name__ == "__main__":
    print("=== Smartwatch Wrist-PPG Production Simulation ===")
    
    MODEL_PATH = "models/cardiac_arrest_v6/"
    predictor_pipeline = EarlyWarningPredictor(MODEL_PATH)
    
    # 1. Evaluate Daily Motion Performance
    healthy_sig, healthy_t = build_healthy_day_stream()
    run_simulation("Healthy Active Day (Motion Testing)", healthy_sig, healthy_t, predictor_pipeline, log_interval=5)
    
    # Re-initialize trends between sessions
    predictor_pipeline.hr_trend_history = []
    
    # 2. Evaluate Post-Exercise Performance
    workout_sig, workout_t = build_workout_recovery_stream()
    run_simulation("Post-Workout Post-Exercise Recovery", workout_sig, workout_t, predictor_pipeline, log_interval=5)
    
    predictor_pipeline.hr_trend_history = []
    
    # 3. Evaluate Long-Term Pre-Arrest Decline
    arrest_sig, arrest_t = build_hours_ahead_arrest_stream()
    run_simulation("3-Hour Gradual Cardiac Collapse", arrest_sig, arrest_t, predictor_pipeline, log_interval=10, arrest_time_min=170)
