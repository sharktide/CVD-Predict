import numpy as np
import pandas as pd
from src.dann.onnx_infer import ProductionONNXPredictor as EarlyWarningPredictor
#from src.dann.inference import FeatureExtractor, CardiacArrestPredictor, EarlyWarningPredictor

def simulate_three_hour_slow_collapse(fs=25):
    """
    Generates a continuous 3-hour streaming PPG signal timeline (180 minutes).
    - Mins 0 to 60 (1 Hr)   : Perfect healthy baseline (72 BPM).
    - Mins 60 to 150 (1.5 Hr): Slow, silent progressive ischemia (HR creeps 72 -> 56 BPM).
    - Mins 150 to 170 (20 Min): Systemic failure onset (HR plunges 56 -> 40 BPM, amplitude drops).
    - Mins 170 to 180 (10 Min): Critical Agonal phase and terminal flatline.
    """
    total_duration_sec = 180 * 60  # 10,800 seconds
    total_samples = fs * total_duration_sec
    t = np.linspace(0, total_duration_sec, total_samples, endpoint=False)
    
    bpm_profile = np.zeros(total_samples)
    amplitude_profile = np.zeros(total_samples)
    
    for i, timestamp in enumerate(t):
        current_minute = timestamp / 60.0
        
        if current_minute < 60.0:      # Hour 1: Normal
            bpm_profile[i] = 72 + 2 * np.sin(2 * np.pi * 0.2 * timestamp)
            amplitude_profile[i] = 0.5
            
        elif current_minute < 150.0:   # Mins 60-150: The Silent 1.5-Hour Decline
            pct = (current_minute - 60.0) / 90.0
            bpm_profile[i] = 72 - (16 * pct) + np.random.normal(0, 1.5) # Gradual degradation
            amplitude_profile[i] = 0.5 - (0.1 * pct)
            
        elif current_minute < 170.0:   # Mins 150-170: Severe Collapse Acceleration
            pct = (current_minute - 150.0) / 20.0
            bpm_profile[i] = 56 - (16 * pct) + np.random.normal(0, 3.0)
            amplitude_profile[i] = 0.4 - (0.25 * pct)
            
        else:                          # Mins 170-180: Death / Flatline
            bpm_profile[i] = 0.0
            amplitude_profile[i] = 0.001
            
    phase = 2 * np.pi * np.cumsum(bpm_profile / 60) / fs
    raw_signal = amplitude_profile * np.sin(phase) + np.random.normal(0, 0.02, total_samples)
    
    return raw_signal, t

if __name__ == "__main__":
    print("=== Constructing 3-Hour Continuous Smartwatch Stream ===")
    raw_stream, time_axis = simulate_three_hour_slow_collapse(fs=25)
    
    MODEL_PATH = "models/cardiac_arrest_v4/cardiac_arrest_detector.onnx"
    pipeline = EarlyWarningPredictor(MODEL_PATH)
    
    window_size = 1500  # 60s window
    step_size = 1500    # Slide forward by 1 minute per log step for readable tracking output
    
    alert_triggered_at_min = None
    time_of_arrest_min = 170.0  # Flatline sets in at Minute 170
    
    print("\nProcessing streaming windows across hours...")
    print(f"{'Time (HH:MM)':<15}{'Risk Level':<15}{'Probability':<15}{'Status':<10}")
    print("-" * 60)

    for start_idx in range(0, len(raw_stream) - window_size, step_size):
        end_idx = start_idx + window_size
        current_time_stamp_sec = time_axis[end_idx]
        current_minute = current_time_stamp_sec / 60.0
        
        segment = raw_stream[start_idx:end_idx]
        output = pipeline.process_stream_step(segment)
        
        # Format timestamps into clean clinical reading format
        hh = int(current_minute // 60)
        mm = int(current_minute % 60)
        time_str = f"{hh:02d}:{mm:02d}"
        
        # Print logs only every 10 minutes to avoid terminal flooding, or always print if an alert triggers
        if mm % 10 == 0 or output['alert']:
            print(f"{time_str:<15}{output['risk_level']:<15}{output['probability']:<15.4f}{'🚨 ALERT' if output['alert'] else 'OK'}")
        
        if output['alert'] and alert_triggered_at_min is None:
            alert_triggered_at_min = current_minute

    print("\n" + "="*45)
    print("=== CLINICAL LONG-TERM PERFORMANCE METRICS ===")
    print("="*45)
    print(f"True Medical Collapse Expected At   : 02:50 (Minute 170)")
    
    if alert_triggered_at_min is not None:
        lead_time_minutes = time_of_arrest_min - alert_triggered_at_min
        alert_hh = int(alert_triggered_at_min // 60)
        alert_mm = int(alert_triggered_at_min % 60)
        
        print(f"First Early Warning Triggered At    : {alert_hh:02d}:{alert_mm:02d} (Minute {int(alert_triggered_at_min)})")
        print(f"✨ SUCCESS: Provided Advanced Lead Time of: {lead_time_minutes:.1f} Minutes")
    else:
        print("❌ CRITICAL FAILURE: Patient collapsed but system never alerted.")
