import numpy as np
import pandas as pd
from src.dann.inference import FeatureExtractor, CardiacArrestPredictor, EarlyWarningPredictor
from collections import deque


def simulate_ten_minute_decline(fs=25):
    """
    Generates a continuous 10-minute streaming PPG signal timeline.
    - Min 0 to 4: Healthy Baseline (75 BPM, normal respiratory modulation)
    - Min 4 to 8: Ischemic Onset (HR degrades, autonomic chaos increases)
    - Min 8 to 9: Agonal Phase (severe bradycardia, fading amplitude)
    - Min 9 to 10: Total Ventricular Arrest (Flatline)
    """
    total_duration_sec = 600
    total_samples = fs * total_duration_sec
    t = np.linspace(0, total_duration_sec, total_samples, endpoint=False)
    
    bpm_profile = np.zeros(total_samples)
    amplitude_profile = np.zeros(total_samples)
    
    for i, timestamp in enumerate(t):
        if timestamp < 240:    # Minutes 0-4 (Baseline)
            bpm_profile[i] = 75 + 3 * np.sin(2 * np.pi * 0.2 * timestamp)
            amplitude_profile[i] = 0.5
        elif timestamp < 480:  # Minutes 4-8 (The target Early Warning window)
            pct = (timestamp - 240) / 240
            bpm_profile[i] = 75 - (30 * pct) + np.random.normal(0, 4) 
            amplitude_profile[i] = 0.5 - (0.2 * pct)
        elif timestamp < 540:  # Minutes 8-9 (Agonal Phase)
            pct = (timestamp - 480) / 60
            bpm_profile[i] = 45 - (20 * pct)
            amplitude_profile[i] = 0.3 - (0.25 * pct)
        else:                  # Minutes 9-10 (Total Collapse / Flatline)
            bpm_profile[i] = 0.0
            amplitude_profile[i] = 0.001  
            
    phase = 2 * np.pi * np.cumsum(bpm_profile / 60) / fs
    raw_signal = amplitude_profile * np.sin(phase) + np.random.normal(0, 0.02, total_samples)
    
    return raw_signal, t


if __name__ == "__main__":
    print("=== Constructing 10-Minute Continuous Physiological Stream ===")
    raw_stream, time_axis = simulate_ten_minute_decline(fs=25)
    
    # Initialize the early warning system pipeline
    MODEL_PATH = "models/cardiac_arrest_v4/"
    pipeline = EarlyWarningPredictor(MODEL_PATH)
    
    window_size = 1500  # 60 seconds of history per inference chunk
    step_size = 375     # Advances forward every 15 seconds
    
    alert_triggered_at = None
    time_of_arrest = 540.0  # 9 minutes mark (540 seconds)
    
    print("\nProcessing sliding time windows...")
    print(f"{'Timestamp':<12}{'Risk Level':<15}{'Probability':<15}{'Status':<10}")
    print("-" * 55)

    alert_triggered_at = None

    for start_idx in range(0, len(raw_stream) - window_size, step_size):
        end_idx = start_idx + window_size
        current_time_stamp = time_axis[end_idx] 
        
        # 1. Slice out the 60-second raw segment (1500 samples)
        segment = raw_stream[start_idx:end_idx]
        
        # 2. Pass ONLY the segment. The wrapper handles extraction internally!
        output = pipeline.process_stream_step(segment)
        
        # Log timeline execution markers to scan progression
        mins = int(current_time_stamp // 60)
        secs = int(current_time_stamp % 60)
        time_str = f"{mins:02d}:{secs:02d}"
        
        print(f"{time_str:<12}{output['risk_level']:<15}{output['probability']:<15.4f}{'🚨 ALERT' if output['alert'] else 'OK'}")
        
        # Catch and pin the exact structural cross-over index
        if output['alert'] and alert_triggered_at is None:
            alert_triggered_at = current_time_stamp

            
    print("\n" + "="*40)
    print("=== EARLY WARNING PERFORMANCE METRICS ===")
    print("="*40)
    print(f"True Cardiac Arrest Flatline Set at : 09:00 (540.0 seconds)")
    
    if alert_triggered_at is not None:
        lead_time_seconds = time_of_arrest - alert_triggered_at
        alert_mins = int(alert_triggered_at // 60)
        alert_secs = int(alert_triggered_at % 60)
        
        print(f"First Critical Warning Flagged At   : {alert_mins:02d}:{alert_secs:02d} ({alert_triggered_at:.1f} seconds)")
        if lead_time_seconds > 0:
            print(f"✨ SUCCESS: Provided Advanced Warning Lead Time of: {lead_time_seconds / 60:.2f} Minutes")
        else:
            print(f"❌ FAILURE: Model reacted late. Alert raised {abs(lead_time_seconds) / 60:.2f} minutes AFTER flatline.")
    else:
        print("❌ CRITICAL FAILURE: Stream completed but no alert was ever raised.")
