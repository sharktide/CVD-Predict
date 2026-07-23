import os
import urllib.request
import numpy as np
import pandas as pd
from scipy.signal import decimate
from src.dann.inference import EarlyWarningPredictor

# Target URL for raw ICU Patient 01 data from PhysioNet
DATA_URL = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv/bidmc_01_Signals.csv"
LOCAL_FILE = "bidmc_01_Signals.csv"

def download_clinical_sample():
    """Downloads a raw patient PPG waveform recording directly from the PhysioNet servers."""
    if not os.path.exists(LOCAL_FILE):
        print(f"📥 Downloading genuine ICU patient recording from PhysioNet...")
        urllib.request.urlretrieve(DATA_URL, LOCAL_FILE)
        print("✅ Download Complete.")
    else:
        print("💾 Using locally cached patient recording.")

def load_and_preprocess_human_wave(target_fs=25, window_sec=60):
    """Parses the patient CSV data cleanly by filtering headers and downsampling from 125Hz to 25Hz."""
    # Read the data file directly, ignoring lines starting with '#'
    df = pd.read_csv(LOCAL_FILE, comment='#')
    
    # Clean whitespace strings from the header definitions
    df.columns = df.columns.str.strip()
    
    # Assert column tracking is intact
    if 'PLETH' not in df.columns:
        raise ValueError(f"Could not locate 'PLETH' column. Available streams are: {list(df.columns)}")
        
    # Extract the raw clinical photoplethysmogram wave stream
    raw_125hz_signal = df['PLETH'].values
    print(f"📊 Extracted Waveform: {len(raw_125hz_signal)} samples @ 125Hz from real human tissue.")
    
    # Decimate from 125Hz to 25Hz using a downsampling factor of 5 (125 / 5 = 25)
    downsampled_25hz = decimate(raw_125hz_signal, q=5, zero_phase=True)
    
    # Slice out exactly 60 seconds of data (1,500 samples @ 25Hz)
    expected_samples = target_fs * window_sec
    if len(downsampled_25hz) >= expected_samples:
        processed_segment = downsampled_25hz[:expected_samples]
    else:
        raise ValueError(f"Signal too short. Expected {expected_samples}, got {len(downsampled_25hz)}")
        
    # Standardize scale dimensions using zero-mean unit-variance normalization
    processed_segment = (processed_segment - np.mean(processed_segment)) / (np.std(processed_segment) + 1e-8)
    
    return processed_segment

if __name__ == "__main__":
    print("=== Clinical Real-World Human Validation Loop ===")
    
    download_clinical_sample()
    try:
        real_human_segment = load_and_preprocess_human_wave()
        print(f"🚀 Normalized Array Ready for Inference: Shape = {real_human_segment.shape} @ 25Hz")
        
        # Initialize your finalized hours-ahead pipeline wrapper
        MODEL_PATH = "models/cardiac_arrest_v4/"
        pipeline = EarlyWarningPredictor(MODEL_PATH)
        
        # Execute the prediction step
        print("\nFeeding real human physiology into the inference loop...")
        result = pipeline.process_stream_step(real_human_segment)
        
        print("\n" + "="*45)
        print("=== MODEL OUTPUT ON GENUINE HUMAN TISSUE ===")
        print("="*45)
        print(f"Calculated Risk Level : {result.get('risk_level')}")
        print(f"Model Probability     : {result.get('probability'):.4f}")
        print(f"Alert Status Triggered: {result.get('alert')}")
        if "early_warning_trigger" in result:
            print(f"Trigger Source        : {result.get('early_warning_trigger')}")
        print(f"Latent Vector Snippet : {result.get('latent_embedding')[:4]}...")
        print("="*45)
        
    except Exception as e:
        print(f"❌ Verification failed due to processing error: {str(e)}")
