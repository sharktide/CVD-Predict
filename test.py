import numpy as np
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoModel
from ohca_predictor.utils.io_utils import WindowSample
# 1. Download and map the custom model wrapper from the cloud Hub repository
model = AutoModel.from_pretrained("sharktide/ohca-predictor-v1", trust_remote_code=True)

patient_record = WindowSample(
    ecg=np.random.randn(24050).astype(np.float32), 
    accelerometer=np.zeros((9620, 3), dtype=np.float32),
    gyroscope=np.zeros((9620, 3), dtype=np.float32),
    ppg=np.random.randn(9250).astype(np.float32),
    respiration=np.zeros(4625, dtype=np.float32),
    spo2=np.zeros(1850, dtype=np.float32),
    temperature=np.zeros(185, dtype=np.float32),
    demographics=np.zeros(24, dtype=np.float32),
    medications=np.zeros(14, dtype=np.float32),
    comorbidities=np.zeros(14, dtype=np.float32),
    lab_values=np.zeros(8, dtype=np.float32),
    heart_rate=72.0, rhythm=0, activity_state=0,
    spo2_mean=97.5, sbp_mean=120.0, dbp_mean=80.0,
    patient_id="live-monitor-case-001", signal_quality={"ecg": 1.0},
    window_start_hours=0.0, window_duration_hours=0.05138,
    ohca_label=0.0, event_indicator=0.0, time_to_event=0.0
)

# 3. Dispatches forward pass execution natively
prediction = model(patient_record)

print(f"Prediction Success!")
print(f"Calculated Patient OHCA Risk: {prediction['ohca_risk'].numpy().item():.4f}")
