# CVD Risk Model v5-watch

## Overview

v5-watch is a **deployment wrapper** around the v4 ICU model with **prediction inversion** for Apple Watch PPG screening. 

The v4 model was trained on MIMIC-IV ICU patients and produces inverted predictions on healthy/outpatient wrist PPG. Inverting the output corrects this population mismatch.

## Performance (Synthetic Apple Watch Test)

| Metric | Value |
|--------|-------|
| AUROC | 0.88 |
| Accuracy | 82.3% |
| Precision | 98.3% |
| Recall | 72.5% |
| F1 | 0.83 |
| False Positives | 1/50 healthy signals |

**At threshold=0.55:** 100% precision (zero false alarms), 71% recall  
**At threshold=0.35 (best F1):** 96% precision, 80% recall, F1=0.87

## Usage

```python
from production.cvd_risk_v5_watch.model import CVDWatchPredictor

predictor = CVDWatchPredictor()

# PPG: 25 Hz, ~120 seconds
# Features: dict with HRV features matching feature_columns.json
result = predictor.predict(ppg_signal, features)

print(result["event_probability"])  # 0-1, higher = more risk
print(result["flagged"])           # True if above threshold
print(result["confidence"])        # "high", "medium", "low"
```

## How It Works

1. Loads the v4 ICU model architecture and weights
2. Runs inference on Apple Watch PPG + features
3. **Inverts** the output probability: `risk_score = 1 - v4_probability`
4. Applies threshold (default 0.55) for binary classification

## Files

| File | Description |
|------|-------------|
| `model.py` | CVDWatchPredictor class with inversion wrapper |
| `best_model.keras` | v4 model weights (copied from cvd_risk_v4) |
| `feature_columns.json` | 97 expected feature columns |
| `optimal_threshold.json` | Threshold config (0.55) |
| `config.yaml` | Model configuration |

## Limitations

- Synthetic test signals only — not validated on real Apple Watch data
- Borderline cases remain unreliable (57% correct)
- Relies on v4's learned features which may not transfer to all wrist PPG
- For production, prefer v6-watch (trained natively on wrist PPG)
