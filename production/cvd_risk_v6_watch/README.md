# CVD Risk Model v6-watch

## Overview

v6-watch is a **natively trained** CVD risk prediction model for Apple Watch wrist PPG. Unlike v5-watch (which wraps the v4 ICU model with inversion), v6-watch was trained from scratch on synthetic wrist PPG signals using a lightweight dual-branch architecture.

## Performance (Synthetic Test Set)

| Metric | Value |
|--------|-------|
| AUROC | 0.998 |
| Accuracy | 96.1% |
| Precision | 100% |
| Recall | 93.3% |
| F1 | 0.97 |
| False Positives | 0/76 healthy signals |
| False Negatives | 7/104 at-risk signals |
| Brier Score | 0.047 |

**Optimal threshold:** 0.11 (tuned on validation set)

## Architecture

Dual-branch neural network (~64K params):

- **PPG branch:** 3-block 1D ResNet-CNN (16→32→64 filters) → BiLSTM (32 units)
- **Feature branch:** MLP (32→32) operating on 34 HRV + signal features
- **Shared:** Concatenated → Dense(32) → event head with sigmoid output

Input: Raw PPG (7500 samples @ 25 Hz) + 34 hand-crafted features.

## Usage

```python
import tensorflow as tf
import numpy as np
import json

# Load model
model = tf.keras.models.load_model("production/cvd_risk_v6_watch/best_model.keras")

# Load metadata
with open("production/cvd_risk_v6_watch/feature_columns.json") as f:
    feature_cols = json.load(f)
with open("production/cvd_risk_v6_watch/optimal_threshold.json") as f:
    threshold = json.load(f)["optimal_threshold"]

# Prepare inputs
ppg = ...  # (7500,) array at 25 Hz
features = ...  # dict with 34 features matching feature_columns.json

feature_array = np.array([[features[col] for col in feature_cols]])
ppg_input = ppg.reshape(1, -1, 1).astype(np.float32)

# Predict
risk_score = model.predict({"ppg_input": ppg_input, "feature_input": feature_array})[0][0]
flagged = risk_score >= threshold
```

## Training Details

- **Dataset:** 1200 synthetic wrist PPG signals (500 healthy, 700 at-risk, 200 borderline)
- **Split:** 70% train / 15% validation / 15% test
- **Optimizer:** AdamW (lr=3e-4, weight decay=1e-4)
- **Loss:** Binary cross-entropy with class weights
- **Callbacks:** Early stopping (patience 15), ReduceLROnPlateau, ModelCheckpoint

## Files

| File | Description |
|------|-------------|
| `best_model.keras` | Best model checkpoint (by validation AUROC) |
| `final_model.keras` | Model after full training |
| `feature_columns.json` | 34 expected feature column names |
| `optimal_threshold.json` | Threshold config (0.11) |
| `config.yaml` | Full model configuration and metadata |

## Limitations

- **Trained on synthetic data only** — not validated on real Apple Watch PPG
- Signal morphology and noise characteristics may differ from real recordings
- Clinical deployment requires prospective validation against ground truth diagnoses
- For production use, validate on actual Apple Watch data before clinical deployment

## Comparison with Other Models

| Model | AUROC | Precision | Recall | Notes |
|-------|-------|-----------|--------|-------|
| v4 (ICU) | 0.12 | 38.5% | 100% | Inverted on wrist PPG |
| v5-watch (inverted v4) | 0.88 | 98.3% | 72.5% | Wrapper around v4 |
| **v6-watch (native)** | **0.998** | **100%** | **93.3%** | Trained for wrist PPG |
