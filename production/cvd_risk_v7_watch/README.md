# CVD Risk Model v7-watch

## Overview

v7-watch is a **hybrid-trained** CVD risk prediction model for Apple Watch wrist PPG. It combines **real clinical data** (MIMIC-IV ICU + MMASH wearable) with synthetic Apple Watch PPG augmentation, achieving **100% accuracy, precision, and recall** on the held-out test set.

## Performance (Test Set — 170 signals)

| Metric | Value |
|--------|-------|
| AUROC | 1.000 |
| Accuracy | 100.0% |
| Precision | 100.0% |
| Recall | 100.0% |
| F1 Score | 1.000 |
| Brier Score | 0.004 |
| False Positives | 0/78 healthy |
| False Negatives | 0/92 at-risk |

**Optimal threshold:** 0.35

## Training Data

| Source | Type | Count | Label |
|--------|------|-------|-------|
| MIMIC-IV ICU | MI crisis/baseline | 290 | At-risk |
| MIMIC-IV ICU | ARREST crisis/baseline | 40 | At-risk |
| MMASH wearable | CONTROL (wrist PPG) | 320 | Healthy |
| Synthetic | Apple Watch simulator | 480 | Mixed |
| **Total** | | **1130** | |

- Real signals: 650 (MIMIC + MMASH)
- Synthetic augmentation: 480
- Split: 70% train / 15% validation / 15% test (stratified)

## Architecture

Dual-branch neural network (~66K params):

- **PPG Branch:** 3-block 1D ResNet-CNN (16->32->64 filters) + BiLSTM (32 units)
- **Feature Branch:** MLP (32->32) on 93 HRV + signal features
- **Shared:** Concatenated -> Dense(32) -> Event head (sigmoid)

Input: Raw PPG (7500 samples @ 25 Hz) + 93 hand-crafted features.

## Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=3e-4, weight_decay=1e-4) |
| Loss | Binary cross-entropy with class weights |
| Batch size | 32 |
| Epochs | 31 (early stopping, patience=20) |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=7) |
| Callbacks | TensorBoard, CSVLogger, ModelCheckpoint |

## Usage

```python
import tensorflow as tf
import numpy as np
import json

# Load model
model = tf.keras.models.load_model("production/cvd_risk_v7_watch/best_model.keras")

# Load metadata
with open("production/cvd_risk_v7_watch/feature_columns.json") as f:
    feature_cols = json.load(f)
with open("production/cvd_risk_v7_watch/optimal_threshold.json") as f:
    threshold = json.load(f)["threshold"]

# Prepare inputs
ppg = ...  # (7500,) array at 25 Hz
features = ...  # dict with 93 features matching feature_columns.json

feature_array = np.array([[features.get(col, 0) for col in feature_cols]])
ppg_input = ppg.reshape(1, -1, 1).astype(np.float32)

# Predict
risk_score = model.predict({"ppg_input": ppg_input, "feature_input": feature_array})[0][0]
flagged = risk_score >= threshold
```

## TensorBoard

```bash
tensorboard --logdir production/cvd_risk_v7_watch/logs
```

Logs include:
- Training/validation loss, AUROC, precision, recall, accuracy per epoch
- Weight histograms
- Computation graph
- Training profiling

## Files

| File | Description |
|------|-------------|
| `best_model.keras` | Best model checkpoint (by validation AUROC) |
| `final_model.keras` | Model after full training |
| `feature_columns.json` | 93 expected feature column names |
| `optimal_threshold.json` | Threshold config (0.35) |
| `config.yaml` | Full model configuration and metadata |
| `training_history.json` | Epoch-level training metrics |
| `training_log.csv` | CSV training log |
| `logs/` | TensorBoard event files |

## Comparison with Other Models

| Model | Data | AUROC | Accuracy | Precision | Recall | F1 |
|-------|------|-------|----------|-----------|--------|----|
| v4 (ICU) | MIMIC only | 0.119 | 30.0% | 43.8% | 48.8% | 0.462 |
| v5-watch (inverted) | v4 wrapper | 0.882 | 82.3% | 100.0% | 71.2% | 0.832 |
| v6-watch (synthetic) | Synthetic only | 0.977 | 96.2% | 98.7% | 95.0% | 0.968 |
| **v7-watch (hybrid)** | **Real + Synthetic** | **1.000** | **100.0%** | **100.0%** | **100.0%** | **1.000** |

## Limitations

- **Test set is small** (170 signals) — perfect scores may not generalize
- Real data includes MIMIC ICU signals (not wrist PPG) — domain gap remains
- MMASH wearable signals are from research devices, not Apple Watch
- Synthetic augmentation may not capture all real-world variability
- Clinical deployment requires prospective validation on actual Apple Watch recordings
- Model was trained on a fixed dataset — no online learning or drift detection

## Data Sources

- **MIMIC-IV:** PhysioNet MIMIC-IV v2.2 emergency department data
- **MMASH:** Multisensor Dataset for Motion, Activity, Sleep and Health
- **Synthetic:** Apple Watch PPG simulator (25 Hz, realistic morphology + noise)
