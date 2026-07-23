# v6-watch Evaluation Report

## Overview

v6-watch is a **natively trained** CVD risk prediction model for Apple Watch wrist PPG. Unlike v5-watch (which wraps the v4 ICU model), v6-watch was trained from scratch on synthetic wrist PPG signals using a lightweight dual-branch architecture.

## Architecture

- **PPG Branch:** 3-block 1D ResNet-CNN (16→32→64 filters) + BiLSTM (32 units)
- **Feature Branch:** MLP (32→32) on 34 HRV + signal features
- **Shared:** Concatenated → Dense(32) → Event head (sigmoid)
- **Total parameters:** ~64K (lightweight, on-device friendly)

## Test Setup

- **Training data:** 1200 synthetic Apple Watch PPG signals (500 healthy, 700 at-risk)
- **Test signals:** 130 synthetic Apple Watch PPG signals (25 Hz, 120 seconds)
  - 50 healthy (true label: 0)
  - 50 at-risk (true label: 1)
  - 30 borderline (true label: 1)
- **Threshold:** 0.11 (optimal from validation sweep)

## Performance Metrics

| Metric | Value |
|--------|-------|
| AUROC | 0.977 |
| Accuracy | 96.2% |
| Precision | 98.7% |
| Recall | 95.0% |
| F1 Score | 0.968 |
| Brier Score | 0.0553 |

## Confusion Matrix

|  | Predicted Healthy | Predicted At-Risk |
|--|-------------------|-------------------|
| **True Healthy** | 49 | 1 |
| **True At-Risk** | 4 | 76 |

## Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | Synthetic Apple Watch PPG |
| Samples | 1200 (500 healthy, 700 at-risk) |
| Split | 70% train / 15% val / 15% test |
| Optimizer | AdamW (lr=3e-4, weight_decay=1e-4) |
| Batch size | 32 |
| Early stopping | patience=15 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |

## Test Set Performance (180 held-out signals)

| Metric | Value |
|--------|-------|
| AUROC | 0.998 |
| Accuracy | 96.1% |
| Precision | 100% |
| Recall | 93.3% |
| F1 | 0.965 |
| True Negatives | 76 |
| False Positives | 0 |
| False Negatives | 7 |
| True Positives | 97 |

## Strengths

- **Highest AUROC** (0.977) among all models on same 130 test signals
- **Near-perfect precision** (98.7%) — almost zero false alarms
- **High recall** (95.0%) — catches 95% of at-risk signals
- Lightweight architecture (~64K params) suitable for on-device inference
- Native wrist PPG training — no inversion hack needed

## Limitations

- **Trained on synthetic data only** — not validated on real Apple Watch PPG
- Signal morphology and noise may differ from real recordings
- Threshold (0.11) is very low — may need recalibration on real data
- Clinical deployment requires prospective validation against ground truth

## Comparison with Other Models

| Model | AUROC | Accuracy | Precision | Recall | F1 |
|-------|-------|----------|-----------|--------|----|
| v4 (raw) | 0.119 | 30.0% | 43.8% | 48.8% | 0.462 |
| v5-watch (inverted) | 0.882 | 82.3% | 100.0% | 71.2% | 0.832 |
| **v6-watch (native)** | **0.977** | **96.2%** | **98.7%** | **95.0%** | **0.968** |

## Recommendation

v6-watch is the **best model** for Apple Watch cardiac event screening. However, all results are on synthetic data. Before clinical deployment:

1. Validate on real Apple Watch PPG recordings with ground truth diagnoses
2. Recalibrate threshold on real data distribution
3. Test across diverse populations (age, skin tone, fitness level)
4. Evaluate edge cases (motion artifacts, poor contact, arrhythmias)
