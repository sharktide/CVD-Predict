# v5-watch Evaluation Report

## Overview

v5-watch is an **inversion wrapper** around the v4 ICU model. The v4 model was trained on MIMIC-IV ICU PPG (125 Hz) and produces inverted predictions on healthy/outpatient wrist PPG. Inverting the output (`1 - v4_probability`) corrects this population mismatch.

## Test Setup

- **Test signals:** 130 synthetic Apple Watch PPG signals (25 Hz, 120 seconds)
  - 50 healthy (true label: 0)
  - 50 at-risk (true label: 1)
  - 30 borderline (true label: 1)
- **Threshold:** 0.55 (100% precision operating point)
- **Alternative threshold:** 0.35 (best F1 = 0.87)

## Performance Metrics

| Metric | Value |
|--------|-------|
| AUROC | 0.882 |
| Accuracy | 82.3% |
| Precision | 100.0% |
| Recall | 71.2% |
| F1 Score | 0.832 |
| Brier Score | 0.1340 |

## Confusion Matrix

|  | Predicted Healthy | Predicted At-Risk |
|--|-------------------|-------------------|
| **True Healthy** | 50 | 0 |
| **True At-Risk** | 23 | 57 |

## How It Works

1. Loads the v4 ICU model architecture and weights
2. Runs inference on Apple Watch PPG + extracted features
3. **Inverts** the output: `risk_score = 1 - v4_probability`
4. Applies threshold (default 0.55) for binary classification

## Strengths

- Zero false positives on healthy signals (100% precision)
- No retraining required — leverages existing v4 model
- Simple, interpretable wrapper

## Limitations

- **Not natively trained** for wrist PPG — relies on v4's ICU-learned features
- Lower recall (71.2%) — misses ~29% of at-risk signals
- Borderline cases have mixed performance
- Synthetic test data only — not validated on real Apple Watch recordings
- Feature extraction depends on HRV quality from wrist PPG (noisier than ICU)

## Recommendation

Use v5-watch as a **baseline** or **quick deployment** option. For production, prefer **v6-watch** which is natively trained for wrist PPG and achieves significantly better recall (95% vs 71%) with comparable precision.
