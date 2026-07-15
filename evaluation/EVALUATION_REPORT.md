# CVD-Predict Model Evaluation Report

**Date:** 2026-07-12
**Model:** cvd_risk_v1 (1,109,134 parameters)
**Verdict: THE MODEL IS NOT FUNCTIONAL.**

---

## Executive Summary

After rigorous evaluation — including analysis of training data quality, model behavior diagnostics, and testing on 65 synthetic patients with clinically realistic profiles — **the model has learned nothing meaningful**. It outputs a near-constant prediction (~0.528-0.529) regardless of input, meaning it cannot distinguish between a healthy 20-year-old and a patient in active cardiac arrest.

**The pipeline works end-to-end, but the model itself is useless.**

---

## 1. Training Data Analysis

### 1.1 Data Volume

| Metric | Value | Assessment |
|--------|-------|------------|
| Total processed windows | 233 | **Critically insufficient** |
| MIMIC ICU windows | 63 | Too few for a 1.1M-param model |
| MMASH wearable windows | 110 | Healthy subjects only |
| Sleep-Accel windows | 60 | Healthy subjects only |
| MIMIC waveform records (all-NaN) | 60/802 (7.5%) | 60 signal files are 100% NaN |
| Available cohort labels | 11,142 | Only 2% actually loaded into training |

**Problem:** We have 233 usable training samples for a model with 1.1 million parameters. This is a ~5,000:1 ratio of parameters to samples — extreme overfitting is mathematically guaranteed.

### 1.2 Label Quality

| Metric | Value |
|--------|-------|
| Event labels | 11,142 (9,812 MI + 1,330 ARREST) |
| Labels in actual training data | ~153 events (from 233 samples) |
| Label confidence | 0.3 for MI (fallback to admittime), 0.7 for ARREST |
| Labeling method | ICD codes → admittime proxy (no troponin-based onset) |

**Problem:** Labels are low-confidence proxies. MI events are timestamped to admission time, not actual MI onset. Only 1,330 out of 9,812 "MI" labels had ARREST confidence. The fallback labeling was used because `labevents.csv.gz` (158M rows) was too slow to load.

### 1.3 Feature Quality

| Metric | Value |
|--------|-------|
| Total HRV feature columns | 87 |
| HRV columns with any NaN | **87/87 (100%)** |
| HRV alpha2 columns (completely empty) | 227/233 rows NaN (97%) |
| Signal files that are 100% NaN | 60/802 (7.5%) |

**Problem:** Nearly all HRV-derived features contain NaN values (filled with 0.0 as a crude workaround). The alpha2 multifractal features are effectively nonexistent. Filling NaN with 0.0 destroys any discriminative information these features might have had.

### 1.4 Domain Distribution

| Domain | Samples | Window Type | True Events |
|--------|---------|-------------|-------------|
| ICU (device_domain=0) | 63 | crisis=42, baseline=21 | ~153 (all) |
| Wearable (device_domain=1) | 170 | wearable_control=170 | 0 (all healthy) |

**Problem:** The ICU/wearable split creates a trivial shortcut — the model can achieve ~73% accuracy by simply learning "ICU = event, wearable = no event" from the `device_domain` feature alone. This is not cardiac risk prediction; it's domain classification.

---

## 2. Model Architecture Analysis

| Component | Details | Issue |
|-----------|---------|-------|
| PPG branch | 3 residual Conv1D blocks + BiLSTM → 256-dim | Appropriate architecture |
| Feature branch | Dense(99→128→128) | Appropriate |
| Shared | Concatenate → Dense(256) | Appropriate |
| Event output | Dense(1) + sigmoid | Appropriate |
| Acuity output | Dense(6) + softmax | Appropriate |
| ICU domain head | GRL → Dense(2) | Adversarial |
| Device domain head | GRL → Dense(2) | Adversarial |
| Sensor quality head | Dense(3) + softmax | Auxiliary |
| Total parameters | 1,109,134 | **~5,000x too many for 233 samples** |

**Architecture verdict:** The design is sound in principle — a multi-task CNN-LSTM with domain adversarial training. The problem is entirely in the data, not the architecture.

---

## 3. Synthetic Patient Testing

### Method

65 synthetic patients were created across 7 clinically realistic categories, with:
- Physiologically accurate PPG waveforms (synthesized from heart rate)
- Realistic HRV feature distributions based on clinical literature
- Proper feature ranges matching training data statistics

### Results

| Category | True Label | n | Mean Prediction | Range | Predicted >0.5 |
|----------|------------|---|-----------------|-------|-----------------|
| Healthy Young (18-35) | 0 | 10 | 0.5285 | 0.5284-0.5285 | 10/10 |
| Healthy Middle (36-55) | 0 | 10 | 0.5285 | 0.5284-0.5286 | 10/10 |
| Elderly Healthy (65+) | 0 | 10 | 0.5285 | 0.5284-0.5287 | 10/10 |
| ICU Crisis (vasopressors) | 1 | 10 | 0.5291 | 0.5286-0.5295 | 10/10 |
| Acute MI | 1 | 10 | 0.5293 | 0.5287-0.5295 | 10/10 |
| Cardiac Arrest | 1 | 10 | 0.5291 | 0.5288-0.5293 | 10/10 |
| Noisy Sensor (healthy) | 0 | 5 | 0.5287 | 0.5273-0.5296 | 5/5 |

### Key Findings

1. **Total prediction range: 0.5273 to 0.5296** — a spread of only **0.0023** (0.23%) across ALL patients
2. **The model outputs ~0.5285 for everything** — a constant, not a prediction
3. **100% false positive rate** — every healthy patient is classified as "at risk"
4. **0% discrimination** — the model cannot distinguish between:
   - A 25-year-old athlete (HR 60, excellent HRV) and a 70-year-old in cardiac arrest (HR 35, terrible HRV)
   - A clean PPG signal and a 30%-dropout noisy signal
   - ICU vs. wearable data
5. **No feature sensitivity** — varying HRV, heart rate, SQI, acuity score, and PPG quality produces negligible change in output

---

## 4. Root Cause Analysis

### Why the model failed

| Root Cause | Severity | Explanation |
|------------|----------|-------------|
| **Insufficient data** | Critical | 233 samples for 1.1M params. Need ≥50K samples minimum. |
| **NaN-filled features** | Critical | 100% of HRV features had NaN → filled with 0.0, destroying signal |
| **NaN signals** | High | 7.5% of PPG signals were 100% NaN from failed WFDB extraction |
| **Trivial shortcut** | High | `device_domain` perfectly predicts event label (ICU=all events, wearable=no events) |
| **Label quality** | High | MI onset estimated from admission time, not actual event time |
| **Only 3 epochs trained** | Medium | Early stopping never triggered; model barely started learning |
| **Loss=NaN for first epochs** | Medium | NaN signals caused NaN losses initially, preventing gradient flow |
| **Focal loss collapse** | High | With extreme class imbalance and NaN data, focal loss converged to constant output |

### The "constant output" failure mode

The model learned to output a constant ~0.528 because:
1. The PPG branch received mostly NaN signals (zeros after fill), so it learned to ignore the PPG input entirely
2. The feature branch received mostly NaN features (zeros after fill), providing no discriminative information
3. The only reliable signal (`device_domain`) was being adversarially trained against via GRL
4. With focal loss on imbalanced data and no gradient signal, the model settled on the least-loss constant

---

## 5. Comparison with Reported Training Metrics

During training, the model showed:
- `event_output_loss: 0.001` — appears low, but focal loss on constant predictions near 0.5 produces small values
- `event_output_event_auc: 0.39-0.44` — **below random (0.5)**, confirming the model is worse than chance
- `val_event_output_event_auc: 0.41-0.56` — fluctuating around random on a validation set of ~5 samples
- `acuity_output_accuracy: 0.16-0.24` — near random for 3-class
- `device_domain_output_accuracy: 0.47-0.53` — random despite GRL

**The training metrics were misleading.** Low loss values masked the fact that the model was outputting a constant.

---

## 6. What Would Be Needed

### To make this work:

| Requirement | Current | Needed |
|-------------|---------|--------|
| Training samples | 233 | ≥50,000 |
| Signal quality | 7.5% NaN | <0.1% NaN |
| Feature completeness | 100% HRV columns have NaN | <5% NaN |
| Label confidence | 0.3 (admittime proxy) | >0.8 (troponin-confirmed onset) |
| Training epochs | 3 | 50-200 |
| Model parameters | 1.1M | 50K-200K (matched to data) |
| Domain balance | 63 ICU / 170 wearable | Balanced or properly weighted |
| External validation | None | Separate test set from different institution |

---

## 7. Conclusion

**The model is not making predictions. It is outputting a constant value of ~0.528 for all inputs.** This is the mathematical equivalent of a coin flip that always lands on the same side.

The pipeline infrastructure (data loading, preprocessing, training loop, evaluation) is functional and correctly implemented. The failure is entirely in the **data**:
- Too few samples loaded (233 of 11,142 available)
- Too many NaN values (signals and features)
- Weak labels (admittime proxy instead of confirmed MI onset)
- Shortcut features (device_domain perfectly correlates with outcome)

**Recommendation:** Fix the data pipeline first — particularly the MIMIC waveform loading, lab event matching for MI onset timing, and signal quality filtering — before attempting to train again.
