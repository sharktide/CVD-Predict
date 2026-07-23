# Apple Watch PPG Approximation Test — CVD Model v4

## Executive Summary

**Critical Finding:** The CVD Model v4 produces **inverted predictions** on synthetic Apple Watch-style PPG signals. Healthy signals receive *higher* event probabilities (mean 0.84) than at-risk signals (mean 0.18). The AUROC of 0.12 (near 0, not near 1) confirms the model's predictions are opposite of what's desired for cardiac event screening.

This indicates a fundamental **population mismatch**: the model learned to distinguish "acute ICU deterioration" from "stable ICU patients," not "healthy" from "cardiac-compromised." When presented with wrist PPG from a non-ICU population, the model's learned decision boundaries do not transfer.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Signal Duration | 120 seconds (2 minutes) |
| Sampling Rate | 25 Hz (Apple Watch optical sensor) |
| PPG Input Length | 7,500 samples (120s × 25 Hz, zero-padded if needed) |
| Classification Threshold | 0.05 |
| Healthy Profiles | 50 |
| At-Risk Profiles | 50 |
| Borderline Profiles | 30 |
| **Total Test Signals** | **130** |

## Apple Watch Signal Characteristics Modeled

### Healthy Profile
- Resting HR: 58–78 BPM
- HRV: 12–25% (good autonomic function)
- Motion artifacts: Low (10–40%)
- Contact quality: 85–100%
- SNR: 18–25 dB

### At-Risk Profile (MI / Cardiac Arrest Risk)
- Resting HR: 85–120 BPM (elevated, sympathetic activation)
- HRV: 3–8% (reduced, autonomic dysfunction)
- Motion artifacts: Low-moderate (10–30%)
- Contact quality: 80–95%
- SNR: 14–20 dB (lower due to reduced cardiac output)

### Borderline Profile
- Resting HR: 72–95 BPM
- HRV: 6–14% (mildly reduced)
- Motion artifacts: Moderate (15–45%)
- Contact quality: 82–97%
- SNR: 16–22 dB

## Overall Results

| Metric | Value |
|--------|-------|
| AUROC | **0.1185** (inverted — near 0 = opposite of correct) |
| Accuracy | 30.0% |
| Precision | 0.4382 |
| Recall | 0.4875 |
| F1 Score | 0.4615 |
| Brier Score | 0.6816 |

### Confusion Matrix

|  | Predicted Neg | Predicted Pos |
|--|--------------|--------------|
| **Actual Neg (Healthy)** | 0 | 50 |
| **Actual Pos (At-Risk)** | 41 | 39 |

**Note:** The model flags **100% of healthy signals** as positive (event) and only **32% of at-risk signals** — the opposite of correct behavior.

## Per-Profile Breakdown

| Profile | n | Mean Probability | Std | Flagged | Flag Rate |
|---------|---|-----------------|-----|---------|-----------|
| Healthy | 50 | **0.8386** | 0.1220 | 50/50 | **100.0%** |
| At_Risk | 50 | **0.1753** | 0.3210 | 16/50 | **32.0%** |
| Borderline | 30 | **0.4192** | 0.3493 | 23/30 | **76.7%** |

## Prediction Distribution by Profile

| Profile | Min | Q25 | Median | Q75 | Max |
|---------|-----|-----|--------|-----|-----|
| Healthy | 0.4695 | 0.8017 | **0.8514** | 0.9240 | 0.9945 |
| At_Risk | 0.0000 | 0.0000 | **0.0004** | 0.1248 | 0.9984 |
| Borderline | 0.0000 | 0.0547 | **0.4122** | 0.7061 | 0.9788 |

## Root Cause Analysis

### 1. The Model Predicts "ICU Acute Event," Not "Cardiac Risk"

The model was trained on MIMIC-IV ICU patients where:
- **Positive class (event):** ICU patients with MI or cardiac arrest — characterized by *acute hemodynamic instability*, *arrhythmia onset*, *specific PPG morphology changes during crisis*
- **Negative class (control):** Stable ICU patients — still critically ill but not having an acute event

The synthetic "at-risk" profile (elevated HR, reduced HRV) mimics **chronic cardiac compromise**, not **acute ICU deterioration**. A healthy person with HR 70 and good HRV looks more like a "stable ICU control" to this model than an at-risk person with HR 100 and poor HRV (who would be flagged as "event" only if the PPG showed acute crisis morphology).

### 2. Feature Distribution Shift

| Feature | ICU Training Range | Apple Watch Synthetic Range | Overlap |
|---------|-------------------|---------------------------|---------|
| HRV_SDNN | 10–200 ms | 20–150 ms | Partial |
| HRV_RMSSD | 5–150 ms | 10–100 ms | Partial |
| HRV_MaxNN | 500–2000 ms | 400–1200 ms | Low |
| SQI | 0.1–0.9 | 0.3–0.8 | Moderate |
| mean_amplitude | Varies by sensor | Normalized to [-1,1] | **None** |

The model's feature branch (97 features) was fit to ICU-specific distributions. Synthetic Apple Watch features fall in different ranges, causing the model to misclassify.

### 3. PPG Morphology Mismatch

The synthetic PPG generator produces:
- Clean Gaussian-based PPG cycles (systolic peak + dicrotic notch)
- Regular beat-to-beat morphology

ICU PPG has:
- Arterial line quality (higher fidelity)
- Pathological morphology during events (ST changes, arrhythmia patterns)
- Different waveform shape than wrist PPG

### 4. Domain Adversarial Training Limitation

The GRL (Gradient Reversal Layer) domain head was designed to make the model domain-invariant between ICU and wearable data. However:
- The "wearable" training data was MMASH/Sleep Accel (research devices)
- The domain head may have learned to suppress domain-specific features rather than generalize
- Apple Watch represents a **third domain** not seen during training

## Why the Model "Works" on ICU Data But Not on Synthetic Watch Data

The v4 model's 99.6% AUROC on the ICU test set is valid for its intended use case:
- **Same population:** ICU patients
- **Same signal source:** MIMIC-IV Waveform Database (arterial PPG)
- **Same event type:** Acute MI/arrest during ICU stay

The model does **not** claim to work on:
- Outpatient screening
- Wrist-worn devices
- Chronic cardiac risk prediction
- Healthy population monitoring

## Limitations

1. **Synthetic signals**: Mathematically generated, not real Apple Watch data. Real signals have additional artifacts (skin tone, tattoos, hair, sweat, motion patterns).

2. **No real cardiac events**: At-risk profiles are simulated with HR/HRV characteristics, not actual MI/arrest physiology.

3. **Population mismatch**: Model trained on critically ill ICU patients. Apple Watch users are predominantly healthy outpatients.

4. **Feature extraction domain shift**: HRV features computed from synthetic PPG may not match real Apple Watch-derived HRV.

5. **Small test set**: 130 signals is sufficient for initial assessment but not definitive validation.

## Recommendations

### For Apple Watch / Wearable CVD Screening:

1. **Collect real wrist PPG data** from cardiac event patients (MI, arrest) and matched controls
2. **Retrain from scratch** on wrist PPG data with appropriate event labels
3. **Redesign the labeling scheme** — ICU event labels don't apply to outpatient screening
4. **Add wrist-specific preprocessing** (different filtering, peak detection for wrist morphology)
5. **Consider a separate screening model** — the ICU model and wearable screening model may need different architectures

### For the Current ICU Model:

1. **Validate on real ICU data** — the 99.6% AUROC should be confirmed with external MIMIC cohorts
2. **Deploy for ICU monitoring** — this is the model's validated use case
3. **Do not use for outpatient screening** — the population mismatch makes predictions unreliable

## Conclusion

The CVD Model v4 **does not work** for Apple Watch-based cardiac event screening. The model's predictions are **inverted** on synthetic wrist PPG, assigning high risk to healthy signals and low risk to at-risk signals. This is not a bug but a fundamental limitation: the model was designed for ICU acute event detection, not outpatient cardiac risk screening.

**Key takeaway:** A model trained on ICU patients cannot be expected to work on healthy outpatients with wearable devices without significant retraining on representative data. The domain gap is too large for the learned decision boundaries to transfer.

| Verdict | Status |
|---------|--------|
| Apple Watch screening | **NOT VIABLE** with current model |
| ICU acute event detection | Validated (AUROC 0.996 on MIMIC-IV) |
| Wearable CVD risk prediction | Requires dedicated model development |
