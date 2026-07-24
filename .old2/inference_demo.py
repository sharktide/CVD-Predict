#!/usr/bin/env python3
"""
OHCA Prediction Inference Demo
===============================
Deployable inference pipeline. Takes raw sensor data, outputs risk assessment.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

MODEL_PATH = Path(__file__).parent / 'models_v3/best_v3.keras'
ECG_FS = 130
ACC_FS = 25


class OHCAPredictor:
    """Production-grade OHCA prediction wrapper."""

    def __init__(self, model_path=MODEL_PATH):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.threshold = 0.40  # Lower threshold for higher sensitivity

    def predict(self, ecg_window, acc_window, age, sex, ischemia_history):
        """
        Run inference on a single 10-second window.

        Args:
            ecg_window: np.array shape (1300,) or (1300,1) — 10s @ 130Hz
            acc_window: np.array shape (250,3) — 10s @ 25Hz [x,y,z] in g
            age: float — patient age in years
            sex: int — 0=male, 1=female
            ischemia_history: int — 0=no, 1=yes

        Returns:
            dict with keys: probability, risk_level, recommendation
        """
        ecg = np.asarray(ecg_window, dtype=np.float32).reshape(1, -1, 1)
        acc = np.asarray(acc_window, dtype=np.float32).reshape(1, -1, 3)
        norm_age = np.clip((age - 18) / (95 - 18), 0, 1).astype(np.float32)
        bio = np.array([[norm_age, float(sex), float(ischemia_history)]], dtype=np.float32)

        prob = float(self.model.predict([ecg, acc, bio], verbose=0).flatten()[0])

        if prob >= 0.7:
            risk = "HIGH"
            rec = "Immediate clinical review. Consider emergency intervention."
        elif prob >= self.threshold:
            risk = "MODERATE"
            rec = "Schedule clinical review within 1 hour. Monitor closely."
        else:
            risk = "LOW"
            rec = "Continue routine monitoring."

        return {
            'probability': prob,
            'risk_level': risk,
            'recommendation': rec,
            'threshold_used': self.threshold,
        }


def demo_random_samples():
    """Run inference on random test samples to demonstrate end-to-end pipeline."""
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║           OHCA INFERENCE DEMO — END-TO-END PIPELINE               ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    predictor = OHCAPredictor()

    data_dir = Path(__file__).parent / 'data'
    X_ecg = np.load(data_dir / 'X_ecg_test.npy')
    X_acc = np.load(data_dir / 'X_acc_test.npy')
    X_bio = np.load(data_dir / 'X_bio_test.npy')
    y = np.load(data_dir / 'y_test.npy')

    np.random.seed(123)
    indices = np.random.choice(len(y), 10, replace=False)

    print(f"\n  Running inference on 10 random test samples...\n")
    print(f"  {'#':>3s}  {'True':>10s}  {'Prob':>7s}  {'Risk':>9s}  {'Prediction':>10s}  {'Correct':>8s}")
    print(f"  {'-'*65}")

    for rank, idx in enumerate(indices):
        true_label = "Pre-Arrest" if y[idx] == 1 else "Stable"

        age_years = X_bio[idx, 0] * 77 + 18
        sex = int(X_bio[idx, 1])
        ischemia = int(X_bio[idx, 2])

        result = predictor.predict(
            ecg_window=X_ecg[idx].flatten(),
            acc_window=X_acc[idx],
            age=age_years,
            sex=sex,
            ischemia_history=ischemia
        )

        pred_label = "Pre-Arrest" if result['probability'] >= 0.40 else "Stable"
        correct = "YES" if (result['probability'] >= 0.40) == (y[idx] == 1) else "NO"

        print(f"  {rank+1:3d}  {true_label:>10s}  {result['probability']:7.4f}  {result['risk_level']:>9s}  {pred_label:>10s}  {correct:>8s}")

    print(f"\n  Inference complete. Model loaded from: {MODEL_PATH}")
    print(f"  Threshold: {predictor.threshold}")
    print(f"  Input shapes: ECG(1300,1) ACC(250,3) Bio(3,)")


if __name__ == '__main__':
    demo_random_samples()
