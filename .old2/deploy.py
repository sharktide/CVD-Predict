#!/usr/bin/env python3
"""Production deployment script for OHCA prediction."""
import numpy as np
import tensorflow as tf

# Optimal threshold from sweep (0.45 balances FPR and sensitivity)
DEFAULT_THRESHOLD = 0.45

class OHCAPredictor:
    def __init__(self, model_path='models_v3/best_v3.keras', threshold=DEFAULT_THRESHOLD):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.threshold = threshold
    
    def predict(self, ecg, acc, bio, threshold=None):
        """Predict OHCA risk.
        
        Args:
            ecg: (1300,) raw ECG signal at 130Hz
            acc: (250, 3) accelerometer data at 25Hz  
            bio: (6,) patient biodata [age, sex, ischemia, bp_sys, bp_dia, spo2]
            threshold: Override decision threshold
        
        Returns:
            dict with risk_level, probability, and recommendation
        """
        thr = threshold or self.threshold
        
        # Preprocess
        ecg = self._preprocess_ecg(ecg)
        acc = self._preprocess_acc(acc)
        bio = bio.reshape(1, -1).astype(np.float32)
        
        # Predict
        prob = float(self.model.predict(
            {'ecg_input': ecg, 'acc_input': acc, 'bio_input': bio},
            verbose=0
        )[0][0])
        
        decision = prob >= thr
        
        return {
            'probability': prob,
            'decision': bool(decision),
            'threshold': thr,
            'risk_level': 'CRITICAL' if prob > 0.7 else 'HIGH' if prob > 0.5 else 'MODERATE' if prob > 0.3 else 'LOW',
            'recommendation': 'Immediate intervention' if decision else 'Continue monitoring'
        }
    
    def _preprocess_ecg(self, ecg):
        ecg = np.asarray(ecg, dtype=np.float32).flatten()
        ecg = ecg[:1300]
        if len(ecg) < 1300:
            ecg = np.pad(ecg, (0, 1300 - len(ecg)))
        # Normalize
        mean = np.mean(ecg)
        std = np.std(ecg) + 1e-8
        ecg = (ecg - mean) / std
        return ecg.reshape(1, 1300, 1)
    
    def _preprocess_acc(self, acc):
        acc = np.asarray(acc, dtype=np.float32)
        if acc.ndim == 1:
            acc = acc.reshape(-1, 3)
        acc = acc[:250]
        if len(acc) < 250:
            acc = np.pad(acc, ((0, 250 - len(acc)), (0, 0)))
        return acc.reshape(1, 250, 3)

if __name__ == '__main__':
    predictor = OHCAPredictor()
    
    # Demo: Simulate one hour before cardiac arrest
    print("=== 1 Hour Before Cardiac Arrest ===")
    ecg = np.random.randn(1300) * 0.8 + 0.2  # Subtle abnormality
    acc = np.random.randn(250, 3) * 0.3       # Mild tremor
    bio = np.array([65.0, 1.0, 0.7, 145.0, 95.0, 92.0])  # 65M, ischemic, hypertensive, borderline O2
    
    result = predictor.predict(ecg, acc, bio)
    print(f"Risk: {result['risk_level']}")
    print(f"Probability: {result['probability']:.3f}")
    print(f"Recommendation: {result['recommendation']}")
