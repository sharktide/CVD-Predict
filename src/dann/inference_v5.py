"""
V5 Inference Pipeline - Multi-Branch Wrist PPG Cardiac Arrest Detection.

Usage:
    from src.dann.inference_v5 import CardiacArrestPredictorV5
    
    predictor = CardiacArrestPredictorV5("models/cardiac_arrest_v5/")
    
    result = predictor.predict(
        ppg=ppg_signal,           # (1500,) wrist PPG @ 25Hz
        accel=accel_signal,       # (1500, 3) 3-axis accelerometer
        biodata={                 # Patient metadata
            "age": 65,
            "sex": 1,
            "bmi": 28,
            "spo2": 0.96,
            "body_temp": 36.5,
            "perfusion_index": 0.08,
            "melanin": 0.3,
            "has_hypertension": 1,
            "has_diabetes": 0,
            "has_hf": 0,
            "has_ckd": 0,
            "has_afib": 0,
            "has_copd": 0,
            "on_betas": 1,
            "on_anticoag": 0,
        }
    )
    
    # Output:
    # {
    #   "risk_level": "HIGH",
    #   "neural_probability": 0.87,      # From deep learning model
    #   "final_probability": 0.91,       # After Edge Decision Gate adjustment
    #   "alert": True,
    #   "confidence": "high",
    #   "motion_quality": "good",        # Estimated from ACC
    #   "ppg_quality": "moderate",       # Estimated from signal
    # }
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from typing import Dict, Optional, Union
from scipy.signal import find_peaks, butter, filtfilt, resample
import warnings
warnings.filterwarnings("ignore")


class FeatureExtractorV5:
    """Extract features from wrist PPG + ACC signals."""
    
    def __init__(self, fs=25):
        self.fs = fs
    
    def extract_ppg_quality(self, ppg: np.ndarray) -> Dict:
        """Estimate PPG signal quality."""
        filt = self._bandpass(ppg)
        threshold = np.mean(filt) + 0.1 * np.std(filt)
        peaks, _ = find_peaks(filt, height=threshold, distance=8, prominence=0.01)
        
        snr = np.var(filt) / (np.var(ppg - filt) + 1e-10)
        peak_regularity = 0.0
        if len(peaks) >= 3:
            intervals = np.diff(peaks)
            peak_regularity = 1.0 - (np.std(intervals) / (np.mean(intervals) + 1e-6))
        
        return {
            "n_peaks": len(peaks),
            "snr_db": float(10 * np.log10(snr)),
            "peak_regularity": float(max(0, peak_regularity)),
            "quality": "good" if snr > 20 and peak_regularity > 0.7 else
                       "moderate" if snr > 10 else "poor",
        }
    
    def extract_motion_quality(self, accel: np.ndarray) -> Dict:
        """Estimate motion artifact level from accelerometer."""
        # Remove gravity (high-pass filter)
        accel_no_grav = accel.copy()
        accel_no_grav[:, 2] = accel_no_grav[:, 2] - np.mean(accel_no_grav[:, 2])
        
        # Motion magnitude
        magnitude = np.sqrt(np.sum(accel_no_grav**2, axis=1))
        motion_energy = np.mean(magnitude**2)
        
        # Jerk (derivative of acceleration) - high jerk = sudden movement
        jerk = np.diff(accel_no_grav, axis=0)
        jerk_mag = np.sqrt(np.sum(jerk**2, axis=1))
        mean_jerk = np.mean(jerk_mag)
        
        # Stationarity: how much does the signal vary over time
        window_size = 250  # 10s windows
        n_windows = len(magnitude) // window_size
        if n_windows > 1:
            window_vars = [np.var(magnitude[i*window_size:(i+1)*window_size]) 
                          for i in range(n_windows)]
            stationarity = 1.0 - (np.std(window_vars) / (np.mean(window_vars) + 1e-6))
        else:
            stationarity = 0.5
        
        quality = "stable" if motion_energy < 0.01 and mean_jerk < 0.5 else \
                  "moderate" if motion_energy < 0.1 else "active"
        
        return {
            "motion_energy": float(motion_energy),
            "mean_jerk": float(mean_jerk),
            "stationarity": float(max(0, min(1, stationarity))),
            "quality": quality,
        }

    def _bandpass(self, sig, low=0.5, high=4.0, order=4):
        nyq = self.fs / 2
        b, a = butter(order, [low / nyq, high / nyq], btype='band')
        try:
            return filtfilt(b, a, sig)
        except Exception:
            return sig


class CardiacArrestPredictorV5:
    """
    Production-ready V5 cardiac arrest predictor.
    
    Pipeline:
    1. Preprocess PPG + ACC signals
    2. Run through multi-branch neural network
    3. Apply Edge Decision Gate (Random Forest)
    4. Return risk assessment
    """
    
    BIODATA_COLUMNS = [
        "age", "sex", "bmi", "spo2", "body_temp", "perfusion_index", "melanin",
        "has_hypertension", "has_diabetes", "has_hf", "has_ckd", "has_afib",
        "has_copd", "on_betas", "on_anticoag", "snr_db",
    ]
    
    DEFAULT_BIODATA = {
        "age": 50, "sex": 0, "bmi": 25, "spo2": 0.97, "body_temp": 36.8,
        "perfusion_index": 0.1, "melanin": 0.3,
        "has_hypertension": 0, "has_diabetes": 0, "has_hf": 0,
        "has_ckd": 0, "has_afib": 0, "has_copd": 0,
        "on_betas": 0, "on_anticoag": 0,
    }
    
    RISK_LEVELS = {
        (0.0, 0.3): "LOW",
        (0.3, 0.6): "MODERATE",
        (0.6, 0.8): "HIGH",
        (0.8, 1.01): "CRITICAL",
    }

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.feature_extractor = FeatureExtractorV5()
        self._load_model()
    
    def _load_model(self):
        """Load neural network + edge gate."""
        import sys
        sys.path.insert(0, str(self.model_dir.parent.parent.parent))
        from src.dann.model_v5 import CardiacArrestDetectorV5
        
        with open(self.model_dir / "config.json") as f:
            self.config = json.load(f)
        
        self.device = torch.device("cpu")
        self.model = CardiacArrestDetectorV5(
            n_biodata=self.config["n_biodata"],
            ppg_dim=self.config["ppg_dim"],
            acc_dim=self.config["acc_dim"],
        )
        
        checkpoint = torch.load(
            self.model_dir / "best_model.pt",
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        # Load edge decision gate
        with open(self.model_dir / "edge_gate.pkl", "rb") as f:
            self.edge_gate = pickle.load(f)
        
        # Load CV metrics
        if (self.model_dir / "cv_results.json").exists():
            with open(self.model_dir / "cv_results.json") as f:
                self.cv_results = json.load(f)
        else:
            self.cv_results = None
    
    def _preprocess_ppg(self, ppg: np.ndarray) -> np.ndarray:
        """Preprocess PPG for model input."""
        ppg = ppg.astype(np.float32)
        ppg = ppg - np.mean(ppg)
        std = np.std(ppg)
        if std > 1e-8:
            ppg = ppg / std
        
        if len(ppg) < 1500:
            ppg = np.pad(ppg, (0, 1500 - len(ppg)), mode='edge')
        elif len(ppg) > 1500:
            ppg = ppg[:1500]
        
        return ppg
    
    def _preprocess_accel(self, accel: np.ndarray) -> np.ndarray:
        """Preprocess accelerometer for model input."""
        accel = accel.astype(np.float32)
        
        # Remove gravity
        accel[:, 2] = accel[:, 2] - np.mean(accel[:, 2])
        
        # Normalize
        accel = accel / (np.std(accel) + 1e-8)
        
        if len(accel) < 1500:
            accel = np.pad(accel, ((0, 1500 - len(accel)), (0, 0)), mode='edge')
        elif len(accel) > 1500:
            accel = accel[:1500]
        
        return accel
    
    def _get_risk_level(self, prob: float) -> str:
        for (low, high), level in self.RISK_LEVELS.items():
            if low <= prob < high:
                return level
        return "CRITICAL"
    
    def _get_confidence(self, prob: float) -> str:
        distance = abs(prob - 0.5)
        if distance > 0.3: return "very_high"
        elif distance > 0.2: return "high"
        elif distance > 0.1: return "moderate"
        else: return "low"

    def predict(
        self,
        ppg: np.ndarray,
        accel: np.ndarray,
        biodata: Optional[Dict] = None,
    ) -> Dict:
        """
        Predict cardiac arrest risk from wrist PPG + ACC + biodata.
        
        Args:
            ppg: PPG signal (1500 samples @ 25Hz, or will be resampled)
            accel: Accelerometer (N×3 or 1500×3, 3-axis @ 25Hz)
            biodata: Patient metadata dict (optional, uses defaults)
        
        Returns:
            Risk assessment dict
        """
        # Merge biodata with defaults
        bio = {**self.DEFAULT_BIODATA, **(biodata or {})}
        
        # Resample if needed
        if len(ppg) != 1500:
            ppg = resample(ppg, 1500)
        if len(accel) != 1500:
            accel = resample(accel, 1500, axis=0)
        
        # Preprocess
        ppg_proc = self._preprocess_ppg(ppg)
        accel_proc = self._preprocess_accel(accel)
        
        # Quality checks
        ppg_quality = self.feature_extractor.extract_ppg_quality(ppg_proc)
        motion_quality = self.feature_extractor.extract_motion_quality(accel_proc)
        
        # Add SNR to biodata
        bio["snr_db"] = ppg_quality["snr_db"]
        
        # Neural network prediction
        ppg_tensor = torch.tensor(ppg_proc).unsqueeze(0).unsqueeze(0)  # (1, 1, 1500)
        accel_tensor = torch.tensor(accel_proc).unsqueeze(0).permute(0, 2, 1)  # (1, 3, 1500)
        biodata_tensor = torch.tensor(
            np.array([bio[col] for col in self.BIODATA_COLUMNS], dtype=np.float32)
        ).unsqueeze(0)  # (1, 16)
        
        with torch.no_grad():
            nn_output = self.model(ppg_tensor, accel_tensor, biodata_tensor)
        
        nn_prob = nn_output["probability"].item()
        
        # Edge Decision Gate
        gate_input = np.array([[nn_prob] + [bio[col] for col in self.BIODATA_COLUMNS]])
        gate_prob = self.edge_gate.predict_proba(gate_input)[0, 1]
        
        # Final probability: weighted combination
        final_prob = 0.7 * nn_prob + 0.3 * gate_prob
        
        return {
            "risk_level": self._get_risk_level(final_prob),
            "neural_probability": round(nn_prob, 4),
            "gate_probability": round(gate_prob, 4),
            "final_probability": round(final_prob, 4),
            "alert": final_prob > 0.5,
            "confidence": self._get_confidence(final_prob),
            "ppg_quality": ppg_quality["quality"],
            "motion_quality": motion_quality["quality"],
            "motion_energy": round(motion_quality["motion_energy"], 4),
            "model_version": "5.0",
        }

    def predict_batch(
        self,
        ppg_batch: np.ndarray,
        accel_batch: np.ndarray,
        biodata_batch: Optional[list] = None,
    ) -> list:
        """Predict on a batch of signals."""
        results = []
        for i in range(len(ppg_batch)):
            bio = biodata_batch[i] if biodata_batch else None
            results.append(self.predict(ppg_batch[i], accel_batch[i], bio))
        return results

    def get_model_info(self) -> Dict:
        """Get model metadata."""
        info = {
            "model_dir": str(self.model_dir),
            "version": "5.0",
            "architecture": "Multi-Branch ResNet + Cross-Attention + Edge Gate",
            "inputs": {
                "ppg": "Wrist PPG (1500 samples @ 25Hz)",
                "accel": "3-axis accelerometer (1500×3 @ 25Hz)",
                "biodata": "Patient metadata (16 features)",
            },
            "n_parameters": sum(p.numel() for p in self.model.parameters()),
        }
        if self.cv_results:
            info["cv_metrics"] = self.cv_results.get("avg_metrics", {})
            info["edge_gate_auroc"] = self.cv_results.get("edge_gate_auroc")
        return info


class EarlyWarningPredictor:
    """
    Streaming early-warning wrapper for V6 model.
    
    Compatible with test_cvd_* scripts:
        pipeline = EarlyWarningPredictor("models/cardiac_arrest_v6/")
        result = pipeline.process_stream_step(ppg_segment)
    
    Takes a 1500-sample PPG segment, generates synthetic resting ACC,
    runs through V6 model, and applies clinical early-warning logic:
    - Sudden flatline / pulse collapse detection
    - HR trend decline tracking (over 15+ minutes)
    - Post-workout fitness gatekeeper
    - Healthy profile baseline gating
    """

    def __init__(self, model_dir: str):
        self.predictor = CardiacArrestPredictorV5(model_dir)
        
        from src.dann.inference import FeatureExtractor
        self.feature_extractor = FeatureExtractor()
        self.hr_trend_history = []

    def _synthetic_resting_accel(self, n_samples: int = 1500) -> np.ndarray:
        """Generate synthetic resting wrist accelerometer."""
        rng = np.random.RandomState()
        accel = np.zeros((n_samples, 3), dtype=np.float32)
        accel[:, 0] = rng.normal(0, 0.02, n_samples)
        accel[:, 1] = rng.normal(0, 0.02, n_samples)
        accel[:, 2] = 1.0 + rng.normal(0, 0.01, n_samples)
        return accel

    def process_stream_step(self, full_60s_signal: np.ndarray) -> dict:
        """
        Process a 60-second PPG window (1500 samples @ 25Hz).
        
        Returns a dict compatible with test_cvd_* scripts:
            risk_level, probability, alert, early_warning_trigger, status_message
        """
        feats_dict = self.feature_extractor.extract_to_dict(full_60s_signal)

        current_hr = feats_dict["heart_rate_bpm"]
        current_rmssd = feats_dict["rmssd"]
        current_snr = feats_dict["snr_db"]
        amp_drop = feats_dict["amplitude_drop_ratio"]

        # Run V6 model with synthetic resting ACC
        accel = self._synthetic_resting_accel(len(full_60s_signal))
        model_result = self.predictor.predict(full_60s_signal, accel)
        nn_prob = model_result["final_probability"]

        # Build base response
        current_res = {
            "risk_level": model_result["risk_level"],
            "probability": nn_prob,
            "neural_probability": model_result["neural_probability"],
            "gate_probability": model_result["gate_probability"],
            "confidence": model_result["confidence"],
            "ppg_quality": model_result["ppg_quality"],
            "motion_quality": model_result["motion_quality"],
            "status_message": "V6 Model Active",
            "model_version": "6.0",
        }

        if model_result["alert"]:
            current_res["alert"] = True
            current_res["early_warning_trigger"] = "Neural Network Alert"
            return current_res

        current_res["alert"] = False

        # Track HR trend
        if current_hr > 0:
            self.hr_trend_history.append(current_hr)
            if len(self.hr_trend_history) > 60:
                self.hr_trend_history.pop(0)

        # 1. IMMEDIATE CRITICAL OVERRIDE: Sudden Flatline
        first_half_std = np.std(full_60s_signal[:750])
        if first_half_std > 0.02 and amp_drop < 0.25:
            current_res["risk_level"] = "CRITICAL"
            current_res["probability"] = 0.9900
            current_res["alert"] = True
            current_res["early_warning_trigger"] = "Sudden Pulse Amplitude Collapse"
            return current_res

        # 2. FITNESS GATEKEEPER CHECK
        is_post_workout = (current_hr > 65.0) and (current_rmssd > 18.0)

        # 3. PRIORITIZED EARLY TREND ENGINE
        if len(self.hr_trend_history) >= 15 and not is_post_workout:
            halfway = len(self.hr_trend_history) // 2
            older_avg = np.mean(self.hr_trend_history[:halfway])
            recent_avg = np.mean(self.hr_trend_history[halfway:])
            true_hr_drop = recent_avg - older_avg

            if true_hr_drop < -1.5 and current_hr < 70.0:
                current_res["risk_level"] = "HIGH"
                current_res["probability"] = 0.8500
                current_res["alert"] = True
                current_res["early_warning_trigger"] = (
                    f"Continuous Long-term Trend Decline (Dropped {abs(true_hr_drop):.1f} BPM)"
                )
                return current_res

        # 4. FALLBACK BASELINE GATE
        is_healthy_profile = (current_hr > 55.0) and (current_rmssd > 8.0) and (current_snr > 5.0)

        if is_healthy_profile:
            current_res["risk_level"] = "LOW"
            current_res["alert"] = False
            current_res["probability"] = min(nn_prob, 0.15)
        else:
            if 0.0 < current_hr < 48.0 and current_rmssd < 6.0:
                current_res["risk_level"] = "CRITICAL"
                current_res["probability"] = 0.9900
                current_res["alert"] = True
                current_res["early_warning_trigger"] = "Pathological Progressive Bradycardia"

        return current_res
