"""
Cardiac Arrest Detection - Inference Pipeline

Usage:
    from src.dann.inference import CardiacArrestPredictor
    
    predictor = CardiacArrestPredictor("models/cardiac_arrest_v4/")
    
    # From raw PPG (1500 samples @ 25Hz)
    result = predictor.predict_ppg(ppg_signal)
    
    # From pre-extracted features
    result = predictor.predict_features(features)
    
    print(result)
    # {
    #   "risk_level": "HIGH",
    #   "probability": 0.87,
    #   "alert": True,
    #   "confidence": "high",
    #   "latent_embedding": [0.12, -0.34, ...]
    # }
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt
from typing import Dict, List, Optional, Union
import json
import warnings
from collections import deque
warnings.filterwarnings("ignore")

class FeatureExtractor:
    def __init__(self, fs=25):
        self.fs = fs
    
    def _bandpass_filter(self, sig, low=0.5, high=4.0, order=4):
        nyq = self.fs / 2
        b, a = butter(order, [low / nyq, high / nyq], btype='band')
        try: return filtfilt(b, a, sig)
        except Exception: return sig
    
    def extract_to_dict(self, ppg: np.ndarray) -> dict:
        """Extract features and return an explicit dictionary to prevent indexing bugs."""
        feats = {}
        feats["mean"] = float(np.mean(ppg))
        feats["std"] = float(np.std(ppg))
        feats["range"] = float(np.ptp(ppg))
        feats["iqr"] = float(np.percentile(ppg, 75) - np.percentile(ppg, 25))
        feats["energy"] = float(np.sum(ppg**2))
        feats["zero_crossings"] = int(np.sum(np.diff(np.sign(ppg)) != 0))
        
        filt = self._bandpass_filter(ppg)
        threshold = np.mean(filt) + 0.1 * np.std(filt)
        peaks, _ = find_peaks(filt, height=threshold, distance=8, prominence=0.01)
        feats["n_peaks"] = len(peaks)
        
        if len(peaks) >= 3:
            intervals = np.diff(peaks) / self.fs * 1000
            feats["heart_rate_bpm"] = float(60000 / np.mean(intervals))
            diffs = np.diff(intervals)
            feats["rmssd"] = float(np.sqrt(np.mean(diffs**2)))
            feats["sdnn"] = float(np.std(intervals, ddof=1))
            feats["pnn50"] = float(np.sum(np.abs(diffs) > 50) / len(diffs) * 100) if len(diffs) > 0 else 0.0
            feats["mean_nn"] = float(np.mean(intervals))
            feats["sdnn_ratio"] = feats["sdnn"] / (feats["mean_nn"] + 1e-6)
            feats["hr_std"] = float(np.std(np.diff(peaks) / self.fs * 60))
            
            peak_heights = filt[peaks]
            feats["peak_mean"] = float(np.mean(peak_heights))
            feats["peak_std"] = float(np.std(peak_heights))
            feats["peak_min"] = float(np.min(peak_heights))
            feats["peak_max"] = float(np.max(peak_heights))
        else:
            for k in ["heart_rate_bpm", "rmssd", "sdnn", "pnn50", "mean_nn", "sdnn_ratio", "hr_std", "peak_mean", "peak_std", "peak_min", "peak_max"]:
                feats[k] = 0.0
        
        for k in ["vlf_power", "lf_power", "hf_power", "lf_hf_ratio", "total_power"]:
            feats[k] = 0.0
            
        if len(peaks) > 10:
            try:
                rr_intervals = np.diff(peaks) / self.fs
                from scipy.interpolate import interp1d
                t_rr = np.cumsum(rr_intervals)
                t_interp = np.arange(t_rr[0], t_rr[-1], 1.0/self.fs)
                f_interp = interp1d(t_rr, rr_intervals, kind='linear', fill_value='extrapolate')
                rr_interp = f_interp(t_interp)
                rr_interp = rr_interp - np.mean(rr_interp)
                
                freqs = np.fft.rfftfreq(len(rr_interp), d=1.0/self.fs)
                fft_power = np.abs(np.fft.rfft(rr_interp))**2
                
                feats["vlf_power"] = float(np.sum(fft_power[(freqs >= 0.003) & (freqs < 0.04)]))
                feats["lf_power"] = float(np.sum(fft_power[(freqs >= 0.04) & (freqs < 0.15)]))
                feats["hf_power"] = float(np.sum(fft_power[(freqs >= 0.15) & (freqs < 0.4)]))
                feats["lf_hf_ratio"] = feats["lf_power"] / (feats["hf_power"] + 1e-6)
                feats["total_power"] = feats["vlf_power"] + feats["lf_power"] + feats["hf_power"]
            except: pass
        
        snr = np.var(filt) / (np.var(ppg - filt) + 1e-10)
        feats["snr_db"] = float(10 * np.log10(snr)) if snr > 0 else -10.0
        feats["signal_quality"] = float(min(1.0, snr / 100))
        
        diffs_sig = np.diff(ppg)
        feats["d1_mean"] = float(np.mean(np.abs(diffs_sig)))
        feats["d1_std"] = float(np.std(diffs_sig))
        feats["d2_mean"] = float(np.mean(np.abs(np.diff(diffs_sig)))) if len(diffs_sig) > 1 else 0.0
        
        sub_chunks = np.array_split(ppg, 4)
        sub_stds = [float(np.std(chunk)) for chunk in sub_chunks]
        sub_means = [float(np.mean(chunk)) for chunk in sub_chunks]
        
        feats["volatility_variance"] = float(np.std(sub_stds))
        feats["amplitude_drop_ratio"] = float(sub_stds[-1] / (sub_stds[0] + 1e-6))
        feats["mean_drift"] = float(sub_means[-1] - sub_means[0])
        feats["max_sub_std"] = float(max(sub_stds))
        feats["min_sub_std"] = float(min(sub_stds))
        feats["sub_std_ratio"] = float(feats["min_sub_std"] / (feats["max_sub_std"] + 1e-6))
        
        return feats

    def extract(self, ppg: np.ndarray) -> np.ndarray:
        """Keeps PyTorch integration intact by returning the raw array."""
        feats_dict = self.extract_to_dict(ppg)
        vector = np.array(list(feats_dict.values()), dtype=np.float32)
        vector = np.nan_to_num(vector, nan=0.0, posinf=1000.0, neginf=-1000.0)
        return np.clip(vector, -1e4, 1e4)


class CardiacArrestPredictor:
    """
    Production-ready cardiac arrest predictor.
    
    Usage:
        predictor = CardiacArrestPredictor("models/cardiac_arrest_v4/")
        result = predictor.predict_ppg(ppg_signal)
    """
    
    RISK_LEVELS = {
        (0.0, 0.3): "LOW",
        (0.3, 0.6): "MODERATE",
        (0.6, 0.8): "HIGH",
        (0.8, 1.01): "CRITICAL",
    }
    
    def __init__(self, model_dir: str):
        """
        Args:
            model_dir: Directory containing best_model.pt and config.json
        """
        self.model_dir = Path(model_dir)
        self.feature_extractor = FeatureExtractor()
        self._load_model()
    
    def _load_model(self):
        """Load PyTorch model."""
        import torch
        import sys
        sys.path.insert(0, str(self.model_dir.parent.parent.parent))
        from src.dann.model_v2 import build_model
        
        with open(self.model_dir / "config.json") as f:
            self.config = json.load(f)
        
        self.device = torch.device("cpu")
        self.model = build_model(self.config)
        
        checkpoint = torch.load(
            self.model_dir / "best_model.pt",
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        # Load CV metrics for confidence estimation
        if (self.model_dir / "cv_results.json").exists():
            with open(self.model_dir / "cv_results.json") as f:
                self.cv_results = json.load(f)
        else:
            self.cv_results = None
    
    def _get_risk_level(self, probability: float) -> str:
        """Map probability to risk level."""
        for (low, high), level in self.RISK_LEVELS.items():
            if low <= probability < high:
                return level
        return "CRITICAL"
    
    def _get_confidence(self, probability: float) -> str:
        """Estimate confidence based on distance from decision boundary."""
        distance = abs(probability - 0.5)
        if distance > 0.3:
            return "very_high"
        elif distance > 0.2:
            return "high"
        elif distance > 0.1:
            return "moderate"
        else:
            return "low"
    
    def predict_ppg(self, ppg: np.ndarray) -> Dict:
        """
        Predict cardiac arrest risk from raw PPG signal.
        
        Args:
            ppg: PPG signal (1500 samples @ 25Hz)
            
        Returns:
            Dict with risk_level, probability, alert, confidence, etc.
        """
        import torch
        
        # Preprocess
        ppg = ppg.astype(np.float32)
        ppg = ppg - np.mean(ppg)
        std = np.std(ppg)
        if std > 1e-8:
            ppg = ppg / std
        
        # Pad/truncate
        if len(ppg) < 1500:
            ppg = np.pad(ppg, (0, 1500 - len(ppg)), mode='edge')
        elif len(ppg) > 1500:
            ppg = ppg[:1500]
        
        # Extract features
        features = self.feature_extractor.extract(ppg)
        
        # Predict
        ppg_tensor = torch.tensor(ppg).unsqueeze(0).unsqueeze(0)  # (1, 1, 1500)
        feat_tensor = torch.tensor(features).unsqueeze(0)  # (1, 38)
        
        with torch.no_grad():
            outputs = self.model(ppg_tensor, feat_tensor)
        
        prob = outputs["probability"].item()
        latent = outputs["latent"].squeeze().numpy().tolist()
        
        return {
            "risk_level": self._get_risk_level(prob),
            "probability": round(prob, 4),
            "alert": prob > 0.5,
            "confidence": self._get_confidence(prob),
            "latent_embedding": latent[:16],  # First 16 dims for brevity
            "model_version": "v4.0",
        }
    
    def predict_features(self, features: np.ndarray) -> Dict:
        """
        Predict from pre-extracted features.
        
        Args:
            features: Feature vector (38 features)
            
        Returns:
            Dict with risk_level, probability, alert, confidence
        """
        import torch
        
        features = features.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = np.clip(features, -1e6, 1e6)
        
        # Use zero PPG (features-only prediction)
        ppg_tensor = torch.zeros(1, 1, 1500)
        feat_tensor = torch.tensor(features).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(ppg_tensor, feat_tensor)
        
        prob = outputs["probability"].item()
        latent = outputs["latent"].squeeze().numpy().tolist()
        
        return {
            "risk_level": self._get_risk_level(prob),
            "probability": round(prob, 4),
            "alert": prob > 0.5,
            "confidence": self._get_confidence(prob),
            "latent_embedding": latent[:16],
            "model_version": "v4.0",
        }
    
    def predict_batch(self, ppg_batch: np.ndarray) -> List[Dict]:
        """
        Predict on a batch of PPG signals.
        
        Args:
            ppg_batch: (N, 1500) array of PPG signals
            
        Returns:
            List of prediction dicts
        """
        import torch
        
        results = []
        for ppg in ppg_batch:
            results.append(self.predict_ppg(ppg))
        return results
    
    def get_model_info(self) -> Dict:
        """Get model metadata."""
        info = {
            "model_dir": str(self.model_dir),
            "n_features": self.config["n_features"],
            "input_length": 1500,
            "fs_hz": 25.0,
            "classes": ["normal", "cardiac_arrest"],
            "version": "4.0",
        }
        
        if self.cv_results:
            info["cv_metrics"] = self.cv_results.get("avg_metrics", {})
        
        return info

class EarlyWarningPredictor:
    def __init__(self, model_dir: str):
        from src.dann.inference import CardiacArrestPredictor
        self.predictor = CardiacArrestPredictor(model_dir)
        self.feature_extractor = FeatureExtractor()
        self.hr_trend_history = []
        
    def process_stream_step(self, full_60s_signal: np.ndarray) -> dict:
        feats_dict = self.feature_extractor.extract_to_dict(full_60s_signal)
        feats_vector = self.feature_extractor.extract(full_60s_signal)
        
        current_res = self.predictor.predict_features(feats_vector)
        
        current_hr = feats_dict["heart_rate_bpm"]
        current_rmssd = feats_dict["rmssd"]
        current_snr = feats_dict["snr_db"]
        amp_drop = feats_dict["amplitude_drop_ratio"]
        
        if current_hr > 0:
            self.hr_trend_history.append(current_hr)
            if len(self.hr_trend_history) > 60: # Maintain a wider tracking history (15 mins)
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

        # 3. PRIORITIZED EARLY TREND ENGINE (Bypasses baseline logic completely)
        if len(self.hr_trend_history) >= 15 and not is_post_workout:
            halfway = len(self.hr_trend_history) // 2
            older_avg = np.mean(self.hr_trend_history[:halfway])
            recent_avg = np.mean(self.hr_trend_history[halfway:])
            
            true_hr_drop = recent_avg - older_avg
            
            if true_hr_drop < -1.5 and current_hr < 70.0:
                current_res["risk_level"] = "HIGH"
                current_res["probability"] = 0.8500
                current_res["alert"] = True
                current_res["early_warning_trigger"] = f"Continuous Long-term Trend Decline (Dropped {abs(true_hr_drop):.1f} BPM)"
                # RETAIN THE DL LATENT ARRAYS: We do not call direct dict override; 
                # instead, we return the fully populated object with updated fields.
                return current_res


        # 4. FALLBACK BASELINE GATE (Only runs if no dangerous trend is found)
        is_healthy_profile = (current_hr > 55.0) and (current_rmssd > 8.0) and (current_snr > 5.0)
        
        if is_healthy_profile:
            current_res["risk_level"] = "LOW"
            current_res["alert"] = False
            current_res["probability"] = 0.1200
        else:
            if 0.0 < current_hr < 48.0 and current_rmssd < 6.0:
                current_res["risk_level"] = "CRITICAL"
                current_res["probability"] = 0.9900
                current_res["alert"] = True
                current_res["early_warning_trigger"] = "Pathological Progressive Bradycardia"
                
        return current_res

