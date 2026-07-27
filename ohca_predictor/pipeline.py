"""OHCAPredictorPipeline – single-import interface for OHCA prediction.

Usage:
    from ohca_predictor.pipeline import OHCAPredictorPipeline

    pipe = OHCAPredictorPipeline.load("models/ohca_model_final.weights.h5")
    result = pipe.predict(
        ecg=ecg_signal,            # np.ndarray, shape (N,) at 130 Hz
        accelerometer=accel_signal, # np.ndarray, shape (M, 3) at 52 Hz
        ppg=ppg_signal,            # np.ndarray, shape (P,) at 50 Hz
        demographics=demo_vec,     # np.ndarray, shape (24,)
        medications=med_vec,       # np.ndarray, shape (14,)
        comorbidities=comorb_vec,  # np.ndarray, shape (14,)
        lab_values=lab_vec,        # np.ndarray, shape (8,)
    )
    print(result.risk)          # float, probability of OHCA in next 4h
    print(result.uncertainty)   # float, epistemic uncertainty (sigma)
    print(result.survival)      # np.ndarray, shape (13,) survival curve
    print(result.high_risk)     # bool, True if risk > 0.5
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np
import tensorflow as tf

from .config import ModelConfig, override_config
from .model.architecture import OHCAPredictionModel


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OHCAResult:
    """Structured prediction result."""

    risk: float
    raw_risk: float
    uncertainty: float
    survival: np.ndarray
    high_risk: bool
    risk_percentiles: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk": self.risk,
            "raw_risk": self.raw_risk,
            "uncertainty": self.uncertainty,
            "survival": self.survival.tolist(),
            "high_risk": self.high_risk,
            "risk_percentiles": self.risk_percentiles,
        }

    def __repr__(self) -> str:
        return (
            f"OHCAResult(risk={self.risk:.4f}, uncertainty={self.uncertainty:.4f}, "
            f"high_risk={self.high_risk})"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class OHCAPredictorPipeline:
    """End-to-end OHCA prediction pipeline.

    Wraps model building, weight loading, preprocessing, and inference
    into a single importable class.

    Example::

        from ohca_predictor.pipeline import OHCAPredictorPipeline

        pipe = OHCAPredictorPipeline.load("model.weights.h5")
        result = pipe.predict(ecg=ecg, accel=accel, ppg=ppg,
                              demographics=demo, medications=meds,
                              comorbidities=comorbs, labs=labs)
    """

    # Signal parameters (must match training config)
    ECG_HZ = 130
    ACCEL_HZ = 52
    PPG_HZ = 50
    TOKENS_PER_MODALITY = 64

    def __init__(
        self,
        model: OHCAPredictionModel,
        config: Optional[ModelConfig] = None,
        threshold: float = 0.20,
    ):
        self.model = model
        self.config = config or model._config
        self.threshold = threshold

    @classmethod
    def load(
        cls,
        weights_path: str,
        config: Optional[ModelConfig] = None,
        threshold: float = 0.20,
    ) -> "OHCAPredictorPipeline":
        """Load a trained pipeline from a weights file.

        The weights file should have been saved by the training pipeline.
        Config is loaded from the sibling ``config.json`` if present,
        otherwise uses defaults.

        Args:
            weights_path: Path to ``.weights.h5`` file.
            config: Optional ModelConfig override.
            threshold: Classification threshold (default 0.20).

        Returns:
            Initialized OHCAPredictorPipeline.
        """
        if config is None:
            cfg_path = os.path.join(os.path.dirname(weights_path), "config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r") as f:
                    cfg_dict = json.load(f)
                config = ModelConfig(**cfg_dict.get("model", cfg_dict))
            else:
                config = ModelConfig()

        # Build model with default dims (overridden on first call)
        model = OHCAPredictionModel(
            config=config,
            demographic_dim=24,
            num_medications=14,
            num_comorbidities=14,
            num_labs=8,
        )

        # Build by running a dummy forward pass
        dummy = cls._make_dummy_batch()
        _ = model(dummy, training=False)

        # Load weights
        model.load_weights(weights_path)
        return cls(model=model, config=config, threshold=threshold)

    @classmethod
    def _make_dummy_batch(cls) -> Dict[str, tf.Tensor]:
        """Create a minimal dummy batch for model building."""
        T = cls.TOKENS_PER_MODALITY
        return {
            "ecg": tf.zeros((1, T * 41, 1)),
            "accelerometer": tf.zeros((1, T * 16, 3)),
            "ppg": tf.zeros((1, T * 16, 1)),
            "demographics": tf.zeros((1, 24)),
            "medications": tf.zeros((1, 14)),
            "comorbidities": tf.zeros((1, 14)),
            "lab_values": tf.zeros((1, 8)),
            "ohca_label": tf.zeros((1, 1)),
            "time_to_event": tf.zeros((1, 1)),
            "event_indicator": tf.zeros((1, 1)),
            "survival_events": tf.zeros((1, 13)),
            "heart_rate": tf.zeros((1, 1)),
            "rhythm": tf.zeros((1, 1)),
            "activity_state": tf.zeros((1, 1)),
            "spo2_mean": tf.zeros((1, 1)),
            "sbp_mean": tf.zeros((1, 1)),
            "dbp_mean": tf.zeros((1, 1)),
        }

    def _resample(self, signal: np.ndarray, src_hz: int, target_len: int) -> np.ndarray:
        """Resample a 1-D signal to target_len via linear interpolation."""
        if signal.ndim > 1:
            signal = signal[:, 0] if signal.shape[1] == 1 else np.mean(signal, axis=1)
        n = len(signal)
        if n == 0:
            return np.zeros(target_len, dtype=np.float32)
        x_old = np.linspace(0, 1, n)
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, signal).astype(np.float32)

    def predict(
        self,
        ecg: np.ndarray,
        accelerometer: np.ndarray,
        ppg: np.ndarray,
        demographics: np.ndarray,
        medications: np.ndarray,
        comorbidities: np.ndarray,
        lab_values: np.ndarray,
    ) -> OHCAResult:
        """Run OHCA prediction on raw sensor data.

        All signals are resampled to the model's expected token count
        internally — you can pass any duration of data.

        Args:
            ecg: ECG waveform, shape ``(N,)`` at 130 Hz.
            accelerometer: 3-axis accelerometer, shape ``(M, 3)`` at 52 Hz.
            ppg: PPG waveform, shape ``(P,)`` at 50 Hz.
            demographics: Static demographics vector, shape ``(24,)``.
            medications: Binary medication vector, shape ``(14,)``.
            comorbidities: Binary comorbidity vector, shape ``(14,)``.
            lab_values: Normalized lab values, shape ``(8,)``.

        Returns:
            OHCAResult with risk, uncertainty, and survival curve.
        """
        T = self.TOKENS_PER_MODALITY

        # Resample signals to token count
        ecg_tokens = self._resample(ecg, self.ECG_HZ, T * 41)
        accel_tokens = self._resample(accelerometer, self.ACCEL_HZ, T * 16)
        ppg_tokens = self._resample(ppg, self.PPG_HZ, T * 16)

        # Build batch
        batch = {
            "ecg": tf.expand_dims(tf.expand_dims(ecg_tokens, 0), -1),
            "accelerometer": tf.expand_dims(accel_tokens, 0),
            "ppg": tf.expand_dims(tf.expand_dims(ppg_tokens, 0), -1),
            "demographics": tf.expand_dims(
                tf.cast(demographics, tf.float32), 0
            ),
            "medications": tf.expand_dims(
                tf.cast(medications, tf.float32), 0
            ),
            "comorbidities": tf.expand_dims(
                tf.cast(comorbidities, tf.float32), 0
            ),
            "lab_values": tf.expand_dims(
                tf.cast(lab_values, tf.float32), 0
            ),
            "ohca_label": tf.zeros((1, 1)),
            "time_to_event": tf.zeros((1, 1)),
            "event_indicator": tf.zeros((1, 1)),
            "survival_events": tf.zeros((1, 13)),
            "heart_rate": tf.zeros((1, 1)),
            "rhythm": tf.zeros((1, 1)),
            "activity_state": tf.zeros((1, 1)),
            "spo2_mean": tf.zeros((1, 1)),
            "sbp_mean": tf.zeros((1, 1)),
            "dbp_mean": tf.zeros((1, 1)),
        }

        outputs = self.model(batch, training=False)

        risk = float(outputs["ohca_risk"].numpy().ravel()[0])
        raw_risk = float(outputs["raw_risk"].numpy().ravel()[0])
        survival = outputs["survival_curve"].numpy().ravel()
        unc = outputs["uncertainty"].numpy().ravel()
        uncertainty = float(np.sqrt(unc[0] ** 2 + unc[1] ** 2))

        return OHCAResult(
            risk=risk,
            raw_risk=raw_risk,
            uncertainty=uncertainty,
            survival=survival,
            high_risk=risk >= self.threshold,
            risk_percentiles={
                "p10": float(np.percentile(survival, 10)),
                "p25": float(np.percentile(survival, 25)),
                "p50": float(np.percentile(survival, 50)),
                "p75": float(np.percentile(survival, 75)),
            },
        )

    def predict_batch(
        self, batch: Dict[str, tf.Tensor]
    ) -> Dict[str, np.ndarray]:
        """Run inference on a pre-built TF batch dict (for evaluation).

        Returns dict with numpy arrays:
            ohca_risk, raw_risk, survival_curve, uncertainty
        """
        outputs = self.model(batch, training=False)
        return {k: v.numpy() for k, v in outputs.items() if k != "auxiliary"}
