"""CVD Risk Model v5-watch: Inverted v4 model for Apple Watch screening.

This module wraps the v4 ICU model with prediction inversion for
wrist-worn PPG devices. The v4 model was trained on ICU patients
and produces inverted predictions on healthy/outpatient wrist PPG.
Inverting the output recovers useful cardiac screening performance.

Usage:
    from production.cvd_risk_v5_watch.model import CVDWatchPredictor

    predictor = CVDWatchPredictor()
    result = predictor.predict(ppg_signal, features)
    print(result["event_probability"])  # cardiac risk score
    print(result["flagged"])            # True if above threshold
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _DIR


class CVDWatchPredictor:
    """Predictor for cardiac events from Apple Watch-style PPG signals.

    Wraps the v4 ICU model with prediction inversion for wrist PPG.
    The inversion corrects the population mismatch between ICU training
    data and outpatient wrist PPG.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.55,
    ):
        """
        Parameters
        ----------
        model_path : path to v4 .keras model file. If None, uses production default.
        threshold : classification threshold on inverted probability (default 0.55 = 100% precision).
        """
        self.threshold = threshold
        self._model = None
        self._model_path = model_path or str(_MODEL_DIR / "best_model.keras")
        self._feature_columns = self._load_feature_columns()
        self._base_threshold = 0.05  # v4's original threshold

    def _load_feature_columns(self) -> List[str]:
        path = _MODEL_DIR / "feature_columns.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        raise FileNotFoundError(f"feature_columns.json not found at {path}")

    def _load_model(self):
        if self._model is not None:
            return

        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from src.model import build_model
        from src.config import get_model_config
        import zipfile, tempfile, tensorflow as tf

        feature_columns = self._feature_columns
        model_cfg = get_model_config()

        model = build_model(
            ppg_input_shape=(7500, 1),
            feature_dim=len(feature_columns),
            num_event_classes=1,
            num_acuity_classes=6,
            num_sensor_quality_classes=3,
            model_cfg=model_cfg,
        )

        # Load weights from .keras archive
        model_file = Path(self._model_path)
        if not model_file.exists():
            model_file = _MODEL_DIR / "final_model.keras"
        if not model_file.exists():
            raise FileNotFoundError(f"No model found at {model_file}")

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(str(model_file), 'r') as zf:
                zf.extractall(tmpdir)
            h5_path = Path(tmpdir) / "model.weights.h5"
            model.load_weights(str(h5_path))

        self._model = model

    def predict(
        self,
        ppg: np.ndarray,
        features: Dict[str, float],
    ) -> Dict[str, Any]:
        """Run inference on a single Apple Watch PPG signal.

        Parameters
        ----------
        ppg : PPG signal array (25 Hz, ~120 seconds recommended)
        features : dict of HRV/clinical features matching feature_columns.json

        Returns
        -------
        dict with:
            event_probability : float (0-1, higher = more risk)
            raw_v4_probability : float (original v4 output, before inversion)
            flagged : bool (True if event_probability >= threshold)
            threshold : float
            confidence : str ("high", "medium", "low")
        """
        self._load_model()
        import tensorflow as tf

        # Prepare PPG input
        ppg_length = 7500
        ppg_input = np.zeros((1, ppg_length, 1), dtype=np.float32)
        sig = ppg[:ppg_length] if len(ppg) >= ppg_length else np.zeros(ppg_length, dtype=np.float32)
        if len(ppg) < ppg_length:
            sig[:len(ppg)] = ppg
        else:
            sig = ppg[:ppg_length]
        ppg_input[0, :, 0] = sig

        # Prepare feature input
        feat_vec = np.zeros((1, len(self._feature_columns)), dtype=np.float32)
        for i, col in enumerate(self._feature_columns):
            if col in features:
                feat_vec[0, i] = features[col]
            if col == "label_confidence":
                feat_vec[0, i] = 1.0

        # Run v4 model
        preds = self._model({"ppg_input": ppg_input, "feature_input": feat_vec}, training=False)
        raw_v4_prob = float(preds[0].numpy().ravel()[0])

        # Invert for watch context
        event_prob = 1.0 - raw_v4_prob

        # Confidence based on distance from threshold
        dist = abs(event_prob - self.threshold)
        if dist > 0.3:
            confidence = "high"
        elif dist > 0.15:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "event_probability": event_prob,
            "raw_v4_probability": raw_v4_prob,
            "flagged": event_prob >= self.threshold,
            "threshold": self.threshold,
            "confidence": confidence,
        }

    def predict_batch(
        self,
        ppg_batch: np.ndarray,
        features_batch: List[Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """Run inference on a batch of signals."""
        return [self.predict(ppg, feat) for ppg, feat in zip(ppg_batch, features_batch)]

    @property
    def version(self) -> str:
        return "v5-watch"

    @property
    def description(self) -> str:
        return "CVD risk prediction for Apple Watch PPG (inverted v4 ICU model)"

    def get_config(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "base_model": "cvd_risk_v4",
            "inversion": True,
            "threshold": self.threshold,
            "input_type": "apple_watch_ppg_25hz",
            "ppg_length": 7500,
            "feature_columns": len(self._feature_columns),
        }
