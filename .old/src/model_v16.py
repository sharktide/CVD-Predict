"""CVD Watch Model v16 — Wrist-only Cardiac Arrest Detection.

Same architecture style as v12 (backbone + binary head), but designed
for wristppg-only training on cardiac arrest detection.

Inputs:
    ppg_input:      (7500, 1) PPG waveform at 25 Hz
    accel_input:    (7500, 3) 3-axis accelerometer at 25 Hz
    feature_input:  (N,) HRV + wristppg-derived features
    biodata_input:  (12,) clinical/demographic features

Output:
    event_output:   (1,) sigmoid probability of cardiac arrest
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from tensorflow.keras import layers, models

from .backbone import build_backbone


def build_v16(
    ppg_input_shape: Tuple[int, ...] = (7500, 1),
    accel_input_shape: Tuple[int, ...] = (7500, 3),
    hrv_feature_dim: int = 50,
    biodata_dim: int = 12,
    cfg: Dict[str, Any] = None,
) -> models.Model:
    if cfg is None:
        cfg = {}

    shared, backbone = build_backbone(
        ppg_input_shape=ppg_input_shape,
        accel_input_shape=accel_input_shape,
        hrv_feature_dim=hrv_feature_dim,
        biodata_dim=biodata_dim,
        cfg=cfg,
    )

    # Binary cardiac arrest head (same style as v12)
    x = layers.Dense(cfg.get("event_hidden", 32), activation="relu", name="event_dense")(shared)
    x = layers.Dropout(0.2, name="event_dropout")(x)
    event_out = layers.Dense(1, activation="sigmoid", name="event_output")(x)

    model = models.Model(
        inputs=backbone.inputs,
        outputs=[event_out],
        name="cvd_v16",
    )
    return model
