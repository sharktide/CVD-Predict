"""CVD Watch Model v13 — Multi-horizon probability prediction.

Shared backbone + 3 sigmoid outputs for 1h, 6h, 24h horizons.

Output:
    horizon_1h:   (1,) P(event within 1 hour)
    horizon_6h:   (1,) P(event within 6 hours)
    horizon_24h:  (1,) P(event within 24 hours)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from tensorflow.keras import layers, models

from .backbone import build_backbone


def build_v13(
    ppg_input_shape: Tuple[int, ...] = (7500, 1),
    accel_input_shape: Tuple[int, ...] = (7500, 3),
    hrv_feature_dim: int = 50,
    biodata_dim: int = 9,
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

    x = layers.Dense(cfg.get("event_hidden", 32), activation="relu", name="event_dense")(shared)
    x = layers.Dropout(0.2, name="event_dropout")(x)
    horizon_1h = layers.Dense(1, activation="sigmoid", name="horizon_1h")(x)
    horizon_6h = layers.Dense(1, activation="sigmoid", name="horizon_6h")(x)
    horizon_24h = layers.Dense(1, activation="sigmoid", name="horizon_24h")(x)

    model = models.Model(
        inputs=backbone.inputs,
        outputs=[horizon_1h, horizon_6h, horizon_24h],
        name="cvd_v13",
    )
    return model
