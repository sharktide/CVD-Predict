"""CVD Watch Model v15 — Multi-horizon + Risk/Severity Categorization.

Shared backbone + 3 horizon probabilities + 3-tier classification.

Outputs:
    horizon_1h:   (1,) P(event within 1 hour)
    horizon_6h:   (1,) P(event within 6 hours)
    horizon_24h:  (1,) P(event within 24 hours)
    tier_output:  (3,) softmax over [Green, Yellow, Red]

Tier logic (postprocessing, not learned):
    Green:  All horizons < 0.3
    Yellow: Any horizon 0.3-0.7 OR 24h > 0.7
    Red:    1h or 6h > 0.7
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from tensorflow.keras import layers, models

from .backbone import build_backbone


def build_v15(
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

    x = layers.Dense(cfg.get("event_hidden", 32), activation="relu", name="horizon_dense")(shared)
    x = layers.Dropout(0.2, name="horizon_dropout")(x)
    horizon_1h = layers.Dense(1, activation="sigmoid", name="horizon_1h")(x)
    horizon_6h = layers.Dense(1, activation="sigmoid", name="horizon_6h")(x)
    horizon_24h = layers.Dense(1, activation="sigmoid", name="horizon_24h")(x)

    # Tier classification from horizon probabilities
    tier_in = layers.Concatenate(name="tier_input")([horizon_1h, horizon_6h, horizon_24h])
    t = layers.Dense(16, activation="relu", name="tier_dense")(tier_in)
    tier_out = layers.Dense(3, activation="softmax", name="tier_output")(t)

    model = models.Model(
        inputs=backbone.inputs,
        outputs=[horizon_1h, horizon_6h, horizon_24h, tier_out],
        name="cvd_v15",
    )
    return model
