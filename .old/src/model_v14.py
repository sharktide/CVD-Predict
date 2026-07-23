"""CVD Watch Model v14 — Risk + Severity classification.

Shared backbone + binary risk + 4-class severity.

Outputs:
    risk_output:    (1,) sigmoid probability of event
    severity_output: (4,) softmax over [none, mild, moderate, severe]
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from tensorflow.keras import layers, models

from .backbone import build_backbone


def build_v14(
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

    # Risk head (binary)
    r = layers.Dense(cfg.get("event_hidden", 32), activation="relu", name="risk_dense")(shared)
    r = layers.Dropout(0.2, name="risk_dropout")(r)
    risk_out = layers.Dense(1, activation="sigmoid", name="risk_output")(r)

    # Severity head (4-class)
    s = layers.Dense(cfg.get("event_hidden", 32), activation="relu", name="severity_dense")(shared)
    s = layers.Dropout(0.2, name="severity_dropout")(s)
    severity_out = layers.Dense(4, activation="softmax", name="severity_output")(s)

    model = models.Model(
        inputs=backbone.inputs,
        outputs=[risk_out, severity_out],
        name="cvd_v14",
    )
    return model
