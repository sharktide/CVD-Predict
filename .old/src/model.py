"""Multi-branch model – PPG CNN-LSTM encoder, feature MLP, event / acuity / domain heads."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models

from src.losses import GradientReversalLayer


# ---------------------------------------------------------------------------
# PPG branch (1-D CNN → temporal module)
# ---------------------------------------------------------------------------

def _build_ppg_branch(
    input_shape: Tuple[int, ...],
    cfg: Dict[str, Any],
) -> Tuple[layers.Input, layers.Layer]:
    """ResNet-style 1-D CNN followed by a temporal module (Bi-LSTM / GRU / TCN)."""
    inp = layers.Input(shape=input_shape, name="ppg_input")
    x = inp

    filters_list = cfg.get("filters", [64, 128, 256])
    k = cfg.get("kernel_size", 3)
    pool = cfg.get("pooling_size", 2)

    for i, filt in enumerate(filters_list):
        shortcut = x
        x = layers.Conv1D(filt, kernel_size=k, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv1D(filt, kernel_size=k, padding="same")(x)
        x = layers.BatchNormalization()(x)

        # Match channels for residual
        if shortcut.shape[-1] != filt:
            shortcut = layers.Conv1D(filt, kernel_size=1, padding="same")(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling1D(pool_size=pool)(x)

    # Temporal module
    temporal_type = cfg.get("temporal_type", "bilstm")
    temporal_units = cfg.get("temporal_units", 128)

    if temporal_type == "bilstm":
        x = layers.Bidirectional(
            layers.LSTM(temporal_units, return_sequences=False)
        )(x)
    elif temporal_type == "gru":
        x = layers.Bidirectional(
            layers.GRU(temporal_units, return_sequences=False)
        )(x)
    else:
        # TCN-style: stacked dilated causal convolutions
        for dilation in (1, 2, 4):
            x = layers.Conv1D(
                temporal_units, kernel_size=3, padding="causal",
                dilation_rate=dilation, activation="relu",
            )(x)
        x = layers.GlobalAveragePooling1D()(x)

    ppg_out = layers.Dense(cfg.get("dense_units", 256), activation="relu", name="ppg_dense")(x)
    return inp, ppg_out


# ---------------------------------------------------------------------------
# Feature (MLP) branch
# ---------------------------------------------------------------------------

def _build_feature_branch(
    input_dim: int,
    cfg: Dict[str, Any],
) -> Tuple[layers.Input, layers.Layer]:
    inp = layers.Input(shape=(input_dim,), name="feature_input")
    x = inp
    for units in cfg.get("dense_layers", [128, 128]):
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(cfg.get("dropout", 0.3))(x)
    return inp, x


# ---------------------------------------------------------------------------
# Event head
# ---------------------------------------------------------------------------

def _build_event_head(shared: layers.Layer, num_classes: int, cfg: Dict[str, Any]) -> layers.Layer:
    x = layers.Dense(cfg.get("hidden_units", 128), activation="relu")(shared)
    x = layers.Dropout(cfg.get("dropout", 0.3))(x)
    return layers.Dense(num_classes, name="event_logits")(x)


# ---------------------------------------------------------------------------
# Acuity head
# ---------------------------------------------------------------------------

def _build_acuity_head(shared: layers.Layer, num_classes: int, cfg: Dict[str, Any]) -> layers.Layer:
    x = layers.Dense(cfg.get("hidden_units", 64), activation="relu")(shared)
    x = layers.Dropout(cfg.get("dropout", 0.2))(x)
    return layers.Dense(num_classes, name="acuity_logits")(x)


# ---------------------------------------------------------------------------
# Domain heads (adversarial via GRL)
# ---------------------------------------------------------------------------

def _build_domain_head(
    shared: layers.Layer,
    grl_lambda: float,
    hidden: int,
    num_classes: int,
    name: str,
) -> Tuple[layers.Layer, layers.Layer]:
    """Return (logits, softmax_output) after gradient reversal."""
    grl = GradientReversalLayer(lambda_=grl_lambda)
    rev = grl(shared)
    rev = layers.Dense(hidden, activation="relu")(rev)
    logits = layers.Dense(num_classes, name=f"{name}_logits")(rev)
    out = layers.Activation("softmax", name=f"{name}_output")(logits)
    return logits, out


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def _build_sensor_quality_head(
    shared: layers.Layer,
    num_classes: int,
    cfg: Dict[str, Any],
) -> Tuple[layers.Layer, layers.Layer]:
    """Predict PPG signal quality class (clean / noisy / dropout).

    Not adversarial — this head is supervised so the encoder learns to
    separate signal quality from clinical content.
    """
    x = layers.Dense(cfg.get("sensor_quality_hidden", 32), activation="relu")(shared)
    x = layers.Dropout(0.2)(x)
    logits = layers.Dense(num_classes, name="sensor_quality_logits")(x)
    out = layers.Activation("softmax", name="sensor_quality_output")(logits)
    return logits, out


def build_model(
    ppg_input_shape: Tuple[int, ...],
    feature_dim: int,
    num_event_classes: int,
    num_acuity_classes: int,
    num_sensor_quality_classes: int = 3,
    model_cfg: Dict[str, Any] = None,
) -> models.Model:
    """Construct the multi-task, multi-branch CVD risk model.

    Outputs
    -------
    - event_output            (sigmoid for binary)
    - acuity_output           (softmax)
    - icu_domain_output       (softmax, adversarial)
    - device_domain_output    (softmax, adversarial)
    - sensor_quality_output   (softmax, supervised)
    """
    if model_cfg is None:
        model_cfg = {}

    ppg_inp, ppg_enc = _build_ppg_branch(ppg_input_shape, model_cfg.get("ppg_branch", {}))
    feat_inp, feat_enc = _build_feature_branch(feature_dim, model_cfg.get("feature_branch", {}))

    # Shared representation
    shared = layers.Concatenate(name="shared_concat")([ppg_enc, feat_enc])
    shared = layers.Dense(model_cfg.get("shared", {}).get("units", 256), activation="relu", name="shared_dense")(shared)
    shared = layers.Dropout(model_cfg.get("shared", {}).get("dropout", 0.4))(shared)

    # Event head
    event_logits = _build_event_head(shared, num_event_classes, model_cfg.get("event_head", {}))
    event_out = layers.Activation("sigmoid", name="event_output")(event_logits)

    # Acuity head
    acuity_logits = _build_acuity_head(shared, num_acuity_classes, model_cfg.get("acuity_head", {}))
    acuity_out = layers.Activation("softmax", name="acuity_output")(acuity_logits)

    # Domain heads (adversarial)
    domain_cfg = model_cfg.get("domain", {})
    grl_lambda = domain_cfg.get("lambda", 1.0)

    _, icu_domain_out = _build_domain_head(
        shared, grl_lambda, domain_cfg.get("icu_domain_hidden", 64), 2, "icu_domain",
    )
    _, device_domain_out = _build_domain_head(
        shared, grl_lambda, domain_cfg.get("device_domain_hidden", 64), 2, "device_domain",
    )

    # Sensor quality head (supervised, not adversarial)
    _, sensor_quality_out = _build_sensor_quality_head(
        shared, num_sensor_quality_classes, domain_cfg,
    )

    model = models.Model(
        inputs=[ppg_inp, feat_inp],
        outputs=[event_out, acuity_out, icu_domain_out, device_domain_out, sensor_quality_out],
        name="cvd_risk_model",
    )
    return model
