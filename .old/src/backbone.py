"""Shared backbone for v12-v15 Apple Watch cardiac arrest prediction models.

4-branch architecture with cross-attention:
    PPG Branch:     3-block ResNet 1D-CNN → BiLSTM → Dense(32)
    Accel Branch:   3-block 1D-CNN → BiLSTM → Dense(32)
    Cross-Attention: PPG attends to Accel (motion-conditioned denoising)
    Biodata Branch: 2-layer MLP → Dense(16)
    HRV Branch:     2-layer MLP → Dense(32)
    Shared Fusion:  Concat → Dense(64) → task-specific heads
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


def _resnet_block(x, filters, kernel_size, name_prefix):
    """Single ResNet block: Conv-BN-ReLU-Conv-BN + skip."""
    shortcut = x
    x = layers.Conv1D(filters, kernel_size=kernel_size, padding="same",
                       name=f"{name_prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_relu1")(x)
    x = layers.Conv1D(filters, kernel_size=kernel_size, padding="same",
                       name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, padding="same",
                                  name=f"{name_prefix}_skip_proj")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name_prefix}_skip_bn")(shortcut)
    x = layers.Add(name=f"{name_prefix}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{name_prefix}_relu2")(x)
    return x


def build_backbone(
    ppg_input_shape: Tuple[int, ...] = (7500, 1),
    accel_input_shape: Tuple[int, ...] = (7500, 3),
    hrv_feature_dim: int = 50,
    biodata_dim: int = 9,
    cfg: Dict[str, Any] = None,
) -> Tuple[layers.Layer, models.Model]:
    """Build the shared backbone and return (shared_output, full_model).

    Returns
    -------
    shared_out : Keras tensor of shape (batch, 64)
    model : Keras Model with 4 inputs and shared_out as single output
    """
    if cfg is None:
        cfg = {}

    # === PPG Branch ===
    ppg_inp = layers.Input(shape=ppg_input_shape, name="ppg_input")
    x = ppg_inp
    for i, (filt, ksize, pool) in enumerate([
        (cfg.get("ppg_filt1", 16), 5, 4),
        (cfg.get("ppg_filt2", 32), 5, 4),
        (cfg.get("ppg_filt3", 64), 3, 2),
    ]):
        x = _resnet_block(x, filt, ksize, name_prefix=f"ppg_block{i+1}")
        x = layers.MaxPooling1D(pool_size=pool, name=f"ppg_pool{i+1}")(x)
    lstm_units = cfg.get("ppg_lstm", 32)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False),
                              name="ppg_lstm")(x)
    ppg_enc = layers.Dense(cfg.get("ppg_dense", 32), activation="relu", name="ppg_enc")(x)

    # === Accel Branch ===
    accel_inp = layers.Input(shape=accel_input_shape, name="accel_input")
    a = accel_inp
    for i, (filt, ksize, pool) in enumerate([
        (cfg.get("accel_filt1", 16), 7, 4),
        (cfg.get("accel_filt2", 32), 5, 4),
        (cfg.get("accel_filt3", 64), 3, 2),
    ]):
        a = _resnet_block(a, filt, ksize, name_prefix=f"accel_block{i+1}")
        a = layers.MaxPooling1D(pool_size=pool, name=f"accel_pool{i+1}")(a)
    accel_lstm_units = cfg.get("accel_lstm", 16)
    a = layers.Bidirectional(layers.LSTM(accel_lstm_units, return_sequences=False),
                              name="accel_lstm")(a)
    accel_enc = layers.Dense(cfg.get("accel_dense", 32), activation="relu", name="accel_enc")(a)

    # === Cross-Attention: PPG attends to Accel ===
    # Project both to same dim for attention
    attn_dim = cfg.get("attn_dim", 32)
    q = layers.Dense(attn_dim, name="attn_query")(ppg_enc)    # (batch, attn_dim)
    k = layers.Dense(attn_dim, name="attn_key")(accel_enc)    # (batch, attn_dim)
    v = layers.Dense(attn_dim, name="attn_value")(accel_enc)  # (batch, attn_dim)

    # Scaled dot-product attention (single-head, batch-level)
    attn_score = layers.Lambda(
        lambda inputs: tf.reduce_sum(
            inputs[0] * inputs[1], axis=-1, keepdims=True
        ) / tf.math.sqrt(tf.cast(attn_dim, tf.float32)),
        name="attn_score"
    )([q, k])  # (batch, 1)
    attn_weights = layers.Activation("softmax", name="attn_weights")(attn_score)  # (batch, 1)
    context = layers.Multiply(name="attn_context")([attn_weights, v])  # (batch, attn_dim)

    # Fuse PPG + attention context
    fused = layers.Concatenate(name="ppg_accel_fused")([ppg_enc, context])  # (batch, 32+32=64)
    fused_enc = layers.Dense(cfg.get("fused_dense", 64), activation="relu", name="fused_dense")(fused)

    # === HRV Feature Branch ===
    feat_inp = layers.Input(shape=(hrv_feature_dim,), name="feature_input")
    y = feat_inp
    for units in cfg.get("feat_layers", [32, 32]):
        y = layers.Dense(units, activation="relu")(y)
        y = layers.BatchNormalization()(y)
        y = layers.Dropout(cfg.get("feat_dropout", 0.3))(y)
    feat_enc = y  # (batch, 32)

    # === Biodata Branch ===
    bio_inp = layers.Input(shape=(biodata_dim,), name="biodata_input")
    b = bio_inp
    for units in cfg.get("bio_layers", [32, 16]):
        b = layers.Dense(units, activation="relu")(b)
        b = layers.BatchNormalization()(b)
        b = layers.Dropout(cfg.get("bio_dropout", 0.3))(b)
    bio_enc = b  # (batch, 16)

    # === Shared Fusion ===
    shared = layers.Concatenate(name="shared_concat")([fused_enc, feat_enc, bio_enc])
    shared = layers.Dense(cfg.get("shared_units", 64), activation="relu", name="shared_dense")(shared)
    shared = layers.Dropout(cfg.get("shared_dropout", 0.3), name="shared_dropout")(shared)

    backbone_model = models.Model(
        inputs=[ppg_inp, accel_inp, feat_inp, bio_inp],
        outputs=shared,
        name="backbone",
    )

    return shared, backbone_model
