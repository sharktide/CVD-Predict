"""CVD Watch Model v6 — lightweight CNN-BiLSTM for 25 Hz wrist PPG.

Designed specifically for Apple Watch / wearable cardiac event screening.
No ICU-specific auxiliary heads — pure binary classification.

Architecture:
    PPG Branch:  3-block ResNet 1D-CNN → BiLSTM → Dense
    Feature Branch: 2-layer MLP
    Shared: Concat → Dense → Event Head (sigmoid)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


def build_watch_model(
    ppg_input_shape: Tuple[int, ...] = (7500, 1),
    feature_dim: int = 50,
    cfg: Dict[str, Any] = None,
) -> models.Model:
    """Build a lightweight model for wrist PPG cardiac event screening.

    Parameters
    ----------
    ppg_input_shape : (T, 1) where T = fs * duration (e.g. 25*120=3000 or padded to 7500)
    feature_dim : number of HRV/clinical features
    cfg : model configuration dict
    """
    if cfg is None:
        cfg = {}

    # --- PPG Branch (lightweight ResNet 1D-CNN) ---
    ppg_inp = layers.Input(shape=ppg_input_shape, name="ppg_input")
    x = ppg_inp

    # Block 1: 16 filters
    filt1 = cfg.get("filt1", 16)
    shortcut1 = x
    x = layers.Conv1D(filt1, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(filt1, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    if shortcut1.shape[-1] != filt1:
        shortcut1 = layers.Conv1D(filt1, kernel_size=1, padding="same")(shortcut1)
        shortcut1 = layers.BatchNormalization()(shortcut1)
    x = layers.Add()([x, shortcut1])
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=4)(x)  # 4x downsample for 25Hz

    # Block 2: 32 filters
    filt2 = cfg.get("filt2", 32)
    shortcut2 = x
    x = layers.Conv1D(filt2, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(filt2, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    if shortcut2.shape[-1] != filt2:
        shortcut2 = layers.Conv1D(filt2, kernel_size=1, padding="same")(shortcut2)
        shortcut2 = layers.BatchNormalization()(shortcut2)
    x = layers.Add()([x, shortcut2])
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=4)(x)

    # Block 3: 64 filters
    filt3 = cfg.get("filt3", 64)
    shortcut3 = x
    x = layers.Conv1D(filt3, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(filt3, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    if shortcut3.shape[-1] != filt3:
        shortcut3 = layers.Conv1D(filt3, kernel_size=1, padding="same")(shortcut3)
        shortcut3 = layers.BatchNormalization()(shortcut3)
    x = layers.Add()([x, shortcut3])
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Temporal: BiLSTM
    lstm_units = cfg.get("lstm_units", 32)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(x)
    ppg_enc = layers.Dense(cfg.get("ppg_dense", 32), activation="relu", name="ppg_dense")(x)

    # --- Feature Branch (MLP) ---
    feat_inp = layers.Input(shape=(feature_dim,), name="feature_input")
    y = feat_inp
    for units in cfg.get("feat_layers", [32, 32]):
        y = layers.Dense(units, activation="relu")(y)
        y = layers.BatchNormalization()(y)
        y = layers.Dropout(cfg.get("feat_dropout", 0.3))(y)
    feat_enc = y

    # --- Shared + Event Head ---
    shared = layers.Concatenate(name="shared_concat")([ppg_enc, feat_enc])
    shared = layers.Dense(cfg.get("shared_units", 32), activation="relu", name="shared_dense")(shared)
    shared = layers.Dropout(cfg.get("shared_dropout", 0.3))(shared)

    # Event head (binary classification)
    x = layers.Dense(cfg.get("event_hidden", 16), activation="relu")(shared)
    x = layers.Dropout(0.2)(x)
    event_out = layers.Dense(1, activation="sigmoid", name="event_output")(x)

    model = models.Model(
        inputs=[ppg_inp, feat_inp],
        outputs=[event_out],
        name="cvd_watch_model",
    )
    return model
