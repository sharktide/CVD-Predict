"""Multi-input LSTM/GRU model for short-term ischemic stroke prediction.

Architecture
------------
- **Time branch**: LSTM / GRU that ingests sliding-window feature sequences.
- **Static branch**: Dense layers for clinical risk factors (CHA₂DS₂-VASc, age, sex, …).
- **Combined head**: Concatenated outputs → Dense → sigmoid risk probability.

The model predicts the probability of stroke within the next 24–72 h and is
designed to be converted to CoreML / TFLite for on-device inference.
"""

from __future__ import annotations

from typing import Literal

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import (
    DENSE_UNITS,
    DROPOUT,
    LSTM_UNITS,
    N_STATIC_FEATURES,
    N_TIME_FEATURES,
    WINDOW_LENGTH_MINUTES,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _time_branch(
    x: layers.Layer,
    rnn_type: Literal["lstm", "gru"] = "lstm",
    units: int = LSTM_UNITS,
    dropout: float = DROPOUT,
) -> layers.Layer:
    """Recurrent branch over the time axis."""
    rnn_cls = layers.LSTM if rnn_type == "lstm" else layers.GRU
    x = rnn_cls(units, return_sequences=False, dropout=dropout, recurrent_dropout=dropout)(x)
    x = layers.BatchNormalization()(x)
    return x


def _static_branch(
    x: layers.Layer,
    units: int = DENSE_UNITS,
    dropout: float = DROPOUT,
) -> layers.Layer:
    """Dense branch for static clinical features."""
    x = layers.Dense(units, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(units // 2, activation="relu")(x)
    return x


def _risk_head(
    combined: layers.Layer,
    dropout: float = DROPOUT,
) -> layers.Layer:
    """Final dense layers producing a stroke-risk probability."""
    x = layers.Dense(DENSE_UNITS, activation="relu")(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(DENSE_UNITS // 2, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(1, activation="sigmoid", name="stroke_risk")(x)
    return x


# ---------------------------------------------------------------------------
# Public model constructors
# ---------------------------------------------------------------------------


def build_model(
    window_length: int = WINDOW_LENGTH_MINUTES,
    n_time_features: int = N_TIME_FEATURES,
    n_static_features: int = N_STATIC_FEATURES,
    rnn_type: Literal["lstm", "gru"] = "lstm",
    rnn_units: int = LSTM_UNITS,
    dense_units: int = DENSE_UNITS,
    dropout: float = DROPOUT,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Build and compile the multi-input stroke-prediction model.

    Parameters
    ----------
    window_length : int
        Number of time steps in each input window (minutes).
    n_time_features : int
        Number of features per time step in the temporal input.
    n_static_features : int
        Number of scalar clinical features.
    rnn_type : ``"lstm"`` or ``"gru"``
        Recurrent cell type.
    rnn_units : int
        Hidden units in the RNN layer.
    dense_units : int
        Width of the dense layers.
    dropout : float
        Dropout rate.
    learning_rate : float
        Adam optimiser learning rate.

    Returns
    -------
    Compiled ``keras.Model``.
    """
    # --- Inputs ---
    time_input = layers.Input(
        shape=(window_length, n_time_features),
        name="time_input",
        dtype=tf.float32,
    )
    static_input = layers.Input(
        shape=(n_static_features,),
        name="static_input",
        dtype=tf.float32,
    )

    # --- Branches ---
    time_branch = _time_branch(time_input, rnn_type=rnn_type, units=rnn_units, dropout=dropout)
    static_branch = _static_branch(static_input, units=dense_units, dropout=dropout)

    # --- Merge ---
    combined = layers.Concatenate(name="combined")([time_branch, static_branch])

    # --- Risk head ---
    output = _risk_head(combined, dropout=dropout)

    model = keras.Model(inputs=[time_input, static_input], outputs=output, name="stroke_predictor")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_gru_variant(**kwargs) -> keras.Model:
    """Shortcut: GRU instead of LSTM."""
    return build_model(rnn_type="gru", **kwargs)


# ---------------------------------------------------------------------------
# Model summary helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_model()
    model.summary()
