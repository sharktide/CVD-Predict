"""Custom losses – focal loss with sample weights, gradient reversal layer, combined training loss."""

from __future__ import annotations

import tensorflow as tf
import keras
from keras import backend as K


# ---------------------------------------------------------------------------
# Focal Loss (with per-sample weight support)
# ---------------------------------------------------------------------------

def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Return a Keras-compatible focal loss function for binary classification.

    Supports per-sample weighting via the ``sample_weight`` argument that
    Keras passes to loss functions automatically when using
    ``model.fit(..., sample_weight=...)``.

    Parameters
    ----------
    gamma : focusing parameter (higher = more focus on hard examples)
    alpha : balancing factor for the positive class
    """

    def _focal(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # Binary cross-entropy per element
        bce = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)

        # Focal modulating factor
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        modulating = tf.pow(1.0 - p_t, gamma)

        # Alpha weighting
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)

        # Per-sample loss (no reduction yet)
        per_sample = alpha_t * modulating * bce
        per_sample = tf.reduce_sum(per_sample, axis=-1)  # sum over classes (1 for binary)

        return per_sample  # Keras applies sample_weight and reduces via mean

    return _focal


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

@tf.custom_gradient
def _grad_reverse(x: tf.Tensor):
    def _grad(dy: tf.Tensor) -> tf.Tensor:
        return -dy
    return x, _grad


@keras.saving.register_keras_serializable(package="src.losses")
class GradientReversalLayer(tf.keras.layers.Layer):
    """Pass-through layer that reverses gradients during backpropagation."""

    def __init__(self, lambda_: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = lambda_

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.lambda_ * _grad_reverse(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"lambda_": self.lambda_})
        return cfg


# ---------------------------------------------------------------------------
# Combined training loss (used by train.py)
# ---------------------------------------------------------------------------

def build_combined_loss(train_config: dict):
    """Return a dict mapping output names -> loss functions and loss weights."""
    event_loss_fn = focal_loss(
        gamma=train_config.get("focal_gamma", 2.0),
        alpha=train_config.get("focal_alpha", 0.25),
    )
    losses = {
        "event_output": event_loss_fn,
        "acuity_output": "sparse_categorical_crossentropy",
        "icu_domain_output": "sparse_categorical_crossentropy",
        "device_domain_output": "sparse_categorical_crossentropy",
        "sensor_quality_output": "sparse_categorical_crossentropy",
    }
    weights = train_config.get("loss_weights", {
        "event_output": 1.0,
        "acuity_output": 0.3,
        "icu_domain_output": 0.2,
        "device_domain_output": 0.2,
        "sensor_quality_output": 0.1,
    })
    return losses, weights
