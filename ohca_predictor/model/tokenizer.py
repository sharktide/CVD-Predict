"""Signal tokenization layers for OHCA prediction.

Implements multi-scale 1D convolutional tokenizers with dilated residual blocks
for ECG, motion (accelerometer/gyroscope), PPG, and auxiliary signals (SpO2,
temperature, respiration).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow import keras


# ---------------------------------------------------------------------------
# Helper: Adaptive pooling that works with dynamic shapes
# ---------------------------------------------------------------------------

def _adaptive_avg_pool1d(target_length: int):
    """Return a Lambda layer that resizes sequence to *target_length*.

    Uses bilinear interpolation via ``tf.image.resize`` which handles dynamic
    input lengths and is fully Keras-traceable (avoids the static-shape
    requirement of ``keras.layers.AdaptiveAveragePooling1D``).
    """
    def _pool(x):
        x = tf.expand_dims(x, axis=1)                # (B, 1, T, D)
        x = tf.image.resize(x, [1, target_length], method='bilinear')
        return tf.squeeze(x, axis=1)                  # (B, target_length, D)
    return keras.layers.Lambda(_pool, name=f"adaptive_pool_{target_length}")


# ---------------------------------------------------------------------------
# Helper: Dilated Residual Block
# ---------------------------------------------------------------------------

class _ResidualBlock(keras.layers.Layer):
    """Dilated 1-D residual block with batch-norm, GELU, and dropout.

    Used as the building block for all signal tokenizers.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        dilation_rate: int = 1,
        strides: int = 1,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.strides = strides
        self.dropout_rate = dropout_rate

    def build(self, input_shape: tf.TensorShape):
        in_channels = int(input_shape[-1])

        # Main path – stride applied in first conv, dilation in second.
        # (tf.keras does not allow stride > 1 with dilation > 1 in the
        #  same Conv1D layer.)
        self.conv1 = keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding="causal",
            dilation_rate=1,
            use_bias=False,
            kernel_initializer="he_normal",
        )
        self.bn1 = keras.layers.BatchNormalization()
        self.act1 = keras.layers.Activation("gelu")
        self.drop1 = keras.layers.Dropout(self.dropout_rate)

        self.conv2 = keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding="causal",
            dilation_rate=self.dilation_rate,
            use_bias=False,
            kernel_initializer="he_normal",
        )
        self.bn2 = keras.layers.BatchNormalization()
        self.drop2 = keras.layers.Dropout(self.dropout_rate)

        # Skip connection – project if dimensions change
        if self.strides != 1 or in_channels != self.filters:
            self.skip_proj = keras.layers.Conv1D(
                filters=self.filters,
                kernel_size=1,
                strides=self.strides,
                use_bias=False,
                kernel_initializer="he_normal",
            )
            self.skip_bn = keras.layers.BatchNormalization()
        else:
            self.skip_proj = None
            self.skip_bn = None

        self.add_layer = keras.layers.Add()
        self.act_out = keras.layers.Activation("gelu")

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.drop1(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.drop2(x, training=training)

        if self.skip_proj is not None:
            shortcut = self.skip_proj(inputs)
            shortcut = self.skip_bn(shortcut, training=training)
        else:
            shortcut = inputs

        x = self.add_layer([x, shortcut])
        x = self.act_out(x)
        return x

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "dilation_rate": self.dilation_rate,
                "strides": self.strides,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


# ---------------------------------------------------------------------------
# ECG Tokenizer
# ---------------------------------------------------------------------------

class ECGTokenizer(keras.layers.Layer):
    """Multi-scale 1-D convolutional tokenizer for raw ECG waveforms.

    Designed for single-lead ECG from a Polar H10 chest strap (130 Hz).
    Accepts variable-length inputs (3 minutes to 48+ hours) and produces
    a fixed number of tokens via adaptive pooling.

    Input:  ``(batch, ecg_samples, 1)``  – ecg_samples = duration_s × 130
    Output: ``(batch, target_tokens, model_dim)``

    Architecture:
        1. Initial 1-D conv  (kernel 15, stride 1, 32 filters)
        2. Residual Block 1  (kernel 7, dilation 1, 64 filters, stride 2)
        3. Residual Block 2  (kernel 5, dilation 2, 128 filters, stride 2)
        4. Residual Block 3  (kernel 3, dilation 4, 256 filters, stride 2)
        5. Residual Block 4  (kernel 3, dilation 8, model_dim filters, stride 2)
        6. AdaptiveAveragePooling1D → target_tokens
        7. Learned positional encoding
    """

    def __init__(
        self,
        model_dim: int = 256,
        target_tokens: int = 512,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.target_tokens = target_tokens
        self.dropout_rate = dropout_rate

    def build(self, input_shape: tf.TensorShape):
        # 1. Initial conv – captures fine waveform structure
        self.init_conv = keras.layers.Conv1D(
            filters=32,
            kernel_size=15,
            strides=1,
            padding="causal",
            use_bias=False,
            kernel_initializer="he_normal",
        )
        self.init_bn = keras.layers.BatchNormalization()
        self.init_act = keras.layers.Activation("gelu")

        # 2-5. Residual blocks with progressive downsampling (16× total)
        self.res_block1 = _ResidualBlock(
            filters=64, kernel_size=7, dilation_rate=1, strides=2,
            dropout_rate=self.dropout_rate, name="ecg_res1",
        )
        self.res_block2 = _ResidualBlock(
            filters=128, kernel_size=5, dilation_rate=2, strides=2,
            dropout_rate=self.dropout_rate, name="ecg_res2",
        )
        self.res_block3 = _ResidualBlock(
            filters=256, kernel_size=3, dilation_rate=4, strides=2,
            dropout_rate=self.dropout_rate, name="ecg_res3",
        )
        self.res_block4 = _ResidualBlock(
            filters=self.model_dim, kernel_size=3, dilation_rate=8, strides=2,
            dropout_rate=self.dropout_rate, name="ecg_res4",
        )

        # Project to model_dim if last residual block output != model_dim
        if self.model_dim != self.model_dim:
            self.proj = keras.layers.Conv1D(
                self.model_dim, 1, use_bias=False, kernel_initializer="he_normal",
            )
        else:
            self.proj = None

        # 6. Adaptive pooling to fixed token count
        self.adaptive_pool = _adaptive_avg_pool1d(self.target_tokens)

        # 7. Learned positional encoding (added in call)
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, self.target_tokens, self.model_dim),
            initializer="random_normal",
            trainable=True,
        )

        self.token_dropout = keras.layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # inputs: (batch, ecg_samples, 1)
        x = self.init_conv(inputs)
        x = self.init_bn(x, training=training)
        x = self.init_act(x)

        x = self.res_block1(x, training=training)
        x = self.res_block2(x, training=training)
        x = self.res_block3(x, training=training)
        x = self.res_block4(x, training=training)

        if self.proj is not None:
            x = self.proj(x)

        # Adaptive pooling → (batch, target_tokens, model_dim)
        x = self.adaptive_pool(x)

        # Add learned positional encodings
        seq_len = tf.shape(x)[1]
        pos = self.pos_embedding[:, :seq_len, :]
        x = x + pos

        x = self.token_dropout(x, training=training)
        return x

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "model_dim": self.model_dim,
                "target_tokens": self.target_tokens,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Motion Tokenizer
# ---------------------------------------------------------------------------

class MotionTokenizer(keras.layers.Layer):
    """1-D convolutional tokenizer for 3-axis accelerometer / gyroscope data.

    Designed for tri-axis data from a Polar H10 chest strap (accelerometer)
    or chest-mounted gyroscope (52 Hz). Accepts variable-length inputs and
    produces a fixed number of tokens.

    Input:  ``(batch, motion_samples, 3)``
    Output: ``(batch, target_tokens, model_dim)``
    """

    def __init__(
        self,
        model_dim: int = 256,
        target_tokens: int = 512,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.target_tokens = target_tokens
        self.dropout_rate = dropout_rate

    def build(self, input_shape: tf.TensorShape):
        # 1. Initial conv – processes 3-axis input
        self.init_conv = keras.layers.Conv1D(
            filters=32,
            kernel_size=15,
            strides=1,
            padding="causal",
            use_bias=False,
            kernel_initializer="he_normal",
        )
        self.init_bn = keras.layers.BatchNormalization()
        self.init_act = keras.layers.Activation("gelu")

        # 2-4. Residual blocks (8× total reduction)
        self.res_block1 = _ResidualBlock(
            filters=64, kernel_size=7, dilation_rate=1, strides=2,
            dropout_rate=self.dropout_rate, name="motion_res1",
        )
        self.res_block2 = _ResidualBlock(
            filters=128, kernel_size=5, dilation_rate=2, strides=2,
            dropout_rate=self.dropout_rate, name="motion_res2",
        )
        self.res_block3 = _ResidualBlock(
            filters=self.model_dim, kernel_size=3, dilation_rate=4, strides=2,
            dropout_rate=self.dropout_rate, name="motion_res3",
        )

        # Adaptive pooling to fixed token count
        self.adaptive_pool = _adaptive_avg_pool1d(self.target_tokens)

        # Learned positional encoding
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, self.target_tokens, self.model_dim),
            initializer="random_normal",
            trainable=True,
        )

        self.token_dropout = keras.layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        # inputs: (batch, motion_samples, 3)
        x = self.init_conv(inputs)
        x = self.init_bn(x, training=training)
        x = self.init_act(x)

        if mask is not None:
            # Expand mask to feature dim for downstream layers
            # mask: (batch, motion_samples) → (batch, motion_samples, 1)
            if len(mask.shape) == 2:
                mask_expanded = tf.cast(mask[:, :, tf.newaxis], dtype=x.dtype)
            else:
                mask_expanded = tf.cast(mask, dtype=x.dtype)
            x = x * mask_expanded

        x = self.res_block1(x, training=training)
        x = self.res_block2(x, training=training)
        x = self.res_block3(x, training=training)

        # Adaptive pooling → (batch, target_tokens, model_dim)
        x = self.adaptive_pool(x)

        # Add learned positional encodings
        seq_len = tf.shape(x)[1]
        pos = self.pos_embedding[:, :seq_len, :]
        x = x + pos

        x = self.token_dropout(x, training=training)
        return x

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "model_dim": self.model_dim,
                "target_tokens": self.target_tokens,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


# ---------------------------------------------------------------------------
# PPG Tokenizer
# ---------------------------------------------------------------------------

class PPGTokenizer(keras.layers.Layer):
    """Tokenizer optimised for wrist-worn photoplethysmogram (PPG) waveforms.

    Designed for PPG from a wrist sensor (50 Hz). Shallower network than ECG
    since PPG signals have lower frequency content. Accepts variable-length
    inputs and produces a fixed number of tokens.

    Input:  ``(batch, ppg_samples, 1)``  – ppg_samples = duration_s × 50
    Output: ``(batch, target_tokens, model_dim)``
    """

    def __init__(
        self,
        model_dim: int = 256,
        target_tokens: int = 512,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.target_tokens = target_tokens
        self.dropout_rate = dropout_rate

    def build(self, input_shape: tf.TensorShape):
        # 1. Initial conv – kernel 11 for PPG's smoother morphology
        self.init_conv = keras.layers.Conv1D(
            filters=32,
            kernel_size=11,
            strides=1,
            padding="causal",
            use_bias=False,
            kernel_initializer="he_normal",
        )
        self.init_bn = keras.layers.BatchNormalization()
        self.init_act = keras.layers.Activation("gelu")

        # 2-4. Residual blocks (8× total reduction)
        self.res_block1 = _ResidualBlock(
            filters=64, kernel_size=7, dilation_rate=1, strides=2,
            dropout_rate=self.dropout_rate, name="ppg_res1",
        )
        self.res_block2 = _ResidualBlock(
            filters=128, kernel_size=5, dilation_rate=2, strides=2,
            dropout_rate=self.dropout_rate, name="ppg_res2",
        )
        self.res_block3 = _ResidualBlock(
            filters=self.model_dim, kernel_size=3, dilation_rate=4, strides=2,
            dropout_rate=self.dropout_rate, name="ppg_res3",
        )

        # Adaptive pooling to fixed token count
        self.adaptive_pool = _adaptive_avg_pool1d(self.target_tokens)

        # Learned positional encoding
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, self.target_tokens, self.model_dim),
            initializer="random_normal",
            trainable=True,
        )

        self.token_dropout = keras.layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # inputs: (batch, ppg_samples, 1)
        x = self.init_conv(inputs)
        x = self.init_bn(x, training=training)
        x = self.init_act(x)

        x = self.res_block1(x, training=training)
        x = self.res_block2(x, training=training)
        x = self.res_block3(x, training=training)

        # Adaptive pooling → (batch, target_tokens, model_dim)
        x = self.adaptive_pool(x)

        # Add learned positional encodings
        seq_len = tf.shape(x)[1]
        pos = self.pos_embedding[:, :seq_len, :]
        x = x + pos

        x = self.token_dropout(x, training=training)
        return x

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "model_dim": self.model_dim,
                "target_tokens": self.target_tokens,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Auxiliary Signal Tokenizer (SpO2, Temperature, Respiration, etc.)
# ---------------------------------------------------------------------------

class AuxSignalTokenizer(keras.layers.Layer):
    """Lightweight tokenizer for lower-frequency auxiliary signals.

    Designed for wrist-mounted sensors: SpO2 (10 Hz), temperature (1 Hz),
    and ECG-derived respiration (25 Hz). Handles signals with very different
    sample counts by:

    1. Optionally interpolating to a fixed intermediate length when the
       input is shorter than ``intermediate_len``.
    2. Applying a shallow 1-D convolutional stem.
    3. Adaptive average pooling to ``target_tokens`` (matching other modalities).

    For very small sample counts (< 64), the signal is first projected via
    a shared MLP to ``intermediate_len`` before convolution.

    Input:  ``(batch, aux_samples, 1)``  – aux_samples is variable
    Output: ``(batch, target_tokens, model_dim)``
    """

    def __init__(
        self,
        model_dim: int = 256,
        target_tokens: int = 512,
        intermediate_len: int = 256,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.target_tokens = target_tokens
        self.intermediate_len = intermediate_len
        self.dropout_rate = dropout_rate

    def build(self, input_shape: tf.TensorShape):
        # --- Short-signal branch: project via MLP to intermediate_len ---
        self.short_proj_dense1 = keras.layers.Dense(
            self.model_dim, activation="gelu", kernel_initializer="he_normal",
        )
        self.short_proj_dense2 = keras.layers.Dense(
            self.intermediate_len * self.model_dim,
            kernel_initializer="he_normal",
        )
        self.short_proj_dropout = keras.layers.Dropout(self.dropout_rate)

        # --- Convolutional stem ---
        self.conv1 = keras.layers.Conv1D(
            filters=self.model_dim,
            kernel_size=5,
            strides=1,
            padding="causal",
            use_bias=False,
            kernel_initializer="he_normal",
        )
        self.bn1 = keras.layers.BatchNormalization()
        self.act1 = keras.layers.Activation("gelu")
        self.drop1 = keras.layers.Dropout(self.dropout_rate)

        # A small residual refinement after the stem
        self.refine_conv1 = keras.layers.Conv1D(
            filters=self.model_dim,
            kernel_size=3,
            strides=1,
            padding="causal",
            dilation_rate=1,
            use_bias=False,
            kernel_initializer="he_normal",
        )
        self.refine_bn1 = keras.layers.BatchNormalization()
        self.refine_act1 = keras.layers.Activation("gelu")

        self.refine_conv2 = keras.layers.Conv1D(
            filters=self.model_dim,
            kernel_size=3,
            strides=1,
            padding="causal",
            dilation_rate=2,
            use_bias=False,
            kernel_initializer="he_normal",
        )
        self.refine_bn2 = keras.layers.BatchNormalization()

        self.refine_add = keras.layers.Add()
        self.refine_act2 = keras.layers.Activation("gelu")
        self.refine_drop = keras.layers.Dropout(self.dropout_rate)

        # Adaptive pooling to fixed token count
        self.adaptive_pool = _adaptive_avg_pool1d(self.target_tokens)

        # Learned positional encoding
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, self.target_tokens, self.model_dim),
            initializer="random_normal",
            trainable=True,
        )

        self.token_dropout = keras.layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    # -----------------------------------------------------------------
    def _interpolate_to_fixed(self, x: tf.Tensor) -> tf.Tensor:
        """Resize time axis to ``intermediate_len`` via linear interpolation.

        x: (batch, aux_samples, channels)
        returns: (batch, intermediate_len, channels)
        """
        target_len = self.intermediate_len
        current_len = tf.shape(x)[1]

        # Only interpolate if necessary
        needs_interp = tf.less(current_len, target_len)

        def _do_interp() -> tf.Tensor:
            # tf.image.resize expects (batch, H, W, C)
            # Treat current_len as H, 1 as W, C as channels
            # Input: (batch, current_len, C) → (batch, current_len, 1, C)
            xt = tf.expand_dims(x, axis=2)
            xt = tf.image.resize(
                xt,
                size=(target_len, 1),
                method="bilinear",
            )
            # (batch, target_len, 1, C) → (batch, target_len, C)
            xt = tf.squeeze(xt, axis=2)
            return xt

        def _no_interp() -> tf.Tensor:
            return x

        return tf.cond(needs_interp, _do_interp, _no_interp)

    # -----------------------------------------------------------------
    def _mlp_project(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        """Project very short signals via shared MLP.

        Flattens the time axis, projects to ``intermediate_len * model_dim``,
        and reshapes back to ``(batch, intermediate_len, model_dim)``.
        """
        batch_size = tf.shape(x)[0]

        x_flat = tf.reshape(x, (batch_size, -1))  # (batch, aux_samples * ch)

        # Pad to a fixed input size if needed (max 1024 raw values)
        max_flat = 1024
        pad_len = tf.maximum(max_flat - tf.shape(x_flat)[1], 0)
        x_padded = tf.pad(x_flat, [(0, 0), (0, pad_len)])

        h = self.short_proj_dense1(x_padded, training=training)
        h = self.short_proj_dropout(h, training=training)
        h = self.short_proj_dense2(h, training=training)

        # Reshape to (batch, intermediate_len, model_dim)
        h = tf.reshape(h, (batch_size, self.intermediate_len, self.model_dim))
        return h

    # -----------------------------------------------------------------
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # inputs: (batch, aux_samples, 1)
        sample_count = tf.shape(inputs)[1]

        # Branch: very short signals (< 64 samples) → MLP projection
        is_short = tf.less(sample_count, 64)

        def _short_path() -> tf.Tensor:
            return self._mlp_project(inputs, training=training)

        def _long_path() -> tf.Tensor:
            # Interpolate to intermediate length if shorter
            x = self._interpolate_to_fixed(inputs)

            # Convolutional stem
            x = self.conv1(x)
            x = self.bn1(x, training=training)
            x = self.act1(x)
            x = self.drop1(x, training=training)

            # Residual refinement
            residual = x
            x = self.refine_conv1(x)
            x = self.refine_bn1(x, training=training)
            x = self.refine_act1(x)

            x = self.refine_conv2(x)
            x = self.refine_bn2(x, training=training)

            x = self.refine_add([x, residual])
            x = self.refine_act2(x)
            x = self.refine_drop(x, training=training)

            return x

        x = tf.cond(is_short, _short_path, _long_path)

        # Adaptive pooling → (batch, target_tokens, model_dim)
        x = self.adaptive_pool(x)

        # Add learned positional encodings
        seq_len = tf.shape(x)[1]
        pos = self.pos_embedding[:, :seq_len, :]
        x = x + pos

        x = self.token_dropout(x, training=training)
        return x

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "model_dim": self.model_dim,
                "target_tokens": self.target_tokens,
                "intermediate_len": self.intermediate_len,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
