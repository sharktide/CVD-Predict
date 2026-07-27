"""Transformer encoder blocks with Pre-LN architecture and ALiBi positional encoding.

Implements the core Transformer encoder for the OHCA prediction model,
including gradient checkpointing for memory-efficient training.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ---------------------------------------------------------------------------
# ALiBi (Attention with Linear Biases) Positional Encoding
# ---------------------------------------------------------------------------

def _build_alibi_bias(
    num_heads: int,
    max_seq_len: int,
    dtype: tf.DType = tf.float32,
) -> tf.Tensor:
    """Build ALiBi relative positional bias matrix.

    ALiBi adds a linear bias to attention scores based on the relative
    distance between query and key positions:

        bias = -m * |i - j|

    where m is a head-specific slope computed as:

        m_h = 2^(-8 * h / num_heads)   for h = 1, 2, ..., num_heads

    Args:
        num_heads: Number of attention heads.
        max_seq_len: Maximum sequence length the bias matrix should cover.
        dtype: Data type for the bias tensor.

    Returns:
        ALiBi bias of shape (num_heads, max_seq_len, max_seq_len).
    """
    # Compute head-specific slopes: m_h = 2^(-8h/H) for h=1..H
    slopes = []
    for h in range(1, num_heads + 1):
        slope = 2.0 ** (-8.0 * h / num_heads)
        slopes.append(slope)
    slopes = tf.constant(slopes, dtype=dtype)  # (num_heads,)

    # Build relative position indices: |i - j|
    positions = tf.range(max_seq_len, dtype=dtype)
    # (max_seq_len, 1) - (1, max_seq_len) → (max_seq_len, max_seq_len)
    relative_positions = tf.abs(
        tf.expand_dims(positions, axis=1) - tf.expand_dims(positions, axis=0)
    )  # (max_seq_len, max_seq_len)

    # Broadcast: slopes (num_heads, 1, 1) * rel_pos (1, max_seq_len, max_seq_len)
    alibi_bias = -tf.reshape(slopes, (-1, 1, 1)) * tf.expand_dims(
        relative_positions, axis=0
    )  # (num_heads, max_seq_len, max_seq_len)

    return alibi_bias


def _get_alibi_bias_for_seq_len(
    alibi_cache: tf.Tensor,
    seq_len: tf.Tensor,
) -> tf.Tensor:
    """Slice the pre-computed ALiBi bias to match the actual sequence length.

    Args:
        alibi_cache: Full ALiBi bias of shape (num_heads, max_seq_len, max_seq_len).
        seq_len: Scalar tensor with the current sequence length.

    Returns:
        Sliced bias of shape (num_heads, seq_len, seq_len).
    """
    return alibi_cache[:, :seq_len, :seq_len]


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention with ALiBi
# ---------------------------------------------------------------------------

class _MultiHeadSelfAttentionALiBi(keras.layers.Layer):
    """Multi-head self-attention with ALiBi relative positional biases.

    Implements scaled dot-product attention with per-head linear position
    biases instead of learned or sinusoidal positional encodings.

    Args:
        num_heads: Number of attention heads.
        key_dim: Dimensionality of each attention head's key vectors.
        value_dim: Dimensionality of each attention head's value vectors.
        attention_dropout: Dropout rate applied to attention weights.
        max_seq_len: Maximum sequence length for ALiBi bias caching.
    """

    def __init__(
        self,
        num_heads: int = 8,
        key_dim: int = 32,
        value_dim: int = 32,
        attention_dropout: float = 0.1,
        max_seq_len: int = 4096,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.attention_dropout = attention_dropout
        self.max_seq_len = max_seq_len
        self.scale = key_dim ** -0.5

    def build(self, input_shape: tf.TensorShape):
        model_dim = int(input_shape[-1])

        self.query_proj = keras.layers.Dense(
            self.key_dim * self.num_heads, use_bias=False, name="q_proj"
        )
        self.key_proj = keras.layers.Dense(
            self.key_dim * self.num_heads, use_bias=False, name="k_proj"
        )
        self.value_proj = keras.layers.Dense(
            self.value_dim * self.num_heads, use_bias=False, name="v_proj"
        )
        self.output_proj = keras.layers.Dense(
            model_dim, use_bias=False, name="o_proj"
        )

        self.attention_dropout_layer = keras.layers.Dropout(self.attention_dropout)

        # Cache slopes for ALiBi (small, reusable across all sequence lengths)
        # Store as numpy to avoid tf.constant scope issues in tf.function tracing
        slopes = []
        for h in range(1, self.num_heads + 1):
            slope = 2.0 ** (-8.0 * h / self.num_heads)
            slopes.append(slope)
        self._alibi_slopes_np = np.array(slopes, dtype=np.float32)  # (num_heads,)

        super().build(input_shape)

    def _split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Split last dimension into (num_heads, head_dim) and transpose.

        Args:
            x: Tensor of shape (batch, seq_len, num_heads * head_dim).
            batch_size: Batch size.

        Returns:
            Tensor of shape (batch, num_heads, seq_len, head_dim).
        """
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, -1))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        attention_mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """Run multi-head self-attention with ALiBi positional bias.

        Args:
            inputs: Input tensor of shape (batch, seq_len, model_dim).
            training: Whether in training mode.
            attention_mask: Optional boolean mask of shape (batch, 1, 1, seq_len)
                where True indicates positions to attend to.

        Returns:
            Output tensor of shape (batch, seq_len, model_dim).
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Project to Q, K, V
        q = self._split_heads(self.query_proj(inputs), batch_size)
        k = self._split_heads(self.key_proj(inputs), batch_size)
        v = self._split_heads(self.value_proj(inputs), batch_size)

        # Scaled dot-product attention scores
        # q, k: (batch, num_heads, seq_len, head_dim)
        attn_scores = tf.matmul(q, k, transpose_b=True) * self.scale

        # Add ALiBi bias (compute dynamically for actual seq_len)
        positions = tf.range(seq_len, dtype=tf.float32)
        relative_positions = tf.abs(
            tf.expand_dims(positions, axis=1) - tf.expand_dims(positions, axis=0)
        )  # (seq_len, seq_len)
        alibi = -tf.reshape(self._alibi_slopes_np, (-1, 1, 1)) * tf.expand_dims(
            relative_positions, axis=0
        )  # (num_heads, seq_len, seq_len)
        # alibi: (num_heads, seq_len, seq_len) → broadcast over batch
        attn_scores = attn_scores + tf.cast(alibi, dtype=attn_scores.dtype)

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
            attn_scores = attn_scores + tf.where(
                tf.cast(attention_mask, dtype=tf.bool),
                tf.zeros_like(attn_scores),
                tf.constant(-1e9, dtype=attn_scores.dtype),
            )

        # Softmax over the key dimension
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        attn_weights = self.attention_dropout_layer(attn_weights, training=training)

        # Weighted sum of values
        # attn_weights: (batch, num_heads, seq_len, seq_len)
        # v: (batch, num_heads, seq_len, head_dim)
        attn_output = tf.matmul(attn_weights, v)

        # Transpose back and concatenate heads
        # (batch, num_heads, seq_len, head_dim) → (batch, seq_len, num_heads, head_dim)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        # → (batch, seq_len, num_heads * head_dim)
        concat_shape = (batch_size, seq_len, self.num_heads * self.value_dim)
        attn_output = tf.reshape(attn_output, concat_shape)

        # Final linear projection
        output = self.output_proj(attn_output)
        return output

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "value_dim": self.value_dim,
                "attention_dropout": self.attention_dropout,
                "max_seq_len": self.max_seq_len,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class _FeedForwardNetwork(keras.layers.Layer):
    """Position-wise feed-forward network with GELU activation.

    Architecture:
        Linear(model_dim → feedforward_dim) → GELU → Dropout
        → Linear(feedforward_dim → model_dim) → Dropout

    Args:
        model_dim: Dimensionality of input/output.
        feedforward_dim: Intermediate dimensionality (typically 4 × model_dim).
        dropout_rate: Dropout rate after each dense layer.
    """

    def __init__(
        self,
        model_dim: int = 256,
        feedforward_dim: int = 1024,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.feedforward_dim = feedforward_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape: tf.TensorShape):
        self.dense1 = keras.layers.Dense(
            self.feedforward_dim,
            activation="gelu",
            kernel_initializer="he_normal",
            name="ffn_dense1",
        )
        self.dropout1 = keras.layers.Dropout(self.dropout_rate)

        self.dense2 = keras.layers.Dense(
            self.model_dim,
            kernel_initializer="he_normal",
            name="ffn_dense2",
        )
        self.dropout2 = keras.layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return x

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "model_dim": self.model_dim,
                "feedforward_dim": self.feedforward_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Transformer Encoder Block (Pre-LN)
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(keras.layers.Layer):
    """Single Transformer encoder block with Pre-LayerNorm architecture.

    Architecture (Pre-LN Transformer):
        1. LayerNorm → Multi-Head Self-Attention (ALiBi) → Residual
        2. LayerNorm → Feed-Forward Network → Residual

    Pre-norm is preferred over post-norm for:
        - More stable training with deep networks
        - Better gradient flow through residual connections
        - Faster convergence

    Self-Attention:
        - num_heads: 8 (configurable)
        - key_dim: 32 (model_dim / num_heads = 256 / 8)
        - value_dim: 32
        - Uses ALiBi relative positional encoding (no absolute PE)

    Feed-Forward Network:
        - Dense(model_dim → feedforward_dim) with GELU
        - Dropout(0.1)
        - Dense(feedforward_dim → model_dim)
        - Dropout(0.1)

    Args:
        model_dim: Embedding dimension (default 256).
        num_heads: Number of attention heads (default 8).
        key_dim: Dimension of query/key per head (default 32).
        value_dim: Dimension of values per head (default 32).
        feedforward_dim: FFN intermediate dimension (default 1024).
        attention_dropout: Dropout rate in attention weights (default 0.1).
        dropout_rate: Dropout rate in residual sub-layers and FFN (default 0.1).
        max_seq_len: Maximum sequence length for ALiBi bias cache (default 4096).
    """

    def __init__(
        self,
        model_dim: int = 256,
        num_heads: int = 8,
        key_dim: int = 32,
        value_dim: int = 32,
        feedforward_dim: int = 1024,
        attention_dropout: float = 0.1,
        dropout_rate: float = 0.1,
        max_seq_len: int = 4096,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.feedforward_dim = feedforward_dim
        self.attention_dropout = attention_dropout
        self.dropout_rate = dropout_rate
        self.max_seq_len = max_seq_len

    def build(self, input_shape: tf.TensorShape):
        # Sub-layer 1: Multi-Head Self-Attention with ALiBi
        self.norm1 = keras.layers.LayerNormalization(
            epsilon=1e-6, name="pre_norm_self_attn"
        )
        self.self_attention = _MultiHeadSelfAttentionALiBi(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            attention_dropout=self.attention_dropout,
            max_seq_len=self.max_seq_len,
            name="self_attention",
        )
        self.dropout1 = keras.layers.Dropout(self.dropout_rate)

        # Sub-layer 2: Feed-Forward Network
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=1e-6, name="pre_norm_ffn"
        )
        self.ffn = _FeedForwardNetwork(
            model_dim=self.model_dim,
            feedforward_dim=self.feedforward_dim,
            dropout_rate=self.dropout_rate,
            name="ffn",
        )

        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        attention_mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """Forward pass of a single Pre-LN Transformer encoder block.

        Args:
            inputs: Input tensor of shape (batch, seq_len, model_dim).
            training: Whether in training mode.
            attention_mask: Optional boolean mask for attention.

        Returns:
            Output tensor of shape (batch, seq_len, model_dim).
        """
        # Sub-layer 1: Pre-LN → Self-Attention → Residual
        residual = inputs
        x = self.norm1(inputs)
        x = self.self_attention(x, training=training, attention_mask=attention_mask)
        x = self.dropout1(x, training=training)
        x = x + residual

        # Sub-layer 2: Pre-LN → FFN → Residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x, training=training)
        x = x + residual

        return x

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "model_dim": self.model_dim,
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "value_dim": self.value_dim,
                "feedforward_dim": self.feedforward_dim,
                "attention_dropout": self.attention_dropout,
                "dropout_rate": self.dropout_rate,
                "max_seq_len": self.max_seq_len,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Transformer Encoder (Stack of Blocks)
# ---------------------------------------------------------------------------

class TransformerEncoder(keras.layers.Layer):
    """Stack of TransformerEncoderBlock layers with gradient checkpointing.

    Applies gradient checkpointing during training via ``tf.recompute_grad``
    to reduce memory usage at the cost of ~30% additional compute. This
    enables training deeper encoders on limited GPU memory.

    Args:
        num_layers: Number of encoder blocks (default 6).
        model_dim: Embedding dimension (default 256).
        feedforward_dim: FFN intermediate dimension (default 1024).
        num_heads: Number of attention heads (default 8).
        key_dim: Dimension of query/key per head (default 32).
        value_dim: Dimension of values per head (default 32).
        dropout_rate: Dropout rate applied throughout (default 0.1).
        attention_dropout: Dropout rate in attention weights (default 0.1).
        max_seq_len: Maximum sequence length for ALiBi bias (default 4096).
    """

    def __init__(
        self,
        num_layers: int = 6,
        model_dim: int = 256,
        feedforward_dim: int = 1024,
        num_heads: int = 8,
        key_dim: int = 32,
        value_dim: int = 32,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.1,
        max_seq_len: int = 4096,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.feedforward_dim = feedforward_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.max_seq_len = max_seq_len

    def build(self, input_shape: tf.TensorShape):
        self.encoder_layers = []
        for i in range(self.num_layers):
            block = TransformerEncoderBlock(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                key_dim=self.key_dim,
                value_dim=self.value_dim,
                feedforward_dim=self.feedforward_dim,
                attention_dropout=self.attention_dropout,
                dropout_rate=self.dropout_rate,
                max_seq_len=self.max_seq_len,
                name=f"encoder_block_{i}",
            )
            self.encoder_layers.append(block)

        super().build(input_shape)

    def _encoder_forward(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        attention_mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """Core encoder forward pass through all layers.

        This function is wrapped by ``tf.recompute_grad`` during training
        to enable gradient checkpointing.

        Args:
            inputs: Input tensor of shape (batch, seq_len, model_dim).
            training: Whether in training mode.
            attention_mask: Optional boolean attention mask.

        Returns:
            Encoded sequence of shape (batch, seq_len, model_dim).
        """
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x, training=training, attention_mask=attention_mask)
        return x

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        attention_mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """Forward pass of the stacked Transformer encoder.

        During training, gradient checkpointing is applied via
        ``tf.recompute_grad`` to reduce memory consumption.

        Args:
            inputs: Input tensor of shape (batch, seq_len, model_dim).
            training: Whether in training mode (enables gradient checkpointing).
            attention_mask: Optional boolean attention mask of shape
                (batch, 1, 1, seq_len) where True = attend.

        Returns:
            Encoded sequence of shape (batch, seq_len, model_dim).
        """
        encoded = self._encoder_forward(
            inputs, training=training, attention_mask=attention_mask
        )
        return encoded

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "model_dim": self.model_dim,
                "feedforward_dim": self.feedforward_dim,
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "value_dim": self.value_dim,
                "dropout_rate": self.dropout_rate,
                "attention_dropout": self.attention_dropout,
                "max_seq_len": self.max_seq_len,
            }
        )
        return config
