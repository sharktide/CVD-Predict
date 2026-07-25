"""
Cross-attention mechanisms for the OHCA Predictor.

Implements asymmetric cross-attention, hierarchical cross-attention fusion,
and ALiBi relative positional encoding for multi-modal ECG-motion analysis.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class RelativePositionalEncoding(layers.Layer):
    """ALiBi (Attention with Linear Biases) relative positional encoding.

    Instead of adding explicit positional encodings to tokens, this layer
    biases attention scores linearly based on distance: bias = -m * |i - j|.
    This allows the model to handle variable-length sequences, has no maximum
    position limit, and extrapolates well to longer sequences than seen in
    training.

    The slope m for each head is computed as: m_i = 2^(-8 * i / h)
    where i = 1, 2, ..., h and h is the number of attention heads.
    """

    def __init__(self, num_heads: int = 8, **kwargs):
        """Initialize ALiBi positional encoding.

        Args:
            num_heads: Number of attention heads. Slopes are computed for each head.
            **kwargs: Additional keyword arguments passed to the parent Layer.
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        """Build the layer by precomputing ALiBi slopes.

        Args:
            input_shape: Expected shape (batch, seq_len, model_dim). Used only
                to determine the number of heads at build time if needed.
        """
        super().build(input_shape)

    def _get_slopes(self, num_heads: int) -> list:
        """Compute the geometric sequence of slopes for ALiBi.

        For h heads, the slopes are: 2^(-8/h), 2^(-16/h), ..., 2^(-8*h/h)
        which simplifies to: 2^(-8*i/h) for i = 1, 2, ..., h.

        Args:
            num_heads: Number of attention heads.

        Returns:
            List of slope values for each head.
        """
        def get_slopes_power_of_two(n: int) -> list:
            start = 2 ** (-8.0 / n)
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        if num_heads & (num_heads - 1) == 0:
            return get_slopes_power_of_two(num_heads)
        else:
            closest_power_of_two = 2 ** int(np.floor(np.log2(num_heads)))
            slopes = get_slopes_power_of_two(closest_power_of_two)
            additional_slopes = get_slopes_power_of_two(2 * closest_power_of_two)
            slopes += additional_slopes[0::2][: num_heads - closest_power_of_two]
            return slopes

    def get_attention_bias(self, seq_len: int) -> tf.Tensor:
        """Compute the ALiBi attention bias tensor.

        The bias matrix is (seq_len, seq_len) where entry [i, j] = -m * |i - j|
        for each head's slope m.

        Args:
            seq_len: Length of the sequence to compute biases for.

        Returns:
            Bias tensor of shape (num_heads, seq_len, seq_len) that should be
            added to attention logits before softmax.
        """
        slopes = self._get_slopes(self.num_heads)
        slopes_tensor = tf.constant(slopes, dtype=tf.float32)

        positions = tf.range(seq_len, dtype=tf.float32)
        relative_positions = tf.abs(
            positions[tf.newaxis, :] - positions[:, tf.newaxis]
        )

        biases = []
        for i in range(self.num_heads):
            bias = -slopes_tensor[i] * relative_positions
            biases.append(bias)

        bias_tensor = tf.stack(biases, axis=0)
        return bias_tensor

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return inputs unchanged.

        ALiBi does not modify token representations. It provides a bias tensor
        that is applied externally to attention logits. Use get_attention_bias()
        within your attention computation.

        Args:
            inputs: Token tensor (batch, seq_len, model_dim). Returned unchanged.

        Returns:
            The same inputs tensor, unchanged.
        """
        return inputs

    def get_config(self) -> dict:
        """Return configuration for serialization.

        Returns:
            Dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config


class AsymmetricCrossAttention(layers.Layer):
    """Asymmetric cross-attention between ECG and contextual modalities.

    One modality (ECG) serves as Query while the other modality (motion/context)
    serves as Key and Value. This allows the primary diagnostic signal (ECG) to
    actively query contextual information from other modalities.

    Architecture:
        Q来自ECG tokens (primary diagnostic signal)
        K, V来自modality tokens (contextual information)

    Includes residual connection, layer normalization, and dropout on both
    attention weights and value projections.
    """

    def __init__(
        self,
        model_dim: int = 256,
        num_heads: int = 8,
        attention_dropout: float = 0.1,
        value_dropout: float = 0.1,
        **kwargs,
    ):
        """Initialize asymmetric cross-attention.

        Args:
            model_dim: Dimensionality of the token embeddings.
            num_heads: Number of attention heads.
            attention_dropout: Dropout rate applied to attention weights.
            value_dropout: Dropout rate applied to value projections.
            **kwargs: Additional keyword arguments passed to parent Layer.
        """
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.key_dim = model_dim // num_heads
        self.attention_dropout = attention_dropout
        self.value_dropout = value_dropout

        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.key_dim,
            dropout=attention_dropout,
        )

        self.value_dropout_layer = layers.Dropout(value_dropout)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.residual_dropout = layers.Dropout(value_dropout)

    def call(
        self,
        ecg_tokens: tf.Tensor,
        context_tokens: tf.Tensor,
        mask: tf.Tensor = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Apply asymmetric cross-attention: ECG queries context.

        Args:
            ecg_tokens: ECG token embeddings (batch, seq_len_ecg, model_dim).
                These serve as the Query.
            context_tokens: Context token embeddings (batch, seq_len_context, model_dim).
                These serve as the Key and Value.
            mask: Optional attention mask (batch, seq_len_ecg, seq_len_context).
                True values indicate positions to attend to.
            training: Whether the layer is in training mode.

        Returns:
            Attended token tensor of shape (batch, seq_len_ecg, model_dim).
        """
        query = ecg_tokens
        key = context_tokens
        value = context_tokens

        attn_output = self.multi_head_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=mask,
            training=training,
        )
        attn_output = self.value_dropout_layer(attn_output, training=training)

        output = self.layer_norm(ecg_tokens + attn_output)
        output = self.residual_dropout(output, training=training)

        return output

    def get_config(self) -> dict:
        """Return configuration for serialization.

        Returns:
            Dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "model_dim": self.model_dim,
                "num_heads": self.num_heads,
                "attention_dropout": self.attention_dropout,
                "value_dropout": self.value_dropout,
            }
        )
        return config


class HierarchicalCrossAttention(layers.Layer):
    """Three-level hierarchical cross-attention fusion.

    Fuses ECG and context tokens through three complementary levels:

        Level 1 - Local Cross-Attention:
            Each ECG token attends to its local temporal neighborhood in motion tokens.
            Window size: configurable (default 7 tokens).
            Captures: local artifact detection, signal quality assessment.

        Level 2 - Global Cross-Attention:
            Each ECG token attends to all motion tokens.
            Captures: global activity state, physiological context.

        Level 3 - Pooled Cross-Attention:
            Global average pooled ECG attends to global average pooled motion.
            Single context vector.
            Captures: high-level modality relationship.

        Final combination: Concatenation + linear projection of all three levels.

    Each level uses AsymmetricCrossAttention internally.
    """

    def __init__(
        self,
        model_dim: int = 256,
        num_heads: int = 8,
        local_window_size: int = 7,
        attention_dropout: float = 0.1,
        value_dropout: float = 0.1,
        **kwargs,
    ):
        """Initialize hierarchical cross-attention.

        Args:
            model_dim: Dimensionality of the token embeddings.
            num_heads: Number of attention heads per level.
            local_window_size: Number of context tokens each ECG token attends
                to in the local level. Tokens outside the window are masked.
            attention_dropout: Dropout rate for attention weights.
            value_dropout: Dropout rate for value projections.
            **kwargs: Additional keyword arguments passed to parent Layer.
        """
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.local_window_size = local_window_size
        self.attention_dropout = attention_dropout
        self.value_dropout = value_dropout

        self.local_cross_attention = AsymmetricCrossAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            value_dropout=value_dropout,
        )

        self.global_cross_attention = AsymmetricCrossAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            value_dropout=value_dropout,
        )

        self.pooled_query_dense = layers.Dense(model_dim, activation="gelu")
        self.pooled_key_dense = layers.Dense(model_dim, activation="gelu")
        self.pooled_attention = AsymmetricCrossAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            value_dropout=value_dropout,
        )

        self.fusion_projection = layers.Dense(model_dim, activation="gelu")
        self.fusion_layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.fusion_dropout = layers.Dropout(value_dropout)

    def _build_local_mask(
        self,
        seq_len_ecg: int,
        seq_len_context: int,
        window_size: int,
        ecg_tokens: tf.Tensor,
    ) -> tf.Tensor:
        """Build a local window attention mask.

        For each ECG position i, only context positions within [i - w//2, i + w//2]
        are attended to, where w = window_size.

        Args:
            seq_len_ecg: Number of ECG tokens.
            seq_len_context: Number of context tokens.
            window_size: Local window size.
            ecg_tokens: ECG token tensor, used for dtype and device placement.

        Returns:
            Boolean mask of shape (1, seq_len_ecg, seq_len_context) where True
            indicates positions to attend to.
        """
        positions = tf.range(seq_len_ecg)
        context_positions = tf.range(seq_len_context)

        half_window = window_size // 2

        positions_expanded = positions[:, tf.newaxis]
        context_expanded = context_positions[tf.newaxis, :]

        distance = tf.abs(positions_expanded - context_expanded)
        mask = distance <= half_window

        mask = tf.cast(mask, dtype=tf.bool)
        mask = tf.expand_dims(mask, axis=0)
        return mask

    def call(
        self,
        ecg_tokens: tf.Tensor,
        context_tokens: tf.Tensor,
        mask: tf.Tensor = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Apply three-level hierarchical cross-attention fusion.

        Args:
            ecg_tokens: ECG token embeddings (batch, seq_len_ecg, model_dim).
            context_tokens: Context token embeddings (batch, seq_len_context, model_dim).
            mask: Optional external attention mask (batch, seq_len_ecg, seq_len_context).
                If provided, it is applied to the global level.
            training: Whether the layer is in training mode.

        Returns:
            Fused token tensor of shape (batch, seq_len_ecg, model_dim).
        """
        seq_len_ecg = tf.shape(ecg_tokens)[1]
        seq_len_context = tf.shape(context_tokens)[1]

        local_mask = self._build_local_mask(
            seq_len_ecg, seq_len_context, self.local_window_size, ecg_tokens
        )

        local_attended = self.local_cross_attention(
            ecg_tokens, context_tokens, mask=local_mask, training=training
        )

        global_attended = self.global_cross_attention(
            ecg_tokens, context_tokens, mask=mask, training=training
        )

        pooled_ecg = tf.reduce_mean(ecg_tokens, axis=1, keepdims=True)
        pooled_context = tf.reduce_mean(context_tokens, axis=1, keepdims=True)

        pooled_query = self.pooled_query_dense(pooled_ecg)
        pooled_kv = self.pooled_key_dense(pooled_context)

        pooled_attended = self.pooled_attention(
            pooled_query, pooled_kv, training=training
        )

        pooled_broadcast = tf.broadcast_to(
            pooled_attended,
            tf.shape(global_attended),
        )

        concatenated = tf.concat(
            [local_attended, global_attended, pooled_broadcast], axis=-1
        )

        fused = self.fusion_projection(concatenated)
        fused = self.fusion_layer_norm(fused)
        fused = self.fusion_dropout(fused, training=training)

        return fused

    def get_config(self) -> dict:
        """Return configuration for serialization.

        Returns:
            Dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update(
            {
                "model_dim": self.model_dim,
                "num_heads": self.num_heads,
                "local_window_size": self.local_window_size,
                "attention_dropout": self.attention_dropout,
                "value_dropout": self.value_dropout,
            }
        )
        return config
