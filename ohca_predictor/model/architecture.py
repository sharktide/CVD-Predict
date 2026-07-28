"""Complete multi-modal OHCA prediction model architecture.

Assembles all signal tokenizers, cross-attention mechanisms, transformer
encoders, and prediction heads into the unified ``OHCAPredictionModel``
and ``SelfSupervisedPretrainer`` classes.

Architecture overview:
    Raw signals → Tokenizers → Cross-Modal Attention → Sequence Alignment →
    Token Assembly → Transformer Encoder → Prediction Heads → Calibrated Output
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow import keras

from ohca_predictor.config import ModelConfig
from ohca_predictor.model.attention import (
    AsymmetricCrossAttention,
    HierarchicalCrossAttention,
)
from ohca_predictor.model.encoder import TransformerEncoder
from ohca_predictor.model.heads import (
    AuxiliaryHeads,
    CalibrationHead,
    ClassificationHead,
    SurvivalAnalysisHead,
    UncertaintyHead,
)
from ohca_predictor.model.tokenizer import (
    AuxSignalTokenizer,
    ECGTokenizer,
    MotionTokenizer,
    PPGTokenizer,
)


# ---------------------------------------------------------------------------
# Static Patient Embedding
# ---------------------------------------------------------------------------

class StaticPatientEmbedding(keras.layers.Layer):
    """Embeds static patient features into the shared model dimension.

    Processes four categories of static features through dedicated MLPs and
    combines them via element-wise addition:

        1. Demographics:  age, sex, BMI, ethnicity one-hots, …
        2. Medications:   binary vector of prescribed medications
        3. Comorbidities: binary vector of diagnosed conditions
        4. Lab Values:    continuous lab measurements (troponin, BNP, …)

    Each MLP: input_dim → 32 (GELU) → static_embedding_dim (linear)

    The combined embedding is broadcast across a singleton time dimension
    so it can be concatenated with token sequences:

        output shape: ``(batch, 1, static_embedding_dim)``

    Args:
        demographic_dim: Dimensionality of the demographics feature vector.
        num_medications: Number of binary medication features.
        num_comorbidities: Number of binary comorbidity features.
        num_labs: Dimensionality of the lab-value feature vector.
        static_embedding_dim: Output embedding dimensionality (must match
            the model's ``model_dim`` downstream).
        dropout_rate: Dropout rate applied after each MLP hidden layer.
    """

    def __init__(
        self,
        demographic_dim: int = 16,
        num_medications: int = 50,
        num_comorbidities: int = 30,
        num_labs: int = 20,
        static_embedding_dim: int = 256,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.demographic_dim = demographic_dim
        self.num_medications = num_medications
        self.num_comorbidities = num_comorbidities
        self.num_labs = num_labs
        self.static_embedding_dim = static_embedding_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape: tf.TensorShape):
        # --- Demographics MLP ---
        self.demo_dense1 = keras.layers.Dense(
            32,
            activation="gelu",
            kernel_initializer="he_normal",
            name="demo_dense1",
        )
        self.demo_dropout = keras.layers.Dropout(self.dropout_rate)
        self.demo_dense2 = keras.layers.Dense(
            self.static_embedding_dim,
            kernel_initializer="he_normal",
            name="demo_dense2",
        )

        # --- Medications MLP ---
        self.med_dense1 = keras.layers.Dense(
            32,
            activation="gelu",
            kernel_initializer="he_normal",
            name="med_dense1",
        )
        self.med_dropout = keras.layers.Dropout(self.dropout_rate)
        self.med_dense2 = keras.layers.Dense(
            self.static_embedding_dim,
            kernel_initializer="he_normal",
            name="med_dense2",
        )

        # --- Comorbidities MLP ---
        self.comorb_dense1 = keras.layers.Dense(
            32,
            activation="gelu",
            kernel_initializer="he_normal",
            name="comorb_dense1",
        )
        self.comorb_dropout = keras.layers.Dropout(self.dropout_rate)
        self.comorb_dense2 = keras.layers.Dense(
            self.static_embedding_dim,
            kernel_initializer="he_normal",
            name="comorb_dense2",
        )

        # --- Lab Values MLP ---
        self.lab_dense1 = keras.layers.Dense(
            32,
            activation="gelu",
            kernel_initializer="he_normal",
            name="lab_dense1",
        )
        self.lab_dropout = keras.layers.Dropout(self.dropout_rate)
        self.lab_dense2 = keras.layers.Dense(
            self.static_embedding_dim,
            kernel_initializer="he_normal",
            name="lab_dense2",
        )

        # --- Final combination ---
        self.combination_norm = keras.layers.LayerNormalization(epsilon=1e-6)

        super().build(input_shape)

    def call(
        self,
        demographics: tf.Tensor,
        medications: tf.Tensor,
        comorbidities: tf.Tensor,
        lab_values: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Compute the static patient embedding.

        Args:
            demographics: Float tensor ``(batch, demographic_dim)``.
            medications: Binary/float tensor ``(batch, num_medications)``.
            comorbidities: Binary/float tensor ``(batch, num_comorbidities)``.
            lab_values: Float tensor ``(batch, num_labs)``.
            training: Whether in training mode (controls dropout).

        Returns:
            Combined embedding ``(batch, 1, static_embedding_dim)``.
        """
        # Demographics branch
        demo = self.demo_dense1(demographics)
        demo = self.demo_dropout(demo, training=training)
        demo = self.demo_dense2(demo)  # (batch, D)

        # Medications branch
        med = self.med_dense1(medications)
        med = self.med_dropout(med, training=training)
        med = self.med_dense2(med)  # (batch, D)

        # Comorbidities branch
        comorb = self.comorb_dense1(comorbidities)
        comorb = self.comorb_dropout(comorb, training=training)
        comorb = self.comorb_dense2(comorb)  # (batch, D)

        # Lab values branch
        lab = self.lab_dense1(lab_values)
        lab = self.lab_dropout(lab, training=training)
        lab = self.lab_dense2(lab)  # (batch, D)

        # Combine via addition + layer norm
        combined = demo + med + comorb + lab  # (batch, D)
        combined = self.combination_norm(combined)

        # Add sequence dimension: (batch, D) → (batch, 1, D)
        combined = tf.expand_dims(combined, axis=1)
        return combined

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "demographic_dim": self.demographic_dim,
                "num_medications": self.num_medications,
                "num_comorbidities": self.num_comorbidities,
                "num_labs": self.num_labs,
                "static_embedding_dim": self.static_embedding_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Adaptive Average Pooling 1-D (standalone helper)
# ---------------------------------------------------------------------------

class _AdaptiveAveragePooling1D(keras.layers.Layer):
    """Adaptive average pooling that resizes any sequence length to a target.

    When the input is longer than the target, we use ``tf.reduce_mean`` over
    windows.  When it is shorter, we repeat the last element.

    This is a lightweight replacement that avoids version-gating issues with
    ``keras.layers.AdaptiveAveragePooling1D``.

    Args:
        target_length: Desired output sequence length.
    """

    def __init__(self, target_length: int = 390, **kwargs):
        super().__init__(**kwargs)
        self.target_length = target_length

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Reshape to (batch, T, D) -> use linear interpolation via tf image resize
        # which handles dynamic shapes and is fully Keras-traceable
        target_len = self.target_length
        shape = tf.shape(inputs)
        batch_size = shape[0]
        depth = inputs.shape[-1] or shape[-1]

        # tf.image.resize expects (N, H, W, C) or (N, H, C) —
        # treat the sequence as height=1, width=T, channels=D
        x = tf.expand_dims(inputs, axis=1)          # (batch, 1, T, D)
        x = tf.image.resize(x, size=[1, target_len], method='bilinear')
        x = tf.squeeze(x, axis=1)                    # (batch, target_len, D)
        return x

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"target_length": self.target_length})
        return config


# ---------------------------------------------------------------------------
# ECG Reconstruction Head (used by SelfSupervisedPretrainer)
# ---------------------------------------------------------------------------

class _ECGReconstructionHead(keras.layers.Layer):
    """Transposes encoded ECG tokens back to the raw waveform space.

    Uses a lightweight 1-D transposed convolution stack to upsample from
    token-resolution back to approximately the original sample count.

    Args:
        ecg_target_length: Number of raw ECG samples to reconstruct.
        model_dim: Token embedding dimension.
    """

    def __init__(
        self,
        ecg_target_length: int = 39000,
        model_dim: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ecg_target_length = ecg_target_length
        self.model_dim = model_dim

    def build(self, input_shape: tf.TensorShape):
        # Progressive upsampling via dense + reshape
        self.dense1 = keras.layers.Dense(
            128, activation="gelu", kernel_initializer="he_normal"
        )
        self.dense2 = keras.layers.Dense(
            256, activation="gelu", kernel_initializer="he_normal"
        )
        self.output_dense = keras.layers.Dense(
            1, kernel_initializer="glorot_uniform"
        )
        super().build(input_shape)

    def call(
        self, encoded_sequence: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        """Reconstruct ECG waveform from encoded tokens.

        Args:
            encoded_sequence: ``(batch, T_tokens, model_dim)``.
            training: Training mode flag.

        Returns:
            Reconstructed ECG ``(batch, ecg_target_length, 1)``.
        """
        batch_size = tf.shape(encoded_sequence)[0]

        # Pool over token dimension to get a single vector
        pooled = tf.reduce_mean(encoded_sequence, axis=1)  # (batch, model_dim)

        # Project to a higher-dimensional space
        x = self.dense1(pooled)
        x = self.dense2(x)

        # Reshape to a 1-D feature map
        # We aim for (batch, target_length, 1)
        # Interpolate from a reasonable intermediate size
        intermediate_len = 4096
        x = tf.reshape(x, (batch_size, 1, self.model_dim))
        x = tf.tile(x, [1, intermediate_len, 1])  # (batch, 4096, model_dim)
        x = self.output_dense(x)  # (batch, 4096, 1)

        # Resize to target length
        x = tf.image.resize(
            tf.expand_dims(x, axis=2),
            size=(self.ecg_target_length, 1),
            method="bilinear",
        )
        x = tf.squeeze(x, axis=2)  # (batch, target_length, 1)
        return x

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "ecg_target_length": self.ecg_target_length,
                "model_dim": self.model_dim,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Temporal Order Prediction Head
# ---------------------------------------------------------------------------

class _TemporalOrderHead(keras.layers.Layer):
    """Predicts whether two sampled token segments are in correct temporal order.

    Given two segments from the same ECG, the head outputs a binary logit
    indicating whether segment A precedes segment B.

    Args:
        model_dim: Token embedding dimension.
    """

    def __init__(self, model_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.model_dim = model_dim

    def build(self, input_shape: tf.TensorShape):
        self.concat_dense1 = keras.layers.Dense(
            128, activation="gelu", kernel_initializer="he_normal"
        )
        self.concat_dropout = keras.layers.Dropout(0.1)
        self.concat_dense2 = keras.layers.Dense(
            64, activation="gelu", kernel_initializer="he_normal"
        )
        self.order_output = keras.layers.Dense(
            1, kernel_initializer="glorot_uniform"
        )
        super().build(input_shape)

    def call(
        self,
        segment_a: tf.Tensor,
        segment_b: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Predict temporal order.

        Args:
            segment_a: ``(batch, T_seg, model_dim)`` – first segment.
            segment_b: ``(batch, T_seg, model_dim)`` – second segment.
            training: Training mode flag.

        Returns:
            Logits ``(batch, 1)``.  Positive = correct order.
        """
        # Pool each segment
        pool_a = tf.reduce_mean(segment_a, axis=1)  # (batch, D)
        pool_b = tf.reduce_mean(segment_b, axis=1)  # (batch, D)

        # Concatenate and classify
        combined = tf.concat([pool_a, pool_b], axis=-1)  # (batch, 2*D)
        x = self.concat_dense1(combined)
        x = self.concat_dropout(x, training=training)
        x = self.concat_dense2(x)
        return self.order_output(x)  # (batch, 1)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"model_dim": self.model_dim})
        return config


# ===========================================================================
# Main Model
# ===========================================================================

class OHCAPredictionModel(keras.Model):
    """End-to-end multi-modal OHCA prediction model.

    Assembles all components into a single callable model. All signal inputs
    are variable-length and must share the same wall-clock duration (e.g.,
    3 min, 7 h, 24 h, or 48 h of continuous data).

    Device mapping:
        - Polar H10 chest strap: ECG (single-lead, 130 Hz) + Accelerometer (tri-axis, 52 Hz)
        - Chest sensor: Gyroscope (tri-axis, 52 Hz)
        - Wrist sensor: PPG (50 Hz) + SpO2 (10 Hz) + Temperature (1 Hz)
        - Derived: Respiration from ECG (25 Hz)

    Architecture:
        1. **Input tokenization** – Each modality is tokenized into
           ``tokens_per_modality`` tokens of dimension ``model_dim``.
        2. **Motion fusion** – Accelerometer and gyroscope tokens are
           concatenated along the time axis.
        3. **Cross-modal attention** – ECG attends to motion, then the
           result attends to PPG via two sequential
           ``AsymmetricCrossAttention`` layers.
        4. **Sequence alignment** – all token sequences are pooled to
           ``tokens_per_modality`` via adaptive pooling.
        5. **Token assembly** – static, attended ECG, SpO2, temperature,
           and respiration tokens are concatenated.
        6. **Transformer encoder** – N-layer Pre-LN transformer with ALiBi.
        7. **Prediction heads** – classification, survival analysis,
           uncertainty, and auxiliary heads.
        8. **Calibration** – Platt scaling on the classification output.

    Args:
        config: A ``ModelConfig`` dataclass with all hyper-parameters.
        demographic_dim: Dimensionality of demographics input vector.
        num_medications: Number of binary medication features.
        num_comorbidities: Number of binary comorbidity features.
        num_labs: Dimensionality of lab-value input vector.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        demographic_dim: int = 16,
        num_medications: int = 50,
        num_comorbidities: int = 30,
        num_labs: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if config is None:
            config = ModelConfig()

        self._config = config
        self._demographic_dim = demographic_dim
        self._num_medications = num_medications
        self._num_comorbidities = num_comorbidities
        self._num_labs = num_labs

        D = config.model_dim
        T_tokens = config.tokens_per_modality

        # ---- Input Tokenizers (variable-length, all produce T_tokens) ----
        self.ecg_tokenizer = ECGTokenizer(
            model_dim=D, target_tokens=T_tokens,
            dropout_rate=config.dropout_rate, name="ecg_tokenizer",
        )
        self.accel_tokenizer = MotionTokenizer(
            model_dim=D, target_tokens=T_tokens,
            dropout_rate=config.dropout_rate, name="accel_tokenizer",
        )
        self.gyro_tokenizer = MotionTokenizer(
            model_dim=D, target_tokens=T_tokens,
            dropout_rate=config.dropout_rate, name="gyro_tokenizer",
        )
        self.ppg_tokenizer = PPGTokenizer(
            model_dim=D, target_tokens=T_tokens,
            dropout_rate=config.dropout_rate, name="ppg_tokenizer",
        )
        self.spo2_tokenizer = AuxSignalTokenizer(
            model_dim=D, target_tokens=T_tokens,
            dropout_rate=config.dropout_rate, name="spo2_tokenizer",
        )
        self.temp_tokenizer = AuxSignalTokenizer(
            model_dim=D, target_tokens=T_tokens,
            dropout_rate=config.dropout_rate, name="temp_tokenizer",
        )
        self.resp_tokenizer = AuxSignalTokenizer(
            model_dim=D, target_tokens=T_tokens,
            dropout_rate=config.dropout_rate, name="resp_tokenizer",
        )

        # ---- Static Patient Embedding ----
        self.static_embedding = StaticPatientEmbedding(
            demographic_dim=demographic_dim,
            num_medications=num_medications,
            num_comorbidities=num_comorbidities,
            num_labs=num_labs,
            static_embedding_dim=D,
            dropout_rate=config.dropout_rate,
            name="static_embedding",
        )

        # ---- Cross-Modal Attention ----
        # ECG attends to fused motion
        self.ecg_motion_attention = AsymmetricCrossAttention(
            model_dim=D,
            num_heads=config.num_attention_heads,
            attention_dropout=config.attention_dropout_rate,
            value_dropout=config.dropout_rate,
            name="ecg_motion_cross_attn",
        )
        self.hierarchical_cross_attention = HierarchicalCrossAttention(
            model_dim=D,
            num_heads=config.num_attention_heads,
            attention_dropout=config.attention_dropout_rate,
            value_dropout=config.dropout_rate,
            name="hierarchical_cross_attn",
        )
        # Multi-modal ECG attends to PPG
        self.ecg_ppg_attention = AsymmetricCrossAttention(
            model_dim=D,
            num_heads=config.num_attention_heads,
            attention_dropout=config.attention_dropout_rate,
            value_dropout=config.dropout_rate,
            name="ecg_ppg_cross_attn",
        )

        # ---- Sequence Alignment (adaptive pooling to common length T) ----
        self.target_seq_length = T_tokens
        self.align_pool_ecg = _AdaptiveAveragePooling1D(
            target_length=T_tokens, name="align_pool_ecg",
        )
        self.align_pool_spo2 = _AdaptiveAveragePooling1D(
            target_length=T_tokens, name="align_pool_spo2",
        )
        self.align_pool_temp = _AdaptiveAveragePooling1D(
            target_length=T_tokens, name="align_pool_temp",
        )
        self.align_pool_resp = _AdaptiveAveragePooling1D(
            target_length=T_tokens, name="align_pool_resp",
        )
        self.assembly_projection = keras.layers.Dense(
            D, activation="gelu", kernel_initializer="he_normal",
            name="assembly_projection",
        )
        self.assembly_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="assembly_norm",
        )
        self.assembly_dropout = keras.layers.Dropout(config.dropout_rate)

        # ---- Transformer Encoder ----
        self.transformer_encoder = TransformerEncoder(
            num_layers=config.num_encoder_layers,
            model_dim=D,
            feedforward_dim=config.feedforward_dim,
            num_heads=config.num_attention_heads,
            key_dim=D // config.num_attention_heads,
            value_dim=D // config.num_attention_heads,
            dropout_rate=config.dropout_rate,
            attention_dropout=config.attention_dropout_rate,
            max_seq_len=config.max_positional_encoding,
            name="transformer_encoder",
        )

        # ---- Prediction Heads ----
        self.classification_head = ClassificationHead(
            model_dim=D, use_cls_token=True, name="classification_head",
        )
        self.survival_head = SurvivalAnalysisHead(
            model_dim=D,
            num_survival_bins=config.num_survival_bins,
            use_cls_token=True,
            name="survival_head",
        )
        self.uncertainty_head = UncertaintyHead(
            model_dim=D,
            num_mc_samples=config.uncertainty_samples,
            use_cls_token=True,
            name="uncertainty_head",
        )
        self.auxiliary_heads = AuxiliaryHeads(
            model_dim=D, use_cls_token=True, name="auxiliary_heads",
        )
        self.calibration_head = CalibrationHead(
            init_temperature=1.0,
            init_platt_a=1.0,
            init_platt_b=0.0,
            name="calibration_head",
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        """Run the full OHCA prediction pipeline.

        Args:
            inputs: Dictionary with the following keys:
                - ``"ecg"``: ``(batch, ecg_samples, 1)``
                - ``"accelerometer"``: ``(batch, accel_samples, 3)``
                - ``"gyroscope"``: ``(batch, gyro_samples, 3)``
                - ``"ppg"``: ``(batch, ppg_samples, 1)``
                - ``"spo2"``: ``(batch, spo2_samples, 1)``
                - ``"temperature"``: ``(batch, temp_samples, 1)``
                - ``"respiration"``: ``(batch, resp_samples, 1)``
                - ``"demographics"``: ``(batch, demographic_dim)``
                - ``"medications"``: ``(batch, num_medications)``
                - ``"comorbidities"``: ``(batch, num_comorbidities)``
                - ``"lab_values"``: ``(batch, num_labs)``
            training: Whether in training mode.

        Returns:
            Dictionary with prediction outputs:
                - ``"ohca_risk"``: calibrated risk ``(batch, 1)``
                - ``"raw_risk"``: uncalibrated risk ``(batch, 1)``
                - ``"survival_curve"``: cumulative survival ``(batch, num_bins+1)``
                - ``"uncertainty"``: ``(batch, 2)`` – [prediction, sigma]
                - ``"auxiliary"``: dict of auxiliary predictions
        """
        # ---- 1–7. Tokenize all inputs ----
        ecg_tokens = self.ecg_tokenizer(inputs["ecg"], training=training)
        accel_tokens = self.accel_tokenizer(
            inputs["accelerometer"], training=training,
        )
        gyro_tokens = self.gyro_tokenizer(
            inputs["gyroscope"], training=training,
        )
        ppg_tokens = self.ppg_tokenizer(inputs["ppg"], training=training)
        spo2_tokens = self.spo2_tokenizer(inputs["spo2"], training=training)
        temp_tokens = self.temp_tokenizer(inputs["temperature"], training=training)
        resp_tokens = self.resp_tokenizer(inputs["respiration"], training=training)

        # ---- 8. Static patient embedding ----
        static_token = self.static_embedding(
            demographics=inputs["demographics"],
            medications=inputs["medications"],
            comorbidities=inputs["comorbidities"],
            lab_values=inputs["lab_values"],
            training=training,
        )  # (batch, 1, D)

        # ---- 9. Motion token fusion ----
        motion_tokens = tf.concat(
            [accel_tokens, gyro_tokens], axis=1,
        )  # (batch, T_accel + T_gyro, D)

        # ---- 10. ECG attends to motion ----
# Concatenate all non-ECG modalities into one context
        context_tokens = tf.concat(
            [motion_tokens, ppg_tokens, spo2_tokens, temp_tokens, resp_tokens],
            axis=1
        )

        # Apply hierarchical cross-attention: ECG queries all modalities jointly
        fused_ecg = self.hierarchical_cross_attention(
            ecg_tokens, context_tokens, training=training
        )


        # ---- 12. Sequence alignment to common length T ----
        aligned_ecg = self.align_pool_ecg(fused_ecg)
        aligned_spo2 = self.align_pool_spo2(spo2_tokens)
        aligned_temp = self.align_pool_temp(temp_tokens)
        aligned_resp = self.align_pool_resp(resp_tokens)

        # ---- 13. Token sequence assembly ----
        # [static(1, D) | attended_ecg(T, D) | multi_modal_ecg(T, D) |
        #  spo2(T, D) | temp(T, D) | resp(T, D)]
        target_dtype = aligned_ecg.dtype
        static_token = tf.cast(static_token, target_dtype)
        full_sequence = tf.concat(
            [
                static_token,     # (batch, 1, D)
                aligned_ecg,      # (batch, T, D)
                aligned_spo2,     # (batch, T, D)
                aligned_temp,     # (batch, T, D)
                aligned_resp,     # (batch, T, D)
            ],
            axis=1,
        )  # (batch, 1 + 4*T, D)

        # Project and normalise
        full_sequence = self.assembly_projection(full_sequence)
        full_sequence = self.assembly_norm(full_sequence)
        full_sequence = self.assembly_dropout(
            full_sequence, training=training,
        )

        # ---- 14. Transformer encoder ----
        encoded_sequence = self.transformer_encoder(
            full_sequence, training=training,
        )  # (batch, T_total, D)

        # ---- 15–18. Prediction heads ----
        ohca_risk = self.classification_head(
            encoded_sequence, training=training,
        )  # (batch, 1)

        survival_hazards = self.survival_head(
            encoded_sequence, training=training,
        )  # (batch, num_bins)
        survival_curve = self.survival_head.compute_cumulative_survival(
            survival_hazards,
        )  # (batch, num_bins + 1)

        uncertainty = self.uncertainty_head(
            encoded_sequence, training=training,
        )  # (batch, 2)

        aux_predictions = self.auxiliary_heads(
            encoded_sequence, training=training,
        )  # dict of tensors

        # ---- 19. Calibration ----
        calibrated_risk = self.calibration_head(ohca_risk, method="platt")

        return {
            "ohca_risk": calibrated_risk,
            "raw_risk": ohca_risk,
            "survival_curve": survival_curve,
            "uncertainty": uncertainty,
            "auxiliary": aux_predictions,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "config": {
                    "model_dim": self._config.model_dim,
                    "num_attention_heads": self._config.num_attention_heads,
                    "num_encoder_layers": self._config.num_encoder_layers,
                    "feedforward_dim": self._config.feedforward_dim,
                    "dropout_rate": self._config.dropout_rate,
                    "attention_dropout_rate": self._config.attention_dropout_rate,
                    "static_embedding_dim": self._config.static_embedding_dim,
                    "max_positional_encoding": self._config.max_positional_encoding,
                    "num_survival_bins": self._config.num_survival_bins,
                    "uncertainty_samples": self._config.uncertainty_samples,
                },
                "demographic_dim": self._demographic_dim,
                "num_medications": self._num_medications,
                "num_comorbidities": self._num_comorbidities,
                "num_labs": self._num_labs,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "OHCAPredictionModel":
        model_config_dict = config.pop("config", {})
        model_config = ModelConfig(**model_config_dict)
        return cls(config=model_config, **config)


# ===========================================================================
# Self-Supervised Pretrainer
# ===========================================================================

class SelfSupervisedPretrainer(keras.Model):
    """Self-supervised pretrainer using three complementary objectives.

    Leverages unlabeled physiological recordings to learn robust
    representations before fine-tuning on the OHCA prediction task.

    Objectives:

        1. **Masked Signal Reconstruction** – Randomly mask 15% of ECG tokens
           (replace with a learned [MASK] embedding) and reconstruct the
           original tokens via an MLP decoder.  Encourages local morphological
           understanding.

        2. **Contrastive Learning (SimCLR-style)** – Generate two augmented
           views of each sample and pull their representations together while
           pushing apart representations of different samples (InfoNCE loss).

        3. **Temporal Order Prediction** – Sample two segments from the same
           ECG, shuffle one, and predict whether the pair is in the correct
           temporal order.  Encourages understanding of cardiac rhythm
           progression.

    The pretrainer shares the tokenizers, cross-attention, and transformer
    encoder with the main ``OHCAPredictionModel`` and adds lightweight task
        heads for each objective.

    Args:
        config: ``ModelConfig`` dataclass (shared with the main model).
        mask_ratio: Fraction of ECG tokens to mask (default 0.15).
        temperature: InfoNCE temperature (default 0.07).
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        mask_ratio: float = 0.15,
        temperature: float = 0.07,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if config is None:
            config = ModelConfig()

        self._config = config
        self._mask_ratio = mask_ratio
        self._temperature = temperature

        D = config.model_dim
        T_tokens = config.tokens_per_modality

        # ---- Shared tokenizers (reused from main model) ----
        self.ecg_tokenizer = ECGTokenizer(
            model_dim=D, target_tokens=T_tokens,
            dropout_rate=config.dropout_rate, name="ecg_tokenizer_pretrain",
        )
        self.accel_tokenizer = MotionTokenizer(
            model_dim=D, target_tokens=T_tokens,
            dropout_rate=config.dropout_rate, name="accel_tokenizer_pretrain",
        )
        self.gyro_tokenizer = MotionTokenizer(
            model_dim=D, target_tokens=T_tokens,
            dropout_rate=config.dropout_rate, name="gyro_tokenizer_pretrain",
        )
        self.ppg_tokenizer = PPGTokenizer(
            model_dim=D, target_tokens=T_tokens,
            dropout_rate=config.dropout_rate, name="ppg_tokenizer_pretrain",
        )
        self.spo2_tokenizer = AuxSignalTokenizer(
            model_dim=D, target_tokens=T_tokens,
            dropout_rate=config.dropout_rate, name="spo2_tokenizer_pretrain",
        )
        self.temp_tokenizer = AuxSignalTokenizer(
            model_dim=D, target_tokens=T_tokens,
            dropout_rate=config.dropout_rate, name="temp_tokenizer_pretrain",
        )
        self.resp_tokenizer = AuxSignalTokenizer(
            model_dim=D, target_tokens=T_tokens,
            dropout_rate=config.dropout_rate, name="resp_tokenizer_pretrain",
        )

        # ---- Cross-modal attention (shared) ----
        self.ecg_motion_attention = AsymmetricCrossAttention(
            model_dim=D,
            num_heads=config.num_attention_heads,
            attention_dropout=config.attention_dropout_rate,
            value_dropout=config.dropout_rate,
            name="ecg_motion_cross_attn_pretrain",
        )
        self.ecg_ppg_attention = AsymmetricCrossAttention(
            model_dim=D,
            num_heads=config.num_attention_heads,
            attention_dropout=config.attention_dropout_rate,
            value_dropout=config.dropout_rate,
            name="ecg_ppg_cross_attn_pretrain",
        )

        # ---- Sequence alignment ----
        self.target_seq_length = T_tokens
        self.align_pool_ecg = _AdaptiveAveragePooling1D(
            target_length=T_tokens, name="align_pool_ecg_pretrain",
        )
        self.align_pool_spo2 = _AdaptiveAveragePooling1D(
            target_length=T_tokens, name="align_pool_spo2_pretrain",
        )
        self.align_pool_temp = _AdaptiveAveragePooling1D(
            target_length=T_tokens, name="align_pool_temp_pretrain",
        )
        self.align_pool_resp = _AdaptiveAveragePooling1D(
            target_length=T_tokens, name="align_pool_resp_pretrain",
        )

        # ---- Assembly ----
        self.assembly_projection = keras.layers.Dense(
            D, activation="gelu", kernel_initializer="he_normal",
            name="assembly_projection_pretrain",
        )
        self.assembly_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="assembly_norm_pretrain",
        )
        self.assembly_dropout = keras.layers.Dropout(config.dropout_rate)

        # ---- Transformer encoder (shared architecture, separate weights) ----
        self.transformer_encoder = TransformerEncoder(
            num_layers=config.num_encoder_layers,
            model_dim=D,
            feedforward_dim=config.feedforward_dim,
            num_heads=config.num_attention_heads,
            key_dim=D // config.num_attention_heads,
            value_dim=D // config.num_attention_heads,
            dropout_rate=config.dropout_rate,
            attention_dropout=config.attention_dropout_rate,
            max_seq_len=config.max_positional_encoding,
            name="transformer_encoder_pretrain",
        )

        # ---- Masked reconstruction components ----
        self.mask_embedding = self.add_weight(
            name="mask_embedding",
            shape=(1, 1, D),
            initializer="random_normal",
            trainable=True,
        )
        self.reconstruction_decoder = _ECGReconstructionHead(
            ecg_target_length=4096,
            model_dim=D,
            name="reconstruction_decoder",
        )

        # ---- Contrastive learning projection head ----
        self.contrastive_projector = keras.Sequential(
            [
                keras.layers.Dense(256, activation="gelu", kernel_initializer="he_normal"),
                keras.layers.Dense(128, kernel_initializer="glorot_uniform"),
            ],
            name="contrastive_projector",
        )

        # ---- Temporal order prediction head ----
        self.temporal_order_head = _TemporalOrderHead(
            model_dim=D, name="temporal_order_head",
        )

    # ------------------------------------------------------------------
    # Masking utility
    # ------------------------------------------------------------------

    def _create_mask(
        self,
        batch_size: tf.Tensor,
        seq_len: int,
        mask_ratio: float,
    ) -> tf.Tensor:
        """Create a random boolean mask for masked token modeling.

        Args:
            batch_size: Scalar tensor with the batch size.
            seq_len: Sequence length (integer).
            mask_ratio: Fraction of tokens to mask.

        Returns:
            Boolean mask ``(batch, seq_len)`` where ``True`` = masked.
        """
        num_masked = int(seq_len * mask_ratio)
        # For each sample, draw ``num_masked`` positions without replacement
        all_positions = tf.range(seq_len)  # (seq_len,)
        # Use random shuffle per sample via tf.random.shuffle on a tiled tensor
        positions_tiled = tf.tile(
            tf.expand_dims(all_positions, 0),
            [batch_size, 1],
        )  # (batch, seq_len)
        # Shuffle along axis 1
        indices = tf.argsort(
            tf.random.uniform((batch_size, seq_len)), axis=1,
        )
        masked_positions = tf.gather(positions_tiled, indices, batch_dims=1)
        masked_positions = masked_positions[:, :num_masked]  # (batch, num_masked)

        # Build sparse mask
        mask = tf.scatter_nd(
            tf.expand_dims(masked_positions, 2),
            tf.ones((batch_size, num_masked), dtype=tf.bool),
            shape=(batch_size, seq_len),
        )
        return mask  # (batch, seq_len) – True where masked

    # ------------------------------------------------------------------
    # Augmentation utilities for contrastive learning
    # ------------------------------------------------------------------

    @staticmethod
    def _augment_ecg(ecg: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Apply random augmentations to an ECG signal for contrastive learning.

        Augmentations:
            - Random Gaussian noise injection
            - Random temporal shifting (±5 % of sequence length)
            - Random amplitude scaling (±20 %)

        Args:
            ecg: Raw ECG tensor ``(batch, samples, 1)``.
            training: Only augment during training.

        Returns:
            Augmented ECG tensor of the same shape.
        """
        if not training:
            return ecg

        x = tf.cast(ecg, tf.float32)

        # 1. Gaussian noise (std = 0.05 × signal range)
        signal_range = tf.reduce_max(x, axis=1, keepdims=True) - tf.reduce_min(
            x, axis=1, keepdims=True,
        )
        noise_std = tf.maximum(signal_range * 0.05, 1e-6)
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=1.0) * noise_std
        x = x + noise

        # 2. Random temporal shift (up to ±5 % of sequence length)
        seq_len = tf.shape(x)[1]
        max_shift = tf.cast(tf.maximum(seq_len // 20, 1), tf.int32)
        shift = tf.random.uniform(
            shape=(), minval=-max_shift, maxval=max_shift + 1, dtype=tf.int32,
        )
        x = tf.roll(x, shift=shift, axis=1)

        # 3. Random amplitude scaling (0.8–1.2)
        scale = tf.random.uniform(
            shape=(tf.shape(x)[0], 1, 1), minval=0.8, maxval=1.2,
        )
        x = x * scale

        return x

    @staticmethod
    def _augment_motion(
        motion: tf.Tensor, training: bool = False,
    ) -> tf.Tensor:
        """Apply random augmentations to motion sensor data.

        Augmentations:
            - Random Gaussian noise
            - Random axis dropout (zero out one axis with 20 % probability)

        Args:
            motion: Motion tensor ``(batch, samples, 3)``.
            training: Only augment during training.

        Returns:
            Augmented motion tensor of the same shape.
        """
        if not training:
            return motion

        x = tf.cast(motion, tf.float32)

        # Gaussian noise
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.1)
        x = x + noise

        # Random axis dropout
        axis_to_drop = tf.random.uniform(
            shape=(tf.shape(x)[0],), minval=0, maxval=3, dtype=tf.int32,
        )
        drop_mask = tf.one_hot(axis_to_drop, depth=3, dtype=tf.float32)
        # drop_mask: (batch, 3) → expand to (batch, 1, 3)
        drop_mask = tf.expand_dims(drop_mask, axis=1)
        # 20 % probability of dropping an axis
        should_drop = tf.random.uniform(
            shape=(tf.shape(x)[0], 1, 1), minval=0.0, maxval=1.0,
        ) < 0.2
        drop_mask = tf.cast(should_drop, tf.float32) * drop_mask
        x = x * (1.0 - drop_mask)

        return x

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(
        self,
        inputs: Dict[str, tf.Tensor],
        training: bool = False,
    ) -> Dict[str, tf.Tensor]:
        """Run the self-supervised pretraining forward pass.

        Args:
            inputs: Same dictionary format as ``OHCAPredictionModel.call``.
            training: Whether in training mode.

        Returns:
            Dictionary with pretraining losses and predictions:
                - ``"reconstruction_loss"``: scalar MSE loss
                - ``"contrastive_loss"``: scalar InfoNCE loss
                - ``"temporal_order_loss"``: scalar BCE loss
                - ``"reconstructed_ecg"``: ``(batch, ecg_samples, 1)``
                - ``"temporal_order_logits"``: ``(batch, 1)``
        """
        ecg_raw = inputs["ecg"]
        batch_size = tf.shape(ecg_raw)[0]
        D = self._config.model_dim

        # ---- Augment for contrastive views ----
        ecg_aug1 = self._augment_ecg(ecg_raw, training=training)
        ecg_aug2 = self._augment_ecg(ecg_raw, training=training)

        # ---- Tokenize view 1 (full pipeline) ----
        ecg_tokens_v1 = self.ecg_tokenizer(ecg_aug1, training=training)
        accel_tokens_v1 = self.accel_tokenizer(
            inputs["accelerometer"], training=training,
        )
        gyro_tokens_v1 = self.gyro_tokenizer(
            inputs["gyroscope"], training=training,
        )
        ppg_tokens_v1 = self.ppg_tokenizer(inputs["ppg"], training=training)
        spo2_tokens_v1 = self.spo2_tokenizer(inputs["spo2"], training=training)
        temp_tokens_v1 = self.temp_tokenizer(
            inputs["temperature"], training=training,
        )
        resp_tokens_v1 = self.resp_tokenizer(
            inputs["respiration"], training=training,
        )

        # ---- Cross-modal attention for view 1 ----
        motion_tokens_v1 = tf.concat(
            [accel_tokens_v1, gyro_tokens_v1], axis=1,
        )
        attended_ecg_v1 = self.ecg_motion_attention(
            ecg_tokens=ecg_tokens_v1,
            context_tokens=motion_tokens_v1,
            training=training,
        )
        multi_modal_ecg_v1 = self.ecg_ppg_attention(
            ecg_tokens=attended_ecg_v1,
            context_tokens=ppg_tokens_v1,
            training=training,
        )

        # ---- Tokenize view 2 (ECG only for contrastive projection) ----
        ecg_tokens_v2 = self.ecg_tokenizer(ecg_aug2, training=training)

        # ---- Align and assemble view 1 ----
        aligned_ecg_v1 = self.align_pool_ecg(multi_modal_ecg_v1)
        aligned_spo2_v1 = self.align_pool_spo2(spo2_tokens_v1)
        aligned_temp_v1 = self.align_pool_temp(temp_tokens_v1)
        aligned_resp_v1 = self.align_pool_resp(resp_tokens_v1)

        # Build a dummy static token (zeros) – pretrainer doesn't use demographics
        static_dummy = tf.zeros(
            (batch_size, 1, D), dtype=aligned_ecg_v1.dtype,
        )

        full_sequence_v1 = tf.concat(
            [
                static_dummy,
                aligned_ecg_v1,
                aligned_spo2_v1,
                aligned_temp_v1,
                aligned_resp_v1,
            ],
            axis=1,
        )
        full_sequence_v1 = self.assembly_projection(full_sequence_v1)
        full_sequence_v1 = self.assembly_norm(full_sequence_v1)
        full_sequence_v1 = self.assembly_dropout(
            full_sequence_v1, training=training,
        )

        encoded_v1 = self.transformer_encoder(
            full_sequence_v1, training=training,
        )

        # ---- 1. Masked Signal Reconstruction ----
        T = tf.shape(ecg_tokens_v1)[1]
        mask = self._create_mask(batch_size, T, self._mask_ratio)
        # mask: (batch, T) bool – True where masked

        # Apply mask: replace masked positions with the learned mask embedding
        mask_expanded = tf.cast(mask[:, :, tf.newaxis], dtype=ecg_tokens_v1.dtype)
        masked_tokens = (
            ecg_tokens_v1 * (1.0 - mask_expanded)
            + self.mask_embedding * mask_expanded
        )

        # Reconstruct the full ECG from the masked input
        reconstructed_ecg = self.reconstruction_decoder(
            masked_tokens, training=training,
        )  # (batch, ecg_samples, 1)

        # Reconstruction loss: MSE on the masked positions' reconstructed
        # waveform vs. original.  For simplicity we compute MSE over the
        # full reconstructed waveform vs. the original.
        reconstruction_loss = tf.reduce_mean(
            tf.square(reconstructed_ecg - ecg_raw),
        )

        # ---- 2. Contrastive Learning (SimCLR-style InfoNCE) ----
        # Pool view 1 and view 2 to get single vectors
        pooled_v1 = tf.reduce_mean(encoded_v1, axis=1)  # (batch, D)

        # Encode view 2 through tokenizer only (simplified – no cross-attn
        # for efficiency; the tokenizer already captures local structure)
        pooled_v2 = tf.reduce_mean(ecg_tokens_v2, axis=1)  # (batch, D)

        # Project to contrastive space
        z_v1 = self.contrastive_projector(pooled_v1)  # (batch, 128)
        z_v2 = self.contrastive_projector(pooled_v2)  # (batch, 128)

        # L2 normalize
        z_v1 = tf.math.l2_normalize(z_v1, axis=-1)
        z_v2 = tf.math.l2_normalize(z_v2, axis=-1)

        # Similarity matrix: (batch, batch)
        similarity = tf.matmul(z_v1, z_v2, transpose_b=True)
        similarity = similarity / self._temperature

        # Labels: diagonal is positive
        labels = tf.range(batch_size)  # (batch,)

        # Symmetric InfoNCE: cross-entropy on both directions
        loss_v1_to_v2 = tf.keras.losses.sparse_categorical_crossentropy(
            labels, similarity, from_logits=True,
        )
        loss_v2_to_v1 = tf.keras.losses.sparse_categorical_crossentropy(
            labels, tf.transpose(similarity), from_logits=True,
        )
        contrastive_loss = tf.reduce_mean(
            (loss_v1_to_v2 + loss_v2_to_v1) / 2.0,
        )

        # ---- 3. Temporal Order Prediction ----
        T_ecg = tf.shape(ecg_tokens_v1)[1]
        # Split ECG tokens into two halves
        half = T_ecg // 2
        segment_a = ecg_tokens_v1[:, :half, :]  # (batch, half, D)
        segment_b = ecg_tokens_v1[:, half : half * 2, :]  # (batch, half, D)

        # Randomly shuffle ~50 % of samples to create negative pairs
        should_shuffle = tf.random.uniform(
            shape=(batch_size, 1, 1), minval=0.0, maxval=1.0,
        ) < 0.5
        # Create shuffled segment_b indices
        indices = tf.argsort(
            tf.random.uniform((batch_size, half)), axis=1,
        )
        shuffled_b = tf.gather(segment_b, indices, batch_dims=1)
        segment_b_final = tf.where(should_shuffle, shuffled_b, segment_b)

        # Ground truth: 1 if not shuffled, 0 if shuffled
        temporal_labels = tf.cast(
            tf.squeeze(~should_shuffle, axis=[1, 2]), tf.float32,
        )  # (batch,)

        temporal_logits = self.temporal_order_head(
            segment_a, segment_b_final, training=training,
        )  # (batch, 1)
        temporal_logits = tf.squeeze(temporal_logits, axis=-1)  # (batch,)

        temporal_order_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                temporal_labels[:, tf.newaxis],
                tf.sigmoid(temporal_logits[:, tf.newaxis]),
                from_logits=False,
            ),
        )

        return {
            "reconstruction_loss": reconstruction_loss,
            "contrastive_loss": contrastive_loss,
            "temporal_order_loss": temporal_order_loss,
            "reconstructed_ecg": reconstructed_ecg,
            "temporal_order_logits": tf.sigmoid(temporal_logits[:, tf.newaxis]),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "config": {
                    "model_dim": self._config.model_dim,
                    "num_attention_heads": self._config.num_attention_heads,
                    "num_encoder_layers": self._config.num_encoder_layers,
                    "feedforward_dim": self._config.feedforward_dim,
                    "dropout_rate": self._config.dropout_rate,
                    "attention_dropout_rate": self._config.attention_dropout_rate,
                    "static_embedding_dim": self._config.static_embedding_dim,
                    "max_positional_encoding": self._config.max_positional_encoding,
                    "num_survival_bins": self._config.num_survival_bins,
                    "uncertainty_samples": self._config.uncertainty_samples,
                    "tokens_per_modality": self._config.tokens_per_modality,
                },
                "mask_ratio": self._mask_ratio,
                "temperature": self._temperature,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SelfSupervisedPretrainer":
        model_config_dict = config.pop("config", {})
        model_config = ModelConfig(**model_config_dict)
        return cls(config=model_config, **config)
