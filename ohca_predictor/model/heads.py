"""Prediction heads for the OHCA prediction model.

Implements classification, survival analysis, uncertainty estimation,
calibration, and auxiliary prediction heads used by the multi-task
OHCA prediction architecture.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import tensorflow as tf
from tensorflow import keras


# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------

class ClassificationHead(keras.layers.Layer):
    """Binary classification head for OHCA risk prediction.

    Extracts the CLS token (prepended static context token) from the
    encoded sequence or falls back to global average pooling, then
    projects through a multi-layer MLP to produce a single risk
    probability in [0.0, 1.0].

    Architecture:
        1. CLS token extraction (index 0) or global average pooling
        2. Dense(model_dim → 256) with GELU
        3. Dropout(0.2)
        4. Dense(256 → 64) with GELU
        5. Dropout(0.1)
        6. Dense(64 → 1) with Sigmoid

    Args:
        model_dim: Dimensionality of input features (default 256).
        use_cls_token: If True, use first token (CLS) as representation.
            If False, use global average pooling over the sequence.
    """

    def __init__(
        self,
        model_dim: int = 256,
        use_cls_token: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.use_cls_token = use_cls_token

    def build(self, input_shape: tf.TensorShape):
        self.dense1 = keras.layers.Dense(
            256, activation="gelu", kernel_initializer="he_normal", name="cls_dense1"
        )
        self.dropout1 = keras.layers.Dropout(0.2)
        self.dense2 = keras.layers.Dense(
            64, activation="gelu", kernel_initializer="he_normal", name="cls_dense2"
        )
        self.dropout2 = keras.layers.Dropout(0.1)
        self.output_dense = keras.layers.Dense(
            1, activation="sigmoid", name="cls_output"
        )

        super().build(input_shape)

    def _extract_representation(self, sequence: tf.Tensor) -> tf.Tensor:
        """Extract a single vector per sample from the encoded sequence.

        Args:
            sequence: Encoded sequence of shape (batch, seq_len, model_dim).

        Returns:
            Representation vector of shape (batch, model_dim).
        """
        if self.use_cls_token:
            # CLS token is the first token in the sequence
            return sequence[:, 0, :]
        else:
            # Global average pooling over the sequence dimension
            return tf.reduce_mean(sequence, axis=1)

    def call(
        self, sequence: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        """Compute OHCA risk probability from encoded sequence.

        Args:
            sequence: Encoded sequence of shape (batch, seq_len, model_dim).
            training: Whether in training mode.

        Returns:
            Risk probabilities of shape (batch, 1), values in [0.0, 1.0].
        """
        x = self._extract_representation(sequence)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return self.output_dense(x)

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "model_dim": self.model_dim,
                "use_cls_token": self.use_cls_token,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Survival Analysis Head
# ---------------------------------------------------------------------------

class SurvivalAnalysisHead(keras.layers.Layer):
    """Discrete-time survival analysis head.

    Predicts per-bin conditional hazard probabilities for a set of
    equally-spaced time bins covering the prediction window.

    Architecture:
        1. CLS token extraction or global average pooling
        2. Dense(model_dim → 128) with GELU
        3. Dense(128 → num_survival_bins) with Sigmoid

    Each output S_i represents:

        P(event in bin i | survival to start of bin i)

    The cumulative survival function is:

        S(t) = product_{j=0}^{i} (1 - S_j)

    where S_0 = 1 (certain to survive the initial instant).

    Default configuration: 12 bins of 20 minutes each = 4 hours total.

    Args:
        model_dim: Dimensionality of input features (default 256).
        num_survival_bins: Number of discrete time bins (default 12).
        use_cls_token: Whether to use CLS token (True) or GAP (False).
    """

    def __init__(
        self,
        model_dim: int = 256,
        num_survival_bins: int = 12,
        use_cls_token: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.num_survival_bins = num_survival_bins
        self.use_cls_token = use_cls_token

    def build(self, input_shape: tf.TensorShape):
        self.dense1 = keras.layers.Dense(
            128, activation="gelu", kernel_initializer="he_normal", name="surv_dense1"
        )
        self.dense2 = keras.layers.Dense(
            self.num_survival_bins,
            activation="sigmoid",
            kernel_initializer="glorot_uniform",
            name="surv_output",
        )

        super().build(input_shape)

    def _extract_representation(self, sequence: tf.Tensor) -> tf.Tensor:
        """Extract representation from encoded sequence.

        Args:
            sequence: (batch, seq_len, model_dim).

        Returns:
            (batch, model_dim).
        """
        if self.use_cls_token:
            return sequence[:, 0, :]
        return tf.reduce_mean(sequence, axis=1)

    def call(
        self, sequence: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        """Compute per-bin hazard probabilities.

        Args:
            sequence: Encoded sequence of shape (batch, seq_len, model_dim).
            training: Whether in training mode.

        Returns:
            Hazard probabilities of shape (batch, num_survival_bins).
            Each value is P(event in bin i | survived to bin i).
        """
        x = self._extract_representation(sequence)
        x = self.dense1(x)
        return self.dense2(x)

    def compute_cumulative_survival(self, hazards: tf.Tensor) -> tf.Tensor:
        """Convert per-bin hazards to cumulative survival probabilities.

        Args:
            hazards: Per-bin hazard probabilities (batch, num_bins).

        Returns:
            Cumulative survival curve S(t) of shape (batch, num_bins + 1).
            S[0] = 1.0 (certain survival at t=0), S[i] = P(survive past bin i).
        """
        # survival_per_bin = 1 - hazard
        survival_per_bin = 1.0 - hazards  # (batch, num_bins)

        # Cumulative product: S(t) = prod(1 - S_i) for i=0..t
        # Pad with leading 1.0 for S(t=0)
        ones = tf.ones((tf.shape(hazards)[0], 1), dtype=hazards.dtype)
        cumulative = tf.math.cumprod(survival_per_bin, axis=1, exclusive=True)
        cumulative = tf.concat([ones, cumulative], axis=1)
        return cumulative

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "model_dim": self.model_dim,
                "num_survival_bins": self.num_survival_bins,
                "use_cls_token": self.use_cls_token,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Uncertainty Estimation Head
# ---------------------------------------------------------------------------

class UncertaintyHead(keras.layers.Layer):
    """Monte Carlo Dropout uncertainty estimation head.

    Estimates both epistemic (model) and aleatoric (data) uncertainty
    for each prediction.

    Architecture:
        1. Dense(model_dim → 64) with GELU
        2. Dense(64 → 2) outputs [mu, log_sigma]
        3. sigma = exp(log_sigma)  (ensures positivity)

    During training (reparameterization trick):
        output = mu + sigma * epsilon,   epsilon ~ N(0, 1)

    During inference (MC sampling):
        - Perform ``num_samples`` forward passes with dropout enabled
        - Compute mean and variance across samples for epistemic uncertainty
        - Use learned sigma for aleatoric uncertainty

    The uncertainty estimate is critical for clinical deployment:
        - High uncertainty → flag for manual review
        - Low uncertainty + high risk → urgent alert
        - Low uncertainty + low risk → routine monitoring

    Args:
        model_dim: Dimensionality of input features (default 256).
        num_mc_samples: Number of Monte Carlo forward passes at inference
            (default 50).
        use_cls_token: Whether to use CLS token (True) or GAP (False).
    """

    def __init__(
        self,
        model_dim: int = 256,
        num_mc_samples: int = 50,
        use_cls_token: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.num_mc_samples = num_mc_samples
        self.use_cls_token = use_cls_token

    def build(self, input_shape: tf.TensorShape):
        self.dense1 = keras.layers.Dense(
            64, activation="gelu", kernel_initializer="he_normal", name="unc_dense1"
        )
        self.output_dense = keras.layers.Dense(
            2, kernel_initializer="glorot_uniform", name="unc_output"
        )

        super().build(input_shape)

    def _extract_representation(self, sequence: tf.Tensor) -> tf.Tensor:
        """Extract representation from encoded sequence.

        Args:
            sequence: (batch, seq_len, model_dim).

        Returns:
            (batch, model_dim).
        """
        if self.use_cls_token:
            return sequence[:, 0, :]
        return tf.reduce_mean(sequence, axis=1)

    def _predict_mu_log_sigma(
        self, sequence: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute raw mu and log_sigma from sequence representation.

        Args:
            sequence: Encoded sequence (batch, seq_len, model_dim).
            training: Whether in training mode.

        Returns:
            Tuple of (mu, log_sigma), each of shape (batch, 1).
        """
        x = self._extract_representation(sequence)
        x = self.dense1(x)
        out = self.output_dense(x)  # (batch, 2)
        mu = out[:, :1]       # (batch, 1)
        log_sigma = out[:, 1:]  # (batch, 1)
        return mu, log_sigma

    def call(
        self, sequence: tf.Tensor, training: bool = False
    ) -> tf.Tensor:
        """Compute uncertainty-aware prediction.

        During training: applies reparameterization trick.
        During inference: performs MC sampling.

        Args:
            sequence: Encoded sequence (batch, seq_len, model_dim).
            training: Whether in training mode.

        Returns:
            Tensor of shape (batch, 2): [predicted_value, sigma].
            - predicted_value: risk estimate (mu during training via
              reparameterization, mean of MC samples during inference).
            - sigma: predicted standard deviation (aleatoric + epistemic).
        """
        mu, log_sigma = self._predict_mu_log_sigma(sequence, training=training)
        log_sigma = tf.clip_by_value(log_sigma, -5.0, 3.0)
        sigma = tf.exp(log_sigma)

        if training:
            # Reparameterization trick: mu + sigma * epsilon
            epsilon = tf.random.normal(
                shape=tf.shape(mu), mean=0.0, stddev=1.0, dtype=mu.dtype
            )
            prediction = mu + sigma * epsilon
        else:
            # MC sampling: run multiple forward passes with dropout enabled
            mc_predictions = []
            for _ in range(self.num_mc_samples):
                mc_mu, _ = self._predict_mu_log_sigma(sequence, training=True)
                mc_predictions.append(mc_mu)

            # Stack: (num_samples, batch, 1)
            mc_stack = tf.stack(mc_predictions, axis=0)

            # Mean prediction across MC samples
            prediction = tf.reduce_mean(mc_stack, axis=0)  # (batch, 1)

            # Epistemic uncertainty: std across MC samples
            epistemic_sigma = tf.math.reduce_std(mc_stack, axis=0)

            # Total uncertainty: sqrt(aleatoric^2 + epistemic^2)
            sigma = tf.sqrt(sigma ** 2 + epistemic_sigma ** 2)

        return tf.concat([prediction, sigma], axis=-1)  # (batch, 2)

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "model_dim": self.model_dim,
                "num_mc_samples": self.num_mc_samples,
                "use_cls_token": self.use_cls_token,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Calibration Head
# ---------------------------------------------------------------------------

class CalibrationHead(keras.layers.Layer):
    """Post-hoc calibration using temperature scaling and Platt scaling.

    Temperature Scaling:
        calibrated_prob = sigmoid(logit(raw_prob) / T)

        - T > 1: model is overconfident (typical for deep networks)
        - T < 1: model is underconfident
        - T ≈ 1: model is well-calibrated

    Platt Scaling:
        calibrated_prob = sigmoid(a * logit(raw_prob) + b)

        More flexible than temperature scaling; learns both a linear
        transformation of the logit and a bias term.

    Both parameters are learned on the validation set using NLL loss.

    Args:
        init_temperature: Initial value for the temperature parameter
            (default 1.0).
        init_platt_a: Initial value for Platt scaling slope (default 1.0).
        init_platt_b: Initial value for Platt scaling intercept (default 0.0).
    """

    def __init__(
        self,
        init_temperature: float = 1.0,
        init_platt_a: float = 1.0,
        init_platt_b: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.init_temperature = init_temperature
        self.init_platt_a = init_platt_a
        self.init_platt_b = init_platt_b

    def build(self, input_shape: tf.TensorShape):
        self.temperature = self.add_weight(
            name="temperature",
            shape=(),
            initializer=keras.initializers.Constant(self.init_temperature),
            trainable=True,
        )

        self.platt_a = self.add_weight(
            name="platt_a",
            shape=(),
            initializer=keras.initializers.Constant(self.init_platt_a),
            trainable=True,
        )

        self.platt_b = self.add_weight(
            name="platt_b",
            shape=(),
            initializer=keras.initializers.Constant(self.init_platt_b),
            trainable=True,
        )

        super().build(input_shape)

    @staticmethod
    def _safe_logit(prob: tf.Tensor, eps: float = 1e-7) -> tf.Tensor:
        """Compute logit with numerical stability.

        logit(p) = log(p / (1 - p))

        Clamps p to [eps, 1-eps] before computing logit.

        Args:
            prob: Probability values in (0, 1).
            eps: Small constant for numerical stability.

        Returns:
            Logit values.
        """
        prob = tf.clip_by_value(prob, eps, 1.0 - eps)
        return tf.math.log(prob / (1.0 - prob))

    def temperature_scaling(self, raw_prob: tf.Tensor) -> tf.Tensor:
        """Apply temperature scaling to raw probability.

        calibrated = sigmoid(logit(raw) / T)

        Args:
            raw_prob: Raw probability from the model, shape (batch, 1).

        Returns:
            Calibrated probability, shape (batch, 1).
        """
        logit = self._safe_logit(raw_prob)
        scaled_logit = logit / self.temperature
        return tf.sigmoid(scaled_logit)

    def platt_scaling(self, raw_prob: tf.Tensor) -> tf.Tensor:
        """Apply Platt scaling to raw probability.

        calibrated = sigmoid(a * logit(raw) + b)

        Args:
            raw_prob: Raw probability from the model, shape (batch, 1).

        Returns:
            Calibrated probability, shape (batch, 1).
        """
        logit = self._safe_logit(raw_prob)
        scaled_logit = self.platt_a * logit + self.platt_b
        return tf.sigmoid(scaled_logit)

    def call(
        self,
        raw_prob: tf.Tensor,
        method: str = "platt",
    ) -> tf.Tensor:
        """Calibrate a raw probability prediction.

        Args:
            raw_prob: Raw probability from the classification head,
                shape (batch, 1).
            method: Calibration method, one of "temperature" or "platt"
                (default "platt").

        Returns:
            Calibrated probability, shape (batch, 1).

        Raises:
            ValueError: If method is not "temperature" or "platt".
        """
        if method == "temperature":
            return self.temperature_scaling(raw_prob)
        elif method == "platt":
            return self.platt_scaling(raw_prob)
        else:
            raise ValueError(
                f"Unknown calibration method: {method!r}. "
                "Choose 'temperature' or 'platt'."
            )

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "init_temperature": self.init_temperature,
                "init_platt_a": self.init_platt_a,
                "init_platt_b": self.init_platt_b,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Auxiliary Prediction Heads
# ---------------------------------------------------------------------------

class AuxiliaryHeads(keras.layers.Layer):
    """Multi-task auxiliary heads for self-supervised pretraining and
    regularized training.

    These heads predict physiological quantities that are easier to learn
    than OHCA and provide useful intermediate representations that serve
    as regularizers during training.

    Heads:
        1. Heart Rate Head: Dense(model_dim → 64 → 1), linear output (MSE)
        2. Rhythm Classification: Dense(model_dim → 64 → 5),
           softmax [sinus, AF, VT, VF, other]
        3. Activity State: Dense(model_dim → 64 → 5),
           softmax [sleep, rest, light, moderate, vigorous]
        4. SpO2 Prediction: Dense(model_dim → 64 → 1), linear output (MSE)
        5. Blood Pressure: Dense(model_dim → 64 → 2), linear output (SBP, DBP)

    Args:
        model_dim: Dimensionality of input features (default 256).
        use_cls_token: Whether to use CLS token (True) or GAP (False).
    """

    def __init__(
        self,
        model_dim: int = 256,
        use_cls_token: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dim = model_dim
        self.use_cls_token = use_cls_token

    def build(self, input_shape: tf.TensorShape):
        # --- Shared feature extractor ---
        # Not strictly necessary but avoids duplicating the initial
        # projection; each head can share the lower layers.

        # --- 1. Heart Rate Head (regression) ---
        self.hr_dense1 = keras.layers.Dense(
            64, activation="gelu", kernel_initializer="he_normal", name="hr_dense1"
        )
        self.hr_output = keras.layers.Dense(
            1, activation="linear", kernel_initializer="glorot_uniform", name="hr_output"
        )

        # --- 2. Rhythm Classification Head ---
        self.rhythm_dense1 = keras.layers.Dense(
            64, activation="gelu", kernel_initializer="he_normal", name="rhythm_dense1"
        )
        self.rhythm_output = keras.layers.Dense(
            5, activation="softmax", kernel_initializer="glorot_uniform", name="rhythm_output"
        )

        # --- 3. Activity State Head ---
        self.activity_dense1 = keras.layers.Dense(
            64, activation="gelu", kernel_initializer="he_normal", name="activity_dense1"
        )
        self.activity_output = keras.layers.Dense(
            5, activation="softmax", kernel_initializer="glorot_uniform", name="activity_output"
        )

        # --- 4. SpO2 Prediction Head (regression) ---
        self.spo2_dense1 = keras.layers.Dense(
            64, activation="gelu", kernel_initializer="he_normal", name="spo2_dense1"
        )
        self.spo2_output = keras.layers.Dense(
            1, activation="linear", kernel_initializer="glorot_uniform", name="spo2_output"
        )

        # --- 5. Blood Pressure Head (regression: SBP, DBP) ---
        self.bp_dense1 = keras.layers.Dense(
            64, activation="gelu", kernel_initializer="he_normal", name="bp_dense1"
        )
        self.bp_output = keras.layers.Dense(
            2, activation="linear", kernel_initializer="glorot_uniform", name="bp_output"
        )

        super().build(input_shape)

    def _extract_representation(self, sequence: tf.Tensor) -> tf.Tensor:
        """Extract representation from encoded sequence.

        Args:
            sequence: (batch, seq_len, model_dim).

        Returns:
            (batch, model_dim).
        """
        if self.use_cls_token:
            return sequence[:, 0, :]
        return tf.reduce_mean(sequence, axis=1)

    def call(
        self, sequence: tf.Tensor, training: bool = False
    ) -> Dict[str, tf.Tensor]:
        """Compute all auxiliary predictions.

        Args:
            sequence: Encoded sequence of shape (batch, seq_len, model_dim).
            training: Whether in training mode.

        Returns:
            Dictionary with keys:
                - "heart_rate": (batch, 1) — predicted mean HR
                - "rhythm": (batch, 5) — rhythm class probabilities
                - "activity": (batch, 5) — activity class probabilities
                - "spo2": (batch, 1) — predicted mean SpO2
                - "blood_pressure": (batch, 2) — [SBP, DBP]
        """
        x = self._extract_representation(sequence)

        # 1. Heart rate regression
        hr = self.hr_dense1(x, training=training)
        hr = self.hr_output(hr)

        # 2. Rhythm classification
        rhythm = self.rhythm_dense1(x, training=training)
        rhythm = self.rhythm_output(rhythm)

        # 3. Activity state classification
        activity = self.activity_dense1(x, training=training)
        activity = self.activity_output(activity)

        # 4. SpO2 regression
        spo2 = self.spo2_dense1(x, training=training)
        spo2 = self.spo2_output(spo2)

        # 5. Blood pressure regression (SBP, DBP)
        bp = self.bp_dense1(x, training=training)
        bp = self.bp_output(bp)

        return {
            "heart_rate": hr,
            "rhythm": rhythm,
            "activity": activity,
            "spo2": spo2,
            "blood_pressure": bp,
        }

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update(
            {
                "model_dim": self.model_dim,
                "use_cls_token": self.use_cls_token,
            }
        )
        return config
