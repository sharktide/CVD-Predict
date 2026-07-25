"""Tests for the OHCA Model module."""

import pytest
import numpy as np
import tensorflow as tf
from ohca_predictor.model.tokenizer import ECGTokenizer, MotionTokenizer, PPGTokenizer, AuxSignalTokenizer
from ohca_predictor.model.attention import AsymmetricCrossAttention, HierarchicalCrossAttention
from ohca_predictor.model.encoder import TransformerEncoder
from ohca_predictor.model.heads import (
    ClassificationHead,
    SurvivalAnalysisHead,
    UncertaintyHead,
    CalibrationHead,
    AuxiliaryHeads,
)
from ohca_predictor.model.losses import (
    ClinicallyWeightedBCE,
    FocalLoss,
    DiscreteTimeSurvivalLoss,
    CombinedOHCALoss,
)
from ohca_predictor.config import get_config


class TestTokenizers:
    """Tests for signal tokenization layers with variable-length inputs."""

    def test_ecg_tokenizer_short(self):
        """3-minute ECG (23,400 samples at 130 Hz)."""
        tokenizer = ECGTokenizer(model_dim=256, target_tokens=512)
        ecg = tf.random.normal((2, 23400, 1))
        output = tokenizer(ecg)
        assert output.shape == (2, 512, 256)

    def test_ecg_tokenizer_long(self):
        """5-minute ECG (39,000 samples at 130 Hz)."""
        tokenizer = ECGTokenizer(model_dim=256, target_tokens=512)
        ecg = tf.random.normal((2, 39000, 1))
        output = tokenizer(ecg)
        assert output.shape == (2, 512, 256)

    def test_motion_tokenizer_variable(self):
        """Variable-length accelerometer data."""
        tokenizer = MotionTokenizer(model_dim=256, target_tokens=512)
        motion = tf.random.normal((2, 1560, 3))
        output = tokenizer(motion)
        assert output.shape == (2, 512, 256)

    def test_ppg_tokenizer_variable(self):
        """Variable-length PPG data."""
        tokenizer = PPGTokenizer(model_dim=256, target_tokens=512)
        ppg = tf.random.normal((2, 1500, 1))
        output = tokenizer(ppg)
        assert output.shape == (2, 512, 256)

    def test_aux_tokenizer_short_signal(self):
        """Short SpO2 signal (50 samples)."""
        tokenizer = AuxSignalTokenizer(model_dim=256, target_tokens=512)
        spo2 = tf.random.normal((2, 50, 1))
        output = tokenizer(spo2)
        assert output.shape == (2, 512, 256)

    def test_aux_tokenizer_long_signal(self):
        """Long respiration signal (30,000 samples)."""
        tokenizer = AuxSignalTokenizer(model_dim=256, target_tokens=512)
        resp = tf.random.normal((2, 30000, 1))
        output = tokenizer(resp)
        assert output.shape == (2, 512, 256)


class TestAttention:
    """Tests for attention layers."""

    def test_asymmetric_cross_attention(self):
        layer = AsymmetricCrossAttention(model_dim=256, num_heads=8)
        ecg_tokens = tf.random.normal((2, 512, 256))
        context_tokens = tf.random.normal((2, 1024, 256))
        output = layer(ecg_tokens, context_tokens)
        assert output.shape == (2, 512, 256)

    def test_hierarchical_cross_attention(self):
        layer = HierarchicalCrossAttention(model_dim=256, num_heads=8)
        ecg_tokens = tf.random.normal((2, 512, 256))
        context_tokens = tf.random.normal((2, 1024, 256))
        output = layer(ecg_tokens, context_tokens)
        assert output.shape[0] == 2
        assert output.shape[2] == 256


class TestEncoder:
    """Tests for Transformer encoder."""

    def test_encoder_variable_length(self):
        config = get_config()
        encoder = TransformerEncoder(
            num_layers=config.model.num_encoder_layers,
            model_dim=config.model.model_dim,
            feedforward_dim=config.model.feedforward_dim,
            num_heads=config.model.num_attention_heads,
        )
        x = tf.random.normal((2, 2048, 256))
        output = encoder(x, training=False)
        assert output.shape == (2, 2048, 256)


class TestHeads:
    """Tests for prediction heads."""

    def test_classification_head(self):
        head = ClassificationHead(model_dim=256)
        x = tf.random.normal((2, 512, 256))
        output = head(x)
        assert output.shape == (2, 1)
        assert tf.reduce_all(output >= 0).numpy()
        assert tf.reduce_all(output <= 1).numpy()

    def test_survival_head(self):
        head = SurvivalAnalysisHead(model_dim=256, num_survival_bins=12)
        x = tf.random.normal((2, 512, 256))
        output = head(x)
        assert output.shape[0] == 2

    def test_uncertainty_head(self):
        head = UncertaintyHead(model_dim=256, num_mc_samples=10)
        x = tf.random.normal((2, 512, 256))
        output = head(x, training=False)
        assert output.shape[0] == 2


class TestLosses:
    """Tests for custom loss functions."""

    def test_bce_loss(self):
        loss_fn = ClinicallyWeightedBCE(fn_weight=10.0, fp_weight=1.0)
        y_true = tf.constant([[1.0], [0.0], [0.0], [1.0]])
        y_pred = tf.constant([[0.8], [0.2], [0.1], [0.9]])
        loss = loss_fn(y_true, y_pred)
        assert loss.numpy() > 0

    def test_focal_loss(self):
        loss_fn = FocalLoss(gamma=2.0)
        y_true = tf.constant([[1.0], [0.0], [0.0], [1.0]])
        y_pred = tf.constant([[0.8], [0.2], [0.1], [0.9]])
        loss = loss_fn(y_true, y_pred)
        assert loss.numpy() > 0

    def test_survival_loss(self):
        loss_fn = DiscreteTimeSurvivalLoss()
        K = 12
        y_true = np.zeros((4, K + 1), dtype=np.float32)
        y_true[0, 3] = 1.0
        y_true[1, -1] = 1.0
        y_true[2, 7] = 1.0
        y_true[3, -1] = 1.0
        y_true = tf.constant(y_true)
        y_pred = tf.random.uniform((4, K), 0.1, 0.9)
        loss = loss_fn(y_true, y_pred)
        assert loss.numpy() > 0


class TestFullModel:
    """Tests for complete model architecture with variable-length inputs."""

    def _make_batch(self, duration_seconds, batch_size=2):
        """Create a batch of inputs for a given duration in seconds."""
        ecg_hz = 130
        accel_hz = 52
        gyro_hz = 52
        ppg_hz = 50
        spo2_hz = 10
        temp_hz = 1
        resp_hz = 25

        return {
            'ecg': tf.random.normal((batch_size, int(ecg_hz * duration_seconds), 1)),
            'accelerometer': tf.random.normal((batch_size, int(accel_hz * duration_seconds), 3)),
            'gyroscope': tf.random.normal((batch_size, int(gyro_hz * duration_seconds), 3)),
            'ppg': tf.random.normal((batch_size, int(ppg_hz * duration_seconds), 1)),
            'spo2': tf.random.normal((batch_size, int(spo2_hz * duration_seconds), 1)),
            'temperature': tf.random.normal((batch_size, int(temp_hz * duration_seconds), 1)),
            'respiration': tf.random.normal((batch_size, int(resp_hz * duration_seconds), 1)),
            'demographics': tf.random.normal((batch_size, 24)),
            'medications': tf.random.normal((batch_size, 14)),
            'comorbidities': tf.random.normal((batch_size, 14)),
            'lab_values': tf.random.normal((batch_size, 8)),
        }

    def test_model_forward_pass_3min(self):
        """3-minute window."""
        config = get_config()
        from ohca_predictor.model.architecture import OHCAPredictionModel
        model = OHCAPredictionModel(config.model)
        batch = self._make_batch(duration_seconds=180)
        outputs = model(batch, training=False)
        assert 'ohca_risk' in outputs
        assert outputs['ohca_risk'].shape == (2, 1)
        assert tf.reduce_all(outputs['ohca_risk'] >= 0).numpy()
        assert tf.reduce_all(outputs['ohca_risk'] <= 1).numpy()

    def test_model_gradient_flow(self):
        config = get_config()
        from ohca_predictor.model.architecture import OHCAPredictionModel
        model = OHCAPredictionModel(config.model)
        batch = self._make_batch(duration_seconds=10, batch_size=1)
        batch['ohca_label'] = tf.constant([[1.0]])
        batch['survival_labels'] = tf.zeros((1, 13))
        batch['heart_rate'] = tf.constant([[72.0]])
        batch['rhythm'] = tf.constant([[0]])
        batch['activity_state'] = tf.constant([[1]])
        batch['spo2_mean'] = tf.constant([[98.0]])
        batch['sbp_mean'] = tf.constant([[120.0]])
        batch['dbp_mean'] = tf.constant([[80.0]])

        with tf.GradientTape() as tape:
            outputs = model(batch, training=True)
            loss_fn = CombinedOHCALoss()
            loss = loss_fn(batch, outputs)
        gradients = tape.gradient(loss, model.trainable_variables)
        non_none_grads = [g for g in gradients if g is not None]
        assert len(non_none_grads) > 0
