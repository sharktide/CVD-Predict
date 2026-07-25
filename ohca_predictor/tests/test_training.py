"""Tests for the OHCA Training module."""

import pytest
import numpy as np
import tensorflow as tf
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ohca_predictor.training.trainer import OHCATrainer, CosineDecayWithWarmup, GradientAccumulation
from ohca_predictor.training.curriculum import CurriculumScheduler, SignalAugmenter
from ohca_predictor.config import get_config


@pytest.fixture
def config():
    return get_config()


class TestCosineDecayWithWarmup:
    """Tests for learning rate schedule."""

    def test_lr_schedule_warmup(self):
        schedule = CosineDecayWithWarmup(
            learning_rate=1e-3,
            warmup_steps=100,
            total_steps=1000,
        )
        lr_warmup = schedule(50)
        assert float(lr_warmup) < 1e-3

    def test_lr_schedule_peak(self):
        schedule = CosineDecayWithWarmup(
            learning_rate=1e-3,
            warmup_steps=100,
            total_steps=1000,
        )
        lr_peak = schedule(100)
        assert abs(float(lr_peak) - 1e-3) < 1e-4

    def test_lr_schedule_decay(self):
        schedule = CosineDecayWithWarmup(
            learning_rate=1e-3,
            warmup_steps=100,
            total_steps=1000,
        )
        lr_end = schedule(1000)
        assert float(lr_end) < 1e-3


class TestGradientAccumulation:
    """Tests for gradient accumulation."""

    def test_accumulation_steps(self):
        accum = GradientAccumulation(accumulation_steps=4)
        assert accum.accumulation_steps == 4

    def test_invalid_steps(self):
        with pytest.raises(ValueError):
            GradientAccumulation(accumulation_steps=0)

    def test_reset(self):
        accum = GradientAccumulation(accumulation_steps=4)
        accum._step_count = 3
        accum.reset()
        assert accum._step_count == 0


class TestCurriculumScheduler:
    """Tests for curriculum learning scheduler."""

    def test_initialization(self, config):
        dataset_info = {
            "total_patients": 100,
            "patient_ids": [f"p{i}" for i in range(100)],
            "labels": np.random.randint(0, 2, 100).tolist(),
        }
        scheduler = CurriculumScheduler(config, dataset_info)
        assert scheduler.total_patients == 100

    def test_phase_progression(self, config):
        dataset_info = {
            "total_patients": 100,
            "patient_ids": [f"p{i}" for i in range(100)],
            "labels": np.random.randint(0, 2, 100).tolist(),
        }
        scheduler = CurriculumScheduler(config, dataset_info)
        phase = scheduler.get_current_phase(epoch=0)
        assert phase >= 0

    def test_get_phase_info(self, config):
        dataset_info = {
            "total_patients": 100,
            "patient_ids": [f"p{i}" for i in range(100)],
            "labels": np.random.randint(0, 2, 100).tolist(),
        }
        scheduler = CurriculumScheduler(config, dataset_info)
        info = scheduler.get_phase_info(epoch=0)
        assert isinstance(info, dict)


class TestSignalAugmenter:
    """Tests for signal augmentation."""

    def test_initialization(self):
        augmenter = SignalAugmenter()
        assert augmenter is not None

    def test_augment_ecg(self):
        augmenter = SignalAugmenter()
        ecg = np.random.randn(2500, 12).astype(np.float32)
        augmented = augmenter.augment_ecg(ecg)
        assert augmented.shape == ecg.shape

    def test_augment_with_noise(self):
        augmenter = SignalAugmenter()
        ecg = np.random.randn(2500, 12).astype(np.float32)
        augmented = augmenter.augment_ecg(ecg)
        assert augmented.shape == ecg.shape
        assert not np.array_equal(ecg, augmented)


class TestOHCATrainer:
    """Tests for the main trainer class."""

    def test_initialization(self, config):
        trainer = OHCATrainer(config)
        assert trainer.config == config
        assert trainer.model is not None
        assert trainer.loss_fn is not None
        assert trainer.optimizer is not None
