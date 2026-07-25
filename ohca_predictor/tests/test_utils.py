"""Tests for the OHCA Utils module."""

import pytest
import numpy as np
import tensorflow as tf
from ohca_predictor.utils.reproducibility import ReproducibilityManager
from ohca_predictor.utils.logging import StructuredLogger


class TestReproducibilityManager:
    """Tests for ReproducibilityManager."""

    def test_initialization(self):
        manager = ReproducibilityManager(seed=42)
        assert manager is not None
        assert manager.seed == 42

    def test_set_all_seeds(self):
        manager = ReproducibilityManager(seed=42)
        manager.set_all_seeds()
        val1 = np.random.randn()
        manager.set_all_seeds()
        val2 = np.random.randn()
        # After setting the same seed, the first random value should be the same
        assert val1 == val2

    def test_reproducible_scope(self):
        manager = ReproducibilityManager(seed=42)
        with manager.reproducible_scope():
            val1 = np.random.randn()
        with manager.reproducible_scope():
            val2 = np.random.randn()
        assert val1 == val2

    def test_reset(self):
        manager = ReproducibilityManager(seed=42)
        manager.set_all_seeds()
        val1 = np.random.randn()
        manager.reset()
        manager.set_all_seeds()
        val2 = np.random.randn()
        assert val1 == val2


class TestStructuredLogger:
    """Tests for StructuredLogger."""

    def test_initialization(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredLogger(log_dir=tmpdir)
            assert logger is not None

    def test_log_levels(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = StructuredLogger(log_dir=tmpdir)
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            # Should not raise
