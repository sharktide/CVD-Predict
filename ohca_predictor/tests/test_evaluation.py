"""Tests for the OHCA Evaluation module."""

import pytest
import numpy as np
from ohca_predictor.config import get_config
from ohca_predictor.evaluation.metrics import (
    bootstrap_confidence_interval,
    compute_expected_calibration_error,
    decision_curve_analysis,
    OHCAEvaluator,
)
from ohca_predictor.evaluation.calibration import CalibrationAnalyzer


class TestBootstrapConfidenceInterval:
    """Tests for bootstrap confidence interval computation."""

    def test_basic_bootstrap(self):
        data = np.random.randn(100)
        ci = bootstrap_confidence_interval(
            lambda y, p: np.mean(p),
            np.ones(100),
            data,
            n_iterations=1000,
            confidence=0.95,
        )
        assert len(ci) == 3
        assert ci[1] <= ci[0] <= ci[2]

    def test_bootstrap_confidence_bounds(self):
        data = np.random.randn(200)
        point, lo, hi = bootstrap_confidence_interval(
            lambda y, p: np.mean(p),
            np.ones(200),
            data,
            n_iterations=500,
            confidence=0.95,
        )
        assert lo <= point <= hi


class TestCalibrationMetrics:
    """Tests for calibration metrics."""

    def test_expected_calibration_error_range(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5])
        ece = compute_expected_calibration_error(y_true, y_prob, n_bins=10)
        assert 0 <= ece <= 1

    def test_perfect_calibration(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([1.0, 0.0, 1.0, 0.0])
        ece = compute_expected_calibration_error(y_true, y_prob, n_bins=4)
        assert ece < 0.01

    def test_worst_calibration(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])
        ece = compute_expected_calibration_error(y_true, y_prob, n_bins=4)
        assert ece > 0.5


class TestDecisionCurveAnalysis:
    """Tests for decision curve analysis."""

    def test_basic_dca(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
        result = decision_curve_analysis(y_true, y_prob)
        assert "thresholds" in result
        assert "net_benefit" in result
        assert "treat_all" in result
        assert "treat_none" in result
        assert len(result["net_benefit"]) == len(result["thresholds"])


class TestOHCAEvaluator:
    """Tests for OHCAEvaluator."""

    def test_initialization(self):
        config = get_config()
        evaluator = OHCAEvaluator(config.evaluation)
        assert evaluator is not None

    def test_auroc(self):
        config = get_config()
        evaluator = OHCAEvaluator(config.evaluation)
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        point, lo, hi = evaluator.compute_auroc(y_true, y_pred)
        assert 0.9 <= point <= 1.0
        assert lo <= point <= hi

    def test_auprc(self):
        config = get_config()
        evaluator = OHCAEvaluator(config.evaluation)
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        point, lo, hi = evaluator.compute_auprc(y_true, y_pred)
        assert point >= 0
        assert lo <= point <= hi

    def test_sensitivity_at_threshold(self):
        config = get_config()
        evaluator = OHCAEvaluator(config.evaluation)
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        point, lo, hi = evaluator.compute_sensitivity_at_threshold(y_true, y_pred, threshold=0.5)
        assert 0 <= point <= 1

    def test_specificity_at_threshold(self):
        config = get_config()
        evaluator = OHCAEvaluator(config.evaluation)
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        point, lo, hi = evaluator.compute_specificity_at_threshold(y_true, y_pred, threshold=0.5)
        assert 0 <= point <= 1

    def test_ppv(self):
        config = get_config()
        evaluator = OHCAEvaluator(config.evaluation)
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.9, 0.8, 0.2, 0.1])
        point, lo, hi = evaluator.compute_ppv(y_true, y_pred, threshold=0.5)
        assert 0 <= point <= 1

    def test_npv(self):
        config = get_config()
        evaluator = OHCAEvaluator(config.evaluation)
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.9, 0.8, 0.2, 0.1])
        point, lo, hi = evaluator.compute_npv(y_true, y_pred, threshold=0.5)
        assert 0 <= point <= 1

    def test_brier_score(self):
        config = get_config()
        evaluator = OHCAEvaluator(config.evaluation)
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0.9, 0.1, 0.8, 0.2])
        point, lo, hi = evaluator.compute_brier_score(y_true, y_pred)
        assert 0 <= point <= 1


class TestCalibrationAnalyzer:
    """Tests for calibration analysis."""

    def test_initialization(self):
        config = get_config()
        analyzer = CalibrationAnalyzer(config.evaluation)
        assert analyzer is not None

    def test_reliability_diagram(self):
        config = get_config()
        analyzer = CalibrationAnalyzer(config.evaluation)
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5])
        result = analyzer.reliability_diagram(y_true, y_prob, n_bins=5)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_temperature_scaling(self):
        config = get_config()
        analyzer = CalibrationAnalyzer(config.evaluation)
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        calibrated, temperature = analyzer.temperature_scaling(y_true, y_prob, y_prob)
        assert temperature > 0

    def test_platt_scaling(self):
        config = get_config()
        analyzer = CalibrationAnalyzer(config.evaluation)
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        calibrated, a, b = analyzer.platt_scaling(y_true, y_prob, y_prob)
        assert isinstance(a, float)
        assert isinstance(b, float)
