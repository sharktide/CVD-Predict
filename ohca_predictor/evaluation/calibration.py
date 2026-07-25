"""
Calibration analysis for the OHCA Predictor.

Provides reliability diagrams, post-hoc calibration methods (temperature
scaling, Platt scaling, isotonic regression), and per-subgroup calibration
assessment.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from ohca_predictor.config import EvaluationConfig


class CalibrationAnalyzer:
    """Post-hoc calibration analysis and recalibration utilities.

    Args:
        config: An ``EvaluationConfig`` dataclass controlling the number
            of calibration bins.
    """

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self.n_bins = config.calibration_bins

    # ------------------------------------------------------------------
    # Reliability diagram
    # ------------------------------------------------------------------

    def reliability_diagram(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct a reliability diagram.

        Partitions predictions into equal-width bins by predicted
        probability and returns bin edges, observed (empirical) frequencies,
        and expected (mean predicted) frequencies for each bin.

        Args:
            y_true: Binary ground-truth labels.
            y_pred: Predicted probabilities.
            n_bins: Number of bins.  Defaults to ``self.n_bins``.

        Returns:
            Tuple of ``(bin_edges, observed_freq, expected_freq)`` where
            ``bin_edges`` has length ``n_bins + 1`` and the frequency
            arrays have length ``n_bins``.
        """
        if n_bins is None:
            n_bins = self.n_bins

        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        observed_freq = np.zeros(n_bins, dtype=np.float64)
        expected_freq = np.zeros(n_bins, dtype=np.float64)

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (y_pred >= lo) & (y_pred <= hi)
            else:
                mask = (y_pred >= lo) & (y_pred < hi)
            if mask.sum() == 0:
                continue
            observed_freq[i] = float(y_true[mask].mean())
            expected_freq[i] = float(y_pred[mask].mean())

        return bin_edges, observed_freq, expected_freq

    # ------------------------------------------------------------------
    # Temperature scaling
    # ------------------------------------------------------------------

    def temperature_scaling(
        self,
        y_true: np.ndarray,
        y_pred_val: np.ndarray,
        y_pred_test: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Temperature scaling calibration.

        Learns a single scalar temperature *T* on the validation set by
        minimising the negative log-likelihood of the binary classification
        problem after dividing the logits by *T*.  The learned temperature
        is then applied to the test set.

        Internally the predicted probabilities are converted to logits
        via the inverse sigmoid: ``logit(p) = log(p / (1 - p))``.

        Args:
            y_true: Ground-truth labels for the **validation** set.
            y_pred_val: Predicted probabilities on the **validation** set.
            y_pred_test: Predicted probabilities on the **test** set.

        Returns:
            Tuple of ``(calibrated_test_probs, temperature)``.
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred_val = np.asarray(y_pred_val, dtype=np.float64).ravel()
        y_pred_test = np.asarray(y_pred_test, dtype=np.float64).ravel()

        eps = 1e-7
        logits_val = np.log(
            np.clip(y_pred_val, eps, 1.0 - eps)
            / np.clip(1.0 - y_pred_val, eps, 1.0 - eps)
        )
        logits_test = np.log(
            np.clip(y_pred_test, eps, 1.0 - eps)
            / np.clip(1.0 - y_pred_test, eps, 1.0 - eps)
        )

        def _nll(T_arr):
            T = T_arr[0]
            if T <= 0:
                return 1e12
            scaled = logits_val / T
            probs = expit(scaled)
            probs = np.clip(probs, eps, 1.0 - eps)
            nll = -np.mean(
                y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs)
            )
            return nll

        result = minimize(_nll, x0=[1.0], method="Nelder-Mead",
                          options={"xatol": 1e-8, "maxiter": 1000})
        temperature = float(result.x[0])

        calibrated_test = expit(logits_test / temperature)
        return calibrated_test, temperature

    # ------------------------------------------------------------------
    # Platt scaling
    # ------------------------------------------------------------------

    def platt_scaling(
        self,
        y_true: np.ndarray,
        y_pred_val: np.ndarray,
        y_pred_test: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """Platt scaling (logistic recalibration).

        Learns parameters *a* and *b* on the validation set such that::

            calibrated = sigmoid(a * logit(p) + b)

        The optimisation minimises the negative log-likelihood of the
        binary targets.  The learned parameters are applied to the test
        predictions.

        Args:
            y_true: Ground-truth labels for the **validation** set.
            y_pred_val: Predicted probabilities on the **validation** set.
            y_pred_test: Predicted probabilities on the **test** set.

        Returns:
            Tuple of ``(calibrated_test_probs, a, b)``.
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred_val = np.asarray(y_pred_val, dtype=np.float64).ravel()
        y_pred_test = np.asarray(y_pred_test, dtype=np.float64).ravel()

        eps = 1e-7
        logits_val = np.log(
            np.clip(y_pred_val, eps, 1.0 - eps)
            / np.clip(1.0 - y_pred_val, eps, 1.0 - eps)
        )
        logits_test = np.log(
            np.clip(y_pred_test, eps, 1.0 - eps)
            / np.clip(1.0 - y_pred_test, eps, 1.0 - eps)
        )

        def _nll(params):
            a, b = params
            scaled = a * logits_val + b
            probs = expit(scaled)
            probs = np.clip(probs, eps, 1.0 - eps)
            return -np.mean(
                y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs)
            )

        result = minimize(_nll, x0=[1.0, 0.0], method="Nelder-Mead",
                          options={"xatol": 1e-8, "maxiter": 1000})
        a, b = float(result.x[0]), float(result.x[1])

        calibrated_test = expit(a * logits_test + b)
        return calibrated_test, a, b

    # ------------------------------------------------------------------
    # Isotonic regression calibration
    # ------------------------------------------------------------------

    def isotonic_regression_calibration(
        self,
        y_true: np.ndarray,
        y_pred_val: np.ndarray,
        y_pred_test: np.ndarray,
    ) -> np.ndarray:
        """Isotonic regression calibration.

        Fits a non-parametric monotonic mapping from predicted
        probabilities to calibrated probabilities on the validation set,
        then applies it to the test set.

        Args:
            y_true: Ground-truth labels for the **validation** set.
            y_pred_val: Predicted probabilities on the **validation** set.
            y_pred_test: Predicted probabilities on the **test** set.

        Returns:
            Calibrated test-set probabilities as a 1-D numpy array.
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred_val = np.asarray(y_pred_val, dtype=np.float64).ravel()
        y_pred_test = np.asarray(y_pred_test, dtype=np.float64).ravel()

        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir.fit(y_pred_val, y_true)
        return ir.predict(y_pred_test)

    # ------------------------------------------------------------------
    # Aggregate calibration metrics
    # ------------------------------------------------------------------

    def compute_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute a suite of calibration metrics.

        Returns:
            Dictionary with keys:

            * ``"ece"`` – Expected Calibration Error
            * ``"mce"`` – Maximum Calibration Error
            * ``"brier"`` – Brier score
            * ``"avg_calibration_error"`` – alias for ECE
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

        n = len(y_true)
        if n == 0:
            return {"ece": 0.0, "mce": 0.0, "brier": 0.0,
                    "avg_calibration_error": 0.0}

        n_bins = self.n_bins
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

        ece = 0.0
        mce = 0.0
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (y_pred >= lo) & (y_pred <= hi)
            else:
                mask = (y_pred >= lo) & (y_pred < hi)
            n_b = mask.sum()
            if n_b == 0:
                continue
            accuracy_b = float(y_true[mask].mean())
            confidence_b = float(y_pred[mask].mean())
            ace_b = abs(accuracy_b - confidence_b)
            ece += (n_b / n) * ace_b
            mce = max(mce, ace_b)

        brier = float(np.mean((y_pred - y_true) ** 2))

        return {
            "ece": float(ece),
            "mce": float(mce),
            "brier": brier,
            "avg_calibration_error": float(ece),
        }

    # ------------------------------------------------------------------
    # Per-subgroup calibration
    # ------------------------------------------------------------------

    def calibrate_by_subgroup(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        subgroup_labels: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate calibration separately for each subgroup.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted probabilities.
            subgroup_labels: Array of the same length as ``y_true`` where
                each element identifies the subgroup (e.g. age group,
                sex, ethnicity).

        Returns:
            Dictionary mapping subgroup identifiers to their calibration
            metric dictionaries (keys ``"ece"``, ``"mce"``, ``"brier"``).
        """
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        subgroup_labels = np.asarray(subgroup_labels).ravel()

        results: Dict[str, Dict[str, float]] = {}
        unique_groups = np.unique(subgroup_labels)
        for grp in unique_groups:
            mask = subgroup_labels == grp
            results[str(grp)] = self.compute_calibration_metrics(
                y_true[mask], y_pred[mask],
            )
        return results
