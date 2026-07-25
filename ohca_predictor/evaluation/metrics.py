"""
Comprehensive evaluation metrics for the OHCA Predictor.

Provides bootstrap confidence intervals for all clinical metrics,
decision curve analysis, and expected calibration error computation.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ohca_predictor.config import EvaluationConfig


def bootstrap_confidence_interval(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iterations: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute a bootstrap confidence interval for an arbitrary metric.

    Algorithm:
        1. Draw ``n_iterations`` samples of the same size as the input arrays,
           with replacement.
        2. Evaluate ``metric_fn`` on each bootstrap sample.
        3. Sort the bootstrap statistics and extract the ``alpha/2`` and
           ``1 - alpha/2`` percentiles as the CI bounds.

    Args:
        metric_fn: A callable ``(y_true, y_pred) -> scalar``.
        y_true: Ground-truth binary labels.
        y_pred: Predicted probabilities or scores.
        n_iterations: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for 95 % CI).

    Returns:
        Tuple of ``(point_estimate, ci_lower, ci_upper)`` where
        ``point_estimate`` is computed on the full dataset.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    n = len(y_true)

    point = metric_fn(y_true, y_pred)

    rng = np.random.RandomState(42)
    boot_stats = np.empty(n_iterations, dtype=np.float64)
    for i in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            boot_stats[i] = point
        else:
            boot_stats[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_stats, 100.0 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha / 2)))
    return float(point), ci_lower, ci_upper


class OHCAEvaluator:
    """Full evaluation suite for OHCA prediction with bootstrap CIs.

    Every metric method returns ``(point, ci_lower, ci_upper)`` so that
    uncertainty is always reported alongside the point estimate.

    Args:
        config: An ``EvaluationConfig`` dataclass controlling bootstrap
            iterations, confidence level, and clinical thresholds.
    """

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self._n_iter = config.bootstrap_iterations
        self._conf = config.confidence_level

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bootstrap(
        self, metric_fn: Callable[[np.ndarray, np.ndarray], float],
        y_true: np.ndarray, y_pred: np.ndarray,
    ) -> Tuple[float, float, float]:
        return bootstrap_confidence_interval(
            metric_fn, y_true, y_pred,
            n_iterations=self._n_iter, confidence=self._conf,
        )

    @staticmethod
    def _to_binary(y_pred: np.ndarray, threshold: float) -> np.ndarray:
        return (np.asarray(y_pred).ravel() >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Discrimination metrics
    # ------------------------------------------------------------------

    def compute_auroc(
        self, y_true: np.ndarray, y_pred: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Area under the ROC curve with bootstrap CI."""
        def _fn(yt, yp):
            return roc_auc_score(yt, yp)
        return self._bootstrap(_fn, np.asarray(y_true).ravel(),
                               np.asarray(y_pred).ravel())

    def compute_auprc(
        self, y_true: np.ndarray, y_pred: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Area under the precision–recall curve with bootstrap CI."""
        def _fn(yt, yp):
            return average_precision_score(yt, yp)
        return self._bootstrap(_fn, np.asarray(y_true).ravel(),
                               np.asarray(y_pred).ravel())

    # ------------------------------------------------------------------
    # Threshold-dependent metrics
    # ------------------------------------------------------------------

    def compute_sensitivity_at_threshold(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float,
    ) -> Tuple[float, float, float]:
        """Sensitivity (recall / true positive rate) at a fixed threshold."""
        yt = np.asarray(y_true).ravel()
        yp = np.asarray(y_pred).ravel()

        def _fn(yt, yp):
            return recall_score(yt, (yp >= threshold).astype(int),
                                zero_division=0.0)
        return self._bootstrap(_fn, yt, yp)

    def compute_specificity_at_threshold(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float,
    ) -> Tuple[float, float, float]:
        """Specificity (true negative rate) at a fixed threshold."""
        yt = np.asarray(y_true).ravel()
        yp = np.asarray(y_pred).ravel()

        def _fn(yt, yp):
            pred_pos = (yp >= threshold).astype(int)
            neg_mask = yt == 0
            if neg_mask.sum() == 0:
                return 0.0
            return float(np.mean(pred_pos[neg_mask] == 0))
        return self._bootstrap(_fn, yt, yp)

    def compute_ppv(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float,
    ) -> Tuple[float, float, float]:
        """Positive predictive value (precision) at a fixed threshold."""
        yt = np.asarray(y_true).ravel()
        yp = np.asarray(y_pred).ravel()

        def _fn(yt, yp):
            return precision_score(yt, (yp >= threshold).astype(int),
                                   zero_division=0.0)
        return self._bootstrap(_fn, yt, yp)

    def compute_npv(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float,
    ) -> Tuple[float, float, float]:
        """Negative predictive value at a fixed threshold."""
        yt = np.asarray(y_true).ravel()
        yp = np.asarray(y_pred).ravel()

        def _fn(yt, yp):
            pred_neg = (yp < threshold).astype(int)
            neg_pred_mask = pred_neg == 1
            if neg_pred_mask.sum() == 0:
                return 0.0
            return float(np.mean(yt[neg_pred_mask] == 0))
        return self._bootstrap(_fn, yt, yp)

    def compute_f1_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float,
    ) -> Tuple[float, float, float]:
        """F1 score at a fixed threshold."""
        yt = np.asarray(y_true).ravel()
        yp = np.asarray(y_pred).ravel()

        def _fn(yt, yp):
            return f1_score(yt, (yp >= threshold).astype(int),
                            zero_division=0.0)
        return self._bootstrap(_fn, yt, yp)

    # ------------------------------------------------------------------
    # Calibration-oriented metrics
    # ------------------------------------------------------------------

    def compute_brier_score(
        self, y_true: np.ndarray, y_pred: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Brier score (mean squared error of probabilistic predictions)."""
        yt = np.asarray(y_true, dtype=np.float64).ravel()
        yp = np.asarray(y_pred, dtype=np.float64).ravel()

        def _fn(yt, yp):
            return float(np.mean((yp - yt) ** 2))
        return self._bootstrap(_fn, yt, yp)

    # ------------------------------------------------------------------
    # Aggregate report
    # ------------------------------------------------------------------

    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.20,
    ) -> Dict[str, Tuple[float, float, float]]:
        """Compute the full metric suite and return a dict of results.

        Args:
            y_true: Binary ground-truth labels.
            y_pred: Predicted probabilities.
            threshold: Classification threshold (default 0.20).

        Returns:
            Dictionary mapping metric names to ``(point, ci_lower, ci_upper)``
            tuples.
        """
        results: Dict[str, Tuple[float, float, float]] = {}
        results["auroc"] = self.compute_auroc(y_true, y_pred)
        results["auprc"] = self.compute_auprc(y_true, y_pred)
        results["sensitivity"] = self.compute_sensitivity_at_threshold(
            y_true, y_pred, threshold,
        )
        results["specificity"] = self.compute_specificity_at_threshold(
            y_true, y_pred, threshold,
        )
        results["ppv"] = self.compute_ppv(y_true, y_pred, threshold)
        results["npv"] = self.compute_npv(y_true, y_pred, threshold)
        results["f1"] = self.compute_f1_score(y_true, y_pred, threshold)
        results["brier"] = self.compute_brier_score(y_true, y_pred)
        results["threshold"] = (threshold, threshold, threshold)
        return results

    def print_report(
        self, metrics: Dict[str, Tuple[float, float, float]],
    ) -> str:
        """Render a human-readable evaluation report.

        Args:
            metrics: Dictionary returned by :meth:`compute_all_metrics`.

        Returns:
            Formatted multi-line string.
        """
        ci_pct = int(100 * self.config.confidence_level)
        lines = [
            f"{'='*60}",
            f"  OHCA Predictor Evaluation Report ({ci_pct}% CI)",
            f"{'='*60}",
            "",
        ]
        header = f"  {'Metric':<25} {'Point':>8} {'CI Lower':>8} {'CI Upper':>8}"
        lines.append(header)
        lines.append(f"  {'-'*53}")
        for name, (point, lo, hi) in metrics.items():
            if name == "threshold":
                lines.append(
                    f"  {name:<25} {point:>8.4f} {'':>8} {'':>8}"
                )
            else:
                lines.append(
                    f"  {name:<25} {point:>8.4f} {lo:>8.4f} {hi:>8.4f}"
                )
        lines.append(f"{'='*60}")
        report = "\n".join(lines)
        return report


# ------------------------------------------------------------------
# Module-level convenience functions
# ------------------------------------------------------------------

def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    ECE = sum_{b=1}^{B} (n_b / N) * |accuracy(b) - confidence(b)|

    Predictions are sorted into ``n_bins`` equal-width bins by predicted
    probability.  For each bin, the absolute difference between the
    fraction of true positives and the mean predicted probability is
    weighted by the proportion of samples in that bin.

    Args:
        y_true: Binary ground-truth labels.
        y_pred: Predicted probabilities.
        n_bins: Number of calibration bins.

    Returns:
        Scalar ECE value in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    n = len(y_true)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
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
        ece += (n_b / n) * abs(accuracy_b - confidence_b)
    return float(ece)


def decision_curve_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Decision Curve Analysis (DCA).

    Net Benefit at threshold *t* is defined as::

        NB(t) = TP/N - (FP/N) * (t / (1 - t))

    The function also computes the net benefit of three reference
    strategies:

    * **Treat All** – classify every patient as positive.
    * **Treat None** – classify every patient as negative (NB = 0).
    * **Perfect prediction** – the oracle upper bound.

    Args:
        y_true: Binary ground-truth labels (1 = OHCA, 0 = no OHCA).
        y_pred: Predicted probabilities.
        thresholds: Array of threshold values to evaluate.  Defaults to
            ``np.linspace(0.01, 0.99, 100)``.

    Returns:
        Dictionary with keys ``"thresholds"``, ``"net_benefit"``,
        ``"treat_all"``, ``"treat_none"``, ``"perfect"`` – each a
        1-D numpy array.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    n = len(y_true)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)

    thresholds = np.asarray(thresholds, dtype=np.float64)
    prevalence = y_true.mean()

    net_benefit = np.zeros_like(thresholds)
    treat_all = np.zeros_like(thresholds)
    perfect = np.zeros_like(thresholds)

    for idx, t in enumerate(thresholds):
        pred_pos = (y_pred >= t).astype(np.float64)
        tp = ((pred_pos == 1) & (y_true == 1)).sum()
        fp = ((pred_pos == 1) & (y_true == 0)).sum()
        if t >= 1.0:
            net_benefit[idx] = 0.0
        else:
            net_benefit[idx] = (tp / n) - (fp / n) * (t / (1.0 - t))

        treat_all[idx] = prevalence - (1.0 - prevalence) * (
            t / (1.0 - t) if t < 1.0 else 0.0
        )
        perfect[idx] = prevalence

    return {
        "thresholds": thresholds,
        "net_benefit": net_benefit,
        "treat_all": treat_all,
        "treat_none": np.zeros_like(thresholds),
        "perfect": perfect,
    }
