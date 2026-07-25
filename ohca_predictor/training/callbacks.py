"""Custom Keras callbacks for OHCA Predictor training.

Implements early stopping, checkpointing, TensorBoard logging, metrics
computation, NaN monitoring, and signal quality monitoring tailored to
the OHCA prediction task with clinical metrics (AUROC, sensitivity,
specificity) and physiological signal constraints.
"""

from __future__ import annotations

import glob
import logging
import math
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    raise ImportError("TensorFlow 2.x is required for ohca_predictor.training.callbacks")

try:
    from sklearn.metrics import (
        confusion_matrix,
        roc_curve,
        auc,
        precision_recall_curve,
        average_precision_score,
        roc_auc_score,
    )

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A. OHCAEarlyStopping
# ---------------------------------------------------------------------------

class OHCAEarlyStopping(keras.callbacks.Callback):
    """Early stopping based on OHCA AUROC with best-weight restoration.

    Monitors a configurable metric (default ``val_ohca_auroc``) and stops
    training when the metric has not improved for ``patience`` epochs.

    Args:
        monitor: Metric name to monitor.
        patience: Number of epochs with no improvement before stopping.
        mode: One of ``"max"`` or ``"min"``.  For AUROC use ``"max"``.
        restore_best_weights: Whether to restore model weights from the
            best epoch upon stopping.
        min_delta: Minimum change in the monitored metric to qualify as
            an improvement.
        verbose: Verbosity mode.  0 = silent, 1 = log on improvement.
    """

    def __init__(
        self,
        monitor: str = "val_ohca_auroc",
        patience: int = 15,
        mode: str = "max",
        restore_best_weights: bool = True,
        min_delta: float = 1e-5,
        verbose: int = 1,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.min_delta = min_delta
        self.verbose = verbose

        if mode == "max":
            self._best = -np.inf
            self._is_better = lambda cur, best: cur > best + min_delta
        elif mode == "min":
            self._best = np.inf
            self._is_better = lambda cur, best: cur < best - min_delta
        else:
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")

        self.wait = 0
        self.best_epoch = 0
        self.best_weights: Optional[List[np.ndarray]] = None
        self.stopped_epoch = 0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.wait = 0
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            logger.warning(
                "EarlyStopping: metric '%s' not found in epoch logs. "
                "Available keys: %s. Skipping check.",
                self.monitor,
                sorted(logs.keys()),
            )
            return

        current = float(current)
        if self._is_better(current, self._best):
            self._best = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = [
                    np.array(w) for w in self.model.get_weights()
                ]
            if self.verbose >= 1:
                logger.info(
                    "Epoch %d: %s improved to %.6f (best so far)",
                    epoch,
                    self.monitor,
                    current,
                )
        else:
            self.wait += 1
            if self.verbose >= 1:
                logger.info(
                    "Epoch %d: %s did not improve (%.6f vs best %.6f). "
                    "Wait %d/%d.",
                    epoch,
                    self.monitor,
                    current,
                    self._best,
                    self.wait,
                    self.patience,
                )
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                    if self.verbose >= 1:
                        logger.info(
                            "Restored model weights from epoch %d (best %s=%.6f)",
                            self.best_epoch,
                            self.monitor,
                            self._best,
                        )

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if self.stopped_epoch > 0:
            logger.info(
                "Training stopped early at epoch %d. "
                "Best %s=%.6f at epoch %d.",
                self.stopped_epoch,
                self.monitor,
                self._best,
                self.best_epoch,
            )


# ---------------------------------------------------------------------------
# B. OHCACheckpoint
# ---------------------------------------------------------------------------

class OHCACheckpoint(keras.callbacks.Callback):
    """Checkpoint manager: saves best model by AUROC, periodic saves,
    keeps only the top K checkpoints.

    Checkpoints are saved as TF SavedModel directories containing both
    the full model and separate weights.

    Args:
        save_dir: Base directory for checkpoints.
        monitor: Metric name to determine the best checkpoint.
        mode: ``"max"`` or ``"min"`` for the monitored metric.
        save_best_only: If ``True``, only overwrite the best checkpoint.
        save_every_n_epochs: Save a checkpoint every N epochs regardless
            of metric value.  Set to 0 or ``None`` to disable.
        keep_top_k: Maximum number of checkpoints to retain on disk.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        monitor: str = "val_ohca_auroc",
        mode: str = "max",
        save_best_only: bool = True,
        save_every_n_epochs: int = 5,
        keep_top_k: int = 3,
        verbose: int = 1,
    ) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_every_n_epochs = save_every_n_epochs
        self.keep_top_k = keep_top_k
        self.verbose = verbose

        if mode == "max":
            self._best = -np.inf
            self._is_better = lambda cur, best: cur > best
        elif mode == "min":
            self._best = np.inf
            self._is_better = lambda cur, best: cur < best
        else:
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")

        self._checkpoints: List[Tuple[float, str]] = []

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        self._best = -np.inf if self.mode == "max" else np.inf
        self._checkpoints = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        should_save = False

        current = logs.get(self.monitor)
        if current is not None:
            current = float(current)
            if self._is_better(current, self._best):
                self._best = current
                should_save = True
                if self.verbose >= 1:
                    logger.info(
                        "Checkpoint: %s improved to %.6f",
                        self.monitor,
                        current,
                    )

        if (
            self.save_every_n_epochs is not None
            and self.save_every_n_epochs > 0
            and (epoch + 1) % self.save_every_n_epochs == 0
        ):
            should_save = True

        if not should_save:
            return

        ckpt_name = f"epoch_{epoch + 1:04d}_{self.monitor}_{current:.6f}" if current is not None else f"epoch_{epoch + 1:04d}"
        ckpt_path = os.path.join(self.save_dir, ckpt_name)

        # Save as SavedModel
        model_path = os.path.join(ckpt_path, "saved_model")
        weights_path = os.path.join(ckpt_path, "weights.weights.h5")

        try:
            self.model.save(model_path)
            self.model.save_weights(weights_path)
        except Exception as exc:
            logger.error("Failed to save checkpoint to %s: %s", ckpt_path, exc)
            return

        metric_val = current if current is not None else float("-inf" if self.mode == "max" else "inf")
        self._checkpoints.append((metric_val, ckpt_path))

        if self.verbose >= 1:
            logger.info("Saved checkpoint: %s", ckpt_path)

        self._prune_checkpoints()

    def _prune_checkpoints(self) -> None:
        """Remove all but the top-K checkpoints."""
        if len(self._checkpoints) <= self.keep_top_k:
            return

        if self.mode == "max":
            self._checkpoints.sort(key=lambda x: x[0], reverse=True)
        else:
            self._checkpoints.sort(key=lambda x: x[0])

        to_remove = self._checkpoints[self.keep_top_k :]
        self._checkpoints = self._checkpoints[: self.keep_top_k]

        for _, path in to_remove:
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                    if self.verbose >= 1:
                        logger.info("Pruned old checkpoint: %s", path)
                except OSError as exc:
                    logger.warning(
                        "Failed to remove old checkpoint %s: %s", path, exc
                    )


# ---------------------------------------------------------------------------
# C. OHCATensorBoard
# ---------------------------------------------------------------------------

class OHCATensorBoard(keras.callbacks.TensorBoard):
    """Extended TensorBoard callback for OHCA training.

    Inherits from ``keras.callbacks.TensorBoard`` and adds:
    - Custom histogram logging every N epochs.
    - Gradient distribution logging.
    - Learning rate schedule logging.
    - Custom clinical scalars (AUROC, sensitivity, specificity).
    - Built-in profiling.

    Args:
        log_dir: Base log directory for TensorBoard.
        histogram_freq: Frequency (in epochs) at which to compute
            and write weight histograms.  Default 5.
        write_graph: Whether to visualize the computation graph.
        write_images: Whether to write model weights as images.
        update_freq: ``"batch"``, ``"epoch"``, or an integer.
        profile_batch: Batch(es) to profile.  ``100`` profiles
            batch 100.  Tuple ``(start, end)`` for a range.
        embeddings_freq: Frequency for embedding visualization.
        log_dir_gradients: If ``True``, also log gradient distributions.
        **kwargs: Additional arguments passed to the parent class.
    """

    def __init__(
        self,
        log_dir: str = "logs/tensorboard",
        histogram_freq: int = 5,
        write_graph: bool = True,
        write_images: bool = True,
        update_freq: str = "epoch",
        profile_batch: int = 100,
        embeddings_freq: int = 0,
        log_dir_gradients: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            log_dir=log_dir,
            histogram_freq=histogram_freq,
            write_graph=write_graph,
            write_images=write_images,
            update_freq=update_freq,
            profile_batch=profile_batch,
            embeddings_freq=embeddings_freq,
            **kwargs,
        )
        self.log_dir_gradients = log_dir_gradients
        self._gradient_writer: Optional[tf.summary.SummaryWriter] = None

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        super().on_train_begin(logs)
        if self.log_dir_gradients:
            grad_dir = os.path.join(self.log_dir, "gradients")
            os.makedirs(grad_dir, exist_ok=True)
            self._gradient_writer = tf.summary.create_file_writer(grad_dir)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        super().on_epoch_end(epoch, logs)
        logs = logs or {}

        if self.histogram_freq > 0 and (epoch + 1) % self.histogram_freq == 0:
            self._log_histograms(epoch)
            if self.log_dir_gradients:
                self._log_gradients(epoch)

        self._log_clinical_scalars(epoch, logs)
        self._log_learning_rate(epoch)

    def _log_histograms(self, epoch: int) -> None:
        """Write weight histograms to TensorBoard."""
        if not hasattr(self, "model") or self.model is None:
            return
        writer = getattr(self, "summary_writer", None)
        if writer is None:
            return

        with writer.as_default():
            for layer in self.model.layers:
                weights = layer.get_weights()
                for i, w in enumerate(weights):
                    if w.size == 0:
                        continue
                    tag = f"weights/{layer.name}/var_{i}"
                    tf.summary.histogram(
                        tag, w, step=epoch, description=tag
                    )

    def _log_gradients(self, epoch: int) -> None:
        """Log gradient distributions for all trainable variables."""
        if self._gradient_writer is None:
            return
        if not hasattr(self.model, "trainable_variables"):
            return

        grads = None
        # Attempt to get the last computed gradients from the optimizer
        if hasattr(self.model, "optimizer") and self.model.optimizer is not None:
            optimizer = self.model.optimizer
            if hasattr(optimizer, "gradients"):
                grads = optimizer.gradients

        if grads is None:
            return

        with self._gradient_writer.as_default():
            for i, grad in enumerate(grads):
                if grad is None:
                    continue
                grad_np = grad.numpy() if hasattr(grad, "numpy") else np.array(grad)
                if grad_np.size == 0:
                    continue
                tag = f"gradients/variable_{i}"
                tf.summary.histogram(tag, grad_np, step=epoch, description=tag)
                tf.summary.scalar(
                    f"gradients/variable_{i}_norm",
                    float(np.linalg.norm(grad_np.ravel())),
                    step=epoch,
                )

    def _log_learning_rate(self, epoch: int) -> None:
        """Log the current learning rate from the optimizer."""
        if not hasattr(self.model, "optimizer") or self.model.optimizer is None:
            return
        optimizer = self.model.optimizer
        lr = None
        if hasattr(optimizer, "learning_rate"):
            lr_val = optimizer.learning_rate
            if callable(lr_val):
                try:
                    lr = float(lr_val())
                except Exception:
                    lr = None
            else:
                lr = float(lr_val)

        if lr is None:
            return

        writer = getattr(self, "summary_writer", None)
        if writer is not None:
            with writer.as_default():
                tf.summary.scalar("learning_rate", lr, step=epoch)

    def _log_clinical_scalars(
        self, epoch: int, logs: Dict[str, Any]
    ) -> None:
        """Log clinical metrics: AUROC, sensitivity, specificity."""
        writer = getattr(self, "summary_writer", None)
        if writer is None:
            return

        metric_map = {
            "ohca_auroc": "clinical/auroc",
            "val_ohca_auroc": "clinical/val_auroc",
            "ohca_sensitivity": "clinical/sensitivity",
            "val_ohca_sensitivity": "clinical/val_sensitivity",
            "ohca_specificity": "clinical/specificity",
            "val_ohca_specificity": "clinical/val_specificity",
        }

        with writer.as_default():
            for key, tag in metric_map.items():
                value = logs.get(key)
                if value is not None:
                    tf.summary.scalar(tag, float(value), step=epoch)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        super().on_train_end(logs)
        if self._gradient_writer is not None:
            self._gradient_writer.close()
            self._gradient_writer = None


# ---------------------------------------------------------------------------
# D. LRLogger
# ---------------------------------------------------------------------------

class LRLogger(keras.callbacks.Callback):
    """Log the current learning rate to TensorBoard every epoch.

    Extracts the learning rate from the optimizer, handling both
    fixed learning rates and learning-rate schedules (e.g.,
    ``tf.keras.optimizers.schedules.LearningRateSchedule``).

    Args:
        log_dir: Directory for the TensorBoard summary writer.
    """

    def __init__(self, log_dir: str = "logs/tensorboard/lr") -> None:
        super().__init__()
        self.log_dir = log_dir
        self._writer: Optional[tf.summary.SummaryWriter] = None

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        self._writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if self._writer is None or not hasattr(self, "model") or self.model is None:
            return
        if not hasattr(self.model, "optimizer") or self.model.optimizer is None:
            return

        lr = self._get_learning_rate()
        if lr is not None:
            with self._writer.as_default():
                tf.summary.scalar("learning_rate", lr, step=epoch)

    def _get_learning_rate(self) -> Optional[float]:
        """Extract the current learning rate as a float."""
        optimizer = self.model.optimizer
        if optimizer is None:
            return None

        lr_attr = getattr(optimizer, "learning_rate", None)
        if lr_attr is None:
            return None

        if callable(lr_attr):
            try:
                lr_val = lr_attr()
                if hasattr(lr_val, "numpy"):
                    return float(lr_val.numpy())
                return float(lr_val)
            except Exception:
                return None
        elif hasattr(lr_attr, "numpy"):
            return float(lr_attr.numpy())
        else:
            try:
                return float(lr_attr)
            except (TypeError, ValueError):
                return None

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


# ---------------------------------------------------------------------------
# E. MetricsLogger
# ---------------------------------------------------------------------------

class MetricsLogger(keras.callbacks.Callback):
    """Compute and log all clinical metrics at the end of each epoch.

    Uses ``sklearn.metrics`` for computation and ``matplotlib`` for
    generating image summaries.  Metrics logged:

    - Confusion matrix
    - AUROC (with ROC curve image)
    - AUPRC (with PR curve image)
    - Sensitivity (recall for positive class)
    - Specificity (recall for negative class)
    - Precision, F1
    - Calibration curve

    Args:
        validation_data: Tuple ``(x_val, y_val)`` for evaluation, or
            ``None`` to skip validation-set metric computation.
        log_dir: Directory for TensorBoard image/summary logging.
        log_freq: Compute and log every N epochs.
        threshold: Decision threshold for binary metrics (default 0.20).
        num_calibration_bins: Number of bins for calibration curve.
    """

    def __init__(
        self,
        validation_data: Optional[Tuple[Any, Any]] = None,
        log_dir: str = "logs/tensorboard/metrics",
        log_freq: int = 1,
        threshold: float = 0.20,
        num_calibration_bins: int = 10,
    ) -> None:
        super().__init__()
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.threshold = threshold
        self.num_calibration_bins = num_calibration_bins
        self._writer: Optional[tf.summary.SummaryWriter] = None

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        self._writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if self._writer is None:
            return
        if (epoch + 1) % self.log_freq != 0:
            return
        if not _HAS_SKLEARN:
            return
        if self.validation_data is None:
            return

        x_val, y_val = self.validation_data
        if x_val is None or y_val is None:
            return

        try:
            y_prob = self.model.predict(x_val, verbose=0)
            if isinstance(y_prob, dict):
                y_prob = y_prob.get("ohca_risk", list(y_prob.values())[0])
            y_prob = np.asarray(y_prob).ravel()
        except Exception as exc:
            logger.warning("MetricsLogger: prediction failed: %s", exc)
            return

        y_true = np.asarray(y_val).ravel()
        y_pred = (y_prob >= self.threshold).astype(int)

        with self._writer.as_default():
            self._log_scalar_metrics(epoch, y_true, y_prob, y_pred)
            self._log_confusion_matrix(epoch, y_true, y_pred)
            if _HAS_MATPLOTLIB:
                self._log_roc_curve(epoch, y_true, y_prob)
                self._log_pr_curve(epoch, y_true, y_prob)
                self._log_calibration_curve(epoch, y_true, y_prob)

    def _log_scalar_metrics(
        self,
        epoch: int,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """Log scalar clinical metrics."""
        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos

        # AUROC
        try:
            auroc = roc_auc_score(y_true, y_prob)
            tf.summary.scalar("metrics/auroc", auroc, step=epoch)
        except ValueError:
            auroc = 0.0

        # AUPRC
        try:
            auprc = average_precision_score(y_true, y_prob)
            tf.summary.scalar("metrics/auprc", auprc, step=epoch)
        except ValueError:
            auprc = 0.0

        # Confusion matrix components
        if n_pos > 0 and n_neg > 0:
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            tn = int(np.sum((y_pred == 0) & (y_true == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (
                2 * precision * sensitivity / (precision + sensitivity)
                if (precision + sensitivity) > 0
                else 0.0
            )
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

            tf.summary.scalar("metrics/sensitivity", sensitivity, step=epoch)
            tf.summary.scalar("metrics/specificity", specificity, step=epoch)
            tf.summary.scalar("metrics/precision", precision, step=epoch)
            tf.summary.scalar("metrics/f1", f1, step=epoch)
            tf.summary.scalar("metrics/npv", npv, step=epoch)
            tf.summary.scalar("metrics/true_positives", float(tp), step=epoch)
            tf.summary.scalar("metrics/false_positives", float(fp), step=epoch)
            tf.summary.scalar("metrics/true_negatives", float(tn), step=epoch)
            tf.summary.scalar("metrics/false_negatives", float(fn), step=epoch)

            tf.summary.scalar(
                "metrics/positive_prevalence",
                float(n_pos) / float(len(y_true)),
                step=epoch,
            )

    def _log_confusion_matrix(
        self, epoch: int, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """Log confusion matrix as a text summary."""
        cm = confusion_matrix(y_true, y_pred)
        cm_str = np.array2string(cm, separator=", ")
        tf.summary.text(
            "metrics/confusion_matrix",
            f"Predicted  Neg  Pos\nActual Neg: {cm_str.split(chr(10))[0]}\nActual Pos: {cm_str.split(chr(10))[1] if len(cm_str.split(chr(10))) > 1 else ''}",
            step=epoch,
        )

    def _log_roc_curve(
        self, epoch: int, y_true: np.ndarray, y_prob: np.ndarray
    ) -> None:
        """Generate and log ROC curve as an image."""
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc_val = auc(fpr, tpr)
        except ValueError:
            return

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc_val:.4f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Chance")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — OHCA Prediction")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        buf = self._fig_to_image(fig)
        plt.close(fig)
        tf.summary.image("plots/roc_curve", buf, step=epoch)

    def _log_pr_curve(
        self, epoch: int, y_true: np.ndarray, y_prob: np.ndarray
    ) -> None:
        """Generate and log Precision-Recall curve as an image."""
        try:
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
        except ValueError:
            return

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(recall_vals, precision_vals, color="blue", lw=2, label=f"PR curve (AP = {ap:.4f})")
        baseline = float(np.sum(y_true == 1)) / float(len(y_true))
        ax.axhline(y=baseline, color="red", lw=1, linestyle="--", label=f"Baseline ({baseline:.3f})")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve — OHCA Prediction")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        buf = self._fig_to_image(fig)
        plt.close(fig)
        tf.summary.image("plots/pr_curve", buf, step=epoch)

    def _log_calibration_curve(
        self, epoch: int, y_true: np.ndarray, y_prob: np.ndarray
    ) -> None:
        """Generate and log calibration curve as an image."""
        bin_edges = np.linspace(0.0, 1.0, self.num_calibration_bins + 1)
        bin_centers = []
        bin_means = []
        bin_counts = []

        for i in range(self.num_calibration_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (y_prob >= lo) & (y_prob < hi)
            if i == self.num_calibration_bins - 1:
                mask = mask | (y_prob >= lo)
            count = int(mask.sum())
            if count > 0:
                bin_centers.append((lo + hi) / 2.0)
                bin_means.append(float(y_true[mask].mean()))
                bin_counts.append(count)

        if len(bin_centers) < 2:
            return

        bin_centers = np.array(bin_centers)
        bin_means = np.array(bin_means)
        bin_counts = np.array(bin_counts, dtype=float)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
        ax.plot(
            bin_centers,
            bin_means,
            "s-",
            color="steelblue",
            lw=2,
            markersize=8,
            label="Model",
        )
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration Curve — OHCA Prediction")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Add histogram of prediction distribution on secondary axis
        ax2 = ax.twinx()
        ax2.hist(
            y_prob,
            bins=bin_edges,
            alpha=0.25,
            color="gray",
            edgecolor="gray",
            label="Prediction distribution",
        )
        ax2.set_ylabel("Count")
        ax2.set_ylim([0, max(bin_counts.max() * 1.2, 1)])
        ax2.legend(loc="lower right")

        buf = self._fig_to_image(fig)
        plt.close(fig)
        tf.summary.image("plots/calibration_curve", buf, step=epoch)

    @staticmethod
    def _fig_to_image(fig: "plt.Figure") -> tf.Tensor:
        """Convert a matplotlib figure to a TensorBoard-compatible image."""
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf, dtype=np.uint8)
        img = img[:, :, :3]  # drop alpha channel
        img = np.expand_dims(img, axis=0)  # (1, H, W, 3)
        return tf.convert_to_tensor(img, dtype=tf.uint8)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


# ---------------------------------------------------------------------------
# F. NaNMonitor
# ---------------------------------------------------------------------------

class NaNMonitor(keras.callbacks.Callback):
    """Detect NaN/Inf in loss or gradients and reduce learning rate.

    If NaN or Inf is detected in the training loss or any gradient,
    the learning rate is halved.  If the loss remains NaN for
    ``max_nan_count`` consecutive epochs, training is stopped.

    Args:
        max_nan_count: Stop training after this many consecutive NaN epochs.
        reduce_lr_factor: Factor by which to reduce learning rate.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        max_nan_count: int = 5,
        reduce_lr_factor: float = 0.5,
        verbose: int = 1,
    ) -> None:
        super().__init__()
        self.max_nan_count = max_nan_count
        self.reduce_lr_factor = reduce_lr_factor
        self.verbose = verbose
        self._nan_count = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        has_nan = False

        # Check training loss
        loss = logs.get("loss")
        if loss is not None:
            loss_val = float(loss)
            if math.isnan(loss_val) or math.isinf(loss_val):
                has_nan = True
                if self.verbose >= 1:
                    logger.warning(
                        "NaN/Inf detected in training loss at epoch %d: %s",
                        epoch,
                        loss_val,
                    )

        # Check validation loss
        val_loss = logs.get("val_loss")
        if val_loss is not None:
            val_loss_val = float(val_loss)
            if math.isnan(val_loss_val) or math.isinf(val_loss_val):
                has_nan = True
                if self.verbose >= 1:
                    logger.warning(
                        "NaN/Inf detected in validation loss at epoch %d: %s",
                        epoch,
                        val_loss_val,
                    )

        # Check gradients if model is available
        if not has_nan and hasattr(self, "model") and self.model is not None:
            has_nan = self._check_gradients()

        if has_nan:
            self._nan_count += 1
            if self.verbose >= 1:
                logger.warning(
                    "NaN/Inf count: %d/%d (epoch %d)",
                    self._nan_count,
                    self.max_nan_count,
                    epoch,
                )
            self._reduce_learning_rate()
            if self._nan_count >= self.max_nan_count:
                self.model.stop_training = True
                logger.error(
                    "Training stopped: NaN/Inf persisted for %d consecutive epochs.",
                    self._nan_count,
                )
        else:
            self._nan_count = max(0, self._nan_count - 1)

    def _check_gradients(self) -> bool:
        """Scan trainable variables for NaN/Inf in their last computed gradient."""
        if not hasattr(self.model, "trainable_variables"):
            return False
        if not hasattr(self.model, "optimizer"):
            return False
        optimizer = self.model.optimizer
        if not hasattr(optimizer, "gradients"):
            return False

        grads = optimizer.gradients
        if grads is None:
            return False

        for grad in grads:
            if grad is None:
                continue
            grad_np = grad.numpy() if hasattr(grad, "numpy") else np.array(grad)
            if np.any(np.isnan(grad_np)) or np.any(np.isinf(grad_np)):
                return True
        return False

    def _reduce_learning_rate(self) -> None:
        """Reduce the optimizer's learning rate by the configured factor."""
        if not hasattr(self, "model") or self.model is None:
            return
        if not hasattr(self.model, "optimizer") or self.model.optimizer is None:
            return

        optimizer = self.model.optimizer
        lr_attr = getattr(optimizer, "learning_rate", None)
        if lr_attr is None:
            return

        current_lr = None
        if callable(lr_attr):
            try:
                current_lr = float(lr_attr().numpy() if hasattr(lr_attr(), "numpy") else float(lr_attr()))
            except Exception:
                return
        elif hasattr(lr_attr, "numpy"):
            current_lr = float(lr_attr.numpy())
        else:
            try:
                current_lr = float(lr_attr)
            except (TypeError, ValueError):
                return

        if current_lr is None or current_lr <= 0:
            return

        new_lr = current_lr * self.reduce_lr_factor
        try:
            optimizer.learning_rate.assign(new_lr)
            if self.verbose >= 1:
                logger.warning(
                    "Learning rate reduced from %.2e to %.2e (factor=%.2f)",
                    current_lr,
                    new_lr,
                    self.reduce_lr_factor,
                )
        except Exception as exc:
            logger.warning("Failed to reduce learning rate: %s", exc)


# ---------------------------------------------------------------------------
# G. SignalQualityMonitor
# ---------------------------------------------------------------------------

class SignalQualityMonitor(keras.callbacks.Callback):
    """Monitor input signal quality (SNR) and alert on degradation.

    Logs the distribution of signal-to-noise ratio for each modality
    and raises a warning if the mean SNR drops below a threshold.

    This callback is designed for use with a data pipeline that provides
    SNR metadata alongside each batch.  The SNR tensor should be stored
    in the batch dictionary or accessible via a callback on the dataset.

    Args:
        snr_threshold: Minimum acceptable mean SNR (in dB).  Values
            below this trigger a warning.
        log_freq: Log SNR statistics every N epochs.
        verbose: Verbosity level.
        modality_keys: Dictionary mapping modality name to the key
            in the input batch dict (or ``None`` for automatic detection).
    """

    def __init__(
        self,
        snr_threshold: float = 10.0,
        log_freq: int = 1,
        verbose: int = 1,
        modality_keys: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        self.snr_threshold = snr_threshold
        self.log_freq = log_freq
        self.verbose = verbose
        self.modality_keys = modality_keys or {
            "ecg": "ecg",
            "accelerometer": "accelerometer",
            "ppg": "ppg",
        }
        self._last_batch_snr: Dict[str, float] = {}

    def on_train_batch_end(
        self, batch: int, logs: Optional[Dict[str, Any]] = None
    ) -> None:
        """Capture SNR from the current batch if available."""
        logs = logs or {}
        for modality_name, key in self.modality_keys.items():
            snr_key = f"snr_{key}"
            if snr_key in logs:
                snr_val = float(logs[snr_key])
                self._last_batch_snr[modality_name] = snr_val

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if (epoch + 1) % self.log_freq != 0:
            return

        logs = logs or {}
        snr_stats = self._collect_snr_stats(logs)

        if not snr_stats:
            return

        for modality, stats in snr_stats.items():
            mean_snr = stats["mean"]
            std_snr = stats["std"]

            if self.verbose >= 1 and (epoch + 1) % (self.log_freq * 10) == 0:
                logger.info(
                    "SNR [%s] epoch %d: mean=%.2f dB, std=%.2f dB, "
                    "min=%.2f dB, max=%.2f dB",
                    modality,
                    epoch,
                    mean_snr,
                    std_snr,
                    stats["min"],
                    stats["max"],
                )

            if mean_snr < self.snr_threshold:
                logger.warning(
                    "Signal quality alert: %s mean SNR %.2f dB is below "
                    "threshold %.2f dB at epoch %d.",
                    modality,
                    mean_snr,
                    self.snr_threshold,
                    epoch,
                )

    def _collect_snr_stats(self, logs: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Aggregate SNR statistics from logs and last batch info."""
        stats: Dict[str, Dict[str, float]] = {}

        # Check for per-modality SNR in logs
        for modality_name in self.modality_keys:
            snr_key = f"{modality_name}_snr_mean"
            if snr_key in logs:
                mean_val = float(logs[snr_key])
                std_val = float(logs.get(f"{modality_name}_snr_std", 0.0))
                min_val = float(logs.get(f"{modality_name}_snr_min", mean_val))
                max_val = float(logs.get(f"{modality_name}_snr_max", mean_val))
                stats[modality_name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "max": max_val,
                }

        # Fallback: use last batch SNR
        if not stats and self._last_batch_snr:
            for modality_name, snr_val in self._last_batch_snr.items():
                stats[modality_name] = {
                    "mean": snr_val,
                    "std": 0.0,
                    "min": snr_val,
                    "max": snr_val,
                }

        return stats

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self._last_batch_snr = {}

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self._last_batch_snr = {}
