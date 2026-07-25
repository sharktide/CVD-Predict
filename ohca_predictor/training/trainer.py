"""Training pipeline for the OHCA Predictor model.

Implements the complete training loop with mixed-precision support,
gradient accumulation, curriculum-based sample weighting, data augmentation,
TensorBoard logging, clinical metrics evaluation, and early stopping.

Architecture overview:
    OHCAPredictionModel -> CombinedOHCALoss -> AdamW with CosineDecayWithWarmup
    -> GradientAccumulation -> CurriculumScheduler -> SignalAugmenter -> Callbacks
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import tensorflow as tf
from tensorflow import keras

from ohca_predictor.config import OHCAConfig, get_config
from ohca_predictor.model.architecture import OHCAPredictionModel
from ohca_predictor.model.losses import CombinedOHCALoss
from ohca_predictor.training.callbacks import (
    OHCAEarlyStopping,
    OHCACheckpoint,
    OHCATensorBoard,
    LRLogger,
    MetricsLogger,
    NaNMonitor,
    SignalQualityMonitor,
)
from ohca_predictor.training.curriculum import CurriculumScheduler, SignalAugmenter
from ohca_predictor.utils.logging import StructuredLogger

logger = logging.getLogger(__name__)


# ===========================================================================
# A. CosineDecayWithWarmup
# ===========================================================================


class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with linear warmup followed by cosine decay.

    Phase 1 (Linear Warmup):
        Steps 0 -> warmup_steps: lr ramps linearly from 0 to ``learning_rate``.

    Phase 2 (Cosine Decay):
        Steps warmup_steps -> total_steps: lr decays following a cosine
        curve from ``learning_rate`` to ``min_learning_rate``:

            lr(t) = min_lr + 0.5 * (lr - min_lr) *
                    (1 + cos(pi * (t - warmup) / (total - warmup)))

    Args:
        learning_rate: Peak learning rate after warmup.
        warmup_steps: Number of steps for linear warmup.
        total_steps: Total number of training steps.
        min_learning_rate: Minimum learning rate at end of decay.
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        min_learning_rate: float = 1e-7,
    ):
        super().__init__()
        self._learning_rate = float(learning_rate)
        self._warmup_steps = float(warmup_steps)
        self._total_steps = float(total_steps)
        self._min_learning_rate = float(min_learning_rate)

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        """Compute the learning rate for a given global step.

        Args:
            step: Current training step (integer scalar tensor).

        Returns:
            Learning rate as a scalar float32 tensor.
        """
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self._warmup_steps, tf.float32)
        total_steps = tf.cast(self._total_steps, tf.float32)
        lr = tf.cast(self._learning_rate, tf.float32)
        min_lr = tf.cast(self._min_learning_rate, tf.float32)

        # Phase 1: Linear warmup
        warmup_lr = lr * (step / tf.maximum(warmup_steps, 1.0))

        # Phase 2: Cosine decay
        decay_steps = tf.maximum(total_steps - warmup_steps, 1.0)
        t = tf.clip_by_value(step - warmup_steps, 0.0, decay_steps)
        cosine_decay = 0.5 * (1.0 + tf.cos(math.pi * t / decay_steps))
        decay_lr = min_lr + (lr - min_lr) * cosine_decay

        return tf.where(step < warmup_steps, warmup_lr, decay_lr)

    def get_config(self) -> Dict[str, Any]:
        return {
            "learning_rate": self._learning_rate,
            "warmup_steps": self._warmup_steps,
            "total_steps": self._total_steps,
            "min_learning_rate": self._min_learning_rate,
        }


# ===========================================================================
# B. GradientAccumulation
# ===========================================================================


class GradientAccumulation:
    """Accumulate gradients over multiple micro-batches before updating.

    Enables training with a larger effective batch size when GPU memory
    is limited. Gradients are computed for each micro-batch and added to
    running accumulators. After ``accumulation_steps`` micro-batches, the
    accumulated gradients are averaged, clipped, and applied.

    Args:
        accumulation_steps: Number of micro-batches to accumulate before
            applying gradients.
    """

    def __init__(self, accumulation_steps: int = 4):
        if accumulation_steps <= 0:
            raise ValueError(
                f"accumulation_steps must be positive, got {accumulation_steps}"
            )
        self.accumulation_steps = accumulation_steps
        self._accumulated_gradients: Optional[List[tf.Tensor]] = None
        self._step_count: int = 0

    def accumulate(
        self,
        model: keras.Model,
        batch: Any,
        loss_fn: keras.losses.Loss,
        optimizer: keras.optimizers.Optimizer,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], bool]:
        """Compute gradients for a single micro-batch and accumulate.

        Args:
            model: The Keras model being trained.
            batch: A training batch (inputs, labels or a dict).
            loss_fn: The loss function to compute the scalar loss.
            optimizer: The optimizer (used for mixed-precision loss scaling).

        Returns:
            Tuple of (loss_value, predictions_dict, should_update):
                - loss_value: Scalar loss for this micro-batch.
                - predictions_dict: Model output dictionary.
                - should_update: True if accumulated gradients are ready
                  to be applied (every ``accumulation_steps`` calls).
        """
        with tf.GradientTape() as tape:
            if isinstance(batch, tuple) and len(batch) == 2:
                inputs, targets = batch
            else:
                inputs = batch
                targets = None

            predictions = model(inputs, training=True)

            if targets is not None:
                loss = loss_fn(targets, predictions)
            else:
                loss = loss_fn(predictions)

            scaled_loss = (
                optimizer.get_scaled_loss(loss)
                if hasattr(optimizer, "get_scaled_loss")
                else loss
            )

        gradients = tape.gradient(scaled_loss, model.trainable_variables)

        if hasattr(optimizer, "get_unscaled_gradients"):
            gradients = optimizer.get_unscaled_gradients(gradients)

        if self._accumulated_gradients is None:
            self._accumulated_gradients = [
                tf.zeros_like(g) if g is not None else None for g in gradients
            ]

        for i, grad in enumerate(gradients):
            if (
                grad is not None
                and self._accumulated_gradients[i] is not None
            ):
                self._accumulated_gradients[i] = (
                    self._accumulated_gradients[i] + grad
                )

        self._step_count += 1
        should_update = (self._step_count % self.accumulation_steps) == 0

        return loss, predictions, should_update

    def apply_gradients(
        self,
        model: keras.Model,
        optimizer: keras.optimizers.Optimizer,
        gradient_clip_norm: float = 1.0,
    ) -> None:
        """Average accumulated gradients, clip, and apply to the optimizer.

        Resets the accumulation state after applying.

        Args:
            model: The Keras model being trained.
            optimizer: The optimizer to apply gradients to.
            gradient_clip_norm: Maximum gradient L2 norm for clipping.
        """
        if self._accumulated_gradients is None or self._step_count == 0:
            return

        averaged_gradients = []
        for grad in self._accumulated_gradients:
            if grad is not None:
                averaged_gradients.append(
                    grad / tf.cast(self.accumulation_steps, grad.dtype)
                )
            else:
                averaged_gradients.append(None)

        non_none_grads = [g for g in averaged_gradients if g is not None]
        non_none_vars = [
            v
            for v, g in zip(model.trainable_variables, averaged_gradients)
            if g is not None
        ]

        if non_none_grads:
            clipped_grads, _ = tf.clip_by_global_norm(
                non_none_grads, gradient_clip_norm
            )
            optimizer.apply_gradients(zip(clipped_grads, non_none_vars))

        self._accumulated_gradients = None
        self._step_count = 0

    def reset(self) -> None:
        """Reset the accumulation state."""
        self._accumulated_gradients = None
        self._step_count = 0


# ===========================================================================
# C. OHCATrainer
# ===========================================================================


class OHCATrainer:
    """Complete training orchestrator for the OHCA prediction model.

    Manages the full training lifecycle: model construction, optimizer setup,
    loss computation, curriculum scheduling, data augmentation, gradient
    accumulation, mixed-precision training, validation, early stopping,
    checkpointing, and TensorBoard logging.

    Args:
        config: An ``OHCAConfig`` instance with all hyperparameters.
    """

    def __init__(self, config: OHCAConfig) -> None:
        self.config = config
        self.train_config = config.training
        self.model_config = config.model

        self.logger = StructuredLogger()

        # ---- Model ----
        self.model = OHCAPredictionModel(config.model)
        self.logger.info("Model built successfully")

        # ---- Loss ----
        self.loss_fn = CombinedOHCALoss(
            cls_weight=1.0,
            surv_weight=self.train_config.survival_loss_weight,
            aux_weight=self.train_config.auxiliary_loss_weight,
            contrast_weight=self.train_config.contrastive_loss_weight,
            recon_weight=self.train_config.reconstruction_loss_weight,
            focal_gamma=self.train_config.focal_loss_gamma,
            fn_weight=self.train_config.false_negative_weight,
            fp_weight=self.train_config.false_positive_weight,
        )

        # ---- Learning rate schedule ----
        self.total_steps = self.train_config.epochs * 1000
        self.lr_schedule = CosineDecayWithWarmup(
            learning_rate=self.train_config.learning_rate,
            warmup_steps=self.train_config.warmup_steps,
            total_steps=self.total_steps,
            min_learning_rate=self.train_config.min_learning_rate,
        )

        # ---- Optimizer (AdamW) ----
        self.optimizer = keras.optimizers.AdamW(
            learning_rate=self.lr_schedule,
            weight_decay=self.train_config.weight_decay,
            clipnorm=self.train_config.gradient_clip_norm,
        )

        # ---- Mixed precision ----
        self._mixed_precision_enabled = False
        if self.train_config.mixed_precision:
            self._setup_mixed_precision()

        # ---- Gradient accumulation ----
        self.gradient_accumulator = GradientAccumulation(
            accumulation_steps=self.train_config.gradient_accumulation_steps,
        )

        # ---- Curriculum scheduler (initialized in train()) ----
        self.curriculum_scheduler: Optional[CurriculumScheduler] = None

        # ---- Signal augmenter ----
        self.augmenter = SignalAugmenter(config=config)

        # ---- Callbacks (initialized in train()) ----
        self._callbacks: List[keras.callbacks.Callback] = []

        # ---- Training state ----
        self.global_step: int = 0
        self.current_epoch: int = 0
        self._best_val_auroc: float = 0.0
        self._history: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Mixed precision setup
    # ------------------------------------------------------------------

    def _setup_mixed_precision(self) -> None:
        """Configure mixed-precision training policy."""
        try:
            policy = keras.mixed_precision.Policy("mixed_float16")
            keras.mixed_precision.set_global_policy(policy)
            self._mixed_precision_enabled = True
            self.logger.info("Mixed precision enabled: mixed_float16")
        except Exception as exc:
            self._mixed_precision_enabled = False
            self.logger.warning(
                f"Mixed precision setup failed, falling back to float32: {exc}"
            )

    # ------------------------------------------------------------------
    # Callback initialization
    # ------------------------------------------------------------------

    def _build_callbacks(
        self, log_dir: str = "logs"
    ) -> List[keras.callbacks.Callback]:
        """Initialize all training callbacks.

        Args:
            log_dir: Base directory for logs and checkpoints.

        Returns:
            List of Keras callback instances.
        """
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        tensorboard_dir = os.path.join(log_dir, "tensorboard")
        lr_dir = os.path.join(log_dir, "lr")

        callbacks = []

        early_stopping = OHCAEarlyStopping(
            monitor="val_ohca_auroc",
            patience=self.train_config.early_stopping_patience,
            mode="max",
            restore_best_weights=True,
            min_delta=1e-5,
            verbose=1,
        )
        callbacks.append(early_stopping)

        checkpoint = OHCACheckpoint(
            save_dir=checkpoint_dir,
            monitor="val_ohca_auroc",
            mode="max",
            save_best_only=self.train_config.checkpoint_save_best_only,
            save_every_n_epochs=5,
            keep_top_k=3,
            verbose=1,
        )
        callbacks.append(checkpoint)

        tensorboard = OHCATensorBoard(
            log_dir=tensorboard_dir,
            histogram_freq=5,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
            profile_batch=100,
            log_dir_gradients=True,
        )
        callbacks.append(tensorboard)

        lr_logger = LRLogger(log_dir=lr_dir)
        callbacks.append(lr_logger)

        nan_monitor = NaNMonitor(
            max_nan_count=5,
            reduce_lr_factor=0.5,
            verbose=1,
        )
        callbacks.append(nan_monitor)

        signal_monitor = SignalQualityMonitor(
            snr_threshold=10.0,
            log_freq=1,
            verbose=1,
        )
        callbacks.append(signal_monitor)

        return callbacks

    # ------------------------------------------------------------------
    # Curriculum initialization
    # ------------------------------------------------------------------

    def _init_curriculum(
        self, dataset_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the curriculum scheduler with dataset metadata.

        Args:
            dataset_info: Dictionary with dataset metadata expected by
                ``CurriculumScheduler``.  If ``None``, a minimal default
                info dict is used.
        """
        if dataset_info is None:
            n = 10000
            rng = np.random.RandomState(42)
            dataset_info = {
                "total_patients": n,
                "patient_ids": [f"patient_{i}" for i in range(n)],
                "labels": rng.choice([0.0, 1.0], size=n, p=[0.999, 0.001]).astype(np.float32),
                "disease_severity": rng.rand(n).astype(np.float32),
                "num_comorbidities": rng.randint(0, 5, n),
                "age": rng.uniform(20, 90, n).astype(np.float32),
                "ef": rng.uniform(0.3, 0.7, n).astype(np.float32),
                "is_lbbb": rng.rand(n) < 0.05,
                "has_pacemaker": rng.rand(n) < 0.03,
                "old_mi": rng.rand(n) < 0.08,
                "medications": rng.randint(0, 10, n),
                "snr_score": rng.uniform(0.5, 1.0, n).astype(np.float32),
            }

        self.curriculum_scheduler = CurriculumScheduler(
            config=self.config,
            dataset_info=dataset_info,
        )
        self.curriculum_scheduler.compute_difficulty_scores()
        self.logger.info("Curriculum scheduler initialized")

    # ------------------------------------------------------------------
    # Train step (tf.function decorated)
    # ------------------------------------------------------------------

    def _make_train_step(self) -> tf.function:
        """Create and compile the training step function.

        Returns:
            A ``tf.function``-decorated training step.
        """
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        model = self.model

        @tf.function(reduce_retracing=True)
        def train_step(
            batch: Any,
        ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
            """Execute one training micro-batch.

            Args:
                batch: Tuple of (inputs_dict, targets_dict).

            Returns:
                Tuple of (loss_value, predictions_dict).
            """
            inputs, targets = batch

            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = loss_fn(targets, predictions)

                if hasattr(optimizer, "get_scaled_loss"):
                    scaled_loss = optimizer.get_scaled_loss(loss)
                else:
                    scaled_loss = loss

            gradients = tape.gradient(scaled_loss, model.trainable_variables)

            if hasattr(optimizer, "get_unscaled_gradients"):
                gradients = optimizer.get_unscaled_gradients(gradients)

            gradients, _ = tf.clip_by_global_norm(
                [g if g is not None else tf.zeros_like(v) for g, v in zip(gradients, model.trainable_variables)],
                optimizer.clipnorm if hasattr(optimizer, "clipnorm") else 1.0,
            )
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )

            return loss, predictions

        return train_step

    def _make_val_step(self) -> tf.function:
        """Create and compile the validation step function.

        Returns:
            A ``tf.function``-decorated validation step.
        """
        loss_fn = self.loss_fn
        model = self.model

        @tf.function(reduce_retracing=True)
        def val_step(
            batch: Any,
        ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
            """Execute one validation batch.

            Args:
                batch: Tuple of (inputs_dict, targets_dict).

            Returns:
                Tuple of (loss_value, predictions_dict).
            """
            inputs, targets = batch
            predictions = model(inputs, training=False)
            loss = loss_fn(targets, predictions)
            return loss, predictions

        return val_step

    # ------------------------------------------------------------------
    # One training epoch
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        dataset: tf.data.Dataset,
        epoch: int,
    ) -> Dict[str, float]:
        """Run one full training epoch.

        Handles:
        - Curriculum phase update
        - Gradient accumulation across micro-batches
        - Data augmentation for training batches
        - Loss and metric accumulation

        Args:
            dataset: Training ``tf.data.Dataset``.
            epoch: Current epoch number (0-indexed).

        Returns:
            Dictionary of averaged training metrics for the epoch.
        """
        self.model.trainable = True

        epoch_loss = 0.0
        num_batches = 0
        all_predictions: List[tf.Tensor] = []
        all_targets: List[tf.Tensor] = []

        # Update curriculum phase
        if self.curriculum_scheduler is not None:
            phase_info = self.curriculum_scheduler.get_phase_info(epoch)
            self.logger.info(
                f"Curriculum phase {phase_info['phase']}: "
                f"{phase_info['description']} "
                f"({phase_info['active_patients']}/{phase_info['total_patients']} patients)"
            )

        self.augmenter.set_epoch(epoch)
        self.gradient_accumulator.reset()
        accum_steps = self.train_config.gradient_accumulation_steps

        for batch in dataset:
            if isinstance(batch, tuple) and len(batch) == 2:
                inputs, targets = batch
                if isinstance(inputs, dict):
                    inputs = self._augment_tf_batch(inputs)
                    batch = (inputs, targets)

            if accum_steps > 1:
                loss_val, predictions, should_update = (
                    self.gradient_accumulator.accumulate(
                        self.model, batch, self.loss_fn, self.optimizer
                    )
                )
                if should_update:
                    self.gradient_accumulator.apply_gradients(
                        self.model,
                        self.optimizer,
                        self.train_config.gradient_clip_norm,
                    )
                    self.global_step += 1
                loss = loss_val
            else:
                train_step_fn = self._make_train_step()
                loss, predictions = train_step_fn(batch)
                self.global_step += 1

            epoch_loss += float(loss)
            num_batches += 1

            if isinstance(predictions, dict) and "ohca_risk" in predictions:
                all_predictions.append(predictions["ohca_risk"])
            if isinstance(targets, dict) and "ohca_label" in targets:
                all_targets.append(targets["ohca_label"])

        # Handle any remaining accumulated gradients
        if accum_steps > 1 and self.gradient_accumulator._step_count > 0:
            self.gradient_accumulator.apply_gradients(
                self.model,
                self.optimizer,
                self.train_config.gradient_clip_norm,
            )
            self.gradient_accumulator.reset()

        metrics: Dict[str, float] = {}
        metrics["loss"] = epoch_loss / max(num_batches, 1)

        if all_predictions and all_targets:
            y_pred = tf.concat(all_predictions, axis=0)
            y_true = tf.concat(all_targets, axis=0)
            computed = self._compute_metrics(y_true, y_pred)
            metrics.update(computed)

        return metrics

    # ------------------------------------------------------------------
    # One validation epoch
    # ------------------------------------------------------------------

    def _validate_epoch(
        self,
        dataset: tf.data.Dataset,
        epoch: int,
    ) -> Dict[str, float]:
        """Run one full validation epoch.

        Args:
            dataset: Validation ``tf.data.Dataset``.
            epoch: Current epoch number (0-indexed).

        Returns:
            Dictionary of averaged validation metrics for the epoch.
        """
        self.model.trainable = False
        val_step_fn = self._make_val_step()

        epoch_loss = 0.0
        num_batches = 0
        all_predictions: List[tf.Tensor] = []
        all_targets: List[tf.Tensor] = []
        all_survival_curves: List[tf.Tensor] = []
        all_survival_targets: List[tf.Tensor] = []
        all_uncertainties: List[tf.Tensor] = []

        for batch in dataset:
            loss, predictions = val_step_fn(batch)

            epoch_loss += float(loss)
            num_batches += 1

            if isinstance(batch, tuple) and len(batch) == 2:
                _, targets = batch
            else:
                targets = None

            if isinstance(predictions, dict):
                if "ohca_risk" in predictions:
                    all_predictions.append(predictions["ohca_risk"])
                if "survival_curve" in predictions:
                    all_survival_curves.append(predictions["survival_curve"])
                if "uncertainty" in predictions:
                    all_uncertainties.append(predictions["uncertainty"])
            if isinstance(targets, dict):
                if "ohca_label" in targets:
                    all_targets.append(targets["ohca_label"])
                if "survival_events" in targets:
                    all_survival_targets.append(targets["survival_events"])

        metrics: Dict[str, float] = {}
        metrics["val_loss"] = epoch_loss / max(num_batches, 1)

        if all_predictions and all_targets:
            y_pred = tf.concat(all_predictions, axis=0)
            y_true = tf.concat(all_targets, axis=0)
            computed = self._compute_metrics(y_true, y_pred, prefix="val_")
            metrics.update(computed)

        if all_survival_curves and all_survival_targets:
            surv_pred = tf.concat(all_survival_curves, axis=0)
            surv_true = tf.concat(all_survival_targets, axis=0)
            surv_metrics = self._compute_survival_metrics(
                surv_true, surv_pred, prefix="val_"
            )
            metrics.update(surv_metrics)

        if all_uncertainties:
            unc = tf.concat(all_uncertainties, axis=0)
            metrics["val_mean_uncertainty"] = float(
                tf.reduce_mean(unc[:, 1]).numpy()
            )
            metrics["val_mean_predicted_risk"] = float(
                tf.reduce_mean(unc[:, 0]).numpy()
            )

        return metrics

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute classification metrics from ground truth and predictions.

        Computes AUROC, sensitivity, specificity, precision, F1, and
        negative predictive value at the primary clinical threshold.

        Args:
            y_true: Binary labels tensor ``(batch, 1)`` or ``(batch,)``.
            y_pred: Predicted risk probabilities ``(batch, 1)`` or ``(batch,)``.
            prefix: Optional string prefix for metric names (e.g., ``"val_"``).

        Returns:
            Dictionary of computed metrics.
        """
        y_true_np = y_true.numpy().astype(np.float32).ravel()
        y_pred_np = y_pred.numpy().astype(np.float32).ravel()

        threshold = self.config.evaluation.primary_threshold
        y_pred_binary = (y_pred_np >= threshold).astype(np.float32)

        metrics: Dict[str, float] = {}

        # AUROC
        try:
            from sklearn.metrics import roc_auc_score

            auroc = float(roc_auc_score(y_true_np, y_pred_np))
        except (ValueError, ImportError):
            auroc = 0.5
        metrics[f"{prefix}ohca_auroc"] = auroc

        # Confusion matrix components
        n_pos = int(y_true_np.sum())
        n_neg = len(y_true_np) - n_pos

        if n_pos > 0 and n_neg > 0:
            tp = int(np.sum((y_pred_binary == 1) & (y_true_np == 1)))
            fp = int(np.sum((y_pred_binary == 1) & (y_true_np == 0)))
            tn = int(np.sum((y_pred_binary == 0) & (y_true_np == 0)))
            fn = int(np.sum((y_pred_binary == 0) & (y_true_np == 1)))

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (
                2 * precision * sensitivity / (precision + sensitivity)
                if (precision + sensitivity) > 0
                else 0.0
            )
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

            metrics[f"{prefix}ohca_sensitivity"] = sensitivity
            metrics[f"{prefix}ohca_specificity"] = specificity
            metrics[f"{prefix}ohca_precision"] = precision
            metrics[f"{prefix}ohca_f1"] = f1
            metrics[f"{prefix}ohca_npv"] = npv
            metrics[f"{prefix}true_positives"] = float(tp)
            metrics[f"{prefix}false_positives"] = float(fp)
            metrics[f"{prefix}true_negatives"] = float(tn)
            metrics[f"{prefix}false_negatives"] = float(fn)
        else:
            metrics[f"{prefix}ohca_sensitivity"] = 0.0
            metrics[f"{prefix}ohca_specificity"] = 0.0
            metrics[f"{prefix}ohca_precision"] = 0.0
            metrics[f"{prefix}ohca_f1"] = 0.0
            metrics[f"{prefix}ohca_npv"] = 0.0

        eps = 1e-7
        y_clipped = np.clip(y_pred_np, eps, 1.0 - eps)
        bce = float(
            -np.mean(
                y_true_np * np.log(y_clipped)
                + (1.0 - y_true_np) * np.log(1.0 - y_clipped)
            )
        )
        metrics[f"{prefix}ohca_bce"] = bce

        metrics[f"{prefix}positive_prevalence"] = (
            float(n_pos) / float(len(y_true_np))
            if len(y_true_np) > 0
            else 0.0
        )

        return metrics

    def _compute_survival_metrics(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute survival analysis metrics.

        Args:
            y_true: Survival event indicators ``(batch, K+1)``.
            y_pred: Predicted cumulative survival curves ``(batch, K+1)``.
            prefix: Optional string prefix for metric names.

        Returns:
            Dictionary of survival metrics.
        """
        y_true_np = y_true.numpy().astype(np.float32)
        y_pred_np = y_pred.numpy().astype(np.float32)

        metrics: Dict[str, float] = {}

        mae = float(np.mean(np.abs(y_true_np - y_pred_np)))
        metrics[f"{prefix}survival_mae"] = mae

        event_indicators = y_true_np[:, :-1]
        has_event = event_indicators.sum(axis=1) > 0

        if has_event.sum() > 0 and (~has_event).sum() > 0:
            event_times = np.argmax(event_indicators[has_event], axis=1)
            event_surv = np.array(
                [y_pred_np[i, t] for i, t in enumerate(event_times)]
            )
            censored_surv = np.mean(y_pred_np[~has_event], axis=1)

            if len(event_surv) > 0 and len(censored_surv) > 0:
                concordant = float(
                    np.mean(event_surv < np.mean(censored_surv))
                )
                metrics[f"{prefix}survival_concordance"] = concordant

        return metrics

    # ------------------------------------------------------------------
    # Augmentation for tf.data batches
    # ------------------------------------------------------------------

    def _augment_tf_batch(
        self,
        inputs: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:
        """Apply signal augmentations to a TensorFlow batch dictionary.

        Applies augmentations only during training. Operates on ECG,
        accelerometer, and PPG tensors.

        Args:
            inputs: Dictionary of input tensors.

        Returns:
            Dictionary with augmented signal tensors.
        """
        augmented = {}

        for key, value in inputs.items():
            if key == "ecg" and isinstance(value, tf.Tensor):
                augmented[key] = self._augment_ecg_tf(value)
            elif key == "accelerometer" and isinstance(value, tf.Tensor):
                augmented[key] = self._augment_motion_tf(value)
            elif key == "gyroscope" and isinstance(value, tf.Tensor):
                augmented[key] = self._augment_motion_tf(value)
            elif key == "ppg" and isinstance(value, tf.Tensor):
                augmented[key] = self._augment_ppg_tf(value)
            else:
                augmented[key] = value

        return augmented

    def _augment_ecg_tf(self, ecg: tf.Tensor) -> tf.Tensor:
        """Apply random augmentations to an ECG tensor batch.

        Augmentations:
        - Gaussian noise injection (std = 0.05 * signal range)
        - Random temporal shift (+-5% of sequence length)
        - Random amplitude scaling (0.8 -- 1.2)

        Args:
            ecg: ECG tensor ``(batch, samples, channels)``.

        Returns:
            Augmented ECG tensor.
        """
        x = tf.cast(ecg, tf.float32)

        signal_range = (
            tf.reduce_max(x, axis=1, keepdims=True)
            - tf.reduce_min(x, axis=1, keepdims=True)
        )
        noise_std = tf.maximum(signal_range * 0.05, 1e-6)
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=1.0) * noise_std
        x = x + noise

        seq_len = tf.shape(x)[1]
        max_shift = tf.cast(tf.maximum(seq_len // 20, 1), tf.int32)
        shift = tf.random.uniform(
            shape=(), minval=-max_shift, maxval=max_shift + 1, dtype=tf.int32
        )
        x = tf.roll(x, shift=shift, axis=1)

        scale = tf.random.uniform(
            shape=(tf.shape(x)[0], 1, 1), minval=0.8, maxval=1.2
        )
        x = x * scale

        return x

    def _augment_motion_tf(self, motion: tf.Tensor) -> tf.Tensor:
        """Apply random augmentations to a motion sensor tensor batch.

        Augmentations:
        - Gaussian noise (std = 0.1)
        - Random axis dropout (zero out one axis with 20% probability)

        Args:
            motion: Motion tensor ``(batch, samples, 3)``.

        Returns:
            Augmented motion tensor.
        """
        x = tf.cast(motion, tf.float32)

        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.1)
        x = x + noise

        axis_to_drop = tf.random.uniform(
            shape=(tf.shape(x)[0],), minval=0, maxval=3, dtype=tf.int32
        )
        drop_mask = tf.one_hot(axis_to_drop, depth=3, dtype=tf.float32)
        drop_mask = tf.expand_dims(drop_mask, axis=1)
        should_drop = (
            tf.random.uniform(
                shape=(tf.shape(x)[0], 1, 1), minval=0.0, maxval=1.0
            )
            < 0.2
        )
        drop_mask = tf.cast(should_drop, tf.float32) * drop_mask
        x = x * (1.0 - drop_mask)

        return x

    def _augment_ppg_tf(self, ppg: tf.Tensor) -> tf.Tensor:
        """Apply random augmentations to a PPG tensor batch.

        Augmentations:
        - Gaussian noise (std = 0.03)
        - Random amplitude scaling (0.75 -- 1.25)

        Args:
            ppg: PPG tensor ``(batch, samples, channels)``.

        Returns:
            Augmented PPG tensor.
        """
        x = tf.cast(ppg, tf.float32)

        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.03)
        x = x + noise

        scale = tf.random.uniform(
            shape=(tf.shape(x)[0], 1, 1), minval=0.75, maxval=1.25
        )
        x = x * scale

        return x

    # ------------------------------------------------------------------
    # Epoch summary logging
    # ------------------------------------------------------------------

    def _log_epoch_summary(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int,
    ) -> None:
        """Log a formatted epoch summary to console and structured logger.

        Args:
            train_metrics: Training metrics dictionary.
            val_metrics: Validation metrics dictionary.
            epoch: Current epoch number (0-indexed).
        """
        train_loss = train_metrics.get("loss", 0.0)
        val_loss = val_metrics.get("val_loss", 0.0)
        train_auroc = train_metrics.get("ohca_auroc", 0.0)
        val_auroc = val_metrics.get("val_ohca_auroc", 0.0)
        val_sens = val_metrics.get("val_ohca_sensitivity", 0.0)
        val_spec = val_metrics.get("val_ohca_specificity", 0.0)

        summary = (
            f"Epoch {epoch + 1:3d} | "
            f"train_loss={train_loss:.4f} train_auroc={train_auroc:.4f} | "
            f"val_loss={val_loss:.4f} val_auroc={val_auroc:.4f} "
            f"val_sens={val_sens:.4f} val_spec={val_spec:.4f}"
        )

        self.logger.info(summary)
        print(summary)

    # ------------------------------------------------------------------
    # Public training API
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: Optional[int] = None,
        log_dir: str = "logs",
        dataset_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[float]]:
        """Run the complete training loop.

        Executes training with:
        - tf.function-decorated train/val steps
        - Mixed-precision loss scaling
        - Gradient accumulation across micro-batches
        - Curriculum-based sample weighting
        - Data augmentation for training only
        - Early stopping based on val_auroc
        - Checkpoint saving on improvement

        Args:
            train_dataset: Training ``tf.data.Dataset``.
            val_dataset: Validation ``tf.data.Dataset``.
            epochs: Maximum number of epochs.  Defaults to config value.
            log_dir: Base directory for logs and checkpoints.
            dataset_info: Optional dataset metadata for curriculum scheduling.

        Returns:
            Dictionary mapping metric names to lists of per-epoch values.
        """
        if epochs is None:
            epochs = self.train_config.epochs

        # Update total_steps for LR schedule based on actual dataset size
        train_batches = 0
        for _ in train_dataset:
            train_batches += 1
        effective_batches_per_epoch = max(
            train_batches // self.train_config.gradient_accumulation_steps, 1
        )
        self.total_steps = epochs * effective_batches_per_epoch
        self.lr_schedule = CosineDecayWithWarmup(
            learning_rate=self.train_config.learning_rate,
            warmup_steps=self.train_config.warmup_steps,
            total_steps=self.total_steps,
            min_learning_rate=self.train_config.min_learning_rate,
        )
        self.optimizer.learning_rate = self.lr_schedule

        self.logger.info(
            f"Training config: {epochs} epochs, {train_batches} batches/epoch, "
            f"{self.total_steps} total steps, accum={self.train_config.gradient_accumulation_steps}"
        )

        # Initialize curriculum
        self._init_curriculum(dataset_info)

        # Initialize callbacks
        self._callbacks = self._build_callbacks(log_dir)
        for callback in self._callbacks:
            callback.set_model(self.model)

        # Trigger on_train_begin for all callbacks
        for callback in self._callbacks:
            callback.on_train_begin()

        self._history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Trigger on_epoch_begin for all callbacks
            for callback in self._callbacks:
                callback.on_epoch_begin(epoch)

            # ---- Train ----
            train_metrics = self._train_epoch(train_dataset, epoch)

            # ---- Validate ----
            val_metrics = self._validate_epoch(val_dataset, epoch)

            # ---- Merge metrics into logs ----
            epoch_logs = {}
            epoch_logs.update(train_metrics)
            epoch_logs.update(val_metrics)
            epoch_logs["epoch"] = epoch

            # ---- Update curriculum difficulty ----
            if self.curriculum_scheduler is not None:
                self.curriculum_scheduler.update_difficulty(epoch)

            # ---- Log epoch summary ----
            self._log_epoch_summary(train_metrics, val_metrics, epoch)

            # ---- Track history ----
            self._history.setdefault("train_loss", []).append(
                train_metrics.get("loss", 0.0)
            )
            self._history.setdefault("val_loss", []).append(
                val_metrics.get("val_loss", 0.0)
            )
            self._history.setdefault("train_auroc", []).append(
                train_metrics.get("ohca_auroc", 0.0)
            )
            self._history.setdefault("val_auroc", []).append(
                val_metrics.get("val_ohca_auroc", 0.0)
            )

            # ---- TensorBoard scalar logging ----
            self._log_scalars_to_tensorboard(epoch_logs, epoch)

            # ---- Trigger on_epoch_end for all callbacks ----
            for callback in self._callbacks:
                callback.on_epoch_end(epoch, epoch_logs)

            epoch_time = time.time() - epoch_start
            self.logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")

            # ---- Check early stopping ----
            val_auroc = val_metrics.get("val_ohca_auroc", 0.0)
            if val_auroc > self._best_val_auroc:
                self._best_val_auroc = val_auroc

            if self.model.stop_training:
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"Best val_auroc={self._best_val_auroc:.4f}"
                )
                break

        # ---- Trigger on_train_end for all callbacks ----
        for callback in self._callbacks:
            callback.on_train_end()

        self.logger.info(
            f"Training complete. Best val_auroc={self._best_val_auroc:.4f}"
        )

        return self._history

    def _log_scalars_to_tensorboard(
        self, epoch_logs: Dict[str, float], epoch: int
    ) -> None:
        """Write key scalars to TensorBoard summary writer.

        Args:
            epoch_logs: Dictionary of all metric values for this epoch.
            epoch: Current epoch number.
        """
        for callback in self._callbacks:
            if isinstance(callback, OHCATensorBoard):
                writer = getattr(callback, "summary_writer", None)
                if writer is not None:
                    with writer.as_default():
                        for key, value in epoch_logs.items():
                            if isinstance(value, (int, float)):
                                tf.summary.scalar(
                                    f"epoch/{key}", float(value), step=epoch
                                )

    # ------------------------------------------------------------------
    # Post-training evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_dataset: tf.data.Dataset,
    ) -> Dict[str, float]:
        """Evaluate the model on a test dataset.

        Runs a full forward pass over the test set and computes all
        classification and survival metrics.

        Args:
            test_dataset: Test ``tf.data.Dataset``.

        Returns:
            Dictionary of final test metrics.
        """
        self.model.trainable = False
        val_step_fn = self._make_val_step()

        epoch_loss = 0.0
        num_batches = 0
        all_predictions: List[tf.Tensor] = []
        all_targets: List[tf.Tensor] = []
        all_survival_curves: List[tf.Tensor] = []
        all_survival_targets: List[tf.Tensor] = []
        all_uncertainties: List[tf.Tensor] = []

        for batch in test_dataset:
            loss, predictions = val_step_fn(batch)

            epoch_loss += float(loss)
            num_batches += 1

            if isinstance(batch, tuple) and len(batch) == 2:
                _, targets = batch
            else:
                targets = None

            if isinstance(predictions, dict):
                if "ohca_risk" in predictions:
                    all_predictions.append(predictions["ohca_risk"])
                if "survival_curve" in predictions:
                    all_survival_curves.append(predictions["survival_curve"])
                if "uncertainty" in predictions:
                    all_uncertainties.append(predictions["uncertainty"])
            if isinstance(targets, dict):
                if "ohca_label" in targets:
                    all_targets.append(targets["ohca_label"])
                if "survival_events" in targets:
                    all_survival_targets.append(targets["survival_events"])

        metrics: Dict[str, float] = {}
        metrics["test_loss"] = epoch_loss / max(num_batches, 1)

        if all_predictions and all_targets:
            y_pred = tf.concat(all_predictions, axis=0)
            y_true = tf.concat(all_targets, axis=0)
            computed = self._compute_metrics(y_true, y_pred, prefix="test_")
            metrics.update(computed)

        if all_survival_curves and all_survival_targets:
            surv_pred = tf.concat(all_survival_curves, axis=0)
            surv_true = tf.concat(all_survival_targets, axis=0)
            surv_metrics = self._compute_survival_metrics(
                surv_true, surv_pred, prefix="test_"
            )
            metrics.update(surv_metrics)

        if all_uncertainties:
            unc = tf.concat(all_uncertainties, axis=0)
            metrics["test_mean_uncertainty"] = float(
                tf.reduce_mean(unc[:, 1]).numpy()
            )
            metrics["test_mean_predicted_risk"] = float(
                tf.reduce_mean(unc[:, 0]).numpy()
            )

        self.logger.info("Test evaluation complete")
        for key, value in sorted(metrics.items()):
            self.logger.info(f"  {key}: {value:.4f}")

        return metrics

    # ------------------------------------------------------------------
    # Model save / load
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        """Save the model weights and architecture.

        Creates a TF SavedModel directory and a separate weights file.

        Args:
            path: Directory path to save the model to.
        """
        os.makedirs(path, exist_ok=True)

        saved_model_path = os.path.join(path, "saved_model")
        weights_path = os.path.join(path, "weights.weights.h5")

        self.model.save(saved_model_path)
        self.model.save_weights(weights_path)

        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load model weights from a checkpoint directory.

        Attempts to load from a TF SavedModel first, then falls back
        to loading weights from an H5 file.

        Args:
            path: Directory path containing the saved model.
        """
        saved_model_path = os.path.join(path, "saved_model")
        weights_path = os.path.join(path, "weights.weights.h5")

        if os.path.isdir(saved_model_path):
            self.model = keras.models.load_model(
                saved_model_path,
                custom_objects={
                    "OHCAPredictionModel": OHCAPredictionModel,
                    "CosineDecayWithWarmup": CosineDecayWithWarmup,
                },
            )
            self.logger.info(f"Model loaded from SavedModel: {saved_model_path}")
        elif os.path.isfile(weights_path):
            self.model.load_weights(weights_path)
            self.logger.info(f"Model weights loaded from: {weights_path}")
        else:
            raise FileNotFoundError(
                f"No model found at {path}. "
                f"Expected {saved_model_path} or {weights_path}"
            )

    # ------------------------------------------------------------------
    # Property accessors
    # ------------------------------------------------------------------

    @property
    def history(self) -> Dict[str, List[float]]:
        """Return the training history dictionary."""
        return self._history

    @property
    def best_val_auroc(self) -> float:
        """Return the best validation AUROC achieved during training."""
        return self._best_val_auroc
