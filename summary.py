#!/usr/bin/env python3
"""
OHCA Prediction — Production Pipeline v2
=========================================
Stable Keras custom training model, comprehensive TensorBoard logging,
enhanced physiological realism, thorough clinical evaluation.

Hardware target: RTX 5070 Ti Mobile (12 GB) + Ultra 9 275HX
Training timeout: 12 hours (wall clock)
"""

import gc
import json
import os
import signal
import sys
import time
import traceback
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf

# ── GPU / mixed precision ──────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.optimizer.set_jit(False)
    try:
        policy = tf.keras.mixed_precision.Policy("mixed_bfloat16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print("[INIT] Mixed precision: mixed_bfloat16")
    except Exception:
        try:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            print("[INIT] Mixed precision: mixed_float16")
        except Exception:
            print("[INIT] Mixed precision disabled")
else:
    print("[INIT] No GPU, using CPU float32")

print(f"[INIT] TF {tf.__version__}  |  GPUs: {gpus}")

from ohca_predictor.config import (
    SimulationConfig, TrainingConfig, ModelConfig, EvaluationConfig,
    override_config,
)
from ohca_predictor.simulator.generator import DataGenerator
from ohca_predictor.utils.io_utils import create_padded_dataset, WindowSample
from ohca_predictor.model.architecture import OHCAPredictionModel
from ohca_predictor.model.losses import CombinedOHCALoss
from ohca_predictor.training.trainer import CosineDecayWithWarmup
from ohca_predictor.evaluation.metrics import OHCAEvaluator
from ohca_predictor.evaluation.calibration import CalibrationAnalyzer

# ── Constants ──────────────────────────────────────────────────────
PIPELINE_TIMEOUT_SECONDS = 12 * 3600
LOG_DIR = "logs/pipeline"
TENSORBOARD_DIR = os.path.join(LOG_DIR, "tensorboard")
RESULTS_DIR = "results"
MODEL_DIR = "models"
LOG_FILE = "pipeline_output.log"

for d in [LOG_DIR, TENSORBOARD_DIR, RESULTS_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

with open(LOG_FILE, "w") as f:
    f.write("")


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
        f.write(line + "\n")


# ── Timeout handler ────────────────────────────────────────────────
_timeout_triggered = False


def _timeout_handler(signum, frame):
    global _timeout_triggered
    _timeout_triggered = True
    log(f"[TIMEOUT] Hard timeout reached. Finishing current epoch...")


signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(PIPELINE_TIMEOUT_SECONDS)


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

def make_config():
    return override_config(
        simulation=SimulationConfig(
            min_window_duration_hours=0.05,
            max_window_duration_hours=0.12,
            population_size=250,
            prevalence_ohca=0.01,
        ),
        training=TrainingConfig(
            batch_size=4,
            epochs=200,
            learning_rate=2e-4,
            warmup_steps=50,
            min_learning_rate=1e-6,
            mixed_precision=True,
            gradient_clip_norm=1.0,
            gradient_accumulation_steps=1,
            focal_loss_gamma=2.0,
            false_negative_weight=10.0,
            false_positive_weight=1.0,
            survival_loss_weight=0.3,
            auxiliary_loss_weight=0.1,
            contrastive_loss_weight=0.0,
            reconstruction_loss_weight=0.0,
            early_stopping_patience=25,
            checkpoint_save_best_only=True,
            weight_decay=0.01,
        ),
        model=ModelConfig(
            model_dim=192,
            num_attention_heads=8,
            num_encoder_layers=4,
            feedforward_dim=512,
            dropout_rate=0.2,
            attention_dropout_rate=0.1,
            static_embedding_dim=64,
            tokens_per_modality=64,
            max_positional_encoding=4096,
            num_survival_bins=12,
            uncertainty_samples=30,
        ),
        evaluation=EvaluationConfig(
            bootstrap_iterations=200,
            confidence_level=0.95,
            clinical_thresholds=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
            primary_threshold=0.15,
            calibration_bins=10,
        ),
    )


# ═══════════════════════════════════════════════════════════════════
# Keras Custom Training Model — eliminates gradient accumulation bugs
# ═══════════════════════════════════════════════════════════════════

class OHCA_TrainableModel(tf.keras.Model):
    """Wraps OHCAPredictionModel with a robust train_step/test_step.

    This avoids manual gradient accumulation entirely — Keras handles
    loss scaling, gradient clipping, and variable management.
    """

    def __init__(self, ohca_model, loss_fn, clip_norm=1.0, **kwargs):
        super().__init__(**kwargs)
        self.ohca_model = ohca_model
        self.loss_fn = loss_fn
        self.clip_norm = clip_norm

        self._train_loss = tf.keras.metrics.Mean(name="loss")
        self._val_loss = tf.keras.metrics.Mean(name="val_loss")

    @property
    def metrics(self):
        return [self._train_loss, self._val_loss]

    def call(self, inputs, training=False):
        return self.ohca_model(inputs, training=training)

    def train_step(self, batch):
        with tf.GradientTape() as tape:
            outputs = self(batch, training=True)
            base_loss = self.loss_fn(batch, outputs)

            # Curriculum time-to-event weighting: penalize late detections
            # Windows closer to OHCA get LOWER weight (easy to detect)
            # Windows 30min-2h before get HIGHER weight (clinically valuable)
            time_to_event = batch.get("time_to_event")
            ohca_label = batch.get("ohca_label")
            if time_to_event is not None and ohca_label is not None:
                tte_hours = tf.cast(time_to_event, tf.float32) / 3600.0
                label_f = tf.cast(ohca_label, tf.float32)
                # Weight peaks at 1-2 hours before OHCA, drops for very close (<30min) and far (>4h)
                time_weight = tf.where(
                    label_f > 0.5,
                    tf.exp(-0.5 * (tte_hours - 1.5) ** 2) * 1.5 + 0.5,  # Gaussian peak at 1.5h
                    1.0  # normal weight for negatives
                )
                time_weight = tf.clip_by_value(time_weight, 0.3, 3.0)
                weighted_loss = tf.reduce_mean(base_loss * time_weight)
            else:
                weighted_loss = base_loss

        trainable_vars = self.ohca_model.trainable_variables
        grads = tape.gradient(weighted_loss, trainable_vars)

        # Filter None grads, NaN/Inf grads, and clip
        grads_and_vars = []
        for g, v in zip(grads, trainable_vars):
            if g is not None:
                g = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                g = tf.clip_by_norm(g, self.clip_norm)
                grads_and_vars.append((g, v))

        self.optimizer.apply_gradients(grads_and_vars)
        self._train_loss.update_state(weighted_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, batch):
        outputs = self(batch, training=False)
        loss = self.loss_fn(batch, outputs)
        self._val_loss.update_state(loss)
        return {"loss": self._val_loss.result()}

    def predict_batch(self, batch):
        """Run inference and return outputs dict."""
        return self(batch, training=False)


# ═══════════════════════════════════════════════════════════════════
# TensorBoard callback for clinical metrics
# ═══════════════════════════════════════════════════════════════════

class ClinicalMetricsCallback(tf.keras.callbacks.Callback):
    """Evaluates clinical metrics at end of each epoch and logs to TensorBoard."""

    def __init__(self, val_samples, config, tb_writer):
        super().__init__()
        self.val_samples = val_samples
        self.config = config
        self.tb_writer = tb_writer
        self.evaluator = OHCAEvaluator(config.evaluation)
        self.cal_analyzer = CalibrationAnalyzer(config.evaluation)
        self.best_auroc = 0.0
        self.patience_counter = 0
        self.history = {
            "train_loss": [], "val_loss": [],
            "val_auroc": [], "val_sens": [], "val_spec": [],
            "val_ppv": [], "val_npv": [], "val_f1": [],
            "val_brier": [], "val_ece": [], "val_auprc": [],
            "lr": [],
        }

        # Pre-build validation dataset
        self.val_ds = create_padded_dataset(
            val_samples, batch_size=self.config.training.batch_size, shuffle=False,
        )

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get("loss", 0.0)
        val_loss = logs.get("val_loss", logs.get("loss", 0.0))
        current_lr = float(self.model.optimizer.learning_rate
                           if hasattr(self.model.optimizer, "learning_rate")
                           else 0.0)

        # Try to get lr from schedule
        try:
            if hasattr(self.model.optimizer, "_decayed_lr"):
                current_lr = float(self.model.optimizer._decayed_lr(tf.float32))
        except Exception:
            pass

        # Collect all validation predictions
        all_risks = []
        all_labels = []
        all_surv = []
        all_uncert = []

        for batch in self.val_ds:
            outputs = self.model.predict_batch(batch)
            all_risks.append(outputs["ohca_risk"].numpy().ravel())
            all_labels.append(batch["ohca_label"].numpy().ravel())
            all_surv.append(outputs["survival_curve"].numpy())
            all_uncert.append(outputs["uncertainty"].numpy())

        all_risks = np.concatenate(all_risks)
        all_labels = np.concatenate(all_labels)
        all_surv = np.concatenate(all_surv)
        all_uncert = np.concatenate(all_uncert)

        # OOD rejection: flag high-uncertainty predictions
        if all_uncert.size > 0:
            epistemic_uncert = all_uncert[:, 1] if all_uncert.ndim > 1 else all_uncert
            ood_threshold = np.percentile(epistemic_uncert, 90)  # top 10% uncertain
            ood_mask = epistemic_uncert > ood_threshold
            if np.sum(ood_mask) > 0 and np.sum(~ood_mask) > 0:
                auroc_in_dist, _, _ = self.evaluator.compute_auroc(
                    all_labels[~ood_mask], all_risks[~ood_mask])
                log(f"    OOD-rejected AUROC (in-dist, n={np.sum(~ood_mask)}): {auroc_in_dist:.3f}")

        # Compute clinical metrics
        auroc = sens = spec = ppv = npv_val = f1 = brier = ece = auprc = 0.0
        auroc_lo = auroc_hi = 0.0
        thr = self.config.evaluation.primary_threshold

        try:
            auroc, auroc_lo, auroc_hi = self.evaluator.compute_auroc(all_labels, all_risks)
            auprc, _, _ = self.evaluator.compute_auprc(all_labels, all_risks)
            sens, _, _ = self.evaluator.compute_sensitivity_at_threshold(all_labels, all_risks, thr)
            spec, _, _ = self.evaluator.compute_specificity_at_threshold(all_labels, all_risks, thr)
            ppv, _, _ = self.evaluator.compute_ppv(all_labels, all_risks, thr)
            npv_val, _, _ = self.evaluator.compute_npv(all_labels, all_risks, thr)
            f1, _, _ = self.evaluator.compute_f1_score(all_labels, all_risks, thr)
            brier, _, _ = self.evaluator.compute_brier_score(all_labels, all_risks)
        except Exception as e:
            log(f"  [WARN] Metric error: {e}")

        try:
            cal_m = self.cal_analyzer.compute_calibration_metrics(all_labels, all_risks)
            ece = cal_m.get("ece", 0.0)
        except Exception:
            pass

        mean_uncert = float(np.mean(all_uncert[:, 1])) if all_uncert.size > 0 else 0.0
        elapsed = logs.get("time", 0.0)

        # Record history
        self.history["train_loss"].append(float(train_loss))
        self.history["val_loss"].append(float(val_loss))
        self.history["val_auroc"].append(float(auroc))
        self.history["val_sens"].append(float(sens))
        self.history["val_spec"].append(float(spec))
        self.history["val_ppv"].append(float(ppv))
        self.history["val_npv"].append(float(npv_val))
        self.history["val_f1"].append(float(f1))
        self.history["val_brier"].append(float(brier))
        self.history["val_ece"].append(float(ece))
        self.history["val_auprc"].append(float(auprc))
        self.history["lr"].append(current_lr)

        # Console log
        log(f"Epoch {epoch+1:3d}/{self.config.training.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"AUROC={auroc:.3f} [{auroc_lo:.3f}–{auroc_hi:.3f}] "
            f"Sens={sens:.3f} Spec={spec:.3f} PPV={ppv:.3f} F1={f1:.3f} | "
            f"Brier={brier:.4f} ECE={ece:.4f} "
            f"uncert={mean_uncert:.2f} lr={current_lr:.2e} "
            f"({elapsed:.0f}s)")

        # TensorBoard scalars
        with self.tb_writer.as_default():
            tf.summary.scalar("train/loss", train_loss, step=epoch)
            tf.summary.scalar("train/learning_rate", current_lr, step=epoch)
            tf.summary.scalar("val/loss", val_loss, step=epoch)
            tf.summary.scalar("val/AUROC", auroc, step=epoch)
            tf.summary.scalar("val/AUPRC", auprc, step=epoch)
            tf.summary.scalar("val/sensitivity", sens, step=epoch)
            tf.summary.scalar("val/specificity", spec, step=epoch)
            tf.summary.scalar("val/PPV", ppv, step=epoch)
            tf.summary.scalar("val/NPV", npv_val, step=epoch)
            tf.summary.scalar("val/F1", f1, step=epoch)
            tf.summary.scalar("val/Brier", brier, step=epoch)
            tf.summary.scalar("val/ECE", ece, step=epoch)
            tf.summary.scalar("val/uncertainty", mean_uncert, step=epoch)
            tf.summary.histogram("val/risk_predictions", all_risks, step=epoch)
            tf.summary.histogram("val/true_labels", all_labels, step=epoch)
            tf.summary.scalar("val/survival_mean", float(np.mean(all_surv)), step=epoch)

            # Gradient norm from last train batch
            if hasattr(self.model, "ohca_model"):
                try:
                    vars_sample = self.model.ohca_model.trainable_variables[:5]
                    grad_norms = [tf.reduce_sum(g**2)
                                  for g in tape.gradient(0.0, vars_sample) if g is not None]
                    if grad_norms:
                        tf.summary.scalar("train/grad_norm_approx",
                                          float(tf.sqrt(tf.add_n(grad_norms))), step=epoch)
                except Exception:
                    pass

        self.tb_writer.flush()

        # Early stopping / checkpointing
        if auroc > self.best_auroc:
            self.best_auroc = auroc
            self.patience_counter = 0
            ckpt_path = os.path.join(MODEL_DIR, "best_checkpoint.weights.h5")
            self.model.ohca_model.save_weights(ckpt_path)
            log(f"  ★ New best AUROC={auroc:.4f} → saved checkpoint")
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.training.early_stopping_patience:
                log(f"  Early stopping at epoch {epoch+1}")
                self.model.stop_training = True

        # Periodic checkpoint
        if (epoch + 1) % 20 == 0:
            p_path = os.path.join(MODEL_DIR, f"checkpoint_epoch{epoch+1}.weights.h5")
            self.model.ohca_model.save_weights(p_path)
            log(f"  Checkpoint saved: {p_path}")


# ═══════════════════════════════════════════════════════════════════
# PHASE 1 — DATA GENERATION
# ═══════════════════════════════════════════════════════════════════

def phase1_generate(config):
    log("=" * 70)
    log("PHASE 1: SYNTHETIC DATA GENERATION (enhanced realism)")
    log("=" * 70)

    gen = DataGenerator(config)
    n_patients = config.simulation.population_size

    all_train, all_val, all_test = [], [], []
    batch_size = 25
    t0 = time.time()

    for i in range(0, n_patients, batch_size):
        if _timeout_triggered:
            break
        bn = min(batch_size, n_patients - i)
        log(f"  Patients {i+1:4d}–{i+bn:4d} / {n_patients} ...")
        t, v, te = gen.generate_dataset(n_patients=bn)
        all_train.extend(t)
        all_val.extend(v)
        all_test.extend(te)

    elapsed = time.time() - t0
    log(f"  Generation done in {elapsed:.0f}s")

    pos_train = sum(1 for s in all_train if s.ohca_label > 0.5)
    pos_val = sum(1 for s in all_val if s.ohca_label > 0.5)
    pos_test = sum(1 for s in all_test if s.ohca_label > 0.5)

    ecg_lens = np.array([s.ecg.shape[0] for s in all_train])
    ecg_durations = ecg_lens / config.simulation.sampling_rates["ecg_hz"]

    log(f"  Train: {len(all_train)} windows ({pos_train} positive, "
        f"{pos_train/max(1,len(all_train))*100:.1f}%)")
    log(f"  Val:   {len(all_val)} windows ({pos_val} positive, "
        f"{pos_val/max(1,len(all_val))*100:.1f}%)")
    log(f"  Test:  {len(all_test)} windows ({pos_test} positive, "
        f"{pos_test/max(1,len(all_test))*100:.1f}%)")
    log(f"  ECG durations (s): min={ecg_durations.min():.1f} "
        f"max={ecg_durations.max():.1f} mean={ecg_durations.mean():.1f}")

    if len(all_train) > 0:
        s0 = all_train[0]
        log(f"  Signal shapes: ECG={s0.ecg.shape} Accel={s0.accelerometer.shape} "
            f"PPG={s0.ppg.shape}")
        log(f"  Static: Demo={s0.demographics.shape} Meds={s0.medications.shape} "
            f"Comorb={s0.comorbidities.shape} Labs={s0.lab_values.shape}")

    return all_train, all_val, all_test


# ═══════════════════════════════════════════════════════════════════
# PHASE 2 — TRAINING
# ═══════════════════════════════════════════════════════════════════

def phase2_train(config, train_samples, val_samples):
    log("=" * 70)
    log("PHASE 2: TRAINING (stable Keras custom train_step)")
    log("=" * 70)

    train_ds = create_padded_dataset(
        train_samples, batch_size=config.training.batch_size, shuffle=True,
    )
    val_ds = create_padded_dataset(
        val_samples, batch_size=config.training.batch_size, shuffle=False,
    )

    train_batches = sum(1 for _ in train_ds)
    val_batches = sum(1 for _ in val_ds)
    log(f"  Train batches: {train_batches}  |  Val batches: {val_batches}")

    # Recreate (datasets consumed)
    train_ds = create_padded_dataset(
        train_samples, batch_size=config.training.batch_size, shuffle=True,
    )

    # Model dimensions from actual data
    demo_dim = int(train_samples[0].demographics.shape[0])
    n_meds = int(train_samples[0].medications.shape[0])
    n_comorb = int(train_samples[0].comorbidities.shape[0])
    n_labs = int(train_samples[0].lab_values.shape[0])

    log(f"  Input dims: demographics={demo_dim} meds={n_meds} "
        f"comorbidities={n_comorb} labs={n_labs}")

    ohca_model = OHCAPredictionModel(
        config=config.model,
        demographic_dim=demo_dim,
        num_medications=n_meds,
        num_comorbidities=n_comorb,
        num_labs=n_labs,
    )

    # Build model
    dummy_batch = next(iter(val_ds))
    _ = ohca_model(dummy_batch, training=False)
    n_params = sum(np.prod(v.shape) for v in ohca_model.trainable_variables)
    log(f"  Model params: {n_params:,.0f}")

    # Sequence length for logging
    seq_len = 1 + 4 * config.model.tokens_per_modality
    log(f"  Sequence length: {seq_len} (1 static + 4×{config.model.tokens_per_modality})")

    # Loss
    loss_fn = CombinedOHCALoss(
        fn_weight=config.training.false_negative_weight,
        fp_weight=config.training.false_positive_weight,
        focal_gamma=config.training.focal_loss_gamma,
    )

    # LR schedule + optimizer
    steps_per_epoch = max(1, train_batches)
    total_steps = config.training.epochs * steps_per_epoch
    log(f"  Steps/epoch: {steps_per_epoch}  |  Total steps: {total_steps}")

    lr_schedule = CosineDecayWithWarmup(
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        total_steps=total_steps,
        min_learning_rate=config.training.min_learning_rate,
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.training.weight_decay,
    )

    # Wrap in custom training model
    trainable_model = OHCA_TrainableModel(
        ohca_model=ohca_model,
        loss_fn=loss_fn,
        clip_norm=config.training.gradient_clip_norm,
    )
    trainable_model.compile(optimizer=optimizer, jit_compile=False)

    # TensorBoard
    tb_writer = tf.summary.create_file_writer(TENSORBOARD_DIR)
    log(f"  TensorBoard: {TENSORBOARD_DIR}")

    # Clinical metrics callback
    clinical_cb = ClinicalMetricsCallback(val_samples, config, tb_writer)

    # ── Train ──────────────────────────────────────────────────────
    t_train_start = time.time()
    log("  Starting training...")
    trainable_model.load_weights(os.path.join(MODEL_DIR, "best_checkpoint.weights.h5"), skip_mismatch=True)
    trainable_model.summary()

    return trainable_model


def main():
    t0 = time.time()
    log("=" * 70)
    log("OHCA PREDICTION — PRODUCTION PIPELINE v2")
    log(f"  Timeout: {PIPELINE_TIMEOUT_SECONDS/3600:.0f}h")
    log(f"  TF: {tf.__version__}")
    log(f"  GPUs: {gpus}")
    log(f"  Mixed precision: {tf.keras.mixed_precision.global_policy().name}")
    log("=" * 70)

    try:
        config = make_config()

        train_samples, val_samples, test_samples = phase1_generate(config)

        if len(train_samples) == 0:
            log("[ERROR] No training data. Aborting.")
            return

        model = phase2_train(config, train_samples, val_samples)

    except Exception as e:
        log(f"\n[FATAL] Pipeline error: {e}")
        traceback.print_exc()
        with open(LOG_FILE, "a") as f:
            traceback.print_exc(file=f)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
