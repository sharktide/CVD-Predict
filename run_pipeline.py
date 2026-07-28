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
import joblib as jb
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

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
tf.config.experimental.enable_tensor_float_32_execution(True)
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
from sklearn.metrics import accuracy_score

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
            population_size=1000,
            prevalence_ohca=0.15,
        ),
        training=TrainingConfig(
            batch_size=4,
            epochs=300,
            learning_rate=4e-4,
            warmup_steps=300,
            min_learning_rate=1e-6,
            mixed_precision=True,
            gradient_clip_norm=0.5,
            gradient_accumulation_steps=8,
            focal_loss_gamma=2.0,
            false_negative_weight=5.0,
            false_positive_weight=1.0,
            survival_loss_weight=0.3,
            auxiliary_loss_weight=0.1,
            contrastive_loss_weight=0.0,
            reconstruction_loss_weight=0.0,
            early_stopping_patience=30,
            checkpoint_save_best_only=True,
            weight_decay=0.01,
        ),
        model=ModelConfig(
            model_dim=192,
            num_attention_heads=8,
            num_encoder_layers=6,
            feedforward_dim=512,
            dropout_rate=0.2,
            attention_dropout_rate=0.1,
            static_embedding_dim=64,
            tokens_per_modality=128,
            max_positional_encoding=4096,
            num_survival_bins=12,
            uncertainty_samples=5,
        ),
        evaluation=EvaluationConfig(
            bootstrap_iterations=200,
            confidence_level=0.95,
            clinical_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            primary_threshold=0.50,
            calibration_bins=10,
        ),
    )


# ═══════════════════════════════════════════════════════════════════
# Keras Custom Training Model — eliminates gradient accumulation bugs
# ═══════════════════════════════════════════════════════════════════
class OHCA_TrainableModel(tf.keras.Model):
    """Wraps OHCAPredictionModel with custom gradient accumulation.
    
    Processes small physical batches to save VRAM, while accumulating 
    gradients over N steps to simulate a large, stable effective batch size.
    """

    def __init__(self, ohca_model, loss_fn, clip_norm=1.0, accum_steps=4, **kwargs):
        super().__init__(**kwargs)
        self.ohca_model = ohca_model
        self.loss_fn = loss_fn
        self.clip_norm = clip_norm
        self.accum_steps = accum_steps

        # Track the current accumulation step (0 to accum_steps)
        self.step_counter = tf.Variable(0, dtype=tf.int32, trainable=False)

        self._train_loss = tf.keras.metrics.Mean(name="loss")
        self._val_loss = tf.keras.metrics.Mean(name="val_loss")

    def compile(self, optimizer, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)
        # Create zero-initialized tracking tensors matching the model's weights
        self.gradient_accumulators = [
            tf.Variable(tf.zeros_like(v), trainable=False)
            for v in self.ohca_model.trainable_variables
        ]

    @property
    def metrics(self):
        return [self._train_loss, self._val_loss]

    def call(self, inputs, training=False):
        return self.ohca_model(inputs, training=training)

    def train_step(self, batch):
        accum_steps_f = tf.cast(self.accum_steps, tf.float32)

        with tf.GradientTape() as tape:
            outputs = self(batch, training=True)
            base_loss = self.loss_fn(batch, outputs)

            # Curriculum time-to-event weighting
            time_to_event = batch.get("time_to_event")
            ohca_label = batch.get("ohca_label")
            if time_to_event is not None and ohca_label is not None:
                tte_hours = tf.cast(time_to_event, tf.float32) / 3600.0
                label_f = tf.cast(ohca_label, tf.float32)
                time_weight = tf.where(
                    label_f > 0.5,
                    tf.exp(-0.3 * (tte_hours - 1.5) ** 2) * 0.8 + 0.6,
                    1.0
                )
                time_weight = tf.clip_by_value(time_weight, 0.6, 1.8)
                weighted_loss = tf.reduce_mean(base_loss * time_weight)
            else:
                weighted_loss = base_loss

            # Scale loss down for accumulation safety
            scaled_loss = weighted_loss / accum_steps_f

        trainable_vars = self.ohca_model.trainable_variables
        grads = tape.gradient(scaled_loss, trainable_vars)

        # Filter and accumulate gradients natively
        for i, g in enumerate(grads):
            if g is not None:
                g_clean = tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                self.gradient_accumulators[i].assign_add(g_clean)

        # Advance our accumulation counter step
        self.step_counter.assign_add(1)

        # Conditional function: Triggers optimizer ONLY when step_counter == accum_steps
        def apply_gradients_stage():
            grads_and_vars = []
            accum_grads = [v.read_value() for v in self.gradient_accumulators]
            
            # Apply global norm clipping across the accumulated total
            clipped_grads, _ = tf.clip_by_global_norm(accum_grads, self.clip_norm)
            
            for g, v in zip(clipped_grads, trainable_vars):
                grads_and_vars.append((g, v))

            # Apply steps to AdamW optimizer
            self.optimizer.apply_gradients(grads_and_vars)

            # Clear out the accumulators entirely for the next macro-step cycle
            for i in range(len(self.gradient_accumulators)):
                self.gradient_accumulators[i].assign(tf.zeros_like(self.gradient_accumulators[i]))
            self.step_counter.assign(0)
            return None  # Explicitly return None (0 outputs)

        # Execute conditional update check
        # Using lambda: None ensures both branches return exactly 0 outputs
        tf.cond(
            tf.equal(self.step_counter, self.accum_steps),
            apply_gradients_stage,
            lambda: None
        )

        # Keep tracking metrics based on the unscaled training loss
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

    def __init__(self, val_samples, config, tb_writer, loss_fn=None):
        super().__init__()
        self.val_samples = val_samples
        self.config = config
        self.tb_writer = tb_writer
        self.loss_fn = loss_fn
        self.evaluator = OHCAEvaluator(config.evaluation)
        self.cal_analyzer = CalibrationAnalyzer(config.evaluation)
        self.best_auroc = 0.0
        self.patience_counter = 0
        self.history = {
            "train_loss": [], "val_loss": [],
            "val_auroc": [], "val_sens": [], "val_spec": [],
            "val_ppv": [], "val_npv": [], "val_f1": [],
            "val_brier": [], "val_ece": [], "val_auprc": [],
            "val_acc_opt": [], "val_thr_opt": [],
            "lr": [],
        }

        self.val_ds = create_padded_dataset(
            val_samples, batch_size=self.config.training.batch_size, shuffle=False,
        )

    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get("loss", 0.0)
        current_lr = float(self.model.optimizer.learning_rate
                           if hasattr(self.model.optimizer, "learning_rate")
                           else 0.0)

        try:
            if hasattr(self.model.optimizer, "_decayed_lr"):
                current_lr = float(self.model.optimizer._decayed_lr(tf.float32))
        except Exception:
            pass

        all_risks, all_labels, all_surv, all_uncert = [], [], [], []
        val_loss_total = 0.0
        val_batches = 0
        for batch in self.val_ds:
            outputs = self.model.predict_batch(batch)
            all_risks.append(outputs["ohca_risk"].numpy().ravel())
            all_labels.append(batch["ohca_label"].numpy().ravel())
            all_surv.append(outputs["survival_curve"].numpy())
            all_uncert.append(outputs["uncertainty"].numpy())
            # Compute val loss from the model's loss function
            try:
                vl = self.model.loss_fn(batch, outputs)
                val_loss_total += float(vl.numpy())
                val_batches += 1
            except Exception:
                pass
        val_loss = val_loss_total / max(1, val_batches)

        all_risks = np.concatenate(all_risks)
        all_labels = np.concatenate(all_labels)
        all_surv = np.concatenate(all_surv)
        all_uncert = np.concatenate(all_uncert)

        auroc = sens = spec = ppv = npv_val = f1 = brier = ece = auprc = 0.0
        auroc_lo = auroc_hi = 0.0
        thr = self.config.evaluation.primary_threshold
        acc_opt = thr_opt = 0.0

        try:
            auroc, auroc_lo, auroc_hi = self.evaluator.compute_auroc(all_labels, all_risks)
            auprc, _, _ = self.evaluator.compute_auprc(all_labels, all_risks)
            sens, _, _ = self.evaluator.compute_sensitivity_at_threshold(all_labels, all_risks, thr)
            spec, _, _ = self.evaluator.compute_specificity_at_threshold(all_labels, all_risks, thr)
            ppv, _, _ = self.evaluator.compute_ppv(all_labels, all_risks, thr)
            npv_val, _, _ = self.evaluator.compute_npv(all_labels, all_risks, thr)
            f1, _, _ = self.evaluator.compute_f1_score(all_labels, all_risks, thr)
            brier, _, _ = self.evaluator.compute_brier_score(all_labels, all_risks)

            # 🔑 Optimal threshold for accuracy
            thresholds = np.linspace(0, 1, 200)
            accs = [accuracy_score(all_labels, (all_risks >= t).astype(int)) for t in thresholds]
            idx = int(np.argmax(accs))
            acc_opt = accs[idx]
            thr_opt = thresholds[idx]

        except Exception as e:
            logger.warning(f"Metric error: {e}")

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
        self.history["val_acc_opt"].append(float(acc_opt))
        self.history["val_thr_opt"].append(float(thr_opt))
        self.history["lr"].append(current_lr)

        # Console output (Rich)
        table = Table(title=f"Epoch {epoch+1}/{self.config.training.epochs}", show_lines=True)
        table.add_column("Metric", style="cyan", justify="right")
        table.add_column("Value", style="magenta", justify="center")

        table.add_row("Train Loss", f"{train_loss:.4f}")
        table.add_row("Val Loss", f"{val_loss:.4f}")
        table.add_row("AUROC", f"{auroc:.3f} [{auroc_lo:.3f}–{auroc_hi:.3f}]")
        table.add_row("AUPRC", f"{auprc:.3f}")
        table.add_row("Acc(opt)", f"{acc_opt:.3f} @ thr={thr_opt:.2f}")
        table.add_row("Sensitivity", f"{sens:.3f}")
        table.add_row("Specificity", f"{spec:.3f}")
        table.add_row("PPV", f"{ppv:.3f}")
        table.add_row("NPV", f"{npv_val:.3f}")
        table.add_row("F1", f"{f1:.3f}")
        table.add_row("Brier", f"{brier:.4f}")
        table.add_row("ECE", f"{ece:.4f}")
        table.add_row("Uncertainty", f"{mean_uncert:.2f}")
        table.add_row("Learning Rate", f"{current_lr:.2e}")
        table.add_row("Elapsed", f"{elapsed:.0f}s")

        console.print(Panel(table, title="Validation Metrics", border_style="green"))

        # File log (plain text)
        logger.info(
            f"Epoch {epoch+1}/{self.config.training.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"AUROC={auroc:.3f} [{auroc_lo:.3f}–{auroc_hi:.3f}] "
            f"AUPRC={auprc:.3f} Acc(opt)={acc_opt:.3f}@thr={thr_opt:.2f} | "
            f"Sens={sens:.3f} Spec={spec:.3f} PPV={ppv:.3f} F1={f1:.3f} | "
            f"Brier={brier:.4f} ECE={ece:.4f} "
            f"uncert={mean_uncert:.2f} lr={current_lr:.2e} "
            f"({elapsed:.0f}s)"
        )

        # TensorBoard scalars
        with self.tb_writer.as_default():
            tf.summary.scalar("train/loss", train_loss, step=epoch)
            tf.summary.scalar("train/learning_rate", current_lr, step=epoch)
            tf.summary.scalar("val/loss", val_loss, step=epoch)
            tf.summary.scalar("val/AUROC", auroc, step=epoch)
            tf.summary.scalar("val/AUPRC", auprc, step=epoch)
            tf.summary.scalar("val/Accuracy_opt", acc_opt, step=epoch)
            tf.summary.scalar("val/Threshold_opt", thr_opt, step=epoch)
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

        self.tb_writer.flush()

        # Early stopping / checkpointing
        if auroc > self.best_auroc:
            self.best_auroc = auroc
            self.patience_counter = 0
            ckpt_path = os.path.join(MODEL_DIR, "best_checkpoint.weights.h5")
            self.model.ohca_model.save_weights(ckpt_path)
            logger.info(f"★ New best AUROC={auroc:.4f} → saved checkpoint")
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.training.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                self.model.stop_training = True

        # Periodic checkpoint
        if (epoch + 1) % 20 == 0:
            p_path = os.path.join(MODEL_DIR, f"checkpoint_epoch{epoch+1}.weights.h5")
            self.model.ohca_model.save_weights(p_path)
            logger.info(f"Checkpoint saved: {p_path}")

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
    batch_size = 100
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
    tf.keras.backend.clear_session()
    gc.collect()
    log("=" * 70)
    log("PHASE 2: TRAINING (stable Keras custom train_step)")
    log("=" * 70)

    import numpy as np

    # 1. Calculate the batch counts mathematically using Python (Lightning fast, 0 memory)
    steps_per_epoch = int(np.ceil(len(train_samples) / config.training.batch_size))  // config.training.gradient_accumulation_steps
    val_batches = int(np.ceil(len(val_samples) / config.training.batch_size))

    log(f"  Train batches: {steps_per_epoch}  |  Val batches: {val_batches}")

    # 2. Build the datasets EXACTLY ONCE and prepare them straight for training
    train_ds = create_padded_dataset(
        train_samples, batch_size=config.training.batch_size, shuffle=True,
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = create_padded_dataset(
        val_samples, batch_size=config.training.batch_size, shuffle=False,
    ).cache().prefetch(tf.data.AUTOTUNE)

    # Now train_ds and val_ds are perfectly fresh and ready for model.fit()!

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
        accum_steps=config.training.gradient_accumulation_steps 
    )
    trainable_model.compile(optimizer=optimizer, jit_compile=False)

    # TensorBoard
    tb_writer = tf.summary.create_file_writer(TENSORBOARD_DIR)
    log(f"  TensorBoard: {TENSORBOARD_DIR}")

    # Clinical metrics callback
    clinical_cb = ClinicalMetricsCallback(val_samples, config, tb_writer, loss_fn=loss_fn)

    # ── Train ──────────────────────────────────────────────────────
    t_train_start = time.time()
    log("  Starting training...")
    trainable_model.summary()
    trainable_model.fit(
        train_ds,
        epochs=config.training.epochs,
        steps_per_epoch=steps_per_epoch, 
        callbacks=[clinical_cb],
        verbose=1
    )

    total_train_time = time.time() - t_train_start
    log(f"  Total training time: {total_train_time/3600:.2f}h ({total_train_time:.0f}s)")

    # Restore best weights
    best_ckpt = os.path.join(MODEL_DIR, "best_checkpoint.weights.h5")
    if os.path.exists(best_ckpt):
        ohca_model.load_weights(best_ckpt)
        log(f"  Restored best weights (AUROC={clinical_cb.best_auroc:.4f})")

    # Save history
    hist_path = os.path.join(RESULTS_DIR, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(clinical_cb.history, f, indent=2, default=str)
    log(f"  Training history saved: {hist_path}")
    ohca_model.save("models/ohca_model_v1.h5")
    ohca_model.summary()
    return ohca_model, clinical_cb.history, clinical_cb.best_auroc


# ═══════════════════════════════════════════════════════════════════
# PHASE 3 — COMPREHENSIVE EVALUATION
# ═══════════════════════════════════════════════════════════════════

def phase3_evaluate(config, model, test_samples, history):
    log("=" * 70)
    log("PHASE 3: COMPREHENSIVE EVALUATION ON HELD-OUT TEST SET")
    log("=" * 70)

    test_ds = create_padded_dataset(
        test_samples, batch_size=config.training.batch_size, shuffle=False,
    )

    all_risks = []
    all_raw_risks = []
    all_labels = []
    all_survival = []
    all_uncertainty = []
    all_tte = []
    all_evt = []

    for batch in test_ds:
        outputs = model(batch, training=False)
        all_risks.append(outputs["ohca_risk"].numpy().ravel())
        all_raw_risks.append(outputs["raw_risk"].numpy().ravel())
        all_labels.append(batch["ohca_label"].numpy().ravel())
        all_survival.append(outputs["survival_curve"].numpy())
        all_uncertainty.append(outputs["uncertainty"].numpy())
        all_tte.append(batch["time_to_event"].numpy().ravel())
        all_evt.append(batch["event_indicator"].numpy().ravel())

    all_risks = np.asarray(np.concatenate(all_risks), dtype=np.float32).ravel()
    all_raw_risks = np.asarray(np.concatenate(all_raw_risks), dtype=np.float32).ravel()
    all_labels = np.asarray(np.concatenate(all_labels), dtype=np.float32).ravel()
    all_survival = np.asarray(np.concatenate(all_survival), dtype=np.float32)
    all_uncertainty = np.asarray(np.concatenate(all_uncertainty), dtype=np.float32)
    all_tte = np.asarray(np.concatenate(all_tte), dtype=np.float32).ravel()
    all_evt = np.asarray(np.concatenate(all_evt), dtype=np.float32).ravel()

    n_total = len(all_labels)
    n_pos = int(all_labels.sum())
    log(f"  Test set: {n_total} samples, {n_pos} positive "
        f"({n_pos/max(1,n_total)*100:.1f}%)")

    evaluator = OHCAEvaluator(config.evaluation)

    # ── 3a. Core metrics at primary threshold ──────────────────────
    thr = config.evaluation.primary_threshold
    log(f"\n  ── Core Metrics (threshold={thr}) ──")
    metrics = evaluator.compute_all_metrics(all_labels, all_risks, threshold=thr)
    evaluator.print_report(metrics)

    # ── 3b. Multi-threshold analysis ───────────────────────────────
    log(f"\n  ── Multi-Threshold Analysis ──")
    threshold_results = []
    for t in config.evaluation.clinical_thresholds:
        s, _, _ = evaluator.compute_sensitivity_at_threshold(all_labels, all_risks, t)
        sp, _, _ = evaluator.compute_specificity_at_threshold(all_labels, all_risks, t)
        p, _, _ = evaluator.compute_ppv(all_labels, all_risks, t)
        n_v, _, _ = evaluator.compute_npv(all_labels, all_risks, t)
        f, _, _ = evaluator.compute_f1_score(all_labels, all_risks, t)
        youden = s + sp - 1.0
        threshold_results.append({
            "threshold": t, "sensitivity": s, "specificity": sp,
            "ppv": p, "npv": n_v, "f1": f, "youden_j": youden,
        })
        log(f"    thr={t:.2f}: Sens={s:.3f} Spec={sp:.3f} "
            f"PPV={p:.3f} NPV={n_v:.3f} F1={f:.3f} J={youden:.3f}")

    best_j_idx = max(range(len(threshold_results)),
                     key=lambda i: threshold_results[i]["youden_j"])
    best_thr = threshold_results[best_j_idx]["threshold"]
    best_j = threshold_results[best_j_idx]["youden_j"]
    log(f"\n  Optimal threshold (Youden's J): {best_thr:.2f} (J={best_j:.3f})")

    log(f"\n  ── Metrics at Optimal Threshold ({best_thr}) ──")
    opt_metrics = evaluator.compute_all_metrics(all_labels, all_risks, threshold=best_thr)
    evaluator.print_report(opt_metrics)

    # ── 3c. Calibration ────────────────────────────────────────────
    log(f"\n  ── Calibration Analysis ──")
    cal_analyzer = CalibrationAnalyzer(config.evaluation)
    cal_metrics = {}
    try:
        bin_edges, obs_freq, exp_freq = cal_analyzer.reliability_diagram(
            all_labels, all_risks,
        )
        cal_metrics = cal_analyzer.compute_calibration_metrics(all_labels, all_risks)
        log(f"    ECE: {cal_metrics.get('ece', 0):.4f}")
        log(f"    MCE: {cal_metrics.get('mce', 0):.4f}")
        log(f"    Brier: {cal_metrics.get('brier', 0):.4f}")
        for i, (o, e) in enumerate(zip(obs_freq, exp_freq)):
            if o > 0 or e > 0:
                log(f"      bin [{bin_edges[i]:.2f}–{bin_edges[i+1]:.2f}]: "
                    f"observed={o:.3f} expected={e:.3f}")
    except Exception as exc:
        log(f"    Calibration error: {exc}")

    # Subgroup-fair calibration
    log(f"\n  ── Subgroup Calibration ──")
    try:
        # Split by age group (from demographics)
        all_demos = None
        for i, batch in enumerate(test_ds):
            demo = batch.get("demographics")
            if demo is not None:
                demo_np = demo.numpy()
                if i == 0:
                    all_demos = demo_np
                else:
                    all_demos = np.concatenate([all_demos, demo_np], axis=0)

        if all_demos is not None and len(all_demos) == len(all_risks):
            age = all_demos[:, 0]  # first feature is age
            for age_lo, age_hi, label in [(18, 45, "young"), (45, 65, "middle"), (65, 100, "elderly")]:
                mask = (age >= age_lo) & (age < age_hi)
                if np.sum(mask) > 5:
                    sub_cal = cal_analyzer.compute_calibration_metrics(all_labels[mask], all_risks[mask])
                    log(f"    {label} (n={np.sum(mask)}): ECE={sub_cal.get('ece',0):.4f} "
                        f"Brier={sub_cal.get('brier',0):.4f}")
    except Exception as exc:
        log(f"    Subgroup calibration error: {exc}")

    # Decision Curve Analysis
    log(f"\n  ── Decision Curve Analysis ──")
    try:
        from ohca_predictor.evaluation.metrics import decision_curve_analysis
        dca_results = decision_curve_analysis(all_labels, all_risks, thresholds=np.arange(0.01, 0.99, 0.01))
        net_benefits = dca_results['net_benefit']
        thresholds_dca = dca_results['thresholds']

        # Find optimal threshold by net benefit
        best_nb_idx = np.argmax(net_benefits)
        best_nb_threshold = thresholds_dca[best_nb_idx]
        best_nb = net_benefits[best_nb_idx]

        log(f"    Max net benefit: {best_nb:.4f} at threshold {best_nb_threshold:.3f}")
        log(f"    Net benefit at 0.10: {net_benefits[np.argmin(np.abs(thresholds_dca - 0.10))]:.4f}")
        log(f"    Net benefit at 0.20: {net_benefits[np.argmin(np.abs(thresholds_dca - 0.20))]:.4f}")
        log(f"    Net benefit at 0.30: {net_benefits[np.argmin(np.abs(thresholds_dca - 0.30))]:.4f}")

        # Clinical utility: at what threshold does model beat treat-all?
        treat_all_nb = dca_results.get('treat_all', np.zeros_like(net_benefits))
        utility_mask = net_benefits > treat_all_nb
        if np.any(utility_mask):
            util_range = thresholds_dca[utility_mask]
            log(f"    Model beats treat-all for thresholds: [{util_range[0]:.2f}, {util_range[-1]:.2f}]")
    except Exception as exc:
        log(f"    DCA error: {exc}")

    # ── 3d. Uncertainty analysis ───────────────────────────────────
    log(f"\n  ── Uncertainty Analysis ──")
    mean_pred = mean_sigma = None
    try:
        all_uncertainty = np.asarray(all_uncertainty, dtype=np.float32)
        mean_pred = all_uncertainty[:, 0]
        mean_sigma = all_uncertainty[:, 1]
        log(f"    Mean predicted risk: {np.mean(mean_pred):.4f}")
        log(f"    Mean uncertainty (σ): {np.mean(mean_sigma):.4f}")
        log(f"    Median σ: {np.median(mean_sigma):.4f}")

        med_sigma = np.median(mean_sigma)
        high_unc = mean_sigma > med_sigma
        low_unc = ~high_unc
        if np.sum(high_unc) > 0 and np.sum(low_unc) > 0:
            log(f"    High-uncertainty (n={np.sum(high_unc)}): "
                f"mean risk={np.mean(all_risks[high_unc]):.4f}, "
                f"true rate={np.mean(all_labels[high_unc]):.4f}")
            log(f"    Low-uncertainty  (n={np.sum(low_unc)}): "
                f"mean risk={np.mean(all_risks[low_unc]):.4f}, "
                f"true rate={np.mean(all_labels[low_unc]):.4f}")
    except Exception as exc:
        log(f"    Uncertainty analysis error: {exc}")
        log(f"    uncertainty shape={np.shape(all_uncertainty)} dtype={getattr(all_uncertainty, 'dtype', 'unknown')}")

    # ── 3e. Survival analysis ──────────────────────────────────────
    log(f"\n  ── Survival Analysis ──")
    log(f"    Survival curve shape: {all_survival.shape}")
    try:
        all_survival = np.asarray(all_survival, dtype=np.float32)
        log(f"    Mean survival at t=0: {float(np.mean(all_survival[:, 0])):.4f}")
        log(f"    Mean survival at t=end: {float(np.mean(all_survival[:, -1])):.4f}")
    except Exception as exc:
        log(f"    Survival analysis error: {exc}")

    # ── 3f. Risk stratification ────────────────────────────────────
    log(f"\n  ── Risk Stratification ──")
    for risk_lo, risk_hi in [(0, 0.05), (0.05, 0.15), (0.15, 0.30),
                              (0.30, 0.50), (0.50, 1.01)]:
        mask = (all_risks >= risk_lo) & (all_risks < risk_hi)
        n = int(mask.sum())
        if n > 0:
            rate = float(all_labels[mask].mean())
            log(f"    [{risk_lo:.2f}, {risk_hi:.2f}): n={n:4d} "
                f"prevalence={rate:.3f}")

    # ── 3g. Prediction distribution ────────────────────────────────
    log(f"\n  ── Prediction Distribution ──")
    try:
        _r = np.asarray(all_risks, dtype=np.float64)
        log(f"    Risk range: [{float(_r.min()):.4f}, {float(_r.max()):.4f}]")
        log(f"    Risk mean±std: {float(_r.mean()):.4f} ± {float(_r.std()):.4f}")
        log(f"    Percentiles: p10={float(np.percentile(_r,10)):.4f} "
            f"p25={float(np.percentile(_r,25)):.4f} "
            f"p50={float(np.percentile(_r,50)):.4f} "
            f"p75={float(np.percentile(_r,75)):.4f} "
            f"p90={float(np.percentile(_r,90)):.4f}")
    except Exception as exc:
        log(f"    Prediction distribution error: {exc}")

    # ── 3h. Bootstrap CIs ──────────────────────────────────────────
    log(f"\n  ── Bootstrap 95% CIs ──")
    try:
        _bl = np.asarray(all_labels, dtype=np.float64).ravel()
        _br = np.asarray(all_risks, dtype=np.float64).ravel()
        auroc_ci, auroc_lo, auroc_hi = evaluator.compute_auroc(_bl, _br)
        log(f"    AUROC: {float(auroc_ci):.4f} [{float(auroc_lo):.4f}, {float(auroc_hi):.4f}]")
        auprc_ci, auprc_lo, auprc_hi = evaluator.compute_auprc(_bl, _br)
        log(f"    AUPRC: {float(auprc_ci):.4f} [{float(auprc_lo):.4f}, {float(auprc_hi):.4f}]")
        sens_ci, sens_lo, sens_hi = evaluator.compute_sensitivity_at_threshold(
            _bl, _br, thr)
        log(f"    Sens@{thr}: {float(sens_ci):.4f} [{float(sens_lo):.4f}, {float(sens_hi):.4f}]")
        spec_ci, spec_lo, spec_hi = evaluator.compute_specificity_at_threshold(
            _bl, _br, thr)
        log(f"    Spec@{thr}: {float(spec_ci):.4f} [{float(spec_lo):.4f}, {float(spec_hi):.4f}]")
    except Exception as e:
        log(f"    Bootstrap CI error: {e}")

    # ── Save results ───────────────────────────────────────────────
    results = {
        "core_metrics": {
            k: (float(v) if isinstance(v, (float, np.floating))
                else [float(x) for x in v] if isinstance(v, tuple) else v)
            for k, v in metrics.items()
        },
        "optimal_threshold": {
            "threshold": float(best_thr),
            "youden_j": float(best_j),
            "metrics": {
                k: (float(v) if isinstance(v, (float, np.floating))
                    else [float(x) for x in v] if isinstance(v, tuple) else v)
                for k, v in opt_metrics.items()
            },
        },
        "multi_threshold": threshold_results,
        "calibration": cal_metrics,
        "uncertainty": {
            "mean_predicted_risk": float(np.mean(mean_pred)) if mean_pred is not None else 0.0,
            "mean_sigma": float(np.mean(mean_sigma)) if mean_sigma is not None else 0.0,
        },
        "survival": {
            "mean_survival_t0": float(np.mean(all_survival[:, 0])) if all_survival.size > 0 else 0.0,
            "mean_survival_end": float(np.mean(all_survival[:, -1])) if all_survival.size > 0 else 0.0,
        },
        "prediction_distribution": {
            "min": float(np.asarray(all_risks, dtype=np.float64).min()),
            "max": float(np.asarray(all_risks, dtype=np.float64).max()),
            "mean": float(np.asarray(all_risks, dtype=np.float64).mean()),
            "std": float(np.asarray(all_risks, dtype=np.float64).std()),
        },
        "test_samples": n_total,
        "positive_rate": float(n_pos / max(1, n_total)),
        "history": history,
    }

    results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\n  Results saved: {results_path}")

    return metrics, results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

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
        if os.path.isfile("pipeline_data.pkl"):
            log("  Loading pre-generated pipeline data from 'pipeline_data.pkl' ...")
            train_samples, val_samples, test_samples = jb.load("pipeline_data.pkl")
        else:
            train_samples, val_samples, test_samples = phase1_generate(config)
            jb.dump((train_samples, val_samples, test_samples), "pipeline_data.pkl")

        if len(train_samples) == 0:
            log("[ERROR] No training data. Aborting.")
            return

        model, history, best_auroc = phase2_train(config, train_samples, val_samples)

        metrics, results = phase3_evaluate(config, model, test_samples, history)

        final_path = os.path.join(MODEL_DIR, "ohca_model_final.weights.h5")
        model.save_weights(final_path)
        log(f"\n  Final model saved: {final_path}")

        # Save model config for OHCAPredictorPipeline.load()
        model_cfg = {
            "model": {
                "model_dim": config.model.model_dim,
                "num_attention_heads": config.model.num_attention_heads,
                "num_encoder_layers": config.model.num_encoder_layers,
                "feedforward_dim": config.model.feedforward_dim,
                "dropout_rate": config.model.dropout_rate,
                "attention_dropout_rate": config.model.attention_dropout_rate,
                "static_embedding_dim": config.model.static_embedding_dim,
                "tokens_per_modality": config.model.tokens_per_modality,
                "max_positional_encoding": config.model.max_positional_encoding,
                "num_survival_bins": config.model.num_survival_bins,
                "uncertainty_samples": config.model.uncertainty_samples,
            }
        }
        cfg_path = os.path.join(MODEL_DIR, "config.json")
        with open(cfg_path, "w") as f:
            json.dump(model_cfg, f, indent=2)
        log(f"  Config saved: {cfg_path}")
        log(f"\n  To use the model:")
        log(f"    from ohca_predictor.pipeline import OHCAPredictorPipeline")
        log(f"    pipe = OHCAPredictorPipeline.load('{final_path}')")
        log(f"    result = pipe.predict(ecg=ecg, accelerometer=accel, ppg=ppg, ...)")
        log(f"    print(result.risk, result.uncertainty, result.high_risk)")

        elapsed = time.time() - t0
        log("\n" + "=" * 70)
        log("PIPELINE COMPLETE")
        log(f"  Total time: {elapsed/3600:.2f}h ({elapsed:.0f}s)")
        log(f"  Best validation AUROC: {best_auroc:.4f}")
        log(f"  Results: {RESULTS_DIR}/evaluation_results.json")
        log(f"  TensorBoard: {TENSORBOARD_DIR}")
        log(f"  Model: {final_path}")
        log("=" * 70)

    except Exception as e:
        log(f"\n[FATAL] Pipeline error: {e}")
        traceback.print_exc()
        with open(LOG_FILE, "a") as f:
            traceback.print_exc(file=f)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
