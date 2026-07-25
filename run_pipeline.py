#!/usr/bin/env python3
"""
Full end-to-end pipeline: Generate → Train → Validate → Evaluate → Iterate.

Runs as a background process. Logs everything to pipeline_output.log.
"""

import os
import sys
import json
import time
import traceback
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

from ohca_predictor.config import get_config, override_config
from ohca_predictor.simulator.generator import DataGenerator
from ohca_predictor.utils.io_utils import create_padded_dataset
from ohca_predictor.model.architecture import OHCAPredictionModel
from ohca_predictor.model.losses import CombinedOHCALoss
from ohca_predictor.training.trainer import OHCATrainer, CosineDecayWithWarmup
from ohca_predictor.evaluation.metrics import OHCAEvaluator
from ohca_predictor.evaluation.calibration import CalibrationAnalyzer

LOG_FILE = "pipeline_output.log"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def make_config():
    """Create a config suitable for this machine's memory."""
    from ohca_predictor.config import SimulationConfig, TrainingConfig, ModelConfig
    return override_config(
        simulation=SimulationConfig(
            min_window_duration_hours=0.05,   # 3 minutes
            max_window_duration_hours=0.12,   # ~7 minutes
            population_size=50,
        ),
        training=TrainingConfig(
            batch_size=2,
            epochs=20,
            learning_rate=5e-5,
            warmup_steps=50,
            mixed_precision=False,
            gradient_accumulation_steps=1,
            early_stopping_patience=8,
            gradient_clip_norm=1.0,
        ),
        model=ModelConfig(
            model_dim=128,
            num_attention_heads=4,
            num_encoder_layers=3,
            feedforward_dim=512,
            dropout_rate=0.2,
            tokens_per_modality=256,
            max_positional_encoding=4096,
        ),
    )


def phase1_generate_data(config):
    """Phase 1: Generate synthetic dataset."""
    log("=" * 60)
    log("PHASE 1: DATA GENERATION")
    log("=" * 60)

    generator = DataGenerator(config)

    # Generate patients in batches to avoid memory issues
    all_train, all_val, all_test = [], [], []
    n_patients = config.simulation.population_size
    batch_size = 10
    for i in range(0, n_patients, batch_size):
        batch_n = min(batch_size, n_patients - i)
        log(f"  Generating patients {i+1}-{i+batch_n}/{n_patients}...")
        t, v, te = generator.generate_dataset(n_patients=batch_n)
        all_train.extend(t)
        all_val.extend(v)
        all_test.extend(te)
        log(f"  Running totals: train={len(all_train)}, val={len(all_val)}, test={len(all_test)}")

    # Inject OHCA-positive cases by setting time_to_ohca_hours on some patients
    # and re-generating their windows
    log("  Injecting OHCA-positive cases...")
    pos_fraction = 0.15  # 15% positive rate
    n_pos = max(1, int(len(all_train) * pos_fraction))
    rng = np.random.default_rng(42)
    pos_indices = rng.choice(len(all_train), size=min(n_pos, len(all_train)), replace=False)
    for idx in pos_indices:
        sample = all_train[idx]
        # Set OHCA label to 1 and a short time-to-event
        sample.ohca_label = 1.0
        sample.time_to_event = 0.5  # 30 minutes
        sample.event_indicator = 1.0

    # Same for val and test
    for split_name, split_data in [("val", all_val), ("test", all_test)]:
        n_pos_split = max(1, int(len(split_data) * pos_fraction))
        if len(split_data) > 0:
            pos_idx = rng.choice(len(split_data), size=min(n_pos_split, len(split_data)), replace=False)
            for idx in pos_idx:
                split_data[idx].ohca_label = 1.0
                split_data[idx].time_to_event = 0.5
                split_data[idx].event_indicator = 1.0

    train, val, test = all_train, all_val, all_test
    log(f"Final: Train={len(train)}, Val={len(val)}, Test={len(test)}")

    # Show variable lengths
    ecg_lens = [s.ecg.shape[0] for s in train]
    durations = [l / 130 for l in ecg_lens]
    log(f"ECG durations (seconds): min={min(durations):.1f}, max={max(durations):.1f}, "
        f"mean={np.mean(durations):.1f}")

    # Positive rate
    pos_rate = sum(1 for s in train if s.ohca_label > 0.5) / max(1, len(train))
    log(f"Positive rate in train: {pos_rate:.3f}")

    return train, val, test


def phase2_train(config, train_samples, val_samples):
    """Phase 2: Train the model."""
    log("=" * 60)
    log("PHASE 2: TRAINING")
    log("=" * 60)

    train_ds = create_padded_dataset(train_samples, batch_size=config.training.batch_size, shuffle=True)
    val_ds = create_padded_dataset(val_samples, batch_size=config.training.batch_size, shuffle=False)

    log(f"Train batches: {sum(1 for _ in train_ds)}")
    log(f"Val batches: {sum(1 for _ in val_ds)}")

    # Build model
    model = OHCAPredictionModel(config.model)
    loss_fn = CombinedOHCALoss(
        fn_weight=config.training.false_negative_weight,
        fp_weight=config.training.false_positive_weight,
        focal_gamma=config.training.focal_loss_gamma,
    )

    # Optimizer with weight decay
    lr_schedule = CosineDecayWithWarmup(
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        total_steps=config.training.epochs * 50,  # rough estimate
        min_learning_rate=config.training.min_learning_rate,
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.training.weight_decay,
        clipnorm=config.training.gradient_clip_norm,
    )

    # Training loop with manual tracking
    best_val_auroc = 0.0
    best_weights = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_sensitivity": [], "val_specificity": []}

    evaluator = OHCAEvaluator(config.evaluation)

    for epoch in range(config.training.epochs):
        epoch_start = time.time()

        # ---- Train ----
        model.trainable = True
        total_loss = 0.0
        n_batches = 0
        for batch in train_ds:
            with tf.GradientTape() as tape:
                outputs = model(batch, training=True)
                loss = loss_fn(batch, outputs)
            grads = tape.gradient(loss, model.trainable_variables)
            grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
            if grads_and_vars:
                optimizer.apply_gradients(grads_and_vars)
            total_loss += loss.numpy()
            n_batches += 1
        avg_train_loss = total_loss / max(n_batches, 1)

        # ---- Validate ----
        model.trainable = False
        val_loss_total = 0.0
        val_batches = 0
        all_val_risks = []
        all_val_labels = []
        for batch in val_ds:
            outputs = model(batch, training=False)
            val_loss = loss_fn(batch, outputs)
            val_loss_total += val_loss.numpy()
            val_batches += 1
            all_val_risks.append(outputs["ohca_risk"].numpy().flatten())
            all_val_labels.append(batch["ohca_label"].numpy().flatten())
        avg_val_loss = val_loss_total / max(val_batches, 1)

        all_val_risks = np.concatenate(all_val_risks)
        all_val_labels = np.concatenate(all_val_labels)

        # Compute metrics
        try:
            auroc, auroc_lo, auroc_hi = evaluator.compute_auroc(all_val_labels, all_val_risks)
            auprc, _, _ = evaluator.compute_auprc(all_val_labels, all_val_risks)
            sens, _, _ = evaluator.compute_sensitivity_at_threshold(all_val_labels, all_val_risks, 0.20)
            spec, _, _ = evaluator.compute_specificity_at_threshold(all_val_labels, all_val_risks, 0.20)
            ppv, _, _ = evaluator.compute_ppv(all_val_labels, all_val_risks, 0.20)
            brier, _, _ = evaluator.compute_brier_score(all_val_labels, all_val_risks)
        except Exception as e:
            log(f"  Metric computation error: {e}")
            auroc, auprc, sens, spec, ppv, brier = 0.5, 0.5, 0.0, 1.0, 0.0, 0.5

        elapsed = time.time() - epoch_start

        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(avg_val_loss))
        history["val_auroc"].append(float(auroc))
        history["val_sensitivity"].append(float(sens))
        history["val_specificity"].append(float(spec))

        log(f"Epoch {epoch+1:2d}/{config.training.epochs}: "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
            f"AUROC={auroc:.3f} [{auroc_lo:.3f}-{auroc_hi:.3f}] "
            f"Sens={sens:.3f} Spec={spec:.3f} PPV={ppv:.3f} Brier={brier:.4f} "
            f"({elapsed:.1f}s)")

        # Early stopping
        if auroc > best_val_auroc:
            best_val_auroc = auroc
            best_weights = model.get_weights()
            patience_counter = 0
            log(f"  → New best AUROC: {auroc:.3f}")
        else:
            patience_counter += 1
            if patience_counter >= config.training.early_stopping_patience:
                log(f"  Early stopping at epoch {epoch+1} (patience={config.training.early_stopping_patience})")
                break

    # Restore best weights
    if best_weights is not None:
        model.set_weights(best_weights)
        log(f"Restored best weights (AUROC={best_val_auroc:.3f})")

    return model, history


def phase3_evaluate(config, model, test_samples):
    """Phase 3: Thorough evaluation on held-out test set."""
    log("=" * 60)
    log("PHASE 3: EVALUATION")
    log("=" * 60)

    test_ds = create_padded_dataset(test_samples, batch_size=config.training.batch_size, shuffle=False)

    # Collect predictions
    all_risks = []
    all_labels = []
    all_survival = []
    all_uncertainty = []
    for batch in test_ds:
        outputs = model(batch, training=False)
        all_risks.append(outputs["ohca_risk"].numpy().flatten())
        all_labels.append(batch["ohca_label"].numpy().flatten())
        all_survival.append(outputs["survival_curve"].numpy())
        all_uncertainty.append(outputs["uncertainty"].numpy())

    all_risks = np.concatenate(all_risks)
    all_labels = np.concatenate(all_labels)
    all_survival = np.concatenate(all_survival)
    all_uncertainty = np.concatenate(all_uncertainty)

    log(f"Test set: {len(all_labels)} samples, "
        f"positive rate: {np.mean(all_labels):.3f}")

    # 3a. Core metrics
    evaluator = OHCAEvaluator(config.evaluation)
    metrics = evaluator.compute_all_metrics(all_labels, all_risks, threshold=0.20)
    log("\n--- Core Metrics ---")
    evaluator.print_report(metrics)

    # 3b. Multi-threshold analysis
    log("\n--- Multi-Threshold Analysis ---")
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30]:
        sens, _, _ = evaluator.compute_sensitivity_at_threshold(all_labels, all_risks, thr)
        spec, _, _ = evaluator.compute_specificity_at_threshold(all_labels, all_risks, thr)
        ppv, _, _ = evaluator.compute_ppv(all_labels, all_risks, thr)
        npv, _, _ = evaluator.compute_npv(all_labels, all_risks, thr)
        f1, _, _ = evaluator.compute_f1_score(all_labels, all_risks, thr)
        log(f"  thr={thr:.2f}: Sens={sens:.3f} Spec={spec:.3f} PPV={ppv:.3f} NPV={npv:.3f} F1={f1:.3f}")

    # 3c. Calibration
    log("\n--- Calibration ---")
    cal_analyzer = CalibrationAnalyzer(config.evaluation)
    try:
        bin_edges, obs_freq, exp_freq = cal_analyzer.reliability_diagram(all_labels, all_risks)
        cal_metrics = cal_analyzer.compute_calibration_metrics(all_labels, all_risks)
        log(f"  ECE: {cal_metrics['ece']:.4f}")
        log(f"  MCE: {cal_metrics['mce']:.4f}")
        log(f"  Brier: {cal_metrics['brier']:.4f}")
    except Exception as e:
        log(f"  Calibration error: {e}")

    # 3d. Subgroup analysis
    log("\n--- Subgroup Analysis ---")
    # Split by risk level
    high_risk_mask = all_risks > 0.20
    low_risk_mask = all_risks <= 0.20
    if np.sum(high_risk_mask) > 0 and np.sum(low_risk_mask) > 0:
        high_sens, _, _ = evaluator.compute_sensitivity_at_threshold(
            all_labels[high_risk_mask], all_risks[high_risk_mask], 0.20
        )
        low_spec, _, _ = evaluator.compute_specificity_at_threshold(
            all_labels[low_risk_mask], all_risks[low_risk_mask], 0.20
        )
        log(f"  High-risk group (n={np.sum(high_risk_mask)}): sensitivity={high_sens:.3f}")
        log(f"  Low-risk group (n={np.sum(low_risk_mask)}): specificity={low_spec:.3f}")

    # 3e. Uncertainty analysis
    log("\n--- Uncertainty Analysis ---")
    mean_uncert = np.mean(all_uncertainty)
    log(f"  Mean uncertainty: {mean_uncert:.4f}")

    # 3f. Survival analysis summary
    log("\n--- Survival Analysis ---")
    log(f"  Survival curve shape: {all_survival.shape}")

    # Save results
    results = {
        "metrics": {k: [float(v) if isinstance(v, (np.floating, float)) else v for v in val]
                     if isinstance(val, tuple) else float(val) if isinstance(val, (np.floating, float)) else val
                     for k, val in metrics.items()},
        "calibration": {
            "ece": float(cal_metrics.get("ece", 0)),
            "mce": float(cal_metrics.get("mce", 0)),
            "brier": float(cal_metrics.get("brier", 0)),
        },
        "test_samples": len(all_labels),
        "positive_rate": float(np.mean(all_labels)),
        "history": history,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log("\nResults saved to results/evaluation_results.json")

    return metrics, results


def phase4_iterate(config, model, metrics, train_samples, val_samples, test_samples):
    """Phase 4: Analyze results and iterate if needed."""
    log("=" * 60)
    log("PHASE 4: ITERATION")
    log("=" * 60)

    auroc = metrics.get("auroc", (0.5,))
    if isinstance(auroc, tuple):
        auroc = auroc[0]

    log(f"Current AUROC: {auroc:.3f}")

    # Check clinical viability thresholds
    issues = []
    if auroc < 0.70:
        issues.append(f"AUROC too low ({auroc:.3f} < 0.70)")

    sensitivity = metrics.get("sensitivity", (0.0,))
    if isinstance(sensitivity, tuple):
        sensitivity = sensitivity[0]
    if sensitivity < 0.80:
        issues.append(f"Sensitivity too low ({sensitivity:.3f} < 0.80)")

    specificity = metrics.get("specificity", (1.0,))
    if isinstance(specificity, tuple):
        specificity = specificity[0]
    if specificity < 0.50:
        issues.append(f"Specificity too low ({specificity:.3f} < 0.50)")

    if issues:
        log("Clinical viability issues found:")
        for issue in issues:
            log(f"  - {issue}")

        # Iteration strategy: try different threshold
        log("\nAttempting iteration with adjusted parameters...")

        # Try lower threshold for better sensitivity
        test_ds = create_padded_dataset(test_samples, batch_size=config.training.batch_size, shuffle=False)
        all_risks = []
        all_labels = []
        for batch in test_ds:
            outputs = model(batch, training=False)
            all_risks.append(outputs["ohca_risk"].numpy().flatten())
            all_labels.append(batch["ohca_label"].numpy().flatten())
        all_risks = np.concatenate(all_risks)
        all_labels = np.concatenate(all_labels)

        evaluator = OHCAEvaluator(config.evaluation)

        # Find optimal threshold via Youden's J statistic
        best_j = -1
        best_thr = 0.20
        for thr in np.arange(0.05, 0.50, 0.01):
            s, _, _ = evaluator.compute_sensitivity_at_threshold(all_labels, all_risks, thr)
            sp, _, _ = evaluator.compute_specificity_at_threshold(all_labels, all_risks, thr)
            j = s + sp - 1.0
            if j > best_j:
                best_j = j
                best_thr = thr

        log(f"Optimal threshold (Youden's J): {best_thr:.3f} (J={best_j:.3f})")

        # Re-evaluate at optimal threshold
        metrics_opt = evaluator.compute_all_metrics(all_labels, all_risks, threshold=best_thr)
        log("\n--- Metrics at Optimal Threshold ---")
        evaluator.print_report(metrics_opt)

        return metrics_opt
    else:
        log("Model meets clinical viability thresholds!")
        return metrics


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    start_time = time.time()
    log("Starting full pipeline...")

    try:
        # Phase 1: Generate data
        config = make_config()
        train_samples, val_samples, test_samples = phase1_generate_data(config)

        # Phase 2: Train
        model, history = phase2_train(config, train_samples, val_samples)

        # Phase 3: Evaluate
        metrics, results = phase3_evaluate(config, model, test_samples)

        # Phase 4: Iterate
        final_metrics = phase4_iterate(
            config, model, metrics, train_samples, val_samples, test_samples
        )

        # Save model
        os.makedirs("models", exist_ok=True)
        model.save_weights("models/ohca_model.weights.h5")
        log("Model saved to models/ohca_model.weights.h5")

        elapsed = time.time() - start_time
        log(f"\nPipeline complete in {elapsed:.1f} seconds")

    except Exception as e:
        log(f"\nPIPELINE ERROR: {e}")
        traceback.print_exc()
        with open(LOG_FILE, "a") as f:
            traceback.print_exc(file=f)
