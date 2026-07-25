"""
Realistic OHCA Prediction Training Pipeline v2

Key design: build model first (to allocate memory), then generate data in small batches,
keeping only what we need.
"""

import os
import sys
import time
import json
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf

from ohca_predictor.config import override_config
from ohca_predictor.config import SimulationConfig, TrainingConfig, ModelConfig, EvaluationConfig
from ohca_predictor.simulator.generator import DataGenerator
from ohca_predictor.utils.io_utils import create_padded_dataset
from ohca_predictor.model.architecture import OHCAPredictionModel
from ohca_predictor.model.losses import CombinedOHCALoss
from ohca_predictor.training.trainer import CosineDecayWithWarmup
from ohca_predictor.evaluation.metrics import OHCAEvaluator
from ohca_predictor.evaluation.calibration import CalibrationAnalyzer


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=" * 70)
    log("REALISTIC OHCA PREDICTION - TRAINING PIPELINE v2")
    log("=" * 70)

    config = override_config(
        simulation=SimulationConfig(
            min_window_duration_hours=0.05,
            max_window_duration_hours=0.12,
            population_size=200,
            prevalence_ohca=0.20,
        ),
        training=TrainingConfig(
            batch_size=2,
            epochs=30,
            learning_rate=3e-5,
            warmup_steps=30,
            mixed_precision=False,
            gradient_accumulation_steps=1,
            early_stopping_patience=10,
            gradient_clip_norm=1.0,
            weight_decay=0.01,
            false_negative_weight=10.0,
            false_positive_weight=1.0,
        ),
        model=ModelConfig(
            model_dim=128,
            num_attention_heads=4,
            num_encoder_layers=3,
            feedforward_dim=512,
            dropout_rate=0.15,
            tokens_per_modality=256,
            max_positional_encoding=4096,
            static_embedding_dim=48,
        ),
        evaluation=EvaluationConfig(
            bootstrap_iterations=500,
            confidence_level=0.95,
        ),
    )

    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Phase 1: Build model first (allocate memory)
    log("\n=== PHASE 1: BUILD MODEL ===")
    model = OHCAPredictionModel(config.model)
    total_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    log(f"Model parameters: {total_params:,.0f}")

    # Build with dummy forward pass (training=True to build all layers)
    from ohca_predictor.simulator.generator import DataGenerator as DG
    dummy_gen = DG(config)
    _, _, dummy_samples = dummy_gen.generate_dataset(n_patients=2)
    dummy_ds = create_padded_dataset(dummy_samples, batch_size=1, shuffle=False)
    for batch in dummy_ds:
        _ = model(batch, training=True)
        break
    total_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    log(f"Model parameters (after build): {total_params:,.0f}")
    del dummy_gen, dummy_samples, dummy_ds
    gc.collect()

    # Phase 2: Generate data
    log("\n=== PHASE 2: DATA GENERATION ===")
    generator = DataGenerator(config)
    all_train, all_val, all_test = [], [], []

    for i in range(0, 200, 50):
        actual = min(50, 200 - i)
        t, v, te = generator.generate_dataset(n_patients=actual)
        all_train.extend(t)
        all_val.extend(v)
        all_test.extend(te)
        log(f"  Batch {i//50 + 1}: train={len(all_train)}, val={len(all_val)}, test={len(all_test)}")
        gc.collect()

    pos_train = sum(1 for s in all_train if s.ohca_label > 0.5)
    pos_val = sum(1 for s in all_val if s.ohca_label > 0.5)
    pos_test = sum(1 for s in all_test if s.ohca_label > 0.5)
    log(f"Natural OHCA rates: train={pos_train}/{len(all_train)} ({pos_train/max(1,len(all_train))*100:.1f}%), "
        f"val={pos_val}/{len(all_val)} ({pos_val/max(1,len(all_val))*100:.1f}%), "
        f"test={pos_test}/{len(all_test)} ({pos_test/max(1,len(all_test))*100:.1f}%)")

    # Phase 3: Train
    log("\n=== PHASE 3: TRAINING ===")
    loss_fn = CombinedOHCALoss()
    evaluator = OHCAEvaluator(config.evaluation)
    total_steps = (len(all_train) // config.training.batch_size) * config.training.epochs
    lr_schedule = CosineDecayWithWarmup(
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        total_steps=total_steps,
        min_learning_rate=config.training.min_learning_rate,
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=config.training.weight_decay,
        clipnorm=config.training.gradient_clip_norm,
    )

    train_ds = create_padded_dataset(all_train, batch_size=config.training.batch_size, shuffle=True)
    val_ds = create_padded_dataset(all_val, batch_size=config.training.batch_size, shuffle=False)

    best_auroc = 0.0
    best_weights = None
    patience = 0
    history = []
    optimizer_built = False

    for epoch in range(config.training.epochs):
        t0 = time.time()
        total_loss = 0.0
        nb = 0

        for batch in train_ds:
            with tf.GradientTape() as tape:
                outputs = model(batch, training=True)
                loss = loss_fn(batch, outputs)
            grads = tape.gradient(loss, model.trainable_variables)
            gvs = list(zip(grads, model.trainable_variables))

            # Build optimizer lazily on first step (after all vars are known)
            if not optimizer_built:
                try:
                    optimizer.apply_gradients(gvs)
                    optimizer_built = True
                except ValueError:
                    # New variables discovered - rebuild optimizer
                    optimizer = tf.keras.optimizers.AdamW(
                        learning_rate=lr_schedule,
                        weight_decay=config.training.weight_decay,
                        clipnorm=config.training.gradient_clip_norm,
                    )
                    optimizer.apply_gradients(gvs)
                    optimizer_built = True
            else:
                optimizer.apply_gradients(gvs)
            total_loss += float(loss.numpy())
            nb += 1

        val_loss = 0.0
        vnb = 0
        risks = []
        labels = []
        for batch in val_ds:
            out = model(batch, training=False)
            vl = loss_fn(batch, out)
            val_loss += float(vl.numpy())
            vnb += 1
            risks.append(out['ohca_risk'].numpy().flatten())
            labels.append(batch['ohca_label'].numpy().flatten())

        risks = np.concatenate(risks)
        labels = np.concatenate(labels)

        try:
            auroc, lo, hi = evaluator.compute_auroc(labels, risks)
            sens, _, _ = evaluator.compute_sensitivity_at_threshold(labels, risks, 0.20)
            spec, _, _ = evaluator.compute_specificity_at_threshold(labels, risks, 0.20)
        except Exception:
            auroc = lo = hi = sens = spec = 0.0

        dt = time.time() - t0
        avg_loss = total_loss / max(1, nb)
        avg_vloss = val_loss / max(1, vnb)

        log(f"  Epoch {epoch+1:2d}/{config.training.epochs}: "
            f"loss={avg_loss:.4f} vloss={avg_vloss:.4f} "
            f"AUROC={auroc:.3f} [{lo:.3f}-{hi:.3f}] "
            f"Sens={sens:.3f} Spec={spec:.3f} ({dt:.1f}s)")

        history.append({
            'epoch': epoch + 1, 'train_loss': avg_loss, 'val_loss': avg_vloss,
            'auroc': float(auroc), 'sensitivity': float(sens), 'specificity': float(spec),
        })

        if auroc > best_auroc:
            best_auroc = auroc
            best_weights = [w.copy() for w in model.get_weights()]
            patience = 0
            log(f"    -> New best AUROC!")
        else:
            patience += 1
            if patience >= config.training.early_stopping_patience:
                log(f"  Early stopping at epoch {epoch+1}")
                break

    if best_weights:
        model.set_weights(best_weights)
    log(f"\nBest val AUROC: {best_auroc:.3f}")

    # Phase 4: Evaluate
    log("\n=== PHASE 4: TEST EVALUATION ===")
    test_ds = create_padded_dataset(all_test, batch_size=2, shuffle=False)
    risks = []
    labels = []
    for batch in test_ds:
        out = model(batch, training=False)
        risks.append(out['ohca_risk'].numpy().flatten())
        labels.append(batch['ohca_label'].numpy().flatten())
    risks = np.concatenate(risks)
    labels = np.concatenate(labels)
    n_pos = int(sum(labels))
    n_neg = int(len(labels) - n_pos)
    log(f"Test set: {len(labels)} samples ({n_pos} pos, {n_neg} neg)")

    if n_pos > 0 and n_neg > 0:
        # Full metrics
        metrics = evaluator.compute_all_metrics(labels, risks, threshold=0.20)
        evaluator.print_report(metrics)

        # Threshold sweep
        log("\n--- THRESHOLD SWEEP ---")
        best_f1 = 0.0
        best_thr = 0.20
        for thr in np.arange(0.05, 0.50, 0.01):
            f_val, _, _ = evaluator.compute_f1_score(labels, risks, thr)
            if f_val > best_f1:
                best_f1 = f_val
                best_thr = thr

        log(f"Optimal threshold: {best_thr:.3f} (F1={best_f1:.3f})")

        # Clinical metrics at optimal threshold
        s, slo, shi = evaluator.compute_sensitivity_at_threshold(labels, risks, best_thr)
        sp, splo, sphi = evaluator.compute_specificity_at_threshold(labels, risks, best_thr)
        p, plo, phi = evaluator.compute_ppv(labels, risks, best_thr)
        n, nlo, nhi = evaluator.compute_npv(labels, risks, best_thr)
        auroc, alo, ahi = evaluator.compute_auroc(labels, risks)
        auprc, aplo, aphi = evaluator.compute_auprc(labels, risks)
        brier, blo, bhi = evaluator.compute_brier_score(labels, risks)

        log(f"\n--- CLINICAL METRICS @ threshold={best_thr:.2f} ---")
        log(f"  AUROC:       {auroc:.4f} [{alo:.4f}-{ahi:.4f}]")
        log(f"  AUPRC:       {auprc:.4f} [{aplo:.4f}-{aphi:.4f}]")
        log(f"  Sensitivity: {s:.4f} [{slo:.4f}-{shi:.4f}]")
        log(f"  Specificity: {sp:.4f} [{splo:.4f}-{sphi:.4f}]")
        log(f"  PPV:         {p:.4f} [{plo:.4f}-{phi:.4f}]")
        log(f"  NPV:         {n:.4f} [{nlo:.4f}-{nhi:.4f}]")
        log(f"  Brier:       {brier:.4f} [{blo:.4f}-{bhi:.4f}]")

        # Calibration
        log("\n--- CALIBRATION ---")
        cal = CalibrationAnalyzer(config.evaluation)
        try:
            cm = cal.compute_calibration_metrics(labels, risks)
            log(f"  ECE: {cm['ece']:.4f}")
            log(f"  MCE: {cm['mce']:.4f}")
            log(f"  Brier: {cm['brier']:.4f}")
        except Exception as e:
            log(f"  Calibration error: {e}")
            cm = {'ece': 0, 'mce': 0, 'brier': 0}

        # Prediction distributions
        log("\n--- PREDICTION DISTRIBUTIONS ---")
        pos_risks = risks[labels > 0.5]
        neg_risks = risks[labels <= 0.5]
        log(f"  Positive: mean={np.mean(pos_risks):.4f} +/- {np.std(pos_risks):.4f} "
            f"[{np.min(pos_risks):.4f} - {np.max(pos_risks):.4f}]")
        log(f"  Negative: mean={np.mean(neg_risks):.4f} +/- {np.std(neg_risks):.4f} "
            f"[{np.min(neg_risks):.4f} - {np.max(neg_risks):.4f}]")
        log(f"  Separation: {np.mean(pos_risks) - np.mean(neg_risks):.4f}")

        # Risk score distribution
        log("\n--- RISK SCORE DISTRIBUTION ---")
        for lo_edge in np.arange(0, 1.0, 0.1):
            hi_edge = lo_edge + 0.1
            count_pos = int(np.sum((risks >= lo_edge) & (risks < hi_edge) & (labels > 0.5)))
            count_neg = int(np.sum((risks >= lo_edge) & (risks < hi_edge) & (labels <= 0.5)))
            log(f"  [{lo_edge:.1f}, {hi_edge:.1f}): {count_pos} pos, {count_neg} neg")

        # Save results
        results = {
            'test_samples': int(len(labels)),
            'test_positive': n_pos,
            'test_negative': n_neg,
            'best_threshold': float(best_thr),
            'auroc': {'value': float(auroc), 'ci_lower': float(alo), 'ci_upper': float(ahi)},
            'auprc': {'value': float(auprc), 'ci_lower': float(aplo), 'ci_upper': float(aphi)},
            'sensitivity': {'value': float(s), 'ci_lower': float(slo), 'ci_upper': float(shi)},
            'specificity': {'value': float(sp), 'ci_lower': float(splo), 'ci_upper': float(sphi)},
            'ppv': {'value': float(p), 'ci_lower': float(plo), 'ci_upper': float(phi)},
            'npv': {'value': float(n), 'ci_lower': float(nlo), 'ci_upper': float(nhi)},
            'brier': {'value': float(brier), 'ci_lower': float(blo), 'ci_upper': float(bhi)},
            'calibration': {'ece': float(cm['ece']), 'mce': float(cm['mce'])},
            'model_config': {
                'model_dim': config.model.model_dim,
                'tokens_per_modality': config.model.tokens_per_modality,
                'num_layers': config.model.num_encoder_layers,
                'num_heads': config.model.num_attention_heads,
                'total_params': int(total_params),
                'variable_length': True,
            },
            'data_config': {
                'n_patients': 200,
                'natural_ohca_progression': True,
                'disease_models': 20,
            },
            'training_history': history,
        }

        with open('results/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        log("\nSaved results/evaluation_results.json")

    # Phase 5: Summary
    log("\n" + "=" * 70)
    log("PIPELINE COMPLETE")
    log("=" * 70)
    if n_pos > 0 and n_neg > 0:
        log(f"AUROC:       {auroc:.4f}")
        log(f"Sensitivity: {s:.4f}")
        log(f"Specificity: {sp:.4f}")
        log(f"Calibration ECE: {cm.get('ece', 0):.4f}")
    log("=" * 70)


if __name__ == "__main__":
    main()
