"""Evaluation script for OHCA Prediction Model."""

import argparse
import sys
from pathlib import Path
import logging
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ohca_predictor.config import get_config, override_config
from ohca_predictor.model.architecture import OHCAPredictionModel
from ohca_predictor.training.trainer import OHCATrainer
from ohca_predictor.evaluation.metrics import (
    bootstrap_confidence_interval,
    expected_calibration_error,
    decision_curve_analysis,
    compute_sensitivity_at_specificity,
    compute_specificity_at_sensitivity,
    compute_ppv,
    compute_npv,
    compute_auc_roc,
    compute_auc_pr,
    compute_brier_score,
)
from ohca_predictor.evaluation.calibration import CalibrationAnalyzer
from ohca_predictor.utils.reproducibility import ReproducibilityManager
from ohca_predictor.utils.logging import StructuredLogger
import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate OHCA Prediction Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (optional)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to evaluation data directory",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ohca_evaluation",
        help="Name of the evaluation experiment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for confidence intervals",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence interval level",
    )
    parser.add_argument(
        "--compute-calibration",
        action="store_true",
        help="Compute calibration metrics",
    )
    parser.add_argument(
        "--compute-dca",
        action="store_true",
        help="Compute decision curve analysis",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger = StructuredLogger.get_instance()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    logger.info("Starting OHCA model evaluation", args=vars(args))
    
    repro_manager = ReproducibilityManager(seed=args.seed)
    repro_manager.set_all_seeds()
    
    config = get_config()
    
    if args.config:
        config = override_config(config, args.config)
    
    logger.info("Loading model from checkpoint", checkpoint=args.checkpoint_path)
    model = OHCAPredictionModel(config)
    model.load_weights(args.checkpoint_path)
    
    trainer = OHCATrainer(config=config, model=model)
    trainer.compile()
    
    logger.info("Loading evaluation data", data_path=args.data_path)
    eval_dataset = trainer.load_data(args.data_path, batch_size=args.batch_size)
    
    logger.info("Running evaluation")
    all_predictions = []
    all_labels = []
    
    for batch in eval_dataset:
        inputs, labels = batch
        predictions = model(inputs, training=False)
        all_predictions.append(predictions)
        all_labels.append(labels)
    
    all_predictions = {k: tf.concat([p[k] for p in all_predictions], axis=0) 
                       for k in all_predictions[0].keys()}
    all_labels = {k: tf.concat([l[k] for l in all_labels], axis=0) 
                  for k in all_labels[0].keys()}
    
    y_true = all_labels["ohca_label"].numpy().flatten()
    y_prob = all_predictions["ohca_prob"].numpy().flatten()
    
    results = {
        "metrics": {},
        "confidence_intervals": {},
    }
    
    results["metrics"]["auc_roc"] = float(compute_auc_roc(y_true, y_prob))
    results["metrics"]["auc_pr"] = float(compute_auc_pr(y_true, y_prob))
    results["metrics"]["brier_score"] = float(compute_brier_score(y_true, y_prob))
    results["metrics"]["ece"] = float(expected_calibration_error(y_true, y_prob))
    
    sens, sens_threshold = compute_sensitivity_at_specificity(y_true, y_prob, target_specificity=0.9)
    results["metrics"]["sensitivity_at_specificity_90"] = float(sens)
    
    spec, spec_threshold = compute_specificity_at_sensitivity(y_true, y_prob, target_sensitivity=0.9)
    results["metrics"]["specificity_at_sensitivity_90"] = float(spec)
    
    results["metrics"]["ppv"] = float(compute_ppv(y_true, (y_prob > 0.5).astype(int)))
    results["metrics"]["npv"] = float(compute_npv(y_true, (y_prob > 0.5).astype(int)))
    
    logger.info("Computing bootstrap confidence intervals")
    results["confidence_intervals"]["auc_roc"] = bootstrap_confidence_interval(
        y_true, lambda y, p: compute_auc_roc(y, p), n_bootstrap=args.n_bootstrap, ci_level=args.ci_level
    )
    results["confidence_intervals"]["auc_pr"] = bootstrap_confidence_interval(
        y_true, lambda y, p: compute_auc_pr(y, p), n_bootstrap=args.n_bootstrap, ci_level=args.ci_level
    )
    results["confidence_intervals"]["brier_score"] = bootstrap_confidence_interval(
        y_true, lambda y, p: compute_brier_score(y, p), n_bootstrap=args.n_bootstrap, ci_level=args.ci_level
    )
    
    if args.compute_calibration:
        logger.info("Computing calibration metrics")
        analyzer = CalibrationAnalyzer()
        reliability = analyzer.reliability_diagram(y_true, y_prob)
        temperature = analyzer.temperature_scale(y_true, y_prob)
        results["calibration"] = {
            "reliability_diagram": reliability,
            "temperature": float(temperature),
        }
    
    if args.compute_dca:
        logger.info("Computing decision curve analysis")
        thresholds = np.linspace(0, 1, 21)
        dca_result = decision_curve_analysis(y_true, y_prob, thresholds)
        results["decision_curve_analysis"] = dca_result
    
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    if args.save_predictions:
        predictions_file = output_dir / "predictions.npz"
        np.savez(
            predictions_file,
            y_true=y_true,
            y_prob=y_prob,
        )
        logger.info("Predictions saved", path=str(predictions_file))
    
    logger.info("Evaluation completed", results=results["metrics"])
    print("\nEvaluation Results:")
    print(json.dumps(results["metrics"], indent=2))
    
    return results


if __name__ == "__main__":
    main()
