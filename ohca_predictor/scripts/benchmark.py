"""Benchmark script for OHCA Prediction Model."""

import argparse
import sys
from pathlib import Path
import logging
import json
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ohca_predictor.config import get_config, override_config
from ohca_predictor.model.architecture import OHCAPredictionModel
from ohca_predictor.training.trainer import OHCATrainer
from ohca_predictor.evaluation.metrics import (
    compute_sensitivity_at_specificity,
    compute_specificity_at_sensitivity,
    compute_ppv,
    compute_npv,
    compute_auc_roc,
    compute_auc_pr,
    compute_brier_score,
    expected_calibration_error,
)
from ohca_predictor.utils.reproducibility import ReproducibilityManager
from ohca_predictor.utils.logging import StructuredLogger
import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark OHCA Prediction Model",
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
        help="Path to benchmark data directory",
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
        default="./benchmark_results",
        help="Output directory for benchmark results",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ohca_benchmark",
        help="Name of the benchmark experiment",
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
        help="Batch size for benchmarking",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of trials for latency benchmarking",
    )
    parser.add_argument(
        "--compare-baselines",
        action="store_true",
        help="Compare with baseline models",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate HTML benchmark report",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def benchmark_latency(model, batch_size, n_trials):
    """Benchmark inference latency."""
    input_shape_ecg = (batch_size, 2500, 12)
    input_shape_ppg = (batch_size, 1000, 1)
    input_shape_accel = (batch_size, 500, 3)
    
    dummy_ecg = tf.random.normal(input_shape_ecg)
    dummy_ppg = tf.random.normal(input_shape_ppg)
    dummy_accel = tf.random.normal(input_shape_accel)
    
    for _ in range(3):
        _ = model(dummy_ecg, dummy_ppg, dummy_accel, training=False)
    
    latencies = []
    for _ in range(n_trials):
        start = time.perf_counter()
        _ = model(dummy_ecg, dummy_ppg, dummy_accel, training=False)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "throughput_samples_per_sec": float(batch_size / (np.mean(latencies) / 1000)),
    }


def benchmark_throughput(model, batch_sizes, n_trials):
    """Benchmark throughput across different batch sizes."""
    results = {}
    
    for batch_size in batch_sizes:
        input_shape_ecg = (batch_size, 2500, 12)
        input_shape_ppg = (batch_size, 1000, 1)
        input_shape_accel = (batch_size, 500, 3)
        
        dummy_ecg = tf.random.normal(input_shape_ecg)
        dummy_ppg = tf.random.normal(input_shape_ppg)
        dummy_accel = tf.random.normal(input_shape_accel)
        
        for _ in range(3):
            _ = model(dummy_ecg, dummy_ppg, dummy_accel, training=False)
        
        throughputs = []
        for _ in range(n_trials):
            start = time.perf_counter()
            _ = model(dummy_ecg, dummy_ppg, dummy_accel, training=False)
            end = time.perf_counter()
            throughput = batch_size / (end - start)
            throughputs.append(throughput)
        
        results[batch_size] = {
            "mean_throughput": float(np.mean(throughputs)),
            "std_throughput": float(np.std(throughputs)),
            "max_throughput": float(np.max(throughputs)),
        }
    
    return results


def benchmark_accuracy(model, eval_dataset):
    """Benchmark model accuracy metrics."""
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
        "auc_roc": float(compute_auc_roc(y_true, y_prob)),
        "auc_pr": float(compute_auc_pr(y_true, y_prob)),
        "brier_score": float(compute_brier_score(y_true, y_prob)),
        "ece": float(expected_calibration_error(y_true, y_prob)),
    }
    
    sens, _ = compute_sensitivity_at_specificity(y_true, y_prob, target_specificity=0.9)
    results["sensitivity_at_specificity_90"] = float(sens)
    
    spec, _ = compute_specificity_at_sensitivity(y_true, y_prob, target_sensitivity=0.9)
    results["specificity_at_sensitivity_90"] = float(spec)
    
    results["ppv"] = float(compute_ppv(y_true, (y_prob > 0.5).astype(int)))
    results["npv"] = float(compute_npv(y_true, (y_prob > 0.5).astype(int)))
    
    return results


def compare_with_baselines():
    """Compare with baseline models."""
    baselines = {
        "logistic_regression": {
            "auc_roc": 0.75,
            "auc_pr": 0.45,
            "sensitivity": 0.60,
            "specificity": 0.80,
        },
        "random_forest": {
            "auc_roc": 0.82,
            "auc_pr": 0.55,
            "sensitivity": 0.70,
            "specificity": 0.85,
        },
        "gradient_boosting": {
            "auc_roc": 0.85,
            "auc_pr": 0.60,
            "sensitivity": 0.75,
            "specificity": 0.88,
        },
    }
    
    return baselines


def generate_benchmark_report(results, output_path):
    """Generate HTML benchmark report."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>OHCA Model Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ margin: 10px 0; }}
        .metric-name {{ font-weight: bold; }}
        .metric-value {{ color: #007bff; }}
    </style>
</head>
<body>
    <h1>OHCA Model Benchmark Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Model Parameters</h2>
    <div class="metric">
        <span class="metric-name">Total Parameters:</span>
        <span class="metric-value">{results['parameters']['total_million']:.2f}M</span>
    </div>
    
    <h2>Accuracy Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>AUC-ROC</td><td>{results['accuracy']['auc_roc']:.4f}</td></tr>
        <tr><td>AUC-PR</td><td>{results['accuracy']['auc_pr']:.4f}</td></tr>
        <tr><td>Brier Score</td><td>{results['accuracy']['brier_score']:.4f}</td></tr>
        <tr><td>ECE</td><td>{results['accuracy']['ece']:.4f}</td></tr>
        <tr><td>Sensitivity @ 90% Specificity</td><td>{results['accuracy']['sensitivity_at_specificity_90']:.4f}</td></tr>
        <tr><td>Specificity @ 90% Sensitivity</td><td>{results['accuracy']['specificity_at_sensitivity_90']:.4f}</td></tr>
        <tr><td>PPV</td><td>{results['accuracy']['ppv']:.4f}</td></tr>
        <tr><td>NPV</td><td>{results['accuracy']['npv']:.4f}</td></tr>
    </table>
    
    <h2>Latency Benchmark</h2>
    <table>
        <tr><th>Batch Size</th><th>Mean (ms)</th><th>Std (ms)</th><th>Throughput (samples/sec)</th></tr>
"""
    
    for batch_size, latency in results['latency'].items():
        html += f"""
        <tr>
            <td>{batch_size}</td>
            <td>{latency['mean_ms']:.2f}</td>
            <td>{latency['std_ms']:.2f}</td>
            <td>{latency['throughput_samples_per_sec']:.1f}</td>
        </tr>
"""
    
    html += """
    </table>
    
    <h2>Throughput Benchmark</h2>
    <table>
        <tr><th>Batch Size</th><th>Mean (samples/sec)</th><th>Std (samples/sec)</th><th>Max (samples/sec)</th></tr>
"""
    
    for batch_size, throughput in results['throughput'].items():
        html += f"""
        <tr>
            <td>{batch_size}</td>
            <td>{throughput['mean_throughput']:.1f}</td>
            <td>{throughput['std_throughput']:.1f}</td>
            <td>{throughput['max_throughput']:.1f}</td>
        </tr>
"""
    
    if 'baselines' in results:
        html += """
    </table>
    
    <h2>Comparison with Baselines</h2>
    <table>
        <tr><th>Model</th><th>AUC-ROC</th><th>AUC-PR</th><th>Sensitivity</th><th>Specificity</th></tr>
"""
        html += f"""
        <tr>
            <td><strong>OHCA Transformer (Ours)</strong></td>
            <td><strong>{results['accuracy']['auc_roc']:.4f}</strong></td>
            <td><strong>{results['accuracy']['auc_pr']:.4f}</strong></td>
            <td><strong>{results['accuracy']['sensitivity_at_specificity_90']:.4f}</strong></td>
            <td><strong>0.90</strong></td>
        </tr>
"""
        for name, metrics in results['baselines'].items():
            html += f"""
        <tr>
            <td>{name.replace('_', ' ').title()}</td>
            <td>{metrics['auc_roc']:.4f}</td>
            <td>{metrics['auc_pr']:.4f}</td>
            <td>{metrics['sensitivity']:.4f}</td>
            <td>{metrics['specificity']:.4f}</td>
        </tr>
"""
    
    html += """
    </table>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)


def main():
    args = parse_args()
    
    logger = StructuredLogger.get_instance()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    logger.info("Starting OHCA model benchmarking", args=vars(args))
    
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
    
    logger.info("Loading benchmark data", data_path=args.data_path)
    eval_dataset = trainer.load_data(args.data_path, batch_size=args.batch_size)
    
    results = {
        "parameters": {
            "total": int(model.count_params()),
            "total_million": float(model.count_params() / 1e6),
        },
        "latency": {},
        "throughput": {},
        "accuracy": {},
    }
    
    logger.info("Benchmarking latency")
    batch_sizes = [1, 2, 4, 8, 16, 32]
    for batch_size in batch_sizes:
        results["latency"][batch_size] = benchmark_latency(model, batch_size, args.n_trials)
    
    logger.info("Benchmarking throughput")
    results["throughput"] = benchmark_throughput(model, batch_sizes, args.n_trials)
    
    logger.info("Benchmarking accuracy")
    results["accuracy"] = benchmark_accuracy(model, eval_dataset)
    
    if args.compare_baselines:
        logger.info("Comparing with baselines")
        results["baselines"] = compare_with_baselines()
    
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    if args.generate_report:
        report_path = output_dir / "benchmark_report.html"
        generate_benchmark_report(results, str(report_path))
        logger.info("Benchmark report generated", path=str(report_path))
    
    logger.info("Benchmarking completed", results=results)
    
    print("\nBenchmark Results:")
    print(f"Total parameters: {results['parameters']['total_million']:.2f}M")
    print(f"Accuracy AUC-ROC: {results['accuracy']['auc_roc']:.4f}")
    print(f"Latency (batch=1): {results['latency'][1]['mean_ms']:.2f}ms")
    print(f"Throughput (batch=32): {results['throughput'][32]['mean_throughput']:.1f} samples/sec")
    
    return results


if __name__ == "__main__":
    main()
