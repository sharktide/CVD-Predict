"""Profiling script for OHCA Prediction Model."""

import argparse
import sys
from pathlib import Path
import logging
import json
import time
from contextlib import contextmanager

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ohca_predictor.config import get_config, override_config
from ohca_predictor.model.architecture import OHCAPredictionModel
from ohca_predictor.utils.reproducibility import ReproducibilityManager
from ohca_predictor.utils.logging import StructuredLogger
import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile OHCA Prediction Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (optional)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to model checkpoint (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./profiling_results",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ohca_profiling",
        help="Name of the profiling experiment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=100,
        help="Number of iterations to profile",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32],
        help="Batch sizes to profile",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Profile GPU memory usage",
    )
    parser.add_argument(
        "--profile-flops",
        action="store_true",
        help="Profile FLOPs",
    )
    parser.add_argument(
        "--profile-latency",
        action="store_true",
        help="Profile inference latency",
    )
    parser.add_argument(
        "--profile-all",
        action="store_true",
        help="Profile all metrics (memory, FLOPs, latency)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


@contextmanager
def timer():
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    return end - start


def profile_latency(model, batch_size, n_warmup, n_iterations):
    """Profile inference latency."""
    input_shape_ecg = (batch_size, 2500, 12)
    input_shape_ppg = (batch_size, 1000, 1)
    input_shape_accel = (batch_size, 500, 3)
    
    dummy_ecg = tf.random.normal(input_shape_ecg)
    dummy_ppg = tf.random.normal(input_shape_ppg)
    dummy_accel = tf.random.normal(input_shape_accel)
    
    for _ in range(n_warmup):
        _ = model(dummy_ecg, dummy_ppg, dummy_accel, training=False)
    
    latencies = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = model(dummy_ecg, dummy_ppg, dummy_accel, training=False)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def profile_memory(model, batch_size):
    """Profile GPU memory usage."""
    input_shape_ecg = (batch_size, 2500, 12)
    input_shape_ppg = (batch_size, 1000, 1)
    input_shape_accel = (batch_size, 500, 3)
    
    dummy_ecg = tf.random.normal(input_shape_ecg)
    dummy_ppg = tf.random.normal(input_shape_ppg)
    dummy_accel = tf.random.normal(input_shape_accel)
    
    tf.keras.backend.clear_session()
    
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.reset_memory_stats('GPU:0')
    
    _ = model(dummy_ecg, dummy_ppg, dummy_accel, training=False)
    
    memory_stats = {}
    if tf.config.list_physical_devices('GPU'):
        try:
            stats = tf.config.experimental.get_memory_stats('GPU:0')
            memory_stats = {
                "current_bytes": stats.get("current", 0),
                "peak_bytes": stats.get("peak", 0),
                "current_mb": stats.get("current", 0) / (1024 * 1024),
                "peak_mb": stats.get("peak", 0) / (1024 * 1024),
            }
        except Exception as e:
            memory_stats = {"error": str(e)}
    
    return memory_stats


def profile_flops(model, batch_size):
    """Profile FLOPs."""
    input_shape_ecg = (batch_size, 2500, 12)
    input_shape_ppg = (batch_size, 1000, 1)
    input_shape_accel = (batch_size, 500, 3)
    
    dummy_ecg = tf.random.normal(input_shape_ecg)
    dummy_ppg = tf.random.normal(input_shape_ppg)
    dummy_accel = tf.random.normal(input_shape_accel)
    
    tf.keras.backend.clear_session()
    
    tf.profiler.experimental.start('logdir')
    _ = model(dummy_ecg, dummy_ppg, dummy_accel, training=False)
    tf.profiler.experimental.stop()
    
    return {"flops": "See TensorBoard profiler logs"}


def count_parameters(model):
    """Count model parameters."""
    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_variables)
    non_trainable_params = total_params - trainable_params
    
    return {
        "total": int(total_params),
        "trainable": int(trainable_params),
        "non_trainable": int(non_trainable_params),
        "total_million": float(total_params / 1e6),
    }


def main():
    args = parse_args()
    
    logger = StructuredLogger.get_instance()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    logger.info("Starting OHCA model profiling", args=vars(args))
    
    repro_manager = ReproducibilityManager(seed=args.seed)
    repro_manager.set_all_seeds()
    
    config = get_config()
    
    if args.config:
        config = override_config(config, args.config)
    
    logger.info("Loading model")
    model = OHCAPredictionModel(config)
    
    if args.checkpoint_path:
        model.load_weights(args.checkpoint_path)
    
    results = {
        "parameters": count_parameters(model),
        "batch_profiles": {},
    }
    
    for batch_size in args.batch_sizes:
        logger.info(f"Profiling batch size {batch_size}")
        batch_results = {}
        
        if args.profile_latency or args.profile_all:
            batch_results["latency"] = profile_latency(
                model, batch_size, args.n_warmup, args.n_iterations
            )
        
        if args.profile_memory or args.profile_all:
            batch_results["memory"] = profile_memory(model, batch_size)
        
        if args.profile_flops or args.profile_all:
            batch_results["flops"] = profile_flops(model, batch_size)
        
        results["batch_profiles"][batch_size] = batch_results
    
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "profiling_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Profiling completed", results=results)
    
    print("\nProfiling Results:")
    print(f"Total parameters: {results['parameters']['total_million']:.2f}M")
    for batch_size, profile in results["batch_profiles"].items():
        print(f"\nBatch size {batch_size}:")
        if "latency" in profile:
            print(f"  Latency: {profile['latency']['mean_ms']:.2f}ms ± {profile['latency']['std_ms']:.2f}ms")
        if "memory" in profile and "peak_mb" in profile["memory"]:
            print(f"  Peak memory: {profile['memory']['peak_mb']:.2f}MB")
    
    return results


if __name__ == "__main__":
    main()
