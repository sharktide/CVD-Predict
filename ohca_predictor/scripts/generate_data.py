"""Data generation script for OHCA Prediction Model."""

import argparse
import sys
from pathlib import Path
import logging
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ohca_predictor.config import get_config, override_config
from ohca_predictor.simulator.population import PopulationGenerator
from ohca_predictor.simulator.generator import DataGenerator
from ohca_predictor.utils.reproducibility import ReproducibilityManager
from ohca_predictor.utils.logging import StructuredLogger
from ohca_predictor.utils.io_utils import DataIOManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic OHCA data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="synthetic_ohca",
        help="Name of the dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=1000,
        help="Number of patients to simulate",
    )
    parser.add_argument(
        "--n-hours",
        type=int,
        default=24,
        help="Number of hours to simulate per patient",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=250,
        help="Sampling rate in Hz",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Window size in seconds",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=5,
        help="Window stride in seconds",
    )
    parser.add_argument(
        "--include-arrhythmias",
        action="store_true",
        help="Include arrhythmia simulation",
    )
    parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="Include signal artifacts",
    )
    parser.add_argument(
        "--artifact-rate",
        type=float,
        default=0.1,
        help="Rate of artifact injection (0-1)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Fraction of data for testing",
    )
    parser.add_argument(
        "--save-format",
        type=str,
        choices=["npz", "tfrecord", "hdf5"],
        default="npz",
        help="Format to save generated data",
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
    
    logger.info("Starting synthetic data generation", args=vars(args))
    
    repro_manager = ReproducibilityManager(seed=args.seed)
    repro_manager.set_all_seeds()
    
    config = get_config()
    
    if args.config:
        config = override_config(config, args.config)
    
    config.simulation.n_patients = args.n_patients
    config.simulation.duration_hours = args.n_hours
    config.simulation.sampling_rate = args.sampling_rate
    config.simulation.window_size_seconds = args.window_size
    config.simulation.window_stride_seconds = args.window_stride
    
    logger.info("Generating patient population", n_patients=args.n_patients)
    population_gen = PopulationGenerator(seed=args.seed)
    patients = population_gen.generate_population(n_patients=args.n_patients)
    
    logger.info("Generating synthetic data from population")
    data_gen = DataGenerator(config=config, seed=args.seed)
    
    if args.include_arrhythmias:
        arrhythmia_types = ["ventricular_tachycardia", "ventricular_fibrillation", "atrial_fibrillation"]
    else:
        arrhythmia_types = None
    
    dataset = data_gen.generate_dataset(
        patients=patients,
        arrhythmia_types=arrhythmia_types,
        include_artifacts=args.include_artifacts,
        artifact_rate=args.artifact_rate,
    )
    
    output_dir = Path(args.output_dir) / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Splitting dataset")
    train_data, val_data, test_data = data_gen.split_dataset(
        dataset,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
    )
    
    logger.info("Saving datasets", output_dir=str(output_dir), format=args.save_format)
    io_manager = DataIOManager(base_path=str(output_dir))
    
    if args.save_format == "npz":
        io_manager.save_dataset(train_data, str(output_dir / "train.npz"))
        io_manager.save_dataset(val_data, str(output_dir / "val.npz"))
        io_manager.save_dataset(test_data, str(output_dir / "test.npz"))
    elif args.save_format == "tfrecord":
        io_manager.save_tfrecord(train_data, str(output_dir / "train.tfrecord"))
        io_manager.save_tfrecord(val_data, str(output_dir / "val.tfrecord"))
        io_manager.save_tfrecord(test_data, str(output_dir / "test.tfrecord"))
    elif args.save_format == "hdf5":
        io_manager.save_hdf5(train_data, str(output_dir / "train.h5"))
        io_manager.save_hdf5(val_data, str(output_dir / "val.h5"))
        io_manager.save_hdf5(test_data, str(output_dir / "test.h5"))
    
    metadata = {
        "n_patients": args.n_patients,
        "n_hours": args.n_hours,
        "sampling_rate": args.sampling_rate,
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "seed": args.seed,
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Data generation completed", metadata=metadata)
    print(f"\nDataset generated successfully:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    print(f"  Saved to: {output_dir}")
    
    return metadata


if __name__ == "__main__":
    main()
