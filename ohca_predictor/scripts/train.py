"""Training script for OHCA Prediction Model."""

import argparse
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ohca_predictor.config import get_config, override_config
from ohca_predictor.model.architecture import OHCAPredictionModel
from ohca_predictor.training.trainer import OHCATrainer
from ohca_predictor.utils.reproducibility import ReproducibilityManager
from ohca_predictor.utils.logging import StructuredLogger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train OHCA Prediction Model",
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
        help="Path to training data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ohca_experiment",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Number of gradient accumulation steps",
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
    
    logger.info("Starting OHCA model training", args=vars(args))
    
    repro_manager = ReproducibilityManager(seed=args.seed)
    repro_manager.set_all_seeds()
    
    config = get_config()
    
    if args.config:
        config = override_config(config, args.config)
    
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.gradient_accumulation_steps is not None:
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    
    config.training.experiment_name = args.experiment_name
    config.training.output_dir = args.output_dir
    
    logger.info("Configuration loaded", config=config.__dict__)
    
    model = OHCAPredictionModel(config)
    
    trainer = OHCATrainer(
        config=config,
        model=model,
        data_path=args.data_path,
        gpus=args.gpus,
        mixed_precision=args.mixed_precision,
    )
    
    trainer.compile()
    
    if args.resume_from:
        logger.info("Resuming training from checkpoint", checkpoint=args.resume_from)
        trainer.load_checkpoint(args.resume_from)
    
    logger.info("Starting training loop")
    history = trainer.fit()
    
    logger.info("Training completed", final_loss=history.history["loss"][-1])
    
    final_checkpoint = Path(args.output_dir) / args.experiment_name / "final_model.ckpt"
    trainer.save_checkpoint(str(final_checkpoint))
    logger.info("Final model saved", path=str(final_checkpoint))
    
    return history


if __name__ == "__main__":
    main()
