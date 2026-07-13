"""Pipeline orchestrator – chains loading, labeling, cohort, preprocessing, training, evaluation.

Usage
-----
    python -m src.run_pipeline                     # run full pipeline
    python -m src.run_pipeline --steps label cohort preprocess  # run specific steps
    python -m src.run_pipeline --steps train       # train only (requires processed data)
    python -m src.run_pipeline --steps eval        # evaluate only (requires trained model)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from src.config import get_paths_config, get_training_config
from src.utils import ensure_dir

logger = logging.getLogger(__name__)

ALL_STEPS = ["label", "cohort", "preprocess", "train", "eval"]


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_label() -> None:
    """Load MIMIC-IV clinical data and build event labels."""
    from src.data_loaders import MIMICClinicalLoader
    from src.labeling import build_event_labels_from_loader

    paths = get_paths_config()
    raw_dir = paths["raw_data_dir"]
    processed_dir = paths["processed_data_dir"]
    ensure_dir(processed_dir)

    loader = MIMICClinicalLoader(raw_dir)
    output_path = os.path.join(processed_dir, "event_labels.parquet")
    build_event_labels_from_loader(loader, output_path=output_path, skip_labs=True)


def step_cohort() -> None:
    """Build cohort metadata from clinical data + event labels."""
    from src.data_loaders import MIMICClinicalLoader
    from src.cohort import build_cohort_from_loader

    paths = get_paths_config()
    raw_dir = paths["raw_data_dir"]
    processed_dir = paths["processed_data_dir"]

    labels_path = os.path.join(processed_dir, "event_labels.parquet")
    if not os.path.exists(labels_path):
        logger.error("event_labels.parquet not found — run 'label' step first")
        return

    import pandas as pd
    labels_df = pd.read_parquet(labels_path)

    loader = MIMICClinicalLoader(raw_dir)
    output_path = os.path.join(processed_dir, "cohort_meta.parquet")
    build_cohort_from_loader(loader, labels_df, output_path=output_path)


def step_preprocess() -> None:
    """Slice windows, extract features, save processed data."""
    from src.preprocess import preprocess_all
    preprocess_all()


def step_train() -> None:
    """Train the model."""
    from src.train import train
    train()


def step_eval() -> None:
    """Evaluate the trained model."""
    from src.eval import evaluate_model

    paths = get_paths_config()
    train_cfg = get_training_config()
    run_name = train_cfg.get("run_name", "cvd_risk_v1")
    model_path = os.path.join(paths["models_dir"], run_name, "best_model.keras")

    if not os.path.exists(model_path):
        model_path = os.path.join(paths["models_dir"], run_name, "final_model.keras")

    if not os.path.exists(model_path):
        logger.error("No trained model found at %s", model_path)
        return

    evaluate_model(model_path, run_name=run_name)


# ---------------------------------------------------------------------------
# Step dispatch
# ---------------------------------------------------------------------------

STEP_FUNCS = {
    "label": step_label,
    "cohort": step_cohort,
    "preprocess": step_preprocess,
    "train": step_train,
    "eval": step_eval,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CVD Risk Prediction Pipeline")
    parser.add_argument(
        "--steps",
        nargs="*",
        choices=ALL_STEPS,
        default=ALL_STEPS,
        help="Pipeline steps to run (default: all).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    for step in args.steps:
        logger.info("=" * 60)
        logger.info("Running step: %s", step)
        logger.info("=" * 60)
        try:
            STEP_FUNCS[step]()
        except Exception:
            logger.exception("Step '%s' failed", step)
            sys.exit(1)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
