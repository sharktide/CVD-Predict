"""Model export – TFLite and ONNX conversion."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import tensorflow as tf

from src.config import get_eval_config, get_paths_config
from src.utils import ensure_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TFLite export
# ---------------------------------------------------------------------------

def export_tflite(
    model_path: str,
    output_dir: Optional[str] = None,
    output_name: str = "model.tflite",
) -> str:
    """Convert a Keras model to TFLite format and write to disk.

    Returns the output path.
    """
    paths = get_paths_config()
    eval_cfg = get_eval_config()

    if output_dir is None:
        version = eval_cfg.get("model_version", "v1")
        output_dir = os.path.join(paths["models_dir"], f"export_{version}")
    ensure_dir(output_dir)

    model = tf.keras.models.load_model(model_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optional quantisation
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    out_path = os.path.join(output_dir, output_name)
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info("TFLite model saved → %s (%.1f MB)", out_path, size_mb)
    return out_path


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(
    model_path: str,
    output_dir: Optional[str] = None,
    output_name: str = "model.onnx",
) -> str:
    """Convert a Keras model to ONNX format via tf2onnx.

    Returns the output path.
    """
    try:
        import tf2onnx
    except ImportError:
        logger.error("tf2onnx is not installed. Install with: pip install tf2onnx")
        raise

    paths = get_paths_config()
    eval_cfg = get_eval_config()

    if output_dir is None:
        version = eval_cfg.get("model_version", "v1")
        output_dir = os.path.join(paths["models_dir"], f"export_{version}")
    ensure_dir(output_dir)

    model = tf.keras.models.load_model(model_path, compile=False)

    # Build input signatures from model inputs
    input_signatures = []
    for inp in model.inputs:
        input_signatures.append(
            tf.TensorSpec(inp.shape, dtype=inp.dtype, name=inp.name.split(":")[0])
        )

    out_path = os.path.join(output_dir, output_name)
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signatures,
        output_path=out_path,
    )

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info("ONNX model saved → %s (%.1f MB)", out_path, size_mb)
    return out_path


# ---------------------------------------------------------------------------
# Save preprocessing metadata alongside export
# ---------------------------------------------------------------------------

def save_export_metadata(
    model_path: str,
    output_dir: Optional[str] = None,
) -> None:
    """Write a metadata.json alongside the exported model with feature columns, preprocessing config."""
    paths = get_paths_config()
    eval_cfg = get_eval_config()
    train_cfg_path = os.path.join(paths.get("processed_data_dir", "data/processed"), "..")

    model_dir = os.path.dirname(model_path)
    feat_cols_path = os.path.join(model_dir, "feature_columns.json")

    metadata = {
        "model_path": model_path,
        "feature_columns": [],
        "preprocessing": {
            "ppg_sample_rate_hz": 125,
            "wearable_target_rate_hz": 25,
            "normalisation": "zscore",
            "window_seconds": 60,
        },
        "calibration": {
            "method": eval_cfg.get("calibration_method", "isotonic"),
        },
    }

    if os.path.exists(feat_cols_path):
        with open(feat_cols_path) as f:
            metadata["feature_columns"] = json.load(f)

    if output_dir is None:
        version = eval_cfg.get("model_version", "v1")
        output_dir = os.path.join(paths["models_dir"], f"export_{version}")
    ensure_dir(output_dir)

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Export metadata saved → %s", meta_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/cvd_risk_v1/final_model.keras"

    eval_cfg = get_eval_config()
    if eval_cfg.get("export_tflite", True):
        export_tflite(model_path)
    if eval_cfg.get("export_onnx", True):
        export_onnx(model_path)
    save_export_metadata(model_path)
