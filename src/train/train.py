#!/usr/bin/env python3
"""End-to-end training script for the stroke-prediction model.

Usage
-----
    # Full pipeline: download → preprocess → train
    python -m src.train.train --full

    # Train only (assumes preprocessed data in data/processed/)
    python -m src.train.train

    # Train a specific window size
    python -m src.train.train --window-hours 48

    # Use GRU instead of LSTM
    python -m src.train.train --rnn-type gru
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

from .config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CLASS_WEIGHT_STROKE,
    DATA_DIR,
    EPOCHS,
    EARLY_STOP_PATIENCE,
    LOG_DIR,
    N_STATIC_FEATURES,
    N_TIME_FEATURES,
    PROCESSED_DIR,
    RANDOM_SEED,
    TEST_SPLIT,
    VAL_SPLIT,
    WINDOW_LENGTH_MINUTES,
)
from .model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_preprocessed(window_hours: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Load preprocessed windows from disk.

    Returns (features, labels) where features has shape
    ``(n, window_length, n_time_features)``.
    """
    npz_path = PROCESSED_DIR / f"windows_{window_hours}h.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {npz_path}\n"
            "Run `python -m src.train.preprocess` first."
        )

    data = np.load(npz_path)
    features = data["features"]
    labels = data["labels"]

    log.info(
        "Loaded %d windows of %dh (shape=%s, label_pos_rate=%.4f)",
        len(features), window_hours, features.shape, labels.mean(),
    )
    return features, labels


def generate_synthetic_static(n: int) -> np.ndarray:
    """Generate synthetic static clinical features for demonstration.

    In production these come from patient records.  Here we sample
    plausible distributions so the model can be trained end-to-end.

    Columns: age, sex, cha2ds2_vasc, has_af, has_hypertension,
             has_diabetes, has_chf, has_vascular_disease
    """
    rng = np.random.RandomState(RANDOM_SEED)
    age = rng.normal(68, 12, size=(n, 1)).clip(30, 100) / 100.0
    sex = rng.binomial(1, 0.45, size=(n, 1))  # 1 = male
    cha2ds2 = rng.poisson(3, size=(n, 1)).clip(0, 9) / 9.0
    af = rng.binomial(1, 0.25, size=(n, 1))
    htn = rng.binomial(1, 0.60, size=(n, 1))
    dm = rng.binomial(1, 0.30, size=(n, 1))
    chf = rng.binomial(1, 0.20, size=(n, 1))
    vasc = rng.binomial(1, 0.15, size=(n, 1))
    return np.hstack([age, sex, cha2ds2, af, htn, dm, chf, vasc]).astype(np.float32)


def generate_synthetic_labels(
    features: np.ndarray,
    base_rate: float = 0.02,
) -> np.ndarray:
    """Generate placeholder labels with a realistic stroke-event rate (~2 %).

    In production, labels come from verified clinical events.
    """
    rng = np.random.RandomState(RANDOM_SEED + 1)
    n = features.shape[0]
    labels = rng.binomial(1, base_rate, size=(n,)).astype(np.float32)
    log.info("Synthetic labels: %d positives out of %d (%.2f%%)", labels.sum(), n, labels.mean() * 100)
    return labels


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_data(
    features: np.ndarray,
    labels: np.ndarray,
    static: np.ndarray,
    val_split: float = VAL_SPLIT,
    test_split: float = TEST_SPLIT,
) -> dict:
    """Stratified train / val / test split."""
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        features, labels, static,
        test_size=test_split, random_state=RANDOM_SEED, stratify=labels,
    )
    X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
        X_train, y_train, s_train,
        test_size=val_split / (1 - test_split), random_state=RANDOM_SEED, stratify=y_train,
    )
    log.info(
        "Split → train=%d  val=%d  test=%d  (positive rates: %.3f / %.3f / %.3f)",
        len(X_train), len(X_val), len(X_test),
        y_train.mean(), y_val.mean(), y_test.mean(),
    )
    return {
        "X_train": X_train, "y_train": y_train, "s_train": s_train,
        "X_val": X_val, "y_val": y_val, "s_val": s_val,
        "X_test": X_test, "y_test": y_test, "s_test": s_test,
    }


# ---------------------------------------------------------------------------
# Class-weight helper
# ---------------------------------------------------------------------------


def compute_class_weights(labels: np.ndarray, stroke_weight: float = CLASS_WEIGHT_STROKE) -> dict:
    """Return class_weight dict for ``model.fit``."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return {0: 1.0, 1: 1.0}
    w_neg = len(labels) / (2 * n_neg)
    w_pos = len(labels) / (2 * n_pos) * stroke_weight
    return {0: float(w_neg), 1: float(w_pos)}


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def make_callbacks(window_hours: int) -> list[keras.callbacks.Callback]:
    """Build standard training callbacks."""
    ckpt_path = CHECKPOINT_DIR / f"stroke_{window_hours}h_best.keras"
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=EARLY_STOP_PATIENCE,
            mode="max",
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            str(ckpt_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(LOG_DIR / f"window_{window_hours}h"),
            histogram_freq=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    window_hours: int = 24,
    rnn_type: str = "lstm",
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    run_download: bool = False,
    run_preprocess: bool = False,
) -> keras.Model:
    """Full training pipeline.

    Parameters
    ----------
    window_hours : int
        Which window size to train on (24, 48, or 72).
    rnn_type : str
        ``"lstm"`` or ``"gru"``.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    run_download : bool
        If True, download datasets first.
    run_preprocess : bool
        If True, run preprocessing first.

    Returns
    -------
    Trained ``keras.Model``.
    """

    # --- Optional upstream steps ---
    if run_download:
        from .download_datasets import download_all
        download_all()

    if run_preprocess:
        from .preprocess import prepare_dataset
        prepare_dataset()

    # --- Load data ---
    features, labels = load_preprocessed(window_hours)

    n_windows = features.shape[0]
    n_time_feats = features.shape[2]

    # --- Static features (synthetic in demo; real in production) ---
    static = generate_synthetic_static(n_windows)

    # --- Synthetic labels (replace with real labels when available) ---
    if labels.sum() == 0:
        log.info("All labels are zero — generating synthetic stroke-event labels for demo.")
        labels = generate_synthetic_labels(features)

    # --- Split ---
    data = split_data(features, labels, static)

    # --- Build model ---
    model = build_model(
        window_length=WINDOW_LENGTH_MINUTES,
        n_time_features=n_time_feats,
        n_static_features=N_STATIC_FEATURES,
        rnn_type=rnn_type,
    )
    model.summary(print_fn=log.info)

    # --- Class weights ---
    class_weights = compute_class_weights(data["y_train"])
    log.info("Class weights: %s", class_weights)

    # --- Train ---
    callbacks = make_callbacks(window_hours)
    history = model.fit(
        [data["X_train"], data["s_train"]],
        data["y_train"],
        validation_data=([data["X_val"], data["s_val"]], data["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # --- Evaluate on test set ---
    log.info("=== Test Set Evaluation ===")
    test_metrics = model.evaluate(
        [data["X_test"], data["s_test"]],
        data["y_test"],
        batch_size=batch_size,
        verbose=0,
        return_dict=True,
    )
    for k, v in test_metrics.items():
        log.info("  %s: %.4f", k, v)

    # --- Save final model ---
    final_path = CHECKPOINT_DIR / f"stroke_{window_hours}h_final.keras"
    model.save(str(final_path))
    log.info("Saved final model → %s", final_path)

    # --- Save training history ---
    hist_path = LOG_DIR / f"history_{window_hours}h.json"
    with open(hist_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    log.info("Saved training history → %s", hist_path)

    # --- Save test metrics ---
    metrics_path = LOG_DIR / f"test_metrics_{window_hours}h.json"
    with open(metrics_path, "w") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)

    return model


# ---------------------------------------------------------------------------
# Post-training: export for on-device deployment
# ---------------------------------------------------------------------------


def export_model(
    model: keras.Model,
    window_hours: int,
    export_coreml: bool = False,
    export_tflite: bool = True,
) -> None:
    """Convert trained model to deployment formats."""
    out_dir = CHECKPOINT_DIR / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    if export_tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        tflite_path = out_dir / f"stroke_{window_hours}h.tflite"
        tflite_path.write_bytes(tflite_model)
        log.info("Exported TFLite → %s (%.1f KB)", tflite_path, len(tflite_model) / 1024)

    if export_coreml:
        try:
            import coremltools as ct  # type: ignore[import-untyped]
        except ImportError:
            log.warning("coremltools not installed — skipping CoreML export.")
            return
        mlmodel = ct.convert(
            model,
            inputs=[ct.TensorType(name="time_input", shape=(1, WINDOW_LENGTH_MINUTES, model.input_shape[0][2])),
                    ct.TensorType(name="static_input", shape=(1, N_STATIC_FEATURES))],
        )
        coreml_path = out_dir / f"stroke_{window_hours}h.mlmodel"
        mlmodel.save(str(coreml_path))
        log.info("Exported CoreML → %s", coreml_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the stroke-prediction model.")
    parser.add_argument("--window-hours", type=int, default=24, choices=[24, 48, 72])
    parser.add_argument("--rnn-type", default="lstm", choices=["lstm", "gru"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--full", action="store_true", help="Run download + preprocess + train.")
    parser.add_argument("--preprocess-only", action="store_true", help="Run download + preprocess, no training.")
    parser.add_argument("--export-coreml", action="store_true", help="Export to CoreML after training.")
    parser.add_argument("--export-tflite", action="store_true", default=True)
    args = parser.parse_args()

    if args.preprocess_only:
        from .download_datasets import download_all
        from .preprocess import prepare_dataset
        download_all()
        prepare_dataset()
        return

    model = train_model(
        window_hours=args.window_hours,
        rnn_type=args.rnn_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        run_download=args.full,
        run_preprocess=args.full,
    )

    export_model(model, args.window_hours, export_coreml=args.export_coreml, export_tflite=args.export_tflite)


if __name__ == "__main__":
    main()
