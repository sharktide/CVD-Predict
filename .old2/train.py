#!/usr/bin/env python3
"""
OHCA Prediction Model — Training Pipeline
==========================================
Multi-modal 1D-CNN + BiGRU + Dense architecture with TensorBoard logging.
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

DATA_DIR = Path(__file__).parent / 'data'
LOG_DIR = Path(__file__).parent / 'runs'
MODEL_PATH = Path(__file__).parent / 'model.keras'

def load_data():
    """Load pre-generated train/test arrays."""
    return {
        'X_ecg_train': np.load(DATA_DIR / 'X_ecg_train.npy'),
        'X_ecg_test':  np.load(DATA_DIR / 'X_ecg_test.npy'),
        'X_acc_train': np.load(DATA_DIR / 'X_acc_train.npy'),
        'X_acc_test':  np.load(DATA_DIR / 'X_acc_test.npy'),
        'X_bio_train': np.load(DATA_DIR / 'X_bio_train.npy'),
        'X_bio_test':  np.load(DATA_DIR / 'X_bio_test.npy'),
        'y_train':     np.load(DATA_DIR / 'y_train.npy'),
        'y_test':      np.load(DATA_DIR / 'y_test.npy'),
    }


def build_model():
    """Construct multi-modal OHCA prediction model."""
    ecg_input = tf.keras.Input(shape=(1300, 1), name='ecg_input')
    acc_input = tf.keras.Input(shape=(250, 3), name='acc_input')
    bio_input = tf.keras.Input(shape=(3,),      name='bio_input')

    # ECG branch (1D CNN)
    x1 = tf.keras.layers.Conv1D(32, 7, activation='relu', padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001))(ecg_input)
    x1 = tf.keras.layers.MaxPooling1D(2)(x1)
    x1 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001))(x1)
    x1 = tf.keras.layers.MaxPooling1D(2)(x1)
    x1 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001))(x1)
    x1 = tf.keras.layers.MaxPooling1D(2)(x1)
    x1 = tf.keras.layers.SpatialDropout1D(0.3)(x1)
    x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)

    # ACC branch (BiGRU)
    x2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(48, dropout=0.3, recurrent_dropout=0.3)
    )(acc_input)

    # Biodata branch
    x3 = tf.keras.layers.Dense(16, activation='relu')(bio_input)

    # Fusion
    fused = tf.keras.layers.Concatenate()([x1, x2, x3])
    fused = tf.keras.layers.Dense(32, activation='relu')(fused)
    fused = tf.keras.layers.Dropout(0.4)(fused)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(fused)

    model = tf.keras.Model(inputs=[ecg_input, acc_input, bio_input],
                           outputs=output, name='OHCA_Predictor')
    return model


def main():
    data = load_data()

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    tb_run = datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=str(LOG_DIR / tb_run),
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=3, monitor='val_loss', restore_best_weights=True
    )

    history = model.fit(
        [data['X_ecg_train'], data['X_acc_train'], data['X_bio_train']],
        data['y_train'],
        validation_split=0.15,
        epochs=20,
        batch_size=64,
        callbacks=[tensorboard_cb, early_stop],
        verbose=1
    )

    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"TensorBoard logs: {LOG_DIR / tb_run}")

    # Quick test metrics
    y_pred_prob = model.predict(
        [data['X_ecg_test'], data['X_acc_test'], data['X_bio_test']]
    ).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    y_true = data['y_test']

    acc = np.mean(y_pred == y_true)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_prob)

    print("\n── Test Metrics ──")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")

    if acc > 0.93:
        print("\n⚠  WARNING: Accuracy > 93%. Check data generation for artificial separability.")


if __name__ == '__main__':
    main()
