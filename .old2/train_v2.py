#!/usr/bin/env python3
"""
OHCA Prediction Model v2 — Autonomous Deployment Grade
=======================================================
Architecture: Temporal Attention CNN + BiGRU + Cross-Modal Fusion
Target: >97% accuracy, >99% sensitivity, <3% false alarm rate
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

DATA_DIR = Path(__file__).parent / 'data'
LOG_DIR = Path(__file__).parent / 'runs_v2'
MODEL_PATH = Path(__file__).parent / 'model_v2.keras'


class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)

        warmup_lr = self.initial_lr * (step / tf.maximum(warmup, 1.0))
        decay_lr = self.initial_lr * 0.5 * (1.0 + tf.cos(np.pi * (step - warmup) / (total - warmup)))

        return tf.where(step < warmup, warmup_lr, decay_lr)

    def get_config(self):
        return {
            'initial_lr': float(self.initial_lr),
            'warmup_steps': int(self.warmup_steps),
            'total_steps': int(self.total_steps),
        }


def load_data():
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


def ecg_augmentation(ecg, training=True):
    """Lightweight ECG augmentation during training."""
    if not training:
        return ecg

    augmented = ecg.copy()

    # Random time shift (circular)
    if np.random.random() < 0.3:
        shift = np.random.randint(-50, 50)
        augmented = np.roll(augmented, shift, axis=1)

    # Random scaling
    if np.random.random() < 0.3:
        scale = np.random.uniform(0.85, 1.15)
        augmented = augmented * scale

    # Random noise injection
    if np.random.random() < 0.3:
        noise_level = np.random.uniform(0.01, 0.05)
        augmented = augmented + np.random.normal(0, noise_level, augmented.shape).astype(np.float32)

    return augmented


def acc_augmentation(acc, training=True):
    if not training:
        return acc
    augmented = acc.copy()
    if np.random.random() < 0.3:
        scale = np.random.uniform(0.9, 1.1, size=(1, 1, 3))
        augmented = augmented * scale
    if np.random.random() < 0.2:
        noise = np.random.normal(0, 0.02, augmented.shape).astype(np.float32)
        augmented = augmented + noise
    return augmented


def build_model_v2():
    """
    Architecture:
    - ECG: Multi-scale CNN + Temporal Self-Attention + GAP
    - ACC: Bidirectional GRU with attention
    - Bio: Deep MLP
    - Fusion: Gated cross-modal attention + classifier
    """
    ecg_input = tf.keras.Input(shape=(1300, 1), name='ecg_input')
    acc_input = tf.keras.Input(shape=(250, 3), name='acc_input')
    bio_input = tf.keras.Input(shape=(6,), name='bio_input')

    # ── ECG Branch: Multi-scale CNN + Self-Attention ──
    # Multi-scale feature extraction
    ecg_small = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(ecg_input)
    ecg_small = tf.keras.layers.BatchNormalization()(ecg_small)

    ecg_med = tf.keras.layers.Conv1D(32, 15, padding='same', activation='relu',
                                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(ecg_input)
    ecg_med = tf.keras.layers.BatchNormalization()(ecg_med)

    ecg_large = tf.keras.layers.Conv1D(32, 31, padding='same', activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(ecg_input)
    ecg_large = tf.keras.layers.BatchNormalization()(ecg_large)

    ecg_fused = tf.keras.layers.Concatenate()([ecg_small, ecg_med, ecg_large])
    ecg_fused = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(ecg_fused)
    ecg_fused = tf.keras.layers.BatchNormalization()(ecg_fused)
    ecg_fused = tf.keras.layers.MaxPooling1D(2)(ecg_fused)

    ecg_fused = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(ecg_fused)
    ecg_fused = tf.keras.layers.BatchNormalization()(ecg_fused)
    ecg_fused = tf.keras.layers.MaxPooling1D(2)(ecg_fused)

    ecg_fused = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(ecg_fused)
    ecg_fused = tf.keras.layers.BatchNormalization()(ecg_fused)

    # Temporal Self-Attention
    attention_scores = tf.keras.layers.Dense(128, activation='tanh')(ecg_fused)
    attention_scores = tf.keras.layers.Dense(1)(attention_scores)
    attention_weights = tf.keras.layers.Softmax(axis=1)(attention_scores)
    ecg_attended = tf.keras.layers.Multiply()([ecg_fused, attention_weights])
    ecg_out = tf.keras.layers.GlobalAveragePooling1D()(ecg_attended)
    ecg_out = tf.keras.layers.Dropout(0.3)(ecg_out)

    # ── ACC Branch: Bidirectional GRU + Attention ──
    acc_x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
    )(acc_input)

    # Attention over time steps
    acc_att = tf.keras.layers.Dense(128, activation='tanh')(acc_x)
    acc_att = tf.keras.layers.Dense(1)(acc_att)
    acc_att = tf.keras.layers.Softmax(axis=1)(acc_att)
    acc_x = tf.keras.layers.Multiply()([acc_x, acc_att])
    acc_out = tf.keras.layers.GlobalAveragePooling1D()(acc_x)
    acc_out = tf.keras.layers.Dropout(0.3)(acc_out)

    # ── Biodata Branch: Deep MLP ──
    bio_x = tf.keras.layers.Dense(32, activation='relu')(bio_input)
    bio_x = tf.keras.layers.BatchNormalization()(bio_x)
    bio_x = tf.keras.layers.Dropout(0.2)(bio_x)
    bio_x = tf.keras.layers.Dense(16, activation='relu')(bio_x)
    bio_out = tf.keras.layers.BatchNormalization()(bio_x)

    # ── Cross-Modal Fusion ──
    fused = tf.keras.layers.Concatenate()([ecg_out, acc_out, bio_out])

    # Gated fusion
    gate = tf.keras.layers.Dense(fused.shape[-1], activation='sigmoid')(fused)
    gated = tf.keras.layers.Multiply()([fused, gate])

    x = tf.keras.layers.Dense(64, activation='relu')(gated)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)

    return tf.keras.Model(inputs=[ecg_input, acc_input, bio_input], outputs=output, name='OHCA_v2')


def focal_loss(gamma=2.0, alpha=0.75):
    """Focal loss for handling hard examples and class imbalance."""
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_weight = alpha_t * tf.pow(1 - p_t, gamma)
        return tf.reduce_mean(focal_weight * bce)
    return loss_fn


class AugmentCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_ecg, X_acc, y, batch_size=64):
        self.X_ecg = X_ecg
        self.X_acc = X_acc
        self.y = y
        self.batch_size = batch_size

    def on_epoch_begin(self, epoch, logs=None):
        indices = np.random.permutation(len(self.y))
        self.X_ecg = self.X_ecg[indices]
        self.X_acc = self.X_acc[indices]
        self.y = self.y[indices]


def main():
    data = load_data()
    print(f"Loaded: ECG {data['X_ecg_train'].shape}, ACC {data['X_acc_train'].shape}, BIO {data['X_bio_train'].shape}")

    model = build_model_v2()
    model.summary()

    total_steps = (len(data['y_train']) // 64) * 30
    warmup_steps = total_steps // 10

    lr_schedule = CosineDecayWithWarmup(
        initial_lr=0.001,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0008)

    # Compute class weights
    n_pos = data['y_train'].sum()
    n_neg = len(data['y_train']) - n_pos
    class_weight = {0: len(data['y_train']) / (2 * n_neg), 1: len(data['y_train']) / (2 * n_pos)}
    print(f"Class weights: {class_weight}")

    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=0.75),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    tb_run = datetime.now().strftime('%Y%m%d_%H%M%S')
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=str(LOG_DIR / tb_run),
            histogram_freq=1, write_graph=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH), monitor='val_auc', mode='max',
            save_best_only=True, verbose=0
        ),
    ]

    history = model.fit(
        [data['X_ecg_train'], data['X_acc_train'], data['X_bio_train']],
        data['y_train'],
        validation_split=0.15,
        epochs=30,
        batch_size=64,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Comprehensive evaluation
    y_prob = model.predict(
        [data['X_ecg_test'], data['X_acc_test'], data['X_bio_test']], verbose=0
    ).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = data['y_test']

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print("\n" + "="*60)
    print("  OHCA MODEL v2 — FINAL METRICS")
    print("="*60)
    print(f"  Accuracy:  {np.mean(y_pred == y_true):.4f}")
    print(f"  AUC:       {roc_auc_score(y_true, y_prob):.4f}")
    print(f"  Sensitivity (Recall): {sensitivity:.4f}  ({tp}/{tp+fn} pre-arrest caught)")
    print(f"  Specificity:          {specificity:.4f}  ({tn}/{tn+fp} stable correct)")
    print(f"  False Positive Rate:  {fpr:.4f}  ({fp} false alarms)")
    print(f"  Precision:  {precision_score(y_true, y_pred):.4f}")
    print(f"  F1-Score:   {f1_score(y_true, y_pred):.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN={tn:5d}  FP={fp:5d}")
    print(f"    FN={fn:5d}  TP={tp:5d}")

    # Threshold analysis
    print(f"\n  Threshold sweep:")
    for thr in [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]:
        yp = (y_prob >= thr).astype(int)
        _tp = ((yp == 1) & (y_true == 1)).sum()
        _fp = ((yp == 1) & (y_true == 0)).sum()
        _fn = ((yp == 0) & (y_true == 1)).sum()
        _tn = ((yp == 0) & (y_true == 0)).sum()
        _sens = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0
        _fpr_val = _fp / (_fp + _tn) if (_fp + _tn) > 0 else 0
        _acc = (_tp + _tn) / len(y_true)
        print(f"    thr={thr:.2f}: acc={_acc:.4f} sens={_sens:.4f} FPR={_fpr_val:.4f} misses={_fn}")

    if sensitivity >= 0.99 and fpr <= 0.03:
        print("\n  ✓ TARGET MET: >99% sensitivity, <3% FPR — AUTONOMOUS DEPLOYMENT READY")
    elif sensitivity >= 0.97:
        print("\n  ~ APPROACHING TARGET: High sensitivity but may need threshold tuning")
    else:
        print("\n  ✗ BELOW TARGET: Needs further iteration")


if __name__ == '__main__':
    main()
