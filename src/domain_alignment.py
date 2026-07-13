"""Domain alignment – MMD loss and feature-space matching between ICU and wearable."""

from __future__ import annotations

import tensorflow as tf


# ---------------------------------------------------------------------------
# Gaussian kernel for MMD
# ---------------------------------------------------------------------------

def gaussian_kernel(
    x: tf.Tensor,
    y: tf.Tensor,
    sigma: float = 1.0,
) -> tf.Tensor:
    """Compute the Gaussian (RBF) kernel matrix between *x* and *y*.

    Parameters
    ----------
    x : (N, D)
    y : (M, D)
    Returns
    -------
    (N, M) kernel matrix
    """
    x_exp = tf.expand_dims(x, 1)  # (N, 1, D)
    y_exp = tf.expand_dims(y, 0)  # (1, M, D)
    diff = x_exp - y_exp
    dist_sq = tf.reduce_sum(tf.square(diff), axis=-1)
    return tf.exp(-dist_sq / (2.0 * sigma ** 2))


# ---------------------------------------------------------------------------
# Maximum Mean Discrepancy
# ---------------------------------------------------------------------------

def mmd_loss(
    x_icu: tf.Tensor,
    x_wearable: tf.Tensor,
    sigma: float = 1.0,
) -> tf.Tensor:
    """Unbiased MMD^2 estimate using a single Gaussian kernel."""
    k_xx = gaussian_kernel(x_icu, x_icu, sigma)
    k_yy = gaussian_kernel(x_wearable, x_wearable, sigma)
    k_xy = gaussian_kernel(x_icu, x_wearable, sigma)

    n = tf.cast(tf.shape(x_icu)[0], tf.float32)
    m = tf.cast(tf.shape(x_wearable)[0], tf.float32)

    # Unbiased estimators (remove diagonal)
    sum_k_xx = tf.reduce_sum(k_xx) - tf.reduce_sum(tf.linalg.diag_part(k_xx))
    sum_k_yy = tf.reduce_sum(k_yy) - tf.reduce_sum(tf.linalg.diag_part(k_yy))
    sum_k_xy = tf.reduce_sum(k_xy)

    mmd2 = sum_k_xx / (n * (n - 1.0)) + sum_k_yy / (m * (m - 1.0)) - 2.0 * sum_k_xy / (n * m)
    return mmd2


# ---------------------------------------------------------------------------
# Multi-kernel MMD (more robust)
# ---------------------------------------------------------------------------

def multi_kernel_mmd(
    x_icu: tf.Tensor,
    x_wearable: tf.Tensor,
    sigmas: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> tf.Tensor:
    """Average MMD^2 over multiple kernel bandwidths."""
    losses = [mmd_loss(x_icu, x_wearable, sigma=s) for s in sigmas]
    return tf.add_n(losses) / float(len(sigmas))


# ---------------------------------------------------------------------------
# Convenience: MMD alignment loss from domain labels
# ---------------------------------------------------------------------------

def compute_alignment_loss(
    shared_features: tf.Tensor,
    device_domain_labels: tf.Tensor,
    weight: float = 0.1,
    sigmas: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> tf.Tensor:
    """Compute MMD alignment loss between ICU-simulated and real wearable features.

    Parameters
    ----------
    shared_features : (batch_size, feature_dim) tensor from the shared encoder.
    device_domain_labels : (batch_size,) tensor, 0=ICU, 1=wearable.
    weight : scalar weight for the alignment loss.
    sigmas : kernel bandwidths for multi-kernel MMD.

    Returns
    -------
    Scalar alignment loss (weighted by *weight*).
    """
    icu_mask = tf.equal(device_domain_labels, 0)
    wear_mask = tf.equal(device_domain_labels, 1)

    x_icu = tf.boolean_mask(shared_features, icu_mask)
    x_wearable = tf.boolean_mask(shared_features, wear_mask)

    # Need at least 2 samples from each domain for unbiased MMD
    n_icu = tf.shape(x_icu)[0]
    n_wear = tf.shape(x_wearable)[0]

    # If either domain has < 2 samples, return zero loss
    has_both = tf.logical_and(n_icu >= 2, n_wear >= 2)

    mmd = tf.cond(
        has_both,
        lambda: multi_kernel_mmd(x_icu, x_wearable, sigmas=sigmas),
        lambda: tf.constant(0.0),
    )

    return weight * mmd
