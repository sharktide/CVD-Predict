"""Custom loss functions for the OHCA prediction model.

Implements clinically-motivated losses that address:
    - Extreme class imbalance (focal loss)
    - Asymmetric misclassification costs (weighted BCE)
    - Right-censored survival data (discrete-time NLL)
    - Self-supervised pretraining objectives (InfoNCE, MSE)
    - Combined multi-task training (weighted sum)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import tensorflow as tf
from tensorflow import keras


# ---------------------------------------------------------------------------
# A. Clinically Weighted Binary Cross-Entropy
# ---------------------------------------------------------------------------

class ClinicallyWeightedBCE(keras.losses.Loss):
    """Binary cross-entropy with asymmetric costs for false negatives vs. false positives.

    In the OHCA prediction context:

        - **False Negative** (missed OHCA): the patient may die.  This is
          catastrophically expensive.
        - **False Positive** (false alarm): triggers a clinical review.
          Annoying but not fatal.

    The loss is:

        L = -[ w_1 * y * log(p) + w_0 * (1 - y) * log(1 - p) ]

    where ``w_1`` penalises missed positives (FN weight) and ``w_0``
    penalises false alarms (FP weight).

    Default: ``w_fn = 10.0``, ``w_fp = 1.0``.

    Args:
        fn_weight: Weight applied to false negatives (missed OHCA).
        fp_weight: Weight applied to false positives (false alarms).
        epsilon: Small constant for numerical stability in log.
    """

    def __init__(
        self,
        fn_weight: float = 10.0,
        fp_weight: float = 1.0,
        epsilon: float = 1e-7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight
        self.epsilon = epsilon

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """Compute the clinically weighted BCE loss.

        Args:
            y_true: Ground-truth binary labels ``(batch, 1)`` or ``(batch,)``.
            y_pred: Predicted probabilities in ``(0, 1)``, same shape.

        Returns:
            Scalar mean loss.
        """
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)

        # Flatten to ensure broadcasting works cleanly
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        pos_loss = -self.fn_weight * y_true * tf.math.log(y_pred)
        neg_loss = -self.fp_weight * (1.0 - y_true) * tf.math.log(1.0 - y_pred)

        loss = pos_loss + neg_loss
        return tf.reduce_mean(loss)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "fn_weight": self.fn_weight,
                "fp_weight": self.fp_weight,
                "epsilon": self.epsilon,
            }
        )
        return config


# ---------------------------------------------------------------------------
# B. Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(keras.losses.Loss):
    """Focal loss for extreme class imbalance.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Standard cross-entropy is dominated by the large number of easy
    (well-classified) negatives.  Focal loss down-weights these easy examples
    so the model focuses on hard, misclassified instances:

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:

        p_t = { p       if y = 1
              { 1 - p   if y = 0

        alpha_t = { alpha_pos   if y = 1
                  { alpha_neg   if y = 0

    With ``gamma = 2.0`` the loss is reduced by ~(1-p_t)^2 for well-classified
    examples, making the model concentrate on hard cases.

    Args:
        gamma: Focusing parameter.  ``gamma = 0`` recovers standard
            weighted BCE.  Higher values increase the focusing effect
            (default 2.0).
        alpha_pos: Balancing weight for the positive class (default 0.75).
        alpha_neg: Balancing weight for the negative class (default 0.25).
        epsilon: Numerical stability constant.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha_pos: float = 0.75,
        alpha_neg: float = 0.25,
        epsilon: float = 1e-7,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.epsilon = epsilon

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """Compute the focal loss.

        Args:
            y_true: Ground-truth binary labels ``(batch, …)``.
            y_pred: Predicted probabilities in ``(0, 1)``.

        Returns:
            Scalar mean loss.
        """
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)

        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])

        # p_t: probability assigned to the true class
        p_t = y_true_f * y_pred_f + (1.0 - y_true_f) * (1.0 - y_pred_f)

        # alpha_t: class-dependent weighting
        alpha_t = y_true_f * self.alpha_pos + (1.0 - y_true_f) * self.alpha_neg

        # Focal modulating factor
        focal_weight = tf.pow(1.0 - p_t, self.gamma)

        # Standard cross-entropy (binary)
        bce = -(y_true_f * tf.math.log(y_pred_f)
                + (1.0 - y_true_f) * tf.math.log(1.0 - y_pred_f))

        loss = alpha_t * focal_weight * bce
        return tf.reduce_mean(loss)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "gamma": self.gamma,
                "alpha_pos": self.alpha_pos,
                "alpha_neg": self.alpha_neg,
                "epsilon": self.epsilon,
            }
        )
        return config


# ---------------------------------------------------------------------------
# C. Discrete-Time Survival Loss
# ---------------------------------------------------------------------------

class DiscreteTimeSurvivalLoss(keras.losses.Loss):
    """Negative log-likelihood for a discrete-time survival model.

    For each sample *i* observed over *K* time bins, the loss is:

        L_i = sum_{k=1}^{K} [
            delta_i(k) * log(S_i(k))  +  (1 - delta_i(k)) * log(1 - S_i(k))
        ]

    where:

        - S_i(k) is the model's predicted survival probability at bin *k*.
        - delta_i(k) = 1 if the event (OHCA) occurred in bin *k*, and 0 otherwise.

    Right-censored samples contribute only through the ``1 - delta`` term up to
    their last observed bin, which is the standard approach for handling
    censoring in discrete-time models.

    The model outputs *K* per-bin conditional hazard probabilities
    ``h_i(k) = P(event in bin k | survived to bin k)``.  We convert these to
    cumulative survival ``S_i(k) = prod_{j=1}^{k} (1 - h_i(j))`` before
    computing the loss.

    Args:
        epsilon: Numerical stability constant for log.
    """

    def __init__(self, epsilon: float = 1e-7, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """Compute the discrete-time survival NLL.

        Args:
            y_true: Tensor of shape ``(batch, K + 1)``.

                Convention:

                    - ``y_true[i, k] = 1`` if the event (OHCA) occurred in
                      bin *k* for sample *i*.  Exactly one entry is 1 for
                      non-censored samples.
                    - ``y_true[i, K] = 1`` if the sample is right-censored
                      (event was not observed within the observation window).
                      For censored samples all of ``y_true[i, 0:K]`` are 0.
                    - All other entries are 0.

            y_pred: Per-bin conditional hazard probabilities ``(batch, K)``
                from the ``SurvivalAnalysisHead``.  Each entry
                ``y_pred[i, k]`` = P(event in bin k | survived to bin k).

        Returns:
            Scalar mean negative log-likelihood (always non-negative).
        """
        y_pred_f = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)

        # Split event indicators and censoring flag
        event_indicators = y_true[:, :-1]  # (batch, K) – one-hot per bin
        censoring_flag = y_true[:, -1]  # (batch,) – 1 if censored

        # Per-bin log-hazard and log-survival contributions
        log_hazard = tf.math.log(y_pred_f)  # (batch, K)
        log_surv = tf.math.log(1.0 - y_pred_f)  # (batch, K)

        # Negative log-likelihood per sample:
        #   For an observed event at bin k:
        #     L = -[ sum_{j<k} log(1-h(j)) + log(h(k)) ]
        #   For a censored sample observed through bin K:
        #     L = -[ sum_{j=1}^{K} log(1-h(j)) ]
        #
        # Unified formula using event_indicators:
        #   L = -sum_k [ delta(k)*log(h(k)) + (1-delta(k))*log(1-h(k)) ]
        # This correctly handles both cases because for censored samples
        # all delta(k) = 0, yielding the censoring likelihood.

        per_bin_nll = (
            event_indicators * log_hazard
            + (1.0 - event_indicators) * log_surv
        )  # (batch, K)

        # Sum over time bins for each sample → log-likelihood (negative value)
        log_likelihood = tf.reduce_sum(per_bin_nll, axis=1)  # (batch,)

        # NLL = -log(likelihood), so negate to get a positive loss
        return tf.reduce_mean(-log_likelihood)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


# ---------------------------------------------------------------------------
# D. Contrastive Loss (InfoNCE)
# ---------------------------------------------------------------------------

class ContrastiveLoss(keras.losses.Loss):
    """InfoNCE contrastive loss for self-supervised pretraining.

    Encourages representations of augmented views of the same sample to be
    similar, while pushing apart representations of different samples.

    For a batch of *N* samples with positive pairs ``(i, i')``:

        L = -log [ exp(z_i · z_i' / tau) /
                   sum_{j=1}^{N} exp(z_i · z_j / tau) ]

    The loss is computed symmetrically over both directions (view 1 → view 2
    and view 2 → view 1).

    Args:
        temperature: Temperature parameter controlling the sharpness of the
            similarity distribution (default 0.07).  Lower values produce
            sharper distributions and harder negatives.
    """

    def __init__(self, temperature: float = 0.07, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(
        self,
        z_a: tf.Tensor,
        z_b: tf.Tensor,
    ) -> tf.Tensor:
        """Compute the symmetric InfoNCE loss.

        Args:
            z_a: L2-normalized embeddings from view A ``(batch, dim)``.
            z_b: L2-normalized embeddings from view B ``(batch, dim)``.

        Returns:
            Scalar mean InfoNCE loss.
        """
        batch_size = tf.shape(z_a)[0]

        # Similarity matrix: (batch, batch)
        similarity = tf.matmul(z_a, z_b, transpose_b=True)
        similarity = similarity / self.temperature

        # Labels: the diagonal elements are the positives
        labels = tf.range(batch_size)

        # Cross-entropy in both directions
        loss_a2b = tf.keras.losses.sparse_categorical_crossentropy(
            labels, similarity, from_logits=True,
        )
        loss_b2a = tf.keras.losses.sparse_categorical_crossentropy(
            labels, tf.transpose(similarity), from_logits=True,
        )

        return tf.reduce_mean((loss_a2b + loss_b2a) / 2.0)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config


# ---------------------------------------------------------------------------
# E. Reconstruction Loss
# ---------------------------------------------------------------------------

class ReconstructionLoss(keras.losses.Loss):
    """Mean squared error loss for the ECG reconstruction head.

    Used during self-supervised pretraining to encourage the model to
    preserve waveform morphology through the tokenization–encoding pipeline.

    The loss is computed element-wise between the original raw ECG signal and
    the reconstructed signal, then averaged:

        L = (1 / N) * sum_i (x_i - x_hat_i)^2

    Args:
        reduction: Keras reduction type (default ``"sum_over_batch_size"``).
    """

    def __init__(self, reduction: str = "sum_over_batch_size", **kwargs):
        super().__init__(reduction=reduction, **kwargs)

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """Compute the MSE reconstruction loss.

        Args:
            y_true: Original ECG signal ``(batch, samples, 1)``.
            y_pred: Reconstructed ECG signal ``(batch, samples, 1)``.

        Returns:
            Scalar MSE loss.
        """
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        return config


# ---------------------------------------------------------------------------
# F. Combined OHCA Loss
# ---------------------------------------------------------------------------

class CombinedOHCALoss(keras.losses.Loss):
    """Weighted combination of all loss components for multi-task training.

    The total loss is:

        L_total = w_cls  * L_cls
                + w_surv * L_surv
                + w_aux  * L_aux
                + w_contrast * L_contrast
                + w_recon * L_recon

    where:

        - **Classification loss** ``L_cls``:
            0.5 * FocalLoss + 0.5 * ClinicallyWeightedBCE

        - **Survival loss** ``L_surv``:
            DiscreteTimeSurvivalLoss

        - **Auxiliary loss** ``L_aux``:
            Weighted sum of auxiliary head losses (heart rate MSE,
            rhythm cross-entropy, activity cross-entropy, SpO2 MSE,
            blood pressure MSE).  Each auxiliary loss is computed
            outside this class and passed in as ``aux_loss``.

        - **Contrastive loss** ``L_contrast``:
            InfoNCE loss (from pretraining or regularisation).

        - **Reconstruction loss** ``L_recon``:
            MSE between original and reconstructed ECG.

    Default weights from the training config:
        w_cls = 1.0, w_surv = 0.3, w_aux = 0.1,
        w_contrast = 0.2, w_recon = 0.15

    Args:
        cls_weight: Weight for the combined classification loss.
        surv_weight: Weight for the survival analysis loss.
        aux_weight: Weight for auxiliary task losses.
        contrast_weight: Weight for contrastive loss.
        recon_weight: Weight for reconstruction loss.
        focal_gamma: Gamma parameter for FocalLoss.
        focal_alpha_pos: Positive class alpha for FocalLoss.
        focal_alpha_neg: Negative class alpha for FocalLoss.
        fn_weight: FN weight for ClinicallyWeightedBCE.
        fp_weight: FP weight for ClinicallyWeightedBCE.
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        surv_weight: float = 0.3,
        aux_weight: float = 0.1,
        contrast_weight: float = 0.2,
        recon_weight: float = 0.15,
        focal_gamma: float = 2.0,
        focal_alpha_pos: float = 0.75,
        focal_alpha_neg: float = 0.25,
        fn_weight: float = 10.0,
        fp_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cls_weight = cls_weight
        self.surv_weight = surv_weight
        self.aux_weight = aux_weight
        self.contrast_weight = contrast_weight
        self.recon_weight = recon_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha_pos = focal_alpha_pos
        self.focal_alpha_neg = focal_alpha_neg
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight

        # Instantiate component losses
        self.focal_loss = FocalLoss(
            gamma=focal_gamma,
            alpha_pos=focal_alpha_pos,
            alpha_neg=focal_alpha_neg,
        )
        self.weighted_bce = ClinicallyWeightedBCE(
            fn_weight=fn_weight,
            fp_weight=fp_weight,
        )
        self.survival_loss = DiscreteTimeSurvivalLoss()
        self.contrastive_loss_fn = ContrastiveLoss()
        self.reconstruction_loss_fn = ReconstructionLoss()

    def _compute_classification_loss(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """Compute the combined classification loss.

        L_cls = 0.5 * focal + 0.5 * weighted_bce

        Args:
            y_true: Binary labels ``(batch, 1)``.
            y_pred: Predicted probabilities ``(batch, 1)``.

        Returns:
            Scalar classification loss.
        """
        focal = self.focal_loss(y_true, y_pred)
        wbce = self.weighted_bce(y_true, y_pred)
        return 0.5 * focal + 0.5 * wbce

    def _compute_auxiliary_loss(
        self,
        auxiliary_predictions: Dict[str, tf.Tensor],
        auxiliary_targets: Dict[str, tf.Tensor],
    ) -> tf.Tensor:
        """Compute the combined auxiliary loss across all auxiliary heads.

        Heads and their losses:
            - heart_rate: MSE
            - rhythm: categorical cross-entropy
            - activity: categorical cross-entropy
            - spo2: MSE
            - blood_pressure: MSE (SBP + DBP)

        Each sub-loss is normalised to roughly the same scale before
        averaging.

        Args:
            auxiliary_predictions: Dict of predicted tensors from
                ``AuxiliaryHeads``.
            auxiliary_targets: Dict of ground-truth tensors with the
                same keys.

        Returns:
            Scalar auxiliary loss.
        """
        total_aux = tf.constant(0.0, dtype=tf.float32)
        num_heads = 0

        # Heart rate regression (MSE)
        if "heart_rate" in auxiliary_predictions and "heart_rate" in auxiliary_targets:
            hr_pred = auxiliary_predictions["heart_rate"]
            hr_true = auxiliary_targets["heart_rate"]
            total_aux = total_aux + tf.reduce_mean(tf.square(hr_true - hr_pred))
            num_heads += 1

        # Rhythm classification (categorical CE)
        if "rhythm" in auxiliary_predictions and "rhythm" in auxiliary_targets:
            rhythm_pred = auxiliary_predictions["rhythm"]
            rhythm_true = auxiliary_targets["rhythm"]
            total_aux = total_aux + tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    rhythm_true, rhythm_pred,
                ),
            )
            num_heads += 1

        # Activity classification (categorical CE)
        if "activity" in auxiliary_predictions and "activity" in auxiliary_targets:
            act_pred = auxiliary_predictions["activity"]
            act_true = auxiliary_targets["activity"]
            total_aux = total_aux + tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    act_true, act_pred,
                ),
            )
            num_heads += 1

        # SpO2 regression (MSE)
        if "spo2" in auxiliary_predictions and "spo2" in auxiliary_targets:
            spo2_pred = auxiliary_predictions["spo2"]
            spo2_true = auxiliary_targets["spo2"]
            total_aux = total_aux + tf.reduce_mean(tf.square(spo2_true - spo2_pred))
            num_heads += 1

        # Blood pressure regression (MSE for SBP + DBP)
        if "blood_pressure" in auxiliary_predictions and "blood_pressure" in auxiliary_targets:
            bp_pred = auxiliary_predictions["blood_pressure"]
            bp_true = auxiliary_targets["blood_pressure"]
            total_aux = total_aux + tf.reduce_mean(tf.square(bp_true - bp_pred))
            num_heads += 1

        # Average over active heads
        if num_heads > 0:
            total_aux = total_aux / tf.cast(num_heads, dtype=tf.float32)

        return total_aux

    def call(
        self,
        y_true: Dict[str, tf.Tensor],
        y_pred: Dict[str, tf.Tensor],
    ) -> tf.Tensor:
        """Compute the combined multi-task loss.

        Args:
            y_true: Dictionary with ground-truth tensors:
                - ``"ohca_label"``: ``(batch, 1)`` binary label
                - ``"survival_events"``: ``(batch, K+1)`` survival targets
                - ``"auxiliary"``: dict of auxiliary ground-truth tensors
                - ``"contrastive_z_a"``: ``(batch, dim)`` embeddings (optional)
                - ``"contrastive_z_b"``: ``(batch, dim)`` embeddings (optional)
                - ``"reconstruction_target"``: ``(batch, samples, 1)`` (optional)

            y_pred: Dictionary with model outputs (from ``OHCAPredictionModel``):
                - ``"ohca_risk"``: ``(batch, 1)``
                - ``"raw_risk"``: ``(batch, 1)`` (uncalibrated – used for loss)
                - ``"survival_curve"``: ``(batch, K+1)``
                - ``"uncertainty"``: ``(batch, 2)``
                - ``"auxiliary"``: dict of auxiliary predictions
                - ``"contrastive_z_a"``: ``(batch, dim)`` (optional)
                - ``"contrastive_z_b"``: ``(batch, dim)`` (optional)
                - ``"reconstructed_ecg"``: ``(batch, samples, 1)`` (optional)

        Returns:
            Scalar combined loss.
        """
        losses: Dict[str, tf.Tensor] = {}

        # ---- Classification loss ----
        if "ohca_label" in y_true and "raw_risk" in y_pred:
            losses["cls"] = self._compute_classification_loss(
                y_true["ohca_label"],
                y_pred["raw_risk"],
            )

        # ---- Survival loss ----
        if "survival_events" in y_true and "survival_curve" in y_pred:
            # survival_curve from model is cumulative S(t); we need per-bin
            # hazard h(k) = 1 - S(k) / S(k-1) for the loss.
            surv_curve = y_pred["survival_curve"]  # (batch, K+1)
            # Recover per-bin hazard: h(k) = 1 - S(k)/S(k-1)
            s_prev = surv_curve[:, :-1]  # (batch, K)  – S(k-1)
            s_curr = surv_curve[:, 1:]  # (batch, K)  – S(k)
            hazards = 1.0 - s_curr / tf.clip_by_value(
                s_prev, 1e-7, 1.0,
            )  # (batch, K)
            losses["surv"] = self.survival_loss(
                y_true["survival_events"], hazards,
            )

        # ---- Auxiliary loss ----
        if "auxiliary" in y_true and "auxiliary" in y_pred:
            losses["aux"] = self._compute_auxiliary_loss(
                y_pred["auxiliary"],
                y_true["auxiliary"],
            )

        # ---- Contrastive loss ----
        if (
            "contrastive_z_a" in y_true
            and "contrastive_z_b" in y_true
            and "contrastive_z_a" in y_pred
            and "contrastive_z_b" in y_pred
        ):
            losses["contrast"] = self.contrastive_loss_fn(
                y_pred["contrastive_z_a"],
                y_pred["contrastive_z_b"],
            )

        # ---- Reconstruction loss ----
        if (
            "reconstruction_target" in y_true
            and "reconstructed_ecg" in y_pred
        ):
            losses["recon"] = self.reconstruction_loss_fn(
                y_true["reconstruction_target"],
                y_pred["reconstructed_ecg"],
            )

        # ---- Weighted combination ----
        total = tf.constant(0.0, dtype=tf.float32)

        if "cls" in losses:
            total = total + self.cls_weight * losses["cls"]
        if "surv" in losses:
            total = total + self.surv_weight * losses["surv"]
        if "aux" in losses:
            total = total + self.aux_weight * losses["aux"]
        if "contrast" in losses:
            total = total + self.contrast_weight * losses["contrast"]
        if "recon" in losses:
            total = total + self.recon_weight * losses["recon"]

        return total

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "cls_weight": self.cls_weight,
                "surv_weight": self.surv_weight,
                "aux_weight": self.aux_weight,
                "contrast_weight": self.contrast_weight,
                "recon_weight": self.recon_weight,
                "focal_gamma": self.focal_gamma,
                "focal_alpha_pos": self.focal_alpha_pos,
                "focal_alpha_neg": self.focal_alpha_neg,
                "fn_weight": self.fn_weight,
                "fp_weight": self.fp_weight,
            }
        )
        return config
