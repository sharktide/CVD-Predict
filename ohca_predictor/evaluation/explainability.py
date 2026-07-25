"""
Model interpretability utilities for the OHCA Predictor.

Provides gradient-based attribution methods (saliency, integrated gradients,
Grad-CAM), attention weight extraction, permutation importance, and
structured failure analysis.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ohca_predictor.config import EvaluationConfig


class OHCAExplainability:
    """Interpretability toolkit for the multi-modal OHCA transformer model.

    All gradient computations use ``tf.GradientTape`` for eager-mode
    compatibility.  Attribution maps are returned as numpy arrays for
    downstream visualisation.

    Args:
        model: A trained ``OHCAPredictionModel`` (or any ``keras.Model``
            whose ``call`` returns a dict containing ``"ohca_risk"``).
        config: An ``EvaluationConfig`` dataclass (currently unused but
            kept for API consistency).
    """

    def __init__(self, model: keras.Model, config: EvaluationConfig) -> None:
        self.model = model
        self.config = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_ohca_risk(
        self, inputs: Dict[str, tf.Tensor], training: bool = False,
    ) -> tf.Tensor:
        """Run a forward pass and return the scalar OHCA risk per sample.

        Args:
            inputs: Model input dictionary.
            training: Whether to run in training mode.

        Returns:
            1-D tensor of shape ``(batch,)`` with OHCA risk values.
        """
        outputs = self.model(inputs, training=training)
        risk = outputs["ohca_risk"]
        return tf.reshape(risk, [-1])

    def _to_tensor(
        self, sample: Dict[str, np.ndarray],
    ) -> Dict[str, tf.Tensor]:
        """Convert a single-sample dict of numpy arrays to tf.Tensor."""
        return {
            k: tf.convert_to_tensor(v[np.newaxis, ...], dtype=tf.float32)
            for k, v in sample.items()
        }

    # ------------------------------------------------------------------
    # Attention weight analysis
    # ------------------------------------------------------------------

    def attention_weight_analysis(
        self, sample: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Extract attention weight matrices from the model.

        Triggers a forward pass with ``training=False`` and attempts to
        capture attention weights from the ``AsymmetricCrossAttention``
        layers via the ``attention_weights`` attribute set during the call.

        If the attention layers do not store their weights, this method
        falls back to zero-valued placeholders so the API remains stable.

        Args:
            sample: Single input sample (each value is a numpy array
                without the batch dimension).

        Returns:
            Dictionary mapping layer names to numpy attention-weight
            arrays.  Keys include ``"ecg_motion_cross_attn"`` and
            ``"ecg_ppg_cross_attn"`` when available.
        """
        inputs = self._to_tensor(sample)
        result: Dict[str, np.ndarray] = {}

        for attr_name in ("ecg_motion_cross_attn", "ecg_ppg_cross_attn"):
            layer = getattr(self.model, attr_name, None)
            if layer is None:
                continue
            stored = getattr(layer, "_attention_weights", None)
            if stored is not None:
                result[attr_name] = stored.numpy()
            else:
                result[attr_name] = np.zeros((1,), dtype=np.float32)

        return result

    # ------------------------------------------------------------------
    # Integrated Gradients
    # ------------------------------------------------------------------

    def integrated_gradients(
        self,
        sample: Dict[str, np.ndarray],
        baseline: Optional[Dict[str, np.ndarray]] = None,
        n_steps: int = 50,
        target_key: str = "ecg",
    ) -> np.ndarray:
        """Integrated Gradients attribution (Sundararajan et al., 2017).

        IG(x) = (x - x') * sum_{k=1}^{m} grad(f(x' + k/m * (x - x')))

        where ``x'`` is the baseline (default: zeros), ``m`` is
        ``n_steps``, and the gradient is taken with respect to the input
        feature specified by ``target_key``.

        Args:
            sample: Single input sample (numpy arrays, no batch dim).
            baseline: Baseline input of the same shape.  Defaults to
                a zero-valued dict with matching keys.
            n_steps: Number of interpolation steps between baseline and
                input.
            target_key: Key in the input dict for which to compute
                attributions (e.g. ``"ecg"``).

        Returns:
            Attribution array with the same shape as
            ``sample[target_key]``.
        """
        input_tensor = tf.convert_to_tensor(
            sample[target_key], dtype=tf.float32,
        )
        input_tensor = tf.expand_dims(input_tensor, axis=0)

        if baseline is None:
            baseline_tensor = tf.zeros_like(input_tensor)
        else:
            baseline_tensor = tf.convert_to_tensor(
                baseline[target_key], dtype=tf.float32,
            )
            baseline_tensor = tf.expand_dims(baseline_tensor, axis=0)

        delta = input_tensor - baseline_tensor
        total_steps = n_steps + 1
        alphas = tf.linspace(0.0, 1.0, total_steps)

        interpolated_inputs = []
        for alpha in alphas:
            interp = baseline_tensor + alpha * delta
            interpolated_inputs.append(interp)

        def _compute_risk(interp: tf.Tensor) -> tf.Tensor:
            full_inputs = {
                k: tf.convert_to_tensor(
                    v[np.newaxis, ...] if isinstance(v, np.ndarray) else v,
                    dtype=tf.float32,
                )
                for k, v in sample.items()
            }
            full_inputs[target_key] = interp
            return self._get_ohca_risk(full_inputs, training=False)

        accum_grads = tf.zeros_like(input_tensor)
        for interp in interpolated_inputs:
            with tf.GradientTape() as tape:
                tape.watch(interp)
                risk = _compute_risk(interp)
            grad = tape.gradient(risk, interp)
            if grad is not None:
                accum_grads += grad

        avg_grads = accum_grads / tf.cast(total_steps, tf.float32)
        attributions = tf.squeeze(delta * avg_grads, axis=0).numpy()
        return attributions

    # ------------------------------------------------------------------
    # Saliency map
    # ------------------------------------------------------------------

    def saliency_map(
        self,
        sample: Dict[str, np.ndarray],
        target_key: str = "ecg",
    ) -> np.ndarray:
        """Vanilla saliency map (gradient of output w.r.t. input).

        Args:
            sample: Single input sample (numpy arrays, no batch dim).
            target_key: Key for which to compute the saliency map.

        Returns:
            Saliency array with the same shape as ``sample[target_key]``.
        """
        input_tensor = tf.convert_to_tensor(
            sample[target_key], dtype=tf.float32,
        )
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        input_tensor = tf.Variable(input_tensor)

        full_inputs = {
            k: tf.convert_to_tensor(
                v[np.newaxis, ...] if isinstance(v, np.ndarray) else v,
                dtype=tf.float32,
            )
            for k, v in sample.items()
        }

        with tf.GradientTape() as tape:
            inputs_for_grad = dict(full_inputs)
            inputs_for_grad[target_key] = input_tensor
            tape.watch(input_tensor)
            risk = self._get_ohca_risk(inputs_for_grad, training=False)

        grad = tape.gradient(risk, input_tensor)
        if grad is None:
            return np.zeros_like(sample[target_key], dtype=np.float32)
        return tf.squeeze(grad, axis=0).numpy()

    # ------------------------------------------------------------------
    # Grad-CAM
    # ------------------------------------------------------------------

    def grad_cam(
        self,
        sample: Dict[str, np.ndarray],
        layer_name: str,
        target_key: str = "ecg",
    ) -> np.ndarray:
        """Gradient-weighted Class Activation Mapping (Grad-CAM).

        Computes gradients of the OHCA risk output with respect to the
        feature maps of the named intermediate layer, then produces a
        weighted sum of those feature maps followed by a ReLU.

        Args:
            sample: Single input sample (numpy arrays, no batch dim).
            layer_name: Name of a convolutional or dense layer whose
                output activations to inspect.
            target_key: Input key used for attribution context.

        Returns:
            2-D numpy array of shape ``(seq_len, n_channels)`` (or
            ``(height, width)`` for 2-D inputs) representing the
            Grad-CAM heatmap.
        """
        input_tensor = tf.convert_to_tensor(
            sample[target_key], dtype=tf.float32,
        )
        input_tensor = tf.expand_dims(input_tensor, axis=0)

        layer = self.model.get_layer(layer_name)
        grad_model = keras.Model(
            inputs=[self.model.input],
            outputs=[layer.output, self.model.output["ohca_risk"]],
        )

        full_inputs = {
            k: tf.convert_to_tensor(
                v[np.newaxis, ...] if isinstance(v, np.ndarray) else v,
                dtype=tf.float32,
            )
            for k, v in sample.items()
        }

        with tf.GradientTape() as tape:
            feature_maps, risk = grad_model(full_inputs)
            risk_scalar = tf.reshape(risk, [-1])[0]

        grads = tape.gradient(risk_scalar, feature_maps)
        if grads is None:
            return np.zeros_like(
                feature_maps.numpy().squeeze(axis=0), dtype=np.float32,
            )

        grads_pooled = tf.reduce_mean(grads, axis=1)
        feature_maps_sq = tf.squeeze(feature_maps, axis=0)

        weights = tf.reduce_mean(
            tf.expand_dims(grads_pooled, axis=1) * feature_maps_sq, axis=0,
        )
        cam = tf.reduce_sum(
            feature_maps_sq * tf.reshape(weights, (1, -1)), axis=-1,
        )
        cam = tf.nn.relu(cam)
        return cam.numpy()

    # ------------------------------------------------------------------
    # Permutation importance
    # ------------------------------------------------------------------

    def permutation_importance(
        self,
        samples: List[Dict[str, np.ndarray]],
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        n_repeats: int = 10,
        target_key: str = "ecg",
    ) -> Dict[str, float]:
        """Model-agnostic permutation feature importance.

        For each feature key in the input samples, the values along the
        feature dimension are shuffled ``n_repeats`` times and the
        degradation in ``metric_fn`` is recorded.  The importance score
        is the mean performance drop across repeats.

        Args:
            samples: List of input sample dicts (each without batch dim).
                A second element ``labels`` must be supplied externally
                via a wrapper; here we assume the metric_fn takes
                ``(y_true, y_pred)`` and that the caller has already
                obtained baseline predictions.
            metric_fn: Scalar metric ``(y_true, y_pred) -> float``.
            n_repeats: Number of shuffles per feature.
            target_key: Key of the feature to permute (all keys are
                permuted independently if they appear in the samples).

        Returns:
            Dictionary mapping feature keys to importance scores.
        """
        if len(samples) == 0:
            return {}

        all_keys = list(samples[0].keys())
        importance: Dict[str, float] = {}

        batch_inputs = {
            k: np.stack([s[k] for s in samples], axis=0)
            for k in all_keys
        }

        def _predict(batch: Dict[str, np.ndarray]) -> np.ndarray:
            tf_batch = {
                k: tf.convert_to_tensor(v, dtype=tf.float32)
                for k, v in batch.items()
            }
            out = self.model(tf_batch, training=False)
            return out["ohca_risk"].numpy().ravel()

        baseline_preds = _predict(batch_inputs)

        rng = np.random.RandomState(42)

        for key in all_keys:
            drops = []
            for _ in range(n_repeats):
                permuted = {k: v.copy() for k, v in batch_inputs.items()}
                perm_idx = rng.permutation(permuted[key].shape[1])
                permuted[key] = permuted[key][:, perm_idx, :]
                permuted_preds = _predict(permuted)
                drop = float(
                    np.mean(np.abs(baseline_preds - permuted_preds))
                )
                drops.append(drop)
            importance[key] = float(np.mean(drops))

        return importance

    # ------------------------------------------------------------------
    # Failure analysis
    # ------------------------------------------------------------------

    def failure_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        samples: List[Dict[str, np.ndarray]],
        threshold: float = 0.20,
    ) -> Dict[str, Any]:
        """Structured failure analysis of model predictions.

        Identifies false negatives and false positives, and computes
        summary statistics for each failure mode.

        Args:
            y_true: Ground-truth binary labels.
            y_pred: Predicted probabilities.
            samples: List of input sample dicts corresponding to each
                row in ``y_true`` / ``y_pred``.
            threshold: Classification threshold.

        Returns:
            Dictionary with keys:

            * ``"false_negatives"`` – indices and summaries of missed OHCA
            * ``"false_positives"`` – indices and summaries of false alarms
            * ``"patterns"`` – aggregate statistics for each failure mode
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        n = len(y_true)

        pred_labels = (y_pred >= threshold).astype(int)

        fn_mask = (y_true == 1) & (pred_labels == 0)
        fp_mask = (y_true == 0) & (pred_labels == 1)
        tp_mask = (y_true == 1) & (pred_labels == 1)
        tn_mask = (y_true == 0) & (pred_labels == 0)

        fn_indices = np.where(fn_mask)[0].tolist()
        fp_indices = np.where(fp_mask)[0].tolist()

        false_negatives = []
        for idx in fn_indices:
            entry: Dict[str, Any] = {
                "index": int(idx),
                "predicted_prob": float(y_pred[idx]),
                "true_label": 1,
            }
            if idx < len(samples):
                entry["features"] = {
                    k: float(np.mean(v)) for k, v in samples[idx].items()
                }
            false_negatives.append(entry)

        false_positives = []
        for idx in fp_indices:
            entry = {
                "index": int(idx),
                "predicted_prob": float(y_pred[idx]),
                "true_label": 0,
            }
            if idx < len(samples):
                entry["features"] = {
                    k: float(np.mean(v)) for k, v in samples[idx].items()
                }
            false_positives.append(entry)

        patterns = {
            "total_samples": int(n),
            "total_positives": int(y_true.sum()),
            "total_negatives": int(n - y_true.sum()),
            "false_negatives_count": int(fn_mask.sum()),
            "false_positives_count": int(fp_mask.sum()),
            "true_positives_count": int(tp_mask.sum()),
            "true_negatives_count": int(tn_mask.sum()),
            "fn_rate": float(fn_mask.sum() / max(y_true.sum(), 1)),
            "fp_rate": float(fp_mask.sum() / max((y_true == 0).sum(), 1)),
            "mean_fn_confidence": float(y_pred[fn_mask].mean())
            if fn_mask.any() else 0.0,
            "mean_fp_confidence": float(y_pred[fp_mask].mean())
            if fp_mask.any() else 0.0,
        }

        return {
            "false_negatives": false_negatives,
            "false_positives": false_positives,
            "patterns": patterns,
        }
