"""
Robustness testing utilities for the OHCA Predictor.

Provides out-of-distribution detection, adversarial robustness evaluation,
missing-modality stress tests, signal-quality degradation analysis,
temporal perturbation studies, and population-shift assessment.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ohca_predictor.config import EvaluationConfig


class RobustnessEvaluator:
    """Comprehensive robustness testing suite for the OHCA model.

    Args:
        model: A trained ``OHCAPredictionModel`` (or any ``keras.Model``
            whose ``call`` returns a dict containing ``"ohca_risk"``).
        config: An ``EvaluationConfig`` dataclass.
    """

    def __init__(self, model: keras.Model, config: EvaluationConfig) -> None:
        self.model = model
        self.config = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_risk(
        self, inputs: Dict[str, tf.Tensor], training: bool = False,
    ) -> np.ndarray:
        """Forward pass returning OHCA risk probabilities as a numpy array."""
        outputs = self.model(inputs, training=training)
        return outputs["ohca_risk"].numpy().ravel()

    def _batch_predict(
        self, batch: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Predict on a batch of numpy arrays."""
        tf_batch = {
            k: tf.convert_to_tensor(v, dtype=tf.float32)
            for k, v in batch.items()
        }
        return self._predict_risk(tf_batch, training=False)

    def _stack_samples(
        self, samples: List[Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """Stack a list of sample dicts into a single batch dict."""
        keys = list(samples[0].keys())
        return {k: np.stack([s[k] for s in samples], axis=0) for k in keys}

    # ------------------------------------------------------------------
    # OOD detection – Mahalanobis distance
    # ------------------------------------------------------------------

    def ood_detection_mahalanobis(
        self,
        train_features: np.ndarray,
        test_features: np.ndarray,
    ) -> np.ndarray:
        """Out-of-distribution detection via Mahalanobis distance.

        Computes the Mahalanobis distance of each test sample from the
        training distribution in the feature (latent) space, then
        converts to an energy-based OOD score:

            E(x) = -log(sum(exp(f_i(x))))

        where ``f_i(x)`` are the unnormalised energies derived from the
        Mahalanobis distances.  Lower energy indicates higher
        in-distribution confidence.

        Args:
            train_features: Training-set feature matrix ``(N_train, D)``.
            test_features: Test-set feature matrix ``(N_test, D)``.

        Returns:
            1-D numpy array of energy-based OOD scores for each test
            sample.
        """
        train_features = np.asarray(train_features, dtype=np.float64)
        test_features = np.asarray(test_features, dtype=np.float64)

        mean = train_features.mean(axis=0)
        centered = train_features - mean
        cov = (centered.T @ centered) / max(len(train_features) - 1, 1)

        eps = 1e-8
        cov_reg = cov + eps * np.eye(cov.shape[0])
        cov_inv = np.linalg.inv(cov_reg)

        diff = test_features - mean
        mahal_sq = np.sum(diff @ cov_inv * diff, axis=1)

        # Energy-based score: E(x) = -log(sum(exp(-0.5 * d^2)))
        energies = -0.5 * mahal_sq
        max_e = energies.max()
        log_sum_exp = max_e + np.log(np.sum(np.exp(energies - max_e)))
        ood_scores = -log_sum_exp

        return ood_scores

    # ------------------------------------------------------------------
    # Adversarial robustness
    # ------------------------------------------------------------------

    def adversarial_robustness(
        self,
        samples: List[Dict[str, np.ndarray]],
        labels: np.ndarray,
        epsilons: Optional[List[float]] = None,
        target_key: str = "ecg",
    ) -> Dict[str, Any]:
        """Evaluate adversarial robustness via FGSM and PGD.

        Args:
            samples: List of input sample dicts.
            labels: Ground-truth binary labels aligned with ``samples``.
            epsilons: Perturbation magnitudes to test.  Defaults to
                ``[0.01, 0.05, 0.1]``.
            target_key: Input feature to perturb.

        Returns:
            Dictionary with keys ``"fgsm"`` and ``"pgd"``, each mapping
            epsilon values to dicts of ``"accuracy"``,
            ``"mean_prediction"``, and ``"max_perturbation"``.
        """
        if epsilons is None:
            epsilons = [0.01, 0.05, 0.1]

        labels = np.asarray(labels, dtype=np.float64).ravel()
        batch = self._stack_samples(samples)

        baseline_preds = self._batch_predict(batch)
        baseline_acc = float(np.mean((baseline_preds >= 0.20).astype(int) == labels))

        results: Dict[str, Dict[float, Dict[str, float]]] = {
            "fgsm": {},
            "pgd": {},
        }

        tf_inputs = {
            k: tf.convert_to_tensor(v, dtype=tf.float32)
            for k, v in batch.items()
        }

        for eps in epsilons:
            # --- FGSM ---
            input_var = tf.Variable(tf_inputs[target_key])

            with tf.GradientTape() as tape:
                current_inputs = dict(tf_inputs)
                current_inputs[target_key] = input_var
                risk = self._get_risk_from_tf(current_inputs)
                loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(labels, risk)
                )

            grad = tape.gradient(loss, input_var)
            if grad is not None:
                perturbation = eps * tf.sign(grad)
                adv_input = tf_inputs[target_key] + perturbation
            else:
                adv_input = tf_inputs[target_key]

            fgsm_inputs = dict(tf_inputs)
            fgsm_inputs[target_key] = adv_input
            fgsm_preds = self._predict_risk(fgsm_inputs, training=False)
            fgsm_acc = float(np.mean((fgsm_preds >= 0.20).astype(int) == labels))

            results["fgsm"][eps] = {
                "accuracy": fgsm_acc,
                "accuracy_drop": baseline_acc - fgsm_acc,
                "mean_prediction": float(fgsm_preds.mean()),
                "max_perturbation": float(eps),
            }

            # --- PGD (iterative FGSM) ---
            pgd_steps = 10
            step_size = eps / pgd_steps
            adv_current = tf.identity(tf_inputs[target_key])

            for _pgd_step in range(pgd_steps):
                input_var_pgd = tf.Variable(adv_current)
                with tf.GradientTape() as tape:
                    current_inputs_pgd = dict(tf_inputs)
                    current_inputs_pgd[target_key] = input_var_pgd
                    risk_pgd = self._get_risk_from_tf(current_inputs_pgd)
                    loss_pgd = tf.reduce_mean(
                        tf.keras.losses.binary_crossentropy(labels, risk_pgd)
                    )
                grad_pgd = tape.gradient(loss_pgd, input_var_pgd)
                if grad_pgd is not None:
                    adv_current = adv_current + step_size * tf.sign(grad_pgd)
                adv_current = tf.clip_by_value(
                    adv_current,
                    tf_inputs[target_key] - eps,
                    tf_inputs[target_key] + eps,
                )

            pgd_inputs = dict(tf_inputs)
            pgd_inputs[target_key] = adv_current
            pgd_preds = self._predict_risk(pgd_inputs, training=False)
            pgd_acc = float(np.mean((pgd_preds >= 0.20).astype(int) == labels))

            results["pgd"][eps] = {
                "accuracy": pgd_acc,
                "accuracy_drop": baseline_acc - pgd_acc,
                "mean_prediction": float(pgd_preds.mean()),
                "max_perturbation": float(eps),
            }

        results["baseline_accuracy"] = baseline_acc
        return results

    def _get_risk_from_tf(
        self, inputs: Dict[str, tf.Tensor],
    ) -> tf.Tensor:
        """Get OHCA risk as a 1-D tensor from tf inputs."""
        out = self.model(inputs, training=False)
        return tf.reshape(out["ohca_risk"], [-1])

    # ------------------------------------------------------------------
    # Missing modality test
    # ------------------------------------------------------------------

    def missing_modality_test(
        self,
        samples: List[Dict[str, np.ndarray]],
        labels: np.ndarray,
        modalities: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate performance when individual modalities are zeroed out.

        For each modality in ``modalities``, the corresponding input
        tensor is set to zeros and the model's accuracy and AUROC are
        measured.

        Args:
            samples: List of input sample dicts.
            labels: Ground-truth binary labels.
            modalities: List of modality keys to ablate.  Defaults to
                ``["ecg", "accelerometer", "gyroscope", "ppg",
                "spo2", "temperature", "respiration"]``.

        Returns:
            Dictionary mapping modality names to dicts of ``"accuracy"``,
            ``"mean_prediction"``, and ``"performance_drop"``.
        """
        if modalities is None:
            modalities = [
                "ecg", "accelerometer", "gyroscope", "ppg",
                "spo2", "temperature", "respiration",
            ]

        labels = np.asarray(labels, dtype=np.float64).ravel()
        batch = self._stack_samples(samples)

        baseline_preds = self._batch_predict(batch)
        baseline_acc = float(np.mean(
            (baseline_preds >= 0.20).astype(int) == labels
        ))

        results: Dict[str, Dict[str, float]] = {}

        for mod in modalities:
            if mod not in batch:
                results[mod] = {
                    "accuracy": baseline_acc,
                    "mean_prediction": float(baseline_preds.mean()),
                    "performance_drop": 0.0,
                }
                continue

            ablated = {k: v.copy() for k, v in batch.items()}
            ablated[mod] = np.zeros_like(ablated[mod])
            ablated_preds = self._batch_predict(ablated)
            ablated_acc = float(np.mean(
                (ablated_preds >= 0.20).astype(int) == labels
            ))

            results[mod] = {
                "accuracy": ablated_acc,
                "mean_prediction": float(ablated_preds.mean()),
                "performance_drop": baseline_acc - ablated_acc,
            }

        results["baseline_accuracy"] = baseline_acc  # type: ignore[assignment]
        return results

    # ------------------------------------------------------------------
    # Signal quality degradation
    # ------------------------------------------------------------------

    def signal_quality_degradation(
        self,
        samples: List[Dict[str, np.ndarray]],
        labels: np.ndarray,
        snr_levels: Optional[List[float]] = None,
        target_key: str = "ecg",
    ) -> Dict[float, Dict[str, float]]:
        """Measure model performance under varying signal-to-noise ratios.

        Adds Gaussian noise to the target modality at each SNR level.

        SNR is defined as ``10 * log10(signal_power / noise_power)``.
        Higher SNR means cleaner signal.

        Args:
            samples: List of input sample dicts.
            labels: Ground-truth binary labels.
            snr_levels: List of SNR values in dB.  Defaults to
                ``[30, 20, 10, 5, 0]``.
            target_key: Modality to corrupt.

        Returns:
            Dictionary mapping SNR levels to dicts of ``"accuracy"``,
            ``"mean_prediction"``, and ``"accuracy_drop"``.
        """
        if snr_levels is None:
            snr_levels = [30.0, 20.0, 10.0, 5.0, 0.0]

        labels = np.asarray(labels, dtype=np.float64).ravel()
        batch = self._stack_samples(samples)

        baseline_preds = self._batch_predict(batch)
        baseline_acc = float(np.mean(
            (baseline_preds >= 0.20).astype(int) == labels
        ))

        rng = np.random.RandomState(42)
        results: Dict[float, Dict[str, float]] = {}

        for snr_db in snr_levels:
            corrupted = {k: v.copy() for k, v in batch.items()}
            signal = corrupted[target_key]
            signal_power = np.mean(signal ** 2)
            if signal_power < 1e-12:
                noise_power = 1e-12
            else:
                noise_power = signal_power / (10.0 ** (snr_db / 10.0))

            noise = rng.normal(
                0.0, np.sqrt(max(noise_power, 1e-12)), size=signal.shape,
            ).astype(np.float32)
            corrupted[target_key] = signal + noise

            corrupted_preds = self._batch_predict(corrupted)
            corrupted_acc = float(np.mean(
                (corrupted_preds >= 0.20).astype(int) == labels
            ))

            results[float(snr_db)] = {
                "accuracy": corrupted_acc,
                "mean_prediction": float(corrupted_preds.mean()),
                "accuracy_drop": baseline_acc - corrupted_acc,
            }

        results[-1.0] = {  # type: ignore[assignment]
            "accuracy": baseline_acc,
            "mean_prediction": float(baseline_preds.mean()),
            "accuracy_drop": 0.0,
        }
        return results

    # ------------------------------------------------------------------
    # Temporal perturbation
    # ------------------------------------------------------------------

    def temporal_perturbation(
        self,
        samples: List[Dict[str, np.ndarray]],
        labels: np.ndarray,
        shifts: Optional[List[int]] = None,
        target_key: str = "ecg",
    ) -> Dict[int, Dict[str, float]]:
        """Evaluate performance under temporal shifts of input signals.

        Circularly shifts the target modality by the specified number of
        samples and measures the impact on accuracy.

        Args:
            samples: List of input sample dicts.
            labels: Ground-truth binary labels.
            shifts: List of integer sample shifts.  Defaults to
                ``[-30, -15, 0, 15, 30]``.
            target_key: Modality to shift.

        Returns:
            Dictionary mapping shift values to dicts of ``"accuracy"``,
            ``"mean_prediction"``, and ``"accuracy_drop"``.
        """
        if shifts is None:
            shifts = [-30, -15, 0, 15, 30]

        labels = np.asarray(labels, dtype=np.float64).ravel()
        batch = self._stack_samples(samples)

        baseline_preds = self._batch_predict(batch)
        baseline_acc = float(np.mean(
            (baseline_preds >= 0.20).astype(int) == labels
        ))

        results: Dict[int, Dict[str, float]] = {}

        for shift_val in shifts:
            shifted = {k: v.copy() for k, v in batch.items()}
            arr = shifted[target_key]
            shift_val_int = int(shift_val)
            if shift_val_int != 0:
                shifted[target_key] = np.roll(arr, shift_val_int, axis=1)

            shifted_preds = self._batch_predict(shifted)
            shifted_acc = float(np.mean(
                (shifted_preds >= 0.20).astype(int) == labels
            ))

            results[shift_val_int] = {
                "accuracy": shifted_acc,
                "mean_prediction": float(shifted_preds.mean()),
                "accuracy_drop": baseline_acc - shifted_acc,
            }

        return results

    # ------------------------------------------------------------------
    # Population shift analysis
    # ------------------------------------------------------------------

    def population_shift_analysis(
        self,
        samples_by_subgroup: Dict[str, List[Dict[str, np.ndarray]]],
        labels_by_subgroup: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance across population subgroups.

        Runs predictions separately for each subgroup and reports per-
        subgroup accuracy and mean predicted risk.

        Args:
            samples_by_subgroup: Dictionary mapping subgroup identifiers
                to lists of input sample dicts.
            labels_by_subgroup: Dictionary mapping the same subgroup
                identifiers to arrays of binary ground-truth labels.

        Returns:
            Dictionary mapping subgroup identifiers to dicts of
            ``"accuracy"``, ``"mean_prediction"``, ``"n_samples"``,
            ``"prevalence"``, and ``"accuracy_drop"`` (relative to the
            overall model).
        """
        all_labels = np.concatenate(
            [labels_by_subgroup[k] for k in labels_by_subgroup]
        )
        all_samples_list = []
        for k in labels_by_subgroup:
            all_samples_list.extend(samples_by_subgroup[k])
        all_batch = self._stack_samples(all_samples_list)
        all_preds = self._batch_predict(all_batch)
        overall_acc = float(np.mean(
            (all_preds >= 0.20).astype(int) == all_labels
        ))

        results: Dict[str, Dict[str, float]] = {}

        for grp_id, grp_labels in labels_by_subgroup.items():
            grp_labels = np.asarray(grp_labels, dtype=np.float64).ravel()
            grp_samples = samples_by_subgroup[grp_id]
            if len(grp_samples) == 0:
                results[grp_id] = {
                    "accuracy": 0.0,
                    "mean_prediction": 0.0,
                    "n_samples": 0.0,
                    "prevalence": 0.0,
                    "accuracy_drop": 0.0,
                }
                continue

            grp_batch = self._stack_samples(grp_samples)
            grp_preds = self._batch_predict(grp_batch)
            grp_acc = float(np.mean(
                (grp_preds >= 0.20).astype(int) == grp_labels
            ))

            results[grp_id] = {
                "accuracy": grp_acc,
                "mean_prediction": float(grp_preds.mean()),
                "n_samples": float(len(grp_labels)),
                "prevalence": float(grp_labels.mean()),
                "accuracy_drop": overall_acc - grp_acc,
            }

        results["overall"] = {  # type: ignore[assignment]
            "accuracy": overall_acc,
            "mean_prediction": float(all_preds.mean()),
            "n_samples": float(len(all_labels)),
            "prevalence": float(all_labels.mean()),
            "accuracy_drop": 0.0,
        }

        return results
