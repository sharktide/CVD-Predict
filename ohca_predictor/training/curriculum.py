"""Curriculum learning and signal augmentation for OHCA Predictor training.

Implements a multi-phase curriculum scheduler that progressively introduces
training examples by difficulty, and a comprehensive signal augmenter with
physiologically motivated transformations for ECG, accelerometer, and PPG
signals including cross-modal augmentations.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.interpolate import CubicSpline
except ImportError:
    CubicSpline = None  # type: ignore[assignment,misc]

try:
    import tensorflow as tf
except ImportError:
    raise ImportError("TensorFlow 2.x is required for ohca_predictor.training.curriculum")


logger = logging.getLogger(__name__)


# ===========================================================================
# A. CurriculumScheduler
# ===========================================================================

class CurriculumScheduler:
    """Multi-phase curriculum scheduler for OHCA training.

    Progressively introduces harder examples across four phases:

    - **Phase 1 (Epochs 1–10)**: Easy examples — clear labels, severe
      multi-vessel disease, cardiogenic shock, healthy young controls.
    - **Phase 2 (Epochs 11–30)**: Moderate examples — single disease
      processes, mild comorbidities.
    - **Phase 3 (Epochs 31–60)**: Hard examples — LBBB, pacemaker,
      old MI, early disease, polypharmacy.
    - **Phase 4 (Epochs 61+)**: All examples including edge cases —
      athlete's heart, sleep apnea, multiple simultaneous diseases.

    Args:
        config: Training configuration (used for epochs, LR, etc.).
        dataset_info: Dictionary with dataset metadata.  Expected keys:
            ``"total_patients"``: int — total number of patients.
            ``"patient_ids"``: List[str] — patient identifier strings.
            ``"labels"``: np.ndarray of shape (N,) — binary OHCA labels.
            ``"disease_severity"``: np.ndarray of shape (N,) in [0, 1].
            ``"num_comorbidities"``: np.ndarray of shape (N,) integers.
            ``"age"``: np.ndarray of shape (N,) — patient ages.
            ``"ef"``: np.ndarray of shape (N,) — ejection fractions.
            ``"is_lbbb"``: np.ndarray of shape (N,) — bool/int.
            ``"has_pacemaker"``: np.ndarray of shape (N,) — bool/int.
            ``"old_mi"``: np.ndarray of shape (N,) — bool/int.
            ``"medications"``: np.ndarray of shape (N,) — number of
            active medications (polypharmacy proxy).
            ``"snr_score"``: np.ndarray of shape (N,) in [0, 1].
    """

    # Phase boundaries (inclusive start, exclusive end)
    PHASE_1_END = 10
    PHASE_2_END = 30
    PHASE_3_END = 60
    # Phase 4 is 61+

    def __init__(
        self,
        config: Any,
        dataset_info: Dict[str, Any],
    ) -> None:
        self.config = config
        self.dataset_info = dataset_info

        self.total_patients = dataset_info["total_patients"]
        self.patient_ids: List[str] = dataset_info["patient_ids"]
        self.labels: np.ndarray = np.asarray(dataset_info["labels"], dtype=np.float32)

        # Difficulty-related attributes
        self.disease_severity: np.ndarray = np.asarray(
            dataset_info.get("disease_severity", np.zeros(self.total_patients)),
            dtype=np.float32,
        )
        self.num_comorbidities: np.ndarray = np.asarray(
            dataset_info.get("num_comorbidities", np.zeros(self.total_patients)),
            dtype=np.int32,
        )
        self.age: np.ndarray = np.asarray(
            dataset_info.get("age", np.full(self.total_patients, 55.0)),
            dtype=np.float32,
        )
        self.ef: np.ndarray = np.asarray(
            dataset_info.get("ef", np.full(self.total_patients, 0.60)),
            dtype=np.float32,
        )
        self.is_lbbb: np.ndarray = np.asarray(
            dataset_info.get("is_lbbb", np.zeros(self.total_patients, dtype=bool)),
            dtype=bool,
        )
        self.has_pacemaker: np.ndarray = np.asarray(
            dataset_info.get("has_pacemaker", np.zeros(self.total_patients, dtype=bool)),
            dtype=bool,
        )
        self.old_mi: np.ndarray = np.asarray(
            dataset_info.get("old_mi", np.zeros(self.total_patients, dtype=bool)),
            dtype=bool,
        )
        self.num_medications: np.ndarray = np.asarray(
            dataset_info.get("medications", np.zeros(self.total_patients)),
            dtype=np.int32,
        )
        self.snr_score: np.ndarray = np.asarray(
            dataset_info.get("snr_score", np.ones(self.total_patients)),
            dtype=np.float32,
        )

        # Computed difficulty scores (updated during training)
        self.difficulty_scores: np.ndarray = np.zeros(
            self.total_patients, dtype=np.float32
        )
        self._phase_sample_counts: Dict[int, int] = {}

    # -------------------------------------------------------------------
    # Phase identification
    # -------------------------------------------------------------------

    def get_current_phase(self, epoch: int) -> int:
        """Return the current curriculum phase (1-indexed) for a given epoch.

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            Phase number: 1, 2, 3, or 4.
        """
        epoch_1indexed = epoch + 1
        if epoch_1indexed <= self.PHASE_1_END:
            return 1
        elif epoch_1indexed <= self.PHASE_2_END:
            return 2
        elif epoch_1indexed <= self.PHASE_3_END:
            return 3
        else:
            return 4

    def _get_phase_description(self, phase: int) -> str:
        """Return a human-readable description of the phase."""
        descriptions = {
            1: "Easy — clear labels, severe disease, healthy controls",
            2: "Moderate — single disease processes, mild comorbidities",
            3: "Hard — LBBB, pacemaker, old MI, early disease, polypharmacy",
            4: "Full — all examples including edge cases",
        }
        return descriptions.get(phase, "Unknown")

    # -------------------------------------------------------------------
    # Difficulty scoring
    # -------------------------------------------------------------------

    def compute_difficulty_scores(
        self,
        patients: Optional[List[Any]] = None,
        baseline_predictions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute a difficulty score for each patient in [0, 1].

        Higher score = harder example.

        Difficulty is determined by:
        1. **Label ambiguity**: Low disease severity or borderline EF.
        2. **Comorbidity count**: More comorbidities → harder.
        3. **ECG confounders**: LBBB, pacemaker, old MI → harder.
        4. **Polypharmacy**: More medications → harder.
        5. **Model disagreement**: High difference between prediction
           and label (if available).
        6. **Age extremes**: Very young OHCA or very old healthy → harder.

        Args:
            patients: Optional list of VirtualPatient objects.  If provided,
                attributes are read directly; otherwise, stored arrays are used.
            baseline_predictions: Optional array of model predictions for
                the current patient set.  Used to compute prediction error
                as a difficulty signal.

        Returns:
            Array of shape (N,) with difficulty scores in [0, 1].
        """
        n = self.total_patients
        scores = np.zeros(n, dtype=np.float32)

        # 1. Disease severity (lower severity for positive class = harder)
        #    For label=1 (OHCA): low severity → hard
        #    For label=0 (healthy): high severity → hard
        positive_mask = self.labels == 1.0
        negative_mask = self.labels == 0.0

        severity_score = np.zeros(n, dtype=np.float32)
        severity_score[positive_mask] = 1.0 - self.disease_severity[positive_mask]
        severity_score[negative_mask] = self.disease_severity[negative_mask]
        scores += 0.25 * np.clip(severity_score, 0.0, 1.0)

        # 2. Comorbidity burden
        max_comorbidities = max(float(self.num_comorbidities.max()), 1.0)
        comorbidity_score = self.num_comorbidities.astype(np.float32) / max_comorbidities
        scores += 0.15 * np.clip(comorbidity_score, 0.0, 1.0)

        # 3. ECG confounders
        confounder_score = np.zeros(n, dtype=np.float32)
        confounder_score[self.is_lbbb] += 0.3
        confounder_score[self.has_pacemaker] += 0.3
        confounder_score[self.old_mi] += 0.2
        scores += 0.20 * np.clip(confounder_score, 0.0, 1.0)

        # 4. Polypharmacy
        max_meds = max(float(self.num_medications.max()), 1.0)
        polypharmacy_score = self.num_medications.astype(np.float32) / max_meds
        scores += 0.10 * np.clip(polypharmacy_score, 0.0, 1.0)

        # 5. Model disagreement (prediction error)
        if baseline_predictions is not None:
            preds = np.asarray(baseline_predictions, dtype=np.float32).ravel()
            if len(preds) == n:
                error = np.abs(preds - self.labels)
                scores += 0.20 * np.clip(error, 0.0, 1.0)

        # 6. Age extremes
        age_deviation = np.abs(self.age - 55.0) / 45.0  # normalised to ~[0, 1]
        scores += 0.10 * np.clip(age_deviation, 0.0, 1.0)

        # Clip final scores
        scores = np.clip(scores, 0.0, 1.0)
        self.difficulty_scores = scores
        return scores

    # -------------------------------------------------------------------
    # Sample weighting
    # -------------------------------------------------------------------

    def get_sample_weights(
        self,
        patients: Optional[List[Any]] = None,
        epoch: int = 0,
    ) -> np.ndarray:
        """Compute per-sample weights for the current epoch.

        The returned weights determine which samples participate in the
        current training step and with what importance.

        Phase-specific behaviour:
        - Phase 1: Binary mask — only easy examples (difficulty < 0.3).
        - Phase 2: Soft weighting — easy samples keep full weight,
          moderate samples (0.3–0.6) get reduced weight.
        - Phase 3: Hard-negative mining — harder samples get higher
          weight, but very hard samples (difficulty > 0.85) are still
          down-weighted to prevent label-noise memorisation.
        - Phase 4: Equal weight for all samples.

        Args:
            patients: Optional list of VirtualPatient objects.
            epoch: Current epoch number (0-indexed).

        Returns:
            Array of shape (N,) with non-negative sample weights.
            Zero weight means the sample is excluded.
        """
        # Ensure difficulty scores are computed
        if np.all(self.difficulty_scores == 0):
            self.compute_difficulty_scores(patients)

        scores = self.difficulty_scores
        n = self.total_patients
        weights = np.zeros(n, dtype=np.float32)

        phase = self.get_current_phase(epoch)

        if phase == 1:
            # Phase 1: Only easy examples (difficulty < 0.3)
            easy_mask = scores < 0.3
            weights[easy_mask] = 1.0

        elif phase == 2:
            # Phase 2: Easy + moderate (difficulty < 0.7)
            # Soft weighting: weight = 1.0 - 0.5 * (difficulty / 0.7)
            eligible = scores < 0.7
            weights[eligible] = np.clip(
                1.0 - 0.5 * (scores[eligible] / 0.7), 0.3, 1.0
            )

        elif phase == 3:
            # Phase 3: All samples, hard-negative mining
            # weight peaks at difficulty ~0.6, drops for very hard (>0.85)
            eligible = scores < 0.9
            # Triangular weighting: peak at difficulty = 0.5
            peak = 0.5
            weights[eligible] = np.where(
                scores[eligible] <= peak,
                0.5 + 0.5 * (scores[eligible] / peak),
                1.0 - 0.6 * ((scores[eligible] - peak) / (0.9 - peak)),
            )
            weights[eligible] = np.clip(weights[eligible], 0.2, 1.0)

        else:
            # Phase 4: Equal weight for all samples
            weights[:] = 1.0

        # Log statistics
        active_count = int((weights > 0).sum())
        self._phase_sample_counts[phase] = active_count
        if active_count > 0:
            logger.debug(
                "Curriculum phase %d (%s): %d/%d samples active, "
                "mean difficulty=%.3f",
                phase,
                self._get_phase_description(phase),
                active_count,
                n,
                float(scores[weights > 0].mean()) if active_count > 0 else 0.0,
            )

        return weights

    # -------------------------------------------------------------------
    # Difficulty update
    # -------------------------------------------------------------------

    def update_difficulty(
        self,
        epoch: int,
        baseline_predictions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Recompute difficulty scores using latest model predictions.

        Should be called at the end of each epoch (or every few epochs)
        to adapt difficulty estimates to the model's improving capability.

        Args:
            epoch: Current epoch number.
            baseline_predictions: Model predictions for all training
                patients, shape (N,).

        Returns:
            Updated difficulty scores, shape (N,).
        """
        scores = self.compute_difficulty_scores(
            patients=None,
            baseline_predictions=baseline_predictions,
        )

        # Log phase transition if any
        if epoch > 0:
            prev_phase = self.get_current_phase(epoch - 1)
            curr_phase = self.get_current_phase(epoch)
            if curr_phase != prev_phase:
                logger.info(
                    "Curriculum phase transition: %d → %d (%s) at epoch %d",
                    prev_phase,
                    curr_phase,
                    self._get_phase_description(curr_phase),
                    epoch,
                )

        return scores

    # -------------------------------------------------------------------
    # Convenience: get dataset subset for current phase
    # -------------------------------------------------------------------

    def get_active_indices(self, epoch: int) -> np.ndarray:
        """Return indices of patients that are active in the current phase.

        Args:
            epoch: Current epoch number.

        Returns:
            Integer array of active patient indices.
        """
        weights = self.get_sample_weights(epoch=epoch)
        return np.where(weights > 0)[0]

    def get_phase_info(self, epoch: int) -> Dict[str, Any]:
        """Return a dictionary summarising the current curriculum state.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary with keys: ``phase``, ``description``,
            ``active_patients``, ``total_patients``, ``difficulty_mean``,
            ``difficulty_std``.
        """
        phase = self.get_current_phase(epoch)
        active = self.get_active_indices(epoch)
        active_difficulties = self.difficulty_scores[active] if len(active) > 0 else np.array([0.0])
        return {
            "phase": phase,
            "description": self._get_phase_description(phase),
            "active_patients": int(len(active)),
            "total_patients": self.total_patients,
            "difficulty_mean": float(active_difficulties.mean()),
            "difficulty_std": float(active_difficulties.std()),
        }


# ===========================================================================
# B. SignalAugmenter
# ===========================================================================

class SignalAugmenter:
    """Physiologically motivated data augmentation for wearable signals.

    Implements augmentations for ECG, accelerometer, and PPG signals,
    as well as cross-modal augmentations that affect multiple modalities
    simultaneously.

    Args:
        config: Configuration object (optional).  If ``None``, default
            augmentation parameters are used.
        rng: NumPy random generator for reproducibility.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

        # ECG augmentation parameters
        self.ecg_time_warp_range: float = 0.10  # ±10%
        self.ecg_amplitude_scale_range: Tuple[float, float] = (0.80, 1.20)  # ±20%
        self.ecg_baseline_wander_freq: Tuple[float, float] = (0.1, 0.5)  # Hz
        self.ecg_baseline_wander_amp: float = 0.15
        self.ecg_snr_range_db: Tuple[float, float] = (10.0, 30.0)
        self.ecg_lead_rotation_range_deg: float = 30.0  # ±30°
        self.ecg_st_perturbation_mv: float = 0.5  # ±0.5 mm
        self.ecg_hr_perturbation_bpm: float = 10.0  # ±10 bpm
        self.ecg_powerline_freqs: List[float] = [50.0, 60.0]

        # Accelerometer augmentation parameters
        self.accel_magnitude_scale_range: Tuple[float, float] = (0.70, 1.30)
        self.accel_time_warp_range: float = 0.15  # ±15%
        self.accel_noise_std: float = 0.05

        # PPG augmentation parameters
        self.ppg_amplitude_scale_range: Tuple[float, float] = (0.75, 1.25)
        self.ppg_baseline_wander_amp: float = 0.10
        self.ppg_ptt_variation_ms: float = 50.0  # ±50 ms
        self.ppg_noise_std: float = 0.03

        # Cross-modal augmentation parameters
        self.cross_time_shift_ms: float = 100.0  # ±100 ms
        self.cross_modality_dropout_prob: float = 0.10
        self.cross_temporal_misalignment_ms: float = 100.0

        # Sampling rates for time-domain augmentations
        self.ecg_hz: float = 130.0
        self.accel_hz: float = 52.0
        self.ppg_hz: float = 50.0

        # Override defaults from config if provided
        if config is not None:
            self._load_config(config)

    def _load_config(self, config: Any) -> None:
        """Load augmentation parameters from a configuration object.

        Only overrides values if the config has the corresponding
        attribute defined.
        """
        if hasattr(config, "simulation"):
            sim = config.simulation
            sr = getattr(sim, "sampling_rates", {})
            self.ecg_hz = sr.get("ecg_hz", self.ecg_hz)
            self.accel_hz = sr.get("accelerometer_hz", self.accel_hz)
            self.ppg_hz = sr.get("ppg_hz", self.ppg_hz)

    # -------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------

    def augment_batch(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply random augmentations to an entire training batch.

        The batch dictionary is expected to have keys for each modality:
        ``"ecg"``, ``"accelerometer"``, ``"ppg"``, each mapping to a
        numpy array of shape ``(batch_size, sequence_length, channels)``.

        Cross-modal augmentations are applied first (shared time shift,
        modality dropout, temporal misalignment), then per-modality
        augmentations.

        Args:
            batch: Dictionary of signal arrays.

        Returns:
            Augmented batch dictionary (new arrays, original not modified).
        """
        augmented = {}
        batch_size = None

        for key, value in batch.items():
            if isinstance(value, np.ndarray) and value.ndim >= 2:
                if batch_size is None:
                    batch_size = value.shape[0]
                augmented[key] = value.copy()
            else:
                augmented[key] = value

        if batch_size is None or batch_size == 0:
            return augmented

        # --- Cross-modal augmentations ---
        augmented = self._augment_cross_modal(augmented, batch_size)

        # --- Per-modality augmentations ---
        if "ecg" in augmented and isinstance(augmented["ecg"], np.ndarray):
            for i in range(batch_size):
                augmented["ecg"][i] = self.augment_ecg(augmented["ecg"][i])

        if "accelerometer" in augmented and isinstance(augmented["accelerometer"], np.ndarray):
            for i in range(batch_size):
                augmented["accelerometer"][i] = self.augment_accelerometer(
                    augmented["accelerometer"][i]
                )

        if "ppg" in augmented and isinstance(augmented["ppg"], np.ndarray):
            for i in range(batch_size):
                augmented["ppg"][i] = self.augment_ppg(augmented["ppg"][i])

        return augmented

    # -------------------------------------------------------------------
    # Cross-modal augmentations
    # -------------------------------------------------------------------

    def _augment_cross_modal(
        self,
        batch: Dict[str, Any],
        batch_size: int,
    ) -> Dict[str, Any]:
        """Apply cross-modal augmentations: time shift, dropout,
        temporal misalignment."""
        modality_keys = [k for k in ("ecg", "accelerometer", "ppg") if k in batch]

        # 1. Random time shift applied to all modalities
        if self.rng.random() < 0.5 and modality_keys:
            shift_ms = self.rng.uniform(
                -self.cross_time_shift_ms, self.cross_time_shift_ms
            )
            for key in modality_keys:
                signal = batch[key]
                if not isinstance(signal, np.ndarray):
                    continue
                hz = self._get_hz(key)
                shift_samples = int(round(shift_ms / 1000.0 * hz))
                if abs(shift_samples) > 0:
                    batch[key] = self._circular_shift(signal, shift_samples)

        # 2. Random modality dropout: zero out one modality
        if modality_keys and self.rng.random() < self.cross_modality_dropout_prob:
            drop_key = self.rng.choice(modality_keys)
            if isinstance(batch[drop_key], np.ndarray):
                batch[drop_key] = np.zeros_like(batch[drop_key])
                logger.debug("Modality dropout: zeroed out '%s'", drop_key)

        # 3. Temporal misalignment: shift one modality relative to others
        if len(modality_keys) >= 2 and self.rng.random() < 0.3:
            misalign_key = self.rng.choice(modality_keys)
            misalign_ms = self.rng.uniform(
                -self.cross_temporal_misalignment_ms,
                self.cross_temporal_misalignment_ms,
            )
            hz = self._get_hz(misalign_key)
            shift_samples = int(round(misalign_ms / 1000.0 * hz))
            if abs(shift_samples) > 0 and isinstance(batch[misalign_key], np.ndarray):
                batch[misalign_key] = self._circular_shift(
                    batch[misalign_key], shift_samples
                )

        return batch

    # -------------------------------------------------------------------
    # ECG augmentations
    # -------------------------------------------------------------------

    def augment_ecg(self, signal: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a single ECG signal.

        Applies a random subset of:
        1. Time warping (±10%) via cubic B-spline.
        2. Amplitude scaling (±20%).
        3. Baseline wander injection (0.1–0.5 Hz).
        4. Gaussian noise (SNR 10–30 dB).
        5. Random lead rotation (±30°).
        6. ST segment perturbation (±0.5 mm).
        7. Heart rate perturbation (±10 bpm).
        8. Powerline interference (50/60 Hz).

        Args:
            signal: ECG array of shape ``(T, C)`` or ``(T,)``.

        Returns:
            Augmented ECG array of same shape.
        """
        aug = signal.copy()
        if aug.size == 0:
            return aug

        # Randomly select which augmentations to apply (1-3 at a time)
        n_augmentations = self.rng.integers(1, 4)
        all_augmentations = [
            self._ecg_time_warp,
            self._ecg_amplitude_scale,
            self._ecg_baseline_wander,
            self._ecg_gaussian_noise,
            self._ecg_lead_rotation,
            self._ecg_st_perturbation,
            self._ecg_hr_perturbation,
            self._ecg_powerline_interference,
        ]
        chosen = self.rng.choice(
            all_augmentations, size=min(n_augmentations, len(all_augmentations)), replace=False
        )

        for aug_fn in chosen:
            aug = aug_fn(aug)

        return aug

    def _ecg_time_warp(self, signal: np.ndarray) -> np.ndarray:
        """Non-linear time distortion via random cubic B-spline.

        Resamples the time axis using a randomly perturbed monotonic
        spline to simulate heart rate variability effects on waveform
        morphology.

        Args:
            signal: ECG array of shape ``(T,)`` or ``(T, C)``.

        Returns:
            Time-warped ECG array.
        """
        if CubicSpline is None:
            return signal

        T = signal.shape[0]
        if T < 4:
            return signal

        t_orig = np.linspace(0, 1, T)

        # Create random anchor points with monotonic perturbation
        n_anchors = max(4, T // 50)
        anchor_indices = np.linspace(0, T - 1, n_anchors, dtype=int)
        anchor_times = t_orig[anchor_indices]

        # Random perturbation ensuring monotonicity
        perturbation = self.rng.uniform(
            -self.ecg_time_warp_range, self.ecg_time_warp_range, size=n_anchors
        )
        # Ensure monotonic: cumulative sum of positive perturbations
        perturbation = np.sort(perturbation)
        new_times = anchor_times + perturbation
        new_times = np.clip(new_times, 0.0, 1.0)
        new_times[0] = 0.0
        new_times[-1] = 1.0
        # Enforce strict monotonicity
        for i in range(1, len(new_times)):
            if new_times[i] <= new_times[i - 1]:
                new_times[i] = new_times[i - 1] + 1e-6
        new_times[-1] = max(new_times[-1], new_times[-2] + 1e-6)

        try:
            cs = CubicSpline(new_times, t_orig[anchor_indices])
            new_t = cs(t_orig)
            new_t = np.clip(new_t, 0.0, float(T - 1))
        except Exception:
            return signal

        if signal.ndim == 1:
            warped = np.interp(new_t, t_orig, signal)
        else:
            warped = np.zeros_like(signal, dtype=np.float64)
            for c in range(signal.shape[1]):
                warped[:, c] = np.interp(new_t, t_orig, signal[:, c].astype(np.float64))

        return warped.astype(signal.dtype)

    def _ecg_amplitude_scale(self, signal: np.ndarray) -> np.ndarray:
        """Scale amplitude by a random factor in [0.80, 1.20]."""
        scale = self.rng.uniform(*self.ecg_amplitude_scale_range)
        return (signal * scale).astype(signal.dtype)

    def _ecg_baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """Inject low-frequency baseline wander (0.1–0.5 Hz).

        Models respiratory-induced baseline oscillation.
        """
        T = signal.shape[0]
        t = np.arange(T, dtype=np.float64) / self.ecg_hz

        freq = self.rng.uniform(*self.ecg_baseline_wander_freq)
        phase = self.rng.uniform(0, 2 * math.pi)
        amplitude = self.ecg_baseline_wander_amp * (
            np.abs(signal).max() if signal.size > 0 else 1.0
        )

        wander = amplitude * np.sin(2 * math.pi * freq * t + phase)

        if signal.ndim == 1:
            return (signal.astype(np.float64) + wander).astype(signal.dtype)
        else:
            return (signal.astype(np.float64) + wander[:, np.newaxis]).astype(signal.dtype)

    def _ecg_gaussian_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add Gaussian noise at a random SNR between 10 and 30 dB."""
        if signal.size == 0:
            return signal

        signal_power = np.mean(signal.astype(np.float64) ** 2)
        if signal_power < 1e-12:
            return signal

        snr_db = self.rng.uniform(*self.ecg_snr_range_db)
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = self.rng.normal(0.0, math.sqrt(max(noise_power, 1e-20)), size=signal.shape)

        return (signal.astype(np.float64) + noise).astype(signal.dtype)

    def _ecg_lead_rotation(self, signal: np.ndarray) -> np.ndarray:
        """Random rotation of multi-lead ECG (±30°).

        Only applied if signal has ≥2 channels.  Simulates electrode
        placement variation.
        """
        if signal.ndim < 2 or signal.shape[1] < 2:
            return signal

        angle_rad = math.radians(
            self.rng.uniform(-self.ecg_lead_rotation_range_deg, self.ecg_lead_rotation_range_deg)
        )

        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # For 2+ leads, apply a rotation matrix to the first two channels
        out = signal.copy().astype(np.float64)
        lead0 = out[:, 0].copy()
        lead1 = out[:, 1].copy()
        out[:, 0] = cos_a * lead0 - sin_a * lead1
        out[:, 1] = sin_a * lead0 + cos_a * lead1

        return out.astype(signal.dtype)

    def _ecg_st_perturbation(self, signal: np.ndarray) -> np.ndarray:
        """Perturb the ST segment by ±0.5 mm (0.05 mV).

        Applies a rectangular pulse in the ST region (~100–250 ms
        after QRS).  This is a simplified physiological model.
        """
        T = signal.shape[0]
        # ST region: approximately 100–250 ms after QRS onset
        # At 130 Hz: sample 13 to 33
        st_start = max(0, int(0.10 * self.ecg_hz))
        st_end = min(T, int(0.25 * self.ecg_hz))

        if st_end <= st_start:
            return signal

        perturbation_mv = self.rng.uniform(
            -self.ecg_st_perturbation_mv, self.ecg_st_perturbation_mv
        )
        # Convert mm to approximate voltage (1 mm ≈ 0.1 mV)
        perturbation = perturbation_mv * 0.1

        out = signal.copy().astype(np.float64)
        if out.ndim == 1:
            out[st_start:st_end] += perturbation
        else:
            out[st_start:st_end, :] += perturbation

        return out.astype(signal.dtype)

    def _ecg_hr_perturbation(self, signal: np.ndarray) -> np.ndarray:
        """Simulate ±10 bpm heart rate perturbation via resampling.

        Stretches/compresses the signal in time to simulate a slightly
        different heart rate.
        """
        T = signal.shape[0]
        base_hr = 72.0  # assumed baseline
        hr_offset = self.rng.uniform(-self.ecg_hr_perturbation_bpm, self.ecg_hr_perturbation_bpm)
        new_hr = max(40.0, min(180.0, base_hr + hr_offset))

        stretch_factor = base_hr / new_hr
        new_T = int(round(T * stretch_factor))
        if new_T < 4:
            return signal

        t_orig = np.linspace(0.0, 1.0, T)
        t_new = np.linspace(0.0, 1.0, new_T)

        if signal.ndim == 1:
            resampled = np.interp(t_new, t_orig, signal)
        else:
            resampled = np.zeros((new_T, signal.shape[1]), dtype=signal.dtype)
            for c in range(signal.shape[1]):
                resampled[:, c] = np.interp(t_new, t_orig, signal[:, c].astype(np.float64))

        # Pad or truncate back to original length
        if new_T > T:
            resampled = resampled[:T]
        elif new_T < T:
            pad_width = T - new_T
            if resampled.ndim == 1:
                resampled = np.pad(resampled, (0, pad_width), mode="edge")
            else:
                resampled = np.pad(resampled, ((0, pad_width), (0, 0)), mode="edge")

        return resampled.astype(signal.dtype)

    def _ecg_powerline_interference(self, signal: np.ndarray) -> np.ndarray:
        """Add 50/60 Hz powerline interference.

        Models electromagnetic interference from mains electricity.
        """
        T = signal.shape[0]
        t = np.arange(T, dtype=np.float64) / self.ecg_hz

        freq = self.rng.choice(self.ecg_powerline_freqs)
        amplitude = self.rng.uniform(0.01, 0.05) * (
            np.abs(signal).max() if signal.size > 0 else 1.0
        )
        phase = self.rng.uniform(0, 2 * math.pi)
        interference = amplitude * np.sin(2 * math.pi * freq * t + phase)

        if signal.ndim == 1:
            return (signal.astype(np.float64) + interference).astype(signal.dtype)
        else:
            return (signal.astype(np.float64) + interference[:, np.newaxis]).astype(signal.dtype)

    # -------------------------------------------------------------------
    # Accelerometer augmentations
    # -------------------------------------------------------------------

    def augment_accelerometer(self, signal: np.ndarray) -> np.ndarray:
        """Apply random augmentations to an accelerometer signal.

        Applies a random subset of:
        1. Magnitude scaling (±30%).
        2. Axis rotation (random 3D rotation).
        3. Time warping (±15%).
        4. Gaussian noise injection.

        Args:
            signal: Accelerometer array of shape ``(T, 3)`` (x, y, z).

        Returns:
            Augmented accelerometer array of same shape.
        """
        aug = signal.copy()
        if aug.size == 0:
            return aug

        n_augmentations = self.rng.integers(1, 4)
        all_augmentations = [
            self._accel_magnitude_scale,
            self._accel_axis_rotation,
            self._accel_time_warp,
            self._accel_noise,
        ]
        chosen = self.rng.choice(
            all_augmentations, size=min(n_augmentations, len(all_augmentations)), replace=False
        )

        for aug_fn in chosen:
            aug = aug_fn(aug)

        return aug

    def _accel_magnitude_scale(self, signal: np.ndarray) -> np.ndarray:
        """Scale all axes by a random magnitude factor (0.70–1.30)."""
        scale = self.rng.uniform(*self.accel_magnitude_scale_range)
        return (signal * scale).astype(signal.dtype)

    def _accel_axis_rotation(self, signal: np.ndarray) -> np.ndarray:
        """Apply a random 3D rotation to the accelerometer axes.

        Simulates different device orientations.
        """
        if signal.ndim < 2 or signal.shape[1] < 3:
            return signal

        # Random rotation angles for ZYZ Euler angles
        alpha = self.rng.uniform(0, 2 * math.pi)
        beta = self.rng.uniform(0, math.pi)
        gamma = self.rng.uniform(0, 2 * math.pi)

        # Build rotation matrix R = Rz(alpha) * Ry(beta) * Rz(gamma)
        ca, sa = math.cos(alpha), math.sin(alpha)
        cb, sb = math.cos(beta), math.sin(beta)
        cg, sg = math.cos(gamma), math.sin(gamma)

        R = np.array(
            [
                [ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb],
                [sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb],
                [-sb * cg, sb * sg, cb],
            ],
            dtype=np.float64,
        )

        out = signal.copy().astype(np.float64)
        # Apply rotation: each time step is a row vector multiplied by R^T
        rotated = out @ R.T
        return rotated.astype(signal.dtype)

    def _accel_time_warp(self, signal: np.ndarray) -> np.ndarray:
        """Time warp the accelerometer signal (±15%).

        Uses linear interpolation for speed.
        """
        T = signal.shape[0]
        if T < 4:
            return signal

        warp_factor = 1.0 + self.rng.uniform(
            -self.accel_time_warp_range, self.accel_time_warp_range
        )
        new_T = max(4, int(round(T * warp_factor)))

        t_orig = np.linspace(0.0, 1.0, T)
        t_new = np.linspace(0.0, 1.0, new_T)

        if signal.ndim == 1:
            warped = np.interp(t_new, t_orig, signal)
        else:
            warped = np.zeros((new_T, signal.shape[1]), dtype=np.float64)
            for c in range(signal.shape[1]):
                warped[:, c] = np.interp(t_new, t_orig, signal[:, c].astype(np.float64))

        # Resize back to original length
        if new_T > T:
            warped = warped[:T]
        elif new_T < T:
            pad_width = T - new_T
            if warped.ndim == 1:
                warped = np.pad(warped, (0, pad_width), mode="edge")
            else:
                warped = np.pad(warped, ((0, pad_width), (0, 0)), mode="edge")

        return warped.astype(signal.dtype)

    def _accel_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to accelerometer axes."""
        noise = self.rng.normal(0.0, self.accel_noise_std, size=signal.shape)
        return (signal.astype(np.float64) + noise).astype(signal.dtype)

    # -------------------------------------------------------------------
    # PPG augmentations
    # -------------------------------------------------------------------

    def augment_ppg(self, signal: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a PPG signal.

        Applies a random subset of:
        1. Amplitude scaling (±25%).
        2. Baseline wander.
        3. Pulse transit time variation (±50 ms).
        4. Gaussian noise injection.

        Args:
            signal: PPG array of shape ``(T,)`` or ``(T, C)``.

        Returns:
            Augmented PPG array of same shape.
        """
        aug = signal.copy()
        if aug.size == 0:
            return aug

        n_augmentations = self.rng.integers(1, 4)
        all_augmentations = [
            self._ppg_amplitude_scale,
            self._ppg_baseline_wander,
            self._ppg_ptt_variation,
            self._ppg_noise,
        ]
        chosen = self.rng.choice(
            all_augmentations, size=min(n_augmentations, len(all_augmentations)), replace=False
        )

        for aug_fn in chosen:
            aug = aug_fn(aug)

        return aug

    def _ppg_amplitude_scale(self, signal: np.ndarray) -> np.ndarray:
        """Scale PPG amplitude by a random factor (0.75–1.25)."""
        scale = self.rng.uniform(*self.ppg_amplitude_scale_range)
        return (signal * scale).astype(signal.dtype)

    def _ppg_baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """Inject low-frequency baseline wander into PPG signal.

        Simulates respiratory modulation and motion artifacts.
        """
        T = signal.shape[0]
        t = np.arange(T, dtype=np.float64) / self.ppg_hz

        # Respiratory frequency (12–20 breaths/min → 0.2–0.33 Hz)
        freq = self.rng.uniform(0.2, 0.33)
        phase = self.rng.uniform(0, 2 * math.pi)
        amplitude = self.ppg_baseline_wander_amp * (
            np.abs(signal).max() if signal.size > 0 else 1.0
        )

        wander = amplitude * np.sin(2 * math.pi * freq * t + phase)

        if signal.ndim == 1:
            return (signal.astype(np.float64) + wander).astype(signal.dtype)
        else:
            return (signal.astype(np.float64) + wander[:, np.newaxis]).astype(signal.dtype)

    def _ppg_ptt_variation(self, signal: np.ndarray) -> np.ndarray:
        """Simulate pulse transit time variation (±50 ms).

        Shifts the signal in time to simulate changes in arterial
        stiffness or blood pressure affecting pulse wave velocity.
        """
        T = signal.shape[0]
        shift_ms = self.rng.uniform(-self.ppg_ptt_variation_ms, self.ppg_ptt_variation_ms)
        shift_samples = int(round(shift_ms / 1000.0 * self.ppg_hz))

        if abs(shift_samples) < 1:
            return signal

        return self._circular_shift(signal, shift_samples)

    def _ppg_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to PPG signal."""
        noise = self.rng.normal(0.0, self.ppg_noise_std, size=signal.shape)
        return (signal.astype(np.float64) + noise).astype(signal.dtype)

    # -------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------

    def _get_hz(self, modality: str) -> float:
        """Return the sampling frequency for a given modality."""
        hz_map = {
            "ecg": self.ecg_hz,
            "accelerometer": self.accel_hz,
            "gyroscope": self.accel_hz,
            "ppg": self.ppg_hz,
        }
        return hz_map.get(modality, 50.0)

    @staticmethod
    def _circular_shift(signal: np.ndarray, shift: int) -> np.ndarray:
        """Circularly shift a signal along the time axis.

        Args:
            signal: Array of shape ``(T, ...)``.
            shift: Number of samples to shift (positive = forward in time).

        Returns:
            Circularly shifted array.
        """
        if shift == 0 or signal.size == 0:
            return signal
        return np.roll(signal, shift, axis=0)

    def set_epoch(self, epoch: int) -> None:
        """Optionally adjust augmentation intensity based on training epoch.

        Early epochs may benefit from stronger augmentation to regularise,
        while later epochs may want weaker augmentation for fine-tuning.
        This is a no-op by default but can be overridden or extended.

        Args:
            epoch: Current epoch number.
        """
        pass
