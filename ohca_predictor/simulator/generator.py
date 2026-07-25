"""
Data generation orchestrator for the OHCA Predictor simulator module.

Coordinates population generation, temporal simulation, signal generation,
label derivation, and dataset packaging into training-ready WindowSample objects.

Usage::

    from ohca_predictor.config import get_config
    from ohca_predictor.simulator.generator import DataGenerator

    config = get_config()
    generator = DataGenerator(config)
    train, val, test = generator.generate_dataset(
        n_patients=1000,
        output_dir="data/synthetic",
    )
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ohca_predictor.config import OHCAConfig, SimulationConfig
from ohca_predictor.simulator.patient import VirtualPatient
from ohca_predictor.simulator.population import PopulationGenerator
from ohca_predictor.simulator.sensors import MultimodalSensorSuite
from ohca_predictor.simulator.diseases import DiseaseType
from ohca_predictor.utils.io_utils import WindowSample


# ---------------------------------------------------------------------------
# Activity state mapping for sensor input
# ---------------------------------------------------------------------------

ACTIVITY_STATE_MAP = {
    "sleeping": 0,
    "resting": 1,
    "light_activity": 2,
    "moderate_activity": 3,
    "vigorous_activity": 4,
}

RHYTHM_MAP = {
    "sinus": 0,
    "af": 1,
    "vt": 2,
    "vf": 3,
    "other": 4,
}


class DataGenerator:
    """Top-level orchestrator for synthetic OHCA dataset generation.

    Manages the full pipeline:
        1. Population generation with realistic demographics.
        2. Temporal simulation of 24-72 hours of continuous physiology.
        3. Multimodal sensor signal generation.
        4. Label derivation from future OHCA events.
        5. Window extraction and feature packaging.

    Args:
        config: Root configuration for all sub-systems.
    """

    def __init__(self, config: OHCAConfig) -> None:
        self.config = config
        self.sim_config: SimulationConfig = config.simulation
        self.sampling_rates: Dict[str, float] = self.sim_config.sampling_rates

        self.population_rng = np.random.default_rng(
            config.reproducibility.seed,
        )
        self.simulation_rng = np.random.default_rng(
            config.reproducibility.seed + 100,
        )
        self.sensor_rng = np.random.default_rng(
            config.reproducibility.seed + 200,
        )
        self.window_rng = np.random.default_rng(
            config.reproducibility.seed + 300,
        )

        self.population_generator = PopulationGenerator(
            config, self.population_rng,
        )
        self.sensor_suite = MultimodalSensorSuite(
            config, self.sensor_rng,
        )

    # ======================================================================
    # Main generation method
    # ======================================================================

    def generate_dataset(
        self,
        n_patients: int,
        output_dir: Optional[str] = None,
        n_workers: int = 1,
    ) -> Tuple[List[WindowSample], List[WindowSample], List[WindowSample]]:
        """Generate a complete synthetic dataset.

        Pipeline:
            1. Population generation.
            2. Temporal simulation per patient.
            3. Signal generation per window.
            4. Label derivation.
            5. Window extraction.
            6. Train/val/test split.

        Args:
            n_patients: Number of virtual patients to generate.
            output_dir: Optional directory for saving dataset metadata.
            n_workers: Number of parallel workers (reserved for future use).

        Returns:
            (train_samples, val_samples, test_samples) tuple of WindowSample lists.
        """
        patients = self.population_generator.generate_population(n_patients)

        all_windows: List[WindowSample] = []

        for patient in patients:
            duration_hours = self.simulation_rng.uniform(
                self.sim_config.min_window_duration_hours,
                self.sim_config.max_window_duration_hours,
            )

            signals, timepoints = self._simulate_patient(patient, duration_hours)

            windows = self._generate_windows(patient, signals, timepoints)
            all_windows.extend(windows)

        train, val, test = self._split_dataset(all_windows)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            metadata = {
                "n_patients": n_patients,
                "total_windows": len(all_windows),
                "train_windows": len(train),
                "val_windows": len(val),
                "test_windows": len(test),
                "positive_rate": (
                    sum(1 for w in all_windows if w.ohca_label > 0.5)
                    / max(1, len(all_windows))
                ),
            }
            meta_path = os.path.join(output_dir, "dataset_metadata.npy")
            np.save(meta_path, metadata)

        return train, val, test

    # ======================================================================
    # Patient simulation
    # ======================================================================

    def _simulate_patient(
        self,
        patient: VirtualPatient,
        duration_hours: float,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Simulate full temporal physiology for a single patient.

        Time step: 1 second for physiological state evolution.
        Applies circadian modulation, activity cycles, medication PK,
        disease progression, and autonomic nervous system dynamics.

        Each patient gets windows of variable duration sampled from
        [min_window_duration_hours, max_window_duration_hours].

        Args:
            patient: VirtualPatient to simulate.
            duration_hours: Total simulation duration in hours.

        Returns:
            (signals, timepoints) where:
                signals: dict mapping modality names to arrays of shape (n_windows, n_samples).
                timepoints: array of window start times in hours.
        """
        total_seconds = int(duration_hours * 3600.0)

        signals: Dict[str, List[np.ndarray]] = {
            "ecg": [],
            "accelerometer": [],
            "gyroscope": [],
            "ppg": [],
            "spo2": [],
            "temperature": [],
            "respiration": [],
        }

        timepoints: List[float] = []

        patient_state = self._build_patient_state(patient)

        current_seconds = 0
        while current_seconds < total_seconds:
            # Sample variable window duration for this window
            window_duration_hours = self.simulation_rng.uniform(
                self.sim_config.min_window_duration_hours,
                self.sim_config.max_window_duration_hours,
            )
            window_duration_seconds = window_duration_hours * 3600.0

            # Don't exceed total simulation time
            if current_seconds + window_duration_seconds > total_seconds:
                window_duration_seconds = total_seconds - current_seconds
                if window_duration_seconds < 60.0:
                    break

            start_hours = current_seconds / 3600.0
            timepoints.append(start_hours)

            window_signals = self._generate_window_signals(
                patient, patient_state, start_hours, window_duration_seconds,
            )

            for modality in signals:
                signals[modality].append(window_signals[modality])

            self._advance_physiology(
                patient, patient_state, start_hours, window_duration_seconds,
            )

            current_seconds += int(window_duration_seconds)

        signals_arrays = {
            mod: np.array(arrs, dtype=object) for mod, arrs in signals.items()
        }
        timepoints_arr = np.array(timepoints, dtype=np.float64)

        return signals_arrays, timepoints_arr

    def _build_patient_state(self, patient: VirtualPatient) -> Dict[str, Any]:
        """Build a state dictionary from a VirtualPatient for sensor generation."""
        comorbidities = getattr(patient, "_comorbidities", [])
        diseases = getattr(patient, "active_diseases", [])

        disease_strs = []
        for d in diseases:
            if isinstance(d, str):
                disease_strs.append(d)
            elif isinstance(d, DiseaseType):
                disease_strs.append(d.value)

        return {
            "age": patient.age,
            "sex": patient.sex,
            "bmi": patient.bmi,
            "heart_rate_bpm": patient.heart_rate_bpm,
            "respiratory_rate_bpm": patient.respiratory_rate_bpm,
            "tidal_volume_ml": 450.0,
            "skin_tone": 0.0,
            "skin_temperature": patient.skin_temperature_c,
            "core_temperature": patient.core_temperature_c,
            "baseline_spo2": patient.spo2_percent,
            "activity_state": "rest",
            "battery_level": 1.0,
            "diseases": disease_strs,
            "comorbidities": comorbidities,
            "hrv_params": {
                "rsa_amplitude": max(1.0, patient.heart_rate_variability_ms * 0.15),
                "lf_amplitude": max(0.5, patient.heart_rate_variability_ms * 0.08),
                "hf_amplitude": max(0.5, patient.heart_rate_variability_ms * 0.12),
                "vlf_amplitude": max(0.3, patient.heart_rate_variability_ms * 0.05),
                "ulf_amplitude": max(1.0, patient.heart_rate_variability_ms * 0.2),
            },
            "perfusion_index": 2.0,
            "potassium_level": patient.blood_potassium_mmol,
            "ef": patient.ejection_fraction,
            "bp_systolic": patient.systolic_bp_mmhg,
            "bp_diastolic": patient.diastolic_bp_mmhg,
            "spo2": patient.spo2_percent,
            "lactate": patient.lactate_mmol,
            "pco2": 40.0,
            "po2": 90.0,
            "respiratory_rate": patient.respiratory_rate_bpm,
            "qtc": 420.0,
            "pr_interval": 180.0,
            "qrs_duration": 100.0,
            "potassium": patient.blood_potassium_mmol,
            "troponin": patient.troponin_ng_l,
            "bnp": patient.bnp_pg_ml,
            "ph": patient.blood_ph,
            "ischemic_burden": patient.ischemic_burden,
            "irritability": patient.myocardial_irritability,
            "exercise_level": 0.0,
            "stress_level": 0.2,
            "sleep_state": 0.0,
            "auditory_stimulus": 0.0,
            "patient_id": patient.patient_id,
        }

    def _generate_window_signals(
        self,
        patient: VirtualPatient,
        patient_state: Dict[str, Any],
        start_hours: float,
        duration_seconds: float,
    ) -> Dict[str, np.ndarray]:
        """Generate sensor signals for a single window.

        Args:
            patient: VirtualPatient instance.
            patient_state: Current patient state dictionary.
            start_hours: Window start time in hours from simulation start.
            duration_seconds: Window duration in seconds.

        Returns:
            Dictionary mapping modality names to signal arrays.
        """
        window_state = dict(patient_state)
        circadian_phase = (start_hours % 24.0)
        alertness = 0.5 + 0.5 * np.sin(2.0 * np.pi * (circadian_phase - 6.0) / 24.0)

        if circadian_phase < 6.0 or circadian_phase > 22.0:
            window_state["activity_state"] = "sleeping"
            window_state["sleep_state"] = 1.0
        elif 6.0 <= circadian_phase < 8.0 or 18.0 <= circadian_phase < 22.0:
            window_state["activity_state"] = "resting"
            window_state["sleep_state"] = 0.0
        elif 8.0 <= circadian_phase < 12.0:
            window_state["activity_state"] = "walking"
            window_state["exercise_level"] = 0.3
        elif 12.0 <= circadian_phase < 14.0:
            window_state["activity_state"] = "resting"
            window_state["exercise_level"] = 0.0
        elif 14.0 <= circadian_phase < 17.0:
            act_choice = self.simulation_rng.random()
            if act_choice < 0.4:
                window_state["activity_state"] = "walking"
                window_state["exercise_level"] = 0.3
            elif act_choice < 0.7:
                window_state["activity_state"] = "resting"
                window_state["exercise_level"] = 0.0
            else:
                window_state["activity_state"] = "running"
                window_state["exercise_level"] = 0.7
        else:
            window_state["activity_state"] = "resting"
            window_state["exercise_level"] = 0.0

        hr_modulation = alertness * 5.0
        window_state["heart_rate_bpm"] = max(
            40.0, patient.heart_rate_bpm + hr_modulation,
        )

        result = self.sensor_suite.generate_all(
            patient_state=window_state,
            duration_seconds=duration_seconds,
            apply_artifacts=True,
            time_hours=start_hours,
        )

        return {
            "ecg": result["ecg"],
            "accelerometer": result["accelerometer"].T,
            "gyroscope": result["gyroscope"].T,
            "ppg": result["ppg"],
            "spo2": result["spo2"],
            "temperature": result["temperature"],
            "respiration": result["respiration"],
        }

    def _advance_physiology(
        self,
        patient: VirtualPatient,
        patient_state: Dict[str, Any],
        current_hours: float,
        dt_seconds: float,
    ) -> None:
        """Advance patient physiology by one window step.

        Updates circadian phase, activity state, medications PK,
        and latent state variables. For patients approaching OHCA,
        applies progressive physiological deterioration:
            - HR increases, HRV decreases
            - BP drops
            - SpO2 declines
            - Troponin rises
            - Lactate rises (metabolic acidosis)
            - Myocardial irritability increases
            - Ejection fraction drops

        Args:
            patient: VirtualPatient to update.
            patient_state: Mutable state dictionary.
            current_hours: Current simulation time in hours.
            dt_seconds: Time step in seconds.
        """
        dt_hours = dt_seconds / 3600.0
        circadian_phase = current_hours % 24.0

        activity_states = [
            "sleeping", "resting", "light_activity",
            "moderate_activity", "vigorous_activity",
        ]
        if circadian_phase < 6.0 or circadian_phase > 22.0:
            activity = "sleeping"
        elif 6.0 <= circadian_phase < 8.0:
            activity = "resting"
        elif 8.0 <= circadian_phase < 12.0:
            activity = "light_activity"
        elif 12.0 <= circadian_phase < 14.0:
            activity = "resting"
        elif 14.0 <= circadian_phase < 17.0:
            activity = "moderate_activity"
        else:
            activity = "resting"

        # Check if patient is approaching OHCA and apply deterioration
        time_to_ohca = getattr(patient, "time_to_ohca_hours", None)
        if time_to_ohca is not None and time_to_ohca > 0:
            hours_remaining = time_to_ohca - current_hours
            if hours_remaining > 0 and hours_remaining < 48.0:
                # Progressive deterioration: severity increases as OHCA approaches
                deterioration = max(0.0, 1.0 - (hours_remaining / 48.0))
                severity = getattr(patient, "_ohca_severity", 1.0) * deterioration

                # HR: increase toward dangerous levels
                hr_increase = severity * 40.0  # Up to 40 bpm increase
                patient.heart_rate_bpm = min(180.0, patient.heart_rate_bpm + hr_increase * dt_hours * 0.5)

                # HRV: decrease (sympathetic overdrive)
                patient.heart_rate_variability_ms = max(
                    5.0, patient.heart_rate_variability_ms * (1.0 - severity * 0.3 * dt_hours)
                )

                # BP: drop (hemodynamic instability)
                bp_drop = severity * 30.0 * dt_hours * 0.3
                patient.systolic_bp_mmhg = max(60.0, patient.systolic_bp_mmhg - bp_drop)
                patient.diastolic_bp_mmhg = max(30.0, patient.diastolic_bp_mmhg - bp_drop * 0.6)

                # SpO2: decline
                spo2_drop = severity * 8.0 * dt_hours * 0.2
                patient.spo2_percent = max(80.0, patient.spo2_percent - spo2_drop)

                # Troponin: rise (myocardial injury)
                patient.troponin_ng_l = min(
                    500.0, patient.troponin_ng_l + severity * 50.0 * dt_hours * 0.5
                )

                # Lactate: rise (metabolic acidosis)
                patient.lactate_mmol = min(
                    15.0, patient.lactate_mmol + severity * 3.0 * dt_hours * 0.3
                )

                # BNP: rise (cardiac stress)
                patient.bnp_pg_ml = min(
                    5000.0, patient.bnp_pg_ml + severity * 200.0 * dt_hours * 0.4
                )

                # Myocardial irritability: increase
                patient.myocardial_irritability = min(
                    1.0, patient.myocardial_irritability + severity * 0.2 * dt_hours
                )

                # EF: decrease
                patient.ejection_fraction = max(
                    0.10, patient.ejection_fraction - severity * 0.15 * dt_hours * 0.3
                )

                # Ischemic burden: increase
                patient.ischemic_burden = min(
                    1.0, patient.ischemic_burden + severity * 0.3 * dt_hours * 0.2
                )

                # pH: drop (acidosis)
                patient.blood_ph = max(7.0, patient.blood_ph - severity * 0.1 * dt_hours * 0.1)

                # Respiratory rate: increase (compensatory)
                patient.respiratory_rate_bpm = min(
                    40.0, patient.respiratory_rate_bpm + severity * 5.0 * dt_hours * 0.2
                )

        # Standard physiology update
        patient.update_physiology(
            dt=dt_hours,
            circadian_phase=circadian_phase,
            activity_state=activity,
        )

        # Update ALL patient state fields from actual patient physiology
        patient_state["heart_rate_bpm"] = patient.heart_rate_bpm
        patient_state["respiratory_rate_bpm"] = patient.respiratory_rate_bpm
        patient_state["spo2"] = patient.spo2_percent
        patient_state["skin_temperature"] = patient.skin_temperature_c
        patient_state["core_temperature"] = patient.core_temperature_c
        patient_state["potassium_level"] = patient.blood_potassium_mmol
        patient_state["potassium"] = patient.blood_potassium_mmol
        patient_state["lactate"] = patient.lactate_mmol
        patient_state["ef"] = patient.ejection_fraction
        patient_state["bp_systolic"] = patient.systolic_bp_mmhg
        patient_state["bp_diastolic"] = patient.diastolic_bp_mmhg
        patient_state["qtc"] = 420.0
        patient_state["troponin"] = patient.troponin_ng_l
        patient_state["bnp"] = patient.bnp_pg_ml
        patient_state["ph"] = patient.blood_ph
        patient_state["ischemic_burden"] = patient.ischemic_burden
        patient_state["irritability"] = patient.myocardial_irritability

        # Update HRV params based on actual HRV
        hrv = patient.heart_rate_variability_ms
        patient_state["hrv_params"] = {
            "rsa_amplitude": max(0.5, hrv * 0.15),
            "lf_amplitude": max(0.3, hrv * 0.08),
            "hf_amplitude": max(0.3, hrv * 0.12),
            "vlf_amplitude": max(0.2, hrv * 0.05),
            "ulf_amplitude": max(0.5, hrv * 0.2),
        }

    # ======================================================================
    # Window generation
    # ======================================================================

    def _generate_windows(
        self,
        patient: VirtualPatient,
        signals: Dict[str, np.ndarray],
        timepoints: np.ndarray,
    ) -> List[WindowSample]:
        """Extract WindowSample objects from simulated signals.

        Args:
            patient: VirtualPatient instance.
            signals: Dict of modality arrays, each (n_windows, n_samples).
                     Note: n_samples varies per window due to variable durations.
            timepoints: Array of window start times in hours.

        Returns:
            List of WindowSample objects.
        """
        windows: List[WindowSample] = []
        n_windows = len(timepoints)

        for i in range(n_windows):
            start_hours = float(timepoints[i])
            # Compute window duration from signal length
            ecg_len = signals["ecg"][i].shape[0]
            ecg_hz = self.sampling_rates["ecg_hz"]
            window_duration = ecg_len / ecg_hz / 3600.0  # hours

            label = self._derive_label(patient, start_hours)
            tte, event_ind = self._derive_survival_labels(patient, start_hours)

            demographics = self._compute_demographics_vector(patient)
            medications = self._compute_medications_vector(patient)
            comorbidities = self._compute_comorbidities_vector(patient)
            lab_values = self._compute_lab_values(patient)

            window_signals = {
                mod: signals[mod][i] for mod in signals
            }
            signal_quality = self._compute_signal_quality(window_signals)

            activity_state_name = self._get_activity_at_time(start_hours)
            activity_int = ACTIVITY_STATE_MAP.get(activity_state_name, 1)

            rhythm_int = self._get_rhythm_at_time(patient, start_hours)

            hr = float(patient.heart_rate_bpm)
            spo2_mean = float(np.mean(window_signals.get("spo2", np.array([98.0]))))
            sbp = float(patient.systolic_bp_mmhg)
            dbp = float(patient.diastolic_bp_mmhg)

            sample = WindowSample(
                ecg=window_signals["ecg"],
                accelerometer=window_signals["accelerometer"],
                gyroscope=window_signals["gyroscope"],
                ppg=window_signals["ppg"],
                spo2=window_signals["spo2"],
                temperature=window_signals["temperature"],
                respiration=window_signals["respiration"],
                demographics=demographics,
                medications=medications,
                comorbidities=comorbidities,
                lab_values=lab_values,
                ohca_label=label,
                time_to_event=tte,
                event_indicator=event_ind,
                heart_rate=hr,
                rhythm=rhythm_int,
                activity_state=activity_int,
                spo2_mean=spo2_mean,
                sbp_mean=sbp,
                dbp_mean=dbp,
                patient_id=patient.patient_id,
                window_start_hours=start_hours,
                window_duration_hours=window_duration,
                signal_quality=signal_quality,
            )
            windows.append(sample)

        return windows

    # ======================================================================
    # Label derivation
    # ======================================================================

    def _derive_label(
        self,
        patient: VirtualPatient,
        timepoint_hours: float,
        pre_arrest_window_hours: float = 4.0,
    ) -> float:
        """Derive OHCA binary label from future events.

        Logic:
            - If OHCA occurs within [timepoint, timepoint + window]: label = 1.0
            - If OHCA occurs after window: label = 0.0 (censored)
            - If no OHCA: label = 0.0

        Args:
            patient: VirtualPatient instance.
            timepoint_hours: Current window start time in hours.
            pre_arrest_window_hours: Look-ahead window in hours (default 4.0).

        Returns:
            1.0 if OHCA is imminent, 0.0 otherwise.
        """
        time_to_ohca = getattr(patient, "time_to_ohca_hours", None)

        if time_to_ohca is None:
            return 0.0

        if time_to_ohca <= 0.0:
            return 0.0

        if time_to_ohca >= timepoint_hours and time_to_ohca <= timepoint_hours + pre_arrest_window_hours:
            return 1.0

        return 0.0

    def _derive_survival_labels(
        self,
        patient: VirtualPatient,
        timepoint_hours: float,
    ) -> Tuple[float, float]:
        """Derive time-to-event and event indicator for survival analysis.

        Args:
            patient: VirtualPatient instance.
            timepoint_hours: Current window start time in hours.

        Returns:
            (time_to_event, event_indicator) where:
                time_to_event: Hours from timepoint to OHCA event (or censoring).
                event_indicator: 1.0 if event observed, 0.0 if censored.
        """
        time_to_ohca = getattr(patient, "time_to_ohca_hours", None)

        if time_to_ohca is None or time_to_ohca <= 0.0:
            max_duration = self.sim_config.max_window_duration_hours
            censor_time = max_duration - timepoint_hours
            return float(max(0.0, censor_time)), 0.0

        tte = time_to_ohca - timepoint_hours

        if tte <= 0.0:
            return 0.0, 0.0

        if tte <= self.sim_config.max_window_duration_hours - timepoint_hours:
            return float(tte), 1.0

        censor_time = self.sim_config.max_window_duration_hours - timepoint_hours
        return float(max(0.0, censor_time)), 0.0

    # ======================================================================
    # Feature computation
    # ======================================================================

    def _compute_demographics_vector(self, patient: VirtualPatient) -> np.ndarray:
        """Encode patient demographics as a fixed-size numpy vector.

        Encoding:
            - age_norm: age / 100.0
            - sex: 0.0 = male, 1.0 = female
            - height_norm: height_cm / 200.0
            - weight_norm: weight_kg / 150.0
            - bmi_norm: bmi / 50.0
            - ethnicity_onehot(5): white, black, hispanic, asian, other
            - occupation_onehot(5): sedentary, light_manual, heavy_manual, athletic, desk_professional
            - smoking_onehot(3): never, former, current
            - alcohol_onehot(4): none, light, moderate, heavy
            - fitness_norm: fitness_level (already 0-1)
            - sleep_quality_norm: sleep_quality (already 0-1)

        Total: 24 features.
        """
        vec = np.zeros(24, dtype=np.float32)

        vec[0] = float(np.clip(patient.age / 100.0, 0.0, 1.0))
        vec[1] = 1.0 if patient.sex == "female" else 0.0
        vec[2] = float(np.clip(patient.height_cm / 200.0, 0.0, 1.0))
        vec[3] = float(np.clip(patient.weight_kg / 150.0, 0.0, 1.0))
        vec[4] = float(np.clip(patient.bmi / 50.0, 0.0, 1.0))

        ethnicity_map = {"white": 5, "black": 6, "hispanic": 7, "asian": 8, "other": 9}
        eth_idx = ethnicity_map.get(patient.ethnicity, 9)
        vec[eth_idx] = 1.0

        occupation_map = {
            "sedentary": 10, "light_manual": 11, "heavy_manual": 12,
            "athletic": 13, "desk_professional": 14,
        }
        occ_idx = occupation_map.get(patient.occupation, 10)
        vec[occ_idx] = 1.0

        smoking_map = {"never": 15, "former": 16, "current": 17}
        smoke_idx = smoking_map.get(patient.smoking_status, 15)
        vec[smoke_idx] = 1.0

        alcohol_map = {"none": 18, "light": 19, "moderate": 20, "heavy": 21}
        alc_idx = alcohol_map.get(patient.alcohol_use, 18)
        vec[alc_idx] = 1.0

        vec[22] = float(np.clip(patient.fitness_level, 0.0, 1.0))
        vec[23] = float(np.clip(patient.sleep_quality, 0.0, 1.0))

        return vec

    def _compute_medications_vector(self, patient: VirtualPatient) -> np.ndarray:
        """Encode current medications as a binary vector.

        Medication classes (15 total):
            0: ace_inhibitor, 1: arb, 2: beta_blocker, 3: ccb, 4: diuretic,
            5: metformin, 6: insulin, 7: sulfonylurea, 8: aldosterone_antagonist,
            9: anticoagulant, 10: rate_control, 11: aspirin, 12: statin,
            13: other

        Total: 14 features.
        """
        med_names = getattr(patient, "_medication_names", [])

        med_classes = [
            "ace_inhibitor", "arb", "beta_blocker", "ccb", "diuretic",
            "metformin", "insulin", "sulfonylurea", "aldosterone_antagonist",
            "anticoagulant", "rate_control", "aspirin", "statin",
        ]

        vec = np.zeros(14, dtype=np.float32)

        for med in med_names:
            if med in med_classes:
                idx = med_classes.index(med)
                vec[idx] = 1.0
            else:
                vec[13] = 1.0

        return vec

    def _compute_comorbidities_vector(self, patient: VirtualPatient) -> np.ndarray:
        """Encode comorbidities as a binary vector.

        Comorbidity classes (14 total):
            0: hypertension, 1: diabetes, 2: coronary_artery_disease,
            3: heart_failure, 4: copd, 5: chronic_kidney_disease,
            6: atrial_fibrillation, 7: obesity, 8: sleep_apnea,
            9: depression, 10: anxiety, 11: anemia, 12: osteoarthritis,
            13: active_disease

        Total: 14 features.
        """
        comorbidities = getattr(patient, "_comorbidities", [])

        comorbidity_classes = [
            "hypertension", "diabetes", "coronary_artery_disease",
            "heart_failure", "copd", "chronic_kidney_disease",
            "atrial_fibrillation", "obesity", "sleep_apnea",
            "depression", "anxiety", "anemia", "osteoarthritis",
        ]

        vec = np.zeros(14, dtype=np.float32)

        for c in comorbidities:
            if c in comorbidity_classes:
                idx = comorbidity_classes.index(c)
                vec[idx] = 1.0

        active_diseases = getattr(patient, "active_diseases", [])
        if len(active_diseases) > 0:
            vec[13] = 1.0

        return vec

    def _compute_lab_values(self, patient: VirtualPatient) -> np.ndarray:
        """Compute lab values from current patient physiological state.

        Lab values (8 total):
            0: potassium (mmol/L), normalized
            1: sodium (mmol/L), normalized
            2: glucose (mg/dL), normalized
            3: calcium (mmol/L), normalized
            4: pH, normalized
            5: lactate (mmol/L), normalized
            6: troponin (ng/L), normalized
            7: BNP (pg/mL), normalized

        Total: 8 features, each normalized to approximately [0, 1] range.
        """
        vec = np.zeros(8, dtype=np.float32)

        vec[0] = float(np.clip(patient.blood_potassium_mmol / 7.0, 0.0, 1.0))
        vec[1] = float(np.clip((patient.blood_sodium_mmol - 120.0) / 40.0, 0.0, 1.0))
        vec[2] = float(np.clip(patient.blood_glucose_mgdl / 400.0, 0.0, 1.0))
        vec[3] = float(np.clip(patient.blood_calcium_mmol / 4.0, 0.0, 1.0))
        vec[4] = float(np.clip((patient.blood_ph - 7.0) / 0.8, 0.0, 1.0))
        vec[5] = float(np.clip(patient.lactate_mmol / 15.0, 0.0, 1.0))
        vec[6] = float(np.clip(patient.troponin_ng_l / 500.0, 0.0, 1.0))
        vec[7] = float(np.clip(patient.bnp_pg_ml / 5000.0, 0.0, 1.0))

        return vec

    # ======================================================================
    # Signal quality
    # ======================================================================

    def _compute_signal_quality(self, signals: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Estimate per-modality signal-to-noise ratio (SNR).

        Uses the ratio of signal power to noise floor estimate as an
        approximate SNR metric for each modality.

        Args:
            signals: Dict mapping modality names to 1-D signal arrays.

        Returns:
            Dict mapping modality names to SNR estimates (linear scale).
        """
        quality: Dict[str, float] = {}

        for modality, sig in signals.items():
            if sig is None or sig.size == 0:
                quality[modality] = 0.0
                continue

            flat = sig.flatten().astype(np.float64)
            signal_power = float(np.mean(flat ** 2))

            if len(flat) > 100:
                noise_segment = flat[:100]
                noise_power = float(np.var(noise_segment))
            else:
                noise_power = float(np.var(flat))

            if noise_power < 1e-10:
                snr = 100.0
            else:
                snr = signal_power / noise_power

            quality[modality] = float(np.clip(snr, 0.0, 1000.0))

        return quality

    # ======================================================================
    # Time-dependent state queries
    # ======================================================================

    def _get_activity_at_time(self, time_hours: float) -> str:
        """Determine activity state at a given time of day."""
        hour_of_day = time_hours % 24.0

        if hour_of_day < 6.0 or hour_of_day > 22.0:
            return "sleeping"
        elif 6.0 <= hour_of_day < 8.0:
            return "resting"
        elif 8.0 <= hour_of_day < 12.0:
            return "walking"
        elif 12.0 <= hour_of_day < 14.0:
            return "resting"
        elif 14.0 <= hour_of_day < 17.0:
            return "walking"
        elif 17.0 <= hour_of_day < 19.0:
            return "resting"
        else:
            return "resting"

    def _get_rhythm_at_time(
        self, patient: VirtualPatient, time_hours: float,
    ) -> int:
        """Determine cardiac rhythm at a given time."""
        active = getattr(patient, "active_diseases", [])

        active_strs = []
        for d in active:
            if isinstance(d, str):
                active_strs.append(d)
            elif isinstance(d, DiseaseType):
                active_strs.append(d.value)

        if "vf" in active_strs:
            return RHYTHM_MAP["vf"]
        if "vt" in active_strs:
            return RHYTHM_MAP["vt"]
        if "af" in active_strs:
            return RHYTHM_MAP["af"]

        return RHYTHM_MAP["sinus"]

    # ======================================================================
    # Dataset splitting
    # ======================================================================

    def _split_dataset(
        self,
        patients: List[WindowSample],
        ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    ) -> Tuple[List[WindowSample], List[WindowSample], List[WindowSample]]:
        """Split dataset into train/val/test with no patient overlap.

        Patients are split at the patient level: each patient's windows
        all go to the same split to prevent data leakage.

        Args:
            patients: Full list of WindowSample objects.
            ratios: (train, val, test) split ratios.

        Returns:
            (train, val, test) tuple of WindowSample lists.
        """
        patient_ids = list({w.patient_id for w in patients})
        self.window_rng.shuffle(patient_ids)

        n_total = len(patient_ids)
        n_train = int(n_total * ratios[0])
        n_val = int(n_total * ratios[1])

        train_ids = set(patient_ids[:n_train])
        val_ids = set(patient_ids[n_train:n_train + n_val])
        test_ids = set(patient_ids[n_train + n_val:])

        train = [w for w in patients if w.patient_id in train_ids]
        val = [w for w in patients if w.patient_id in val_ids]
        test = [w for w in patients if w.patient_id in test_ids]

        return train, val, test
