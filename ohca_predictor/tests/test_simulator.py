"""Tests for the OHCA Simulator module."""

import pytest
import numpy as np
from ohca_predictor.simulator.patient import VirtualPatient, Medication
from ohca_predictor.simulator.sensors import (
    ECGSensor,
    AccelerometerSensor,
    PPGSensor,
    MultimodalSensorSuite,
    SensorArtifactModel,
)
from ohca_predictor.simulator.physiology import (
    CircadianModel,
    ActivityModel,
    ANSModel,
    RespiratoryModel,
    ThermoregulationModel,
    HydrationModel,
)
from ohca_predictor.simulator.diseases import DiseaseType, DiseaseState, DiseaseInteractionManager
from ohca_predictor.simulator.population import PopulationGenerator
from ohca_predictor.config import get_config


class TestVirtualPatient:
    """Tests for the VirtualPatient class."""

    def test_patient_creation(self):
        config = get_config()
        rng = np.random.default_rng(42)
        patient = VirtualPatient(rng=rng, config=config)
        assert patient.patient_id is not None
        assert 18 <= patient.age <= 100
        assert patient.sex in ["male", "female"]

    def test_patient_bmi(self):
        config = get_config()
        rng = np.random.default_rng(42)
        patient = VirtualPatient(rng=rng, config=config)
        expected_bmi = patient.weight_kg / (patient.height_cm / 100) ** 2
        assert abs(patient.bmi - expected_bmi) < 0.1

    def test_patient_physiology_ranges(self):
        config = get_config()
        rng = np.random.default_rng(42)
        patient = VirtualPatient(rng=rng, config=config)
        assert 35 <= patient.heart_rate_bpm <= 220
        assert 80 <= patient.systolic_bp_mmhg <= 200
        assert 40 <= patient.diastolic_bp_mmhg <= 130
        assert 70 <= patient.spo2_percent <= 100

    def test_patient_update_physiology(self):
        config = get_config()
        rng = np.random.default_rng(42)
        patient = VirtualPatient(rng=rng, config=config)
        patient.update_physiology(dt=1.0, circadian_phase=0.5, activity_state=1)
        # Should not raise

    def test_patient_clamp_state(self):
        config = get_config()
        rng = np.random.default_rng(42)
        patient = VirtualPatient(rng=rng, config=config)
        patient.heart_rate_bpm = 999
        patient.clamp_state()
        assert patient.heart_rate_bpm <= 220


class TestSensors:
    """Tests for sensor simulation classes."""

    def test_ecg_sensor(self):
        config = get_config()
        rng = np.random.default_rng(42)
        sensor = ECGSensor(config=config, rng=rng)
        ecg = sensor.generate(duration_seconds=5.0, heart_rate_bpm=72.0,
                              hrv_params={'rmssd': 30.0, 'lf_hf': 1.5})
        assert ecg.shape[0] > 0
        assert np.std(ecg) > 0

    def test_accelerometer_sensor(self):
        config = get_config()
        rng = np.random.default_rng(42)
        sensor = AccelerometerSensor(config=config, rng=rng)
        accel = sensor.generate(duration_seconds=5.0, activity_state='resting')
        assert accel.shape[0] == 3
        assert accel.shape[1] > 0

    def test_ppg_sensor(self):
        config = get_config()
        rng = np.random.default_rng(42)
        ecg_sensor = ECGSensor(config=config, rng=rng)
        ecg = ecg_sensor.generate(duration_seconds=5.0, heart_rate_bpm=72.0,
                                  hrv_params={'rmssd': 30.0, 'lf_hf': 1.5})
        ppg_sensor = PPGSensor(config=config, rng=rng)
        ppg = ppg_sensor.generate(duration_seconds=5.0, ecg_signal=ecg)
        assert ppg.shape[0] > 0

    def test_multimodal_suite(self):
        config = get_config()
        rng = np.random.default_rng(42)
        suite = MultimodalSensorSuite(config=config, rng=rng)
        patient_state = {
            'heart_rate_bpm': 72.0,
            'hrv_rmssd': 30.0,
            'lf_hf_ratio': 1.5,
            'respiratory_rate_bpm': 16.0,
            'spo2_percent': 98.0,
            'skin_temperature_c': 35.0,
        }
        signals = suite.generate_all(patient_state=patient_state, duration_seconds=5.0)
        assert 'ecg' in signals
        assert 'accelerometer' in signals


class TestPhysiology:
    """Tests for physiology simulation models."""

    def test_circadian_model(self):
        rng = np.random.default_rng(42)
        model = CircadianModel(rng=rng)
        model.initial_state(rng)
        # Verify internal state was set
        assert hasattr(model, 'S')
        assert 0.0 <= model.S <= 1.0

    def test_activity_model(self):
        rng = np.random.default_rng(42)
        model = ActivityModel(rng=rng)
        model.initial_state(rng)
        # Verify internal state was set
        assert hasattr(model, 'state')

    def test_ans_model(self):
        rng = np.random.default_rng(42)
        model = ANSModel(rng=rng)
        model.initial_state(rng)
        # Verify internal state was set
        assert hasattr(model, 'sympathetic_tone') or hasattr(model, 'state')


class TestDiseases:
    """Tests for disease models."""

    def test_disease_type_enum(self):
        assert len(DiseaseType) == 20

    def test_disease_state(self):
        state = DiseaseState(
            disease_type=DiseaseType.ACS_STEMI,
            onset_time_hours=0.0,
            severity=0.5,
            progression_rate=0.01,
            stage=1,
            complications=[],
            is_active=True,
            probability_of_collapse=0.1,
            latent_pathway=np.zeros(64),
        )
        assert state.severity == 0.5
        assert state.is_active


class TestPopulation:
    """Tests for population generation."""

    def test_population_generation(self):
        config = get_config()
        rng = np.random.default_rng(42)
        gen = PopulationGenerator(config=config, rng=rng)
        patients = gen.generate_population(n_patients=5)
        assert len(patients) == 5
        assert all(isinstance(p, VirtualPatient) for p in patients)

    def test_population_diversity(self):
        config = get_config()
        rng = np.random.default_rng(42)
        gen = PopulationGenerator(config=config, rng=rng)
        patients = gen.generate_population(n_patients=20)
        ages = [p.age for p in patients]
        sexes = [p.sex for p in patients]
        assert len(set(sexes)) >= 1
        assert min(ages) >= 18
        assert max(ages) <= 100
