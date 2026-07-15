"""Unit tests for the wristppg package. Run with: pytest -q"""
import numpy as np
import pytest

from wristppg.cardiac import CardiacState, LeftVentricleModel
from wristppg.windkessel import WindkesselModel, params_from_physiology
from wristppg.arterial_tree import ArterialTreeModel, ArterialTreeParams, pwv_from_state
from wristppg.autonomic import AutonomicSimulator
from wristppg.arrhythmia import RhythmGenerator, ArrhythmiaConfig, RHYTHM_TYPES
from wristppg.optics import SkinOpticalModel, SkinOpticalParams
from wristppg.microvasculature import MicrovascularBedModel
from wristppg.motion import MotionArtifactModel, MotionEvent, ACTIVITY_PROFILES
from wristppg.contact import ContactModel, ContactState
from wristppg.noise import NoiseModel, NoiseParams
from wristppg.sensor_pipeline import SensorPipeline
from wristppg.simulator import WristPPGSimulator
from wristppg.disease import PROFILES
from wristppg import validation as val


RNG = np.random.default_rng(0)


def test_lv_model_produces_valid_pv_loop():
    lv = LeftVentricleModel(RNG)
    state = CardiacState(contractility=1.0, preload_state=0.6, afterload_Ea=1.1, hr_bpm=72)
    beat = lv.beat(state, n_samples=200)
    assert beat["edv_ml"] > beat["esv_ml"] > 0
    assert 0.05 < beat["ef"] < 0.85
    assert np.all(beat["lv_pressure_mmHg"] >= 0)
    assert np.all(beat["flow_ml_s"] >= 0)


def test_lv_frank_starling_direction():
    """Higher preload should not decrease stroke volume (Frank-Starling)."""
    lv = LeftVentricleModel(RNG)
    low = lv.beat(CardiacState(preload_state=0.3, hr_bpm=72), 200)
    high = lv.beat(CardiacState(preload_state=1.0, hr_bpm=72), 200)
    assert high["sv_ml"] >= low["sv_ml"]


def test_windkessel_pressure_positive_and_pulsatile():
    lv = LeftVentricleModel(RNG)
    beat = lv.beat(CardiacState(hr_bpm=72), 200)
    dt = beat["period_s"] / 200
    params = params_from_physiology(stiffness=1.0, resistance=1.0)
    wk = WindkesselModel().simulate(beat["flow_ml_s"], dt, params, p0_mmHg=80.0)
    assert wk["sys_pressure_mmHg"] > wk["dia_pressure_mmHg"]
    assert wk["sys_pressure_mmHg"] < 300


def test_pwv_increases_with_stiffness_and_age():
    params = ArterialTreeParams()
    pwv_low = pwv_from_state(stiffness=0.8, age_years=25, mean_pressure_mmHg=90, params=params)
    pwv_high = pwv_from_state(stiffness=2.2, age_years=75, mean_pressure_mmHg=90, params=params)
    assert pwv_high > pwv_low


def test_arterial_tree_ptt_positive():
    lv = LeftVentricleModel(RNG)
    beat = lv.beat(CardiacState(hr_bpm=72), 300)
    dt = beat["period_s"] / 300
    wk = WindkesselModel().simulate(beat["flow_ml_s"], dt, params_from_physiology(1.0, 1.0), 80.0)
    tree = ArterialTreeModel().propagate(wk["P_aortic_mmHg"], dt, stiffness=1.0, age_years=40,
                                          mean_pressure_mmHg=90, params=ArterialTreeParams())
    assert tree["ptt_s"] > 0
    assert tree["reflection_return_time_s"] > 0
    assert len(tree["radial_pressure_mmHg"]) == 300


def test_autonomic_state_bounds():
    auto = AutonomicSimulator(RNG)
    state = auto.simulate(duration_s=30, base_hr_bpm=70)
    assert np.all(state["hr_bpm"] > 0)
    assert np.all(state["vascular_tone"] >= 0) and np.all(state["vascular_tone"] <= 1)


@pytest.mark.parametrize("rhythm", RHYTHM_TYPES)
def test_all_rhythms_generate_valid_beats(rhythm):
    gen = RhythmGenerator()
    cfg = ArrhythmiaConfig(rhythm=rhythm, base_hr_bpm=75, duration_s=30, rng=np.random.default_rng(1))
    beats = gen.generate(cfg)
    assert len(beats) > 0
    for b in beats:
        assert b.rr_s > 0
        assert 0 < b.sv_scale <= 2.0


def test_afib_more_irregular_than_sinus():
    gen = RhythmGenerator()
    sinus = gen.generate(ArrhythmiaConfig(rhythm="sinus", base_hr_bpm=75, duration_s=60, rng=np.random.default_rng(2)))
    afib = gen.generate(ArrhythmiaConfig(rhythm="afib", base_hr_bpm=75, duration_s=60, rng=np.random.default_rng(2)))
    sinus_rr = np.array([b.rr_s for b in sinus])
    afib_rr = np.array([b.rr_s for b in afib])
    assert np.std(afib_rr) > np.std(sinus_rr)


def test_optics_melanin_reduces_signal():
    optics = SkinOpticalModel()
    light = SkinOpticalParams(melanin_fraction=0.05)
    dark = SkinOpticalParams(melanin_fraction=0.9)
    bf = np.full(50, 0.02)
    out_light = optics.generate_ppg_from_blood_volume("green", light, bf)
    out_dark = optics.generate_ppg_from_blood_volume("green", dark, bf)
    assert out_dark["dc"] < out_light["dc"]


def test_microvasculature_range():
    mv = MicrovascularBedModel()
    p = np.linspace(70, 120, 100)
    bf = mv.pressure_to_blood_fraction(p, baseline_fraction=0.02, vascular_tone=0.5, compliance_scale=1.0)
    assert np.all(bf > 0) and np.all(bf < 0.5)


def test_motion_artifact_activity_scales_with_intensity():
    mm = MotionArtifactModel(np.random.default_rng(3), fs_hz=128)
    rest_acc = mm.accelerometer_signal(500, MotionEvent(activity="rest"))
    run_acc = mm.accelerometer_signal(500, MotionEvent(activity="running"))
    assert np.std(run_acc) > np.std(rest_acc)


def test_contact_water_worse_than_good():
    cm = ContactModel(np.random.default_rng(4), fs_hz=128)
    good = cm.coupling_trace(500, ContactState(mode="good"))
    water = cm.coupling_trace(500, ContactState(mode="water"))
    assert np.mean(water["efficiency"]) < np.mean(good["efficiency"])


def test_noise_model_adds_variance():
    """A single noise realization isn't guaranteed to increase total
    signal variance (it could anti-correlate by chance), but the
    injected noise itself must have nonzero power and the output must
    differ meaningfully from the clean input."""
    nm = NoiseModel(np.random.default_rng(5))
    clean = np.sin(np.linspace(0, 20, 500))
    noisy = nm.apply(clean, fs=128, params=NoiseParams())
    residual = noisy - clean
    assert np.var(residual) > 1e-8
    assert not np.allclose(noisy, clean)


def test_sensor_pipeline_output_length_matches_target_rate():
    sp = SensorPipeline(np.random.default_rng(6))
    sp.params.fs_output_hz = 25.0
    fs_in = 128.0
    n = int(10 * fs_in)
    clean = np.sin(2 * np.pi * 1.2 * np.arange(n) / fs_in)
    out = sp.run(clean, fs_in)
    expected_n = n // int(round(fs_in / 25.0))
    assert abs(len(out["raw_sensor_output"]) - expected_n) <= 1


def test_end_to_end_simulator_healthy():
    sim = WristPPGSimulator(seed=42)
    result = sim.generate(profile="healthy", duration_s=20.0, activity="rest")
    assert result.ppg.ndim == 1
    assert len(result.ppg) > 0
    assert np.all(np.isfinite(result.ppg))
    assert len(result.beat_times_s) > 5
    assert result.latent_physiology["profile"] == "healthy"


def test_end_to_end_simulator_afib_profile():
    sim = WristPPGSimulator(seed=7)
    result = sim.generate(profile="afib_isolated", duration_s=20.0, activity="rest")
    assert np.all(np.isfinite(result.ppg))
    assert "afib" in result.rhythm_labels


def test_all_disease_profiles_run():
    sim = WristPPGSimulator(seed=1)
    for name in PROFILES:
        result = sim.generate(profile=name, duration_s=10.0, activity="rest")
        assert np.all(np.isfinite(result.ppg)), f"profile {name} produced non-finite signal"


def test_latent_override_decouples_hr_from_label():
    sim = WristPPGSimulator(seed=2)
    result = sim.generate(profile="healthy", duration_s=10.0, latent_overrides={"hr_bpm": 130.0})
    assert result.latent_physiology["hr_bpm"] == 130.0


def test_pulse_metrics_on_synthetic_signal():
    sim = WristPPGSimulator(seed=3)
    result = sim.generate(profile="healthy", duration_s=30.0, activity="rest")
    metrics = val.compute_pulse_metrics(result.ppg, result.fs_hz)
    assert metrics.pulse_width_ms is None or np.isnan(metrics.pulse_width_ms) or metrics.pulse_width_ms > 0


def test_statistical_distance_functions():
    a = np.random.default_rng(0).normal(0, 1, 200)
    b = np.random.default_rng(1).normal(0.5, 1, 200)
    ks = val.ks_test(a, b)
    assert 0 <= ks["statistic"] <= 1
    wd = val.wasserstein_distance(a, b)
    assert wd > 0
    mmd = val.maximum_mean_discrepancy(a, b)
    assert mmd >= 0
    dtw = val.dtw_distance(a[:20], b[:20])
    assert dtw >= 0
    fd = val.frechet_distance_gaussian(a.reshape(-1, 1), b.reshape(-1, 1))
    assert fd >= 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))