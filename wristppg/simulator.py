"""
WristPPGSimulator: orchestrates the full pipeline

    LV pressure -> aortic flow -> Windkessel -> arterial tree propagation
    -> PTT -> radial waveform -> microvascular bed -> optical tissue model
    -> photodiode -> ADC output

beat-by-beat, driven by continuously evolving autonomic/systemic state and
a rhythm generator, then passes the resulting "clean" optical signal
through contact, motion, noise, and sensor-acquisition models.

This module is the integration layer; see the evidence-base docstrings in
each component module (cardiac.py, windkessel.py, arterial_tree.py,
autonomic.py, arrhythmia.py, optics.py, microvasculature.py, motion.py,
contact.py, noise.py, sensor_pipeline.py, disease.py) for citations and
explicit heuristic/assumption disclosures. Nothing here should be
interpreted as validated against a specific real device or clinical
population without the comparisons described in validation.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .cardiac import CardiacState, LeftVentricleModel
from .windkessel import WindkesselModel, params_from_physiology
from .arterial_tree import ArterialTreeModel, ArterialTreeParams
from .autonomic import AutonomicSimulator
from .arrhythmia import RhythmGenerator, ArrhythmiaConfig
from .optics import SkinOpticalModel, SkinOpticalParams
from .microvasculature import MicrovascularBedModel
from .motion import MotionArtifactModel, MotionEvent, ACTIVITY_PROFILES
from .contact import ContactModel, ContactState
from .noise import NoiseModel, NoiseParams
from .sensor_pipeline import SensorPipeline, SensorPipelineParams
from .disease import PROFILES, DiseaseProfile


FS_INTERNAL_HZ = 128.0  # internal physiological/optical simulation rate


@dataclass
class SimulationResult:
    ppg: np.ndarray
    fs_hz: float
    clean_ppg_internal: np.ndarray
    fs_internal_hz: float
    beat_times_s: np.ndarray
    beat_types: list
    rhythm_labels: list
    hr_instantaneous_bpm: np.ndarray
    stroke_volume_ml: np.ndarray
    ejection_fraction: np.ndarray
    ptt_s: np.ndarray
    augmentation_index: np.ndarray
    pwv_m_s: np.ndarray
    latent_physiology: dict
    meta: dict


class WristPPGSimulator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.lv_model = LeftVentricleModel(self.rng)
        self.windkessel = WindkesselModel()
        self.arterial_tree = ArterialTreeModel()
        self.autonomic = AutonomicSimulator(self.rng, fs_state_hz=4.0)
        self.rhythm_gen = RhythmGenerator()
        self.optics = SkinOpticalModel()
        self.microvasc = MicrovascularBedModel()
        self.motion_model = MotionArtifactModel(self.rng, fs_hz=FS_INTERNAL_HZ)
        self.contact_model = ContactModel(self.rng, fs_hz=FS_INTERNAL_HZ)
        self.noise_model = NoiseModel(self.rng)
        self.sensor_pipeline = SensorPipeline(self.rng, SensorPipelineParams(fs_output_hz=25.0))

    # ------------------------------------------------------------------
    def generate(self, profile: str | DiseaseProfile = "healthy",
                 duration_s: float = 60.0,
                 activity: str = "rest",
                 contact_mode: str = "good",
                 wavelength: str = "green",
                 fs_output_hz: float = 25.0,
                 exercise_profile: Optional[str] = None,
                 circadian_hour: Optional[float] = None,
                 orthostatic_event_s: Optional[float] = None,
                 latent_overrides: Optional[dict] = None) -> SimulationResult:
        """Generate one recording.

        Parameters
        ----------
        profile:
            Disease/physiological-state profile name (see disease.PROFILES)
            or a DiseaseProfile instance.
        latent_overrides:
            Optional dict to directly override any sampled latent
            physiological variable (hr_bpm, age_years, stiffness,
            resistance, contractility, hrv_frac, rhythm, vascular_tone,
            preload_bias, melanin_fraction). Use this to decouple
            observable features (e.g. heart rate) from the disease label
            when constructing ML training sets, per the "avoid shortcut
            learning" requirement — e.g. force a "healthy" profile to a
            high HR, or an "hfref" profile to a normal HR, so that a
            classifier cannot key on HR alone.
        """
        prof = PROFILES[profile] if isinstance(profile, str) else profile
        latent = prof.sample(self.rng)
        if latent_overrides:
            latent.update(latent_overrides)

        self.sensor_pipeline.params.fs_output_hz = fs_output_hz

        exercise_profile = exercise_profile or ("exercise" if activity in ("running", "lifting_weights") else "rest")

        auto_state = self.autonomic.simulate(
            duration_s=duration_s + 5.0,
            base_hr_bpm=latent["hr_bpm"],
            base_resistance=latent["resistance"],
            base_vascular_tone=latent["vascular_tone"],
            circadian_hour=circadian_hour,
            orthostatic_event_s=orthostatic_event_s,
            exercise_profile=exercise_profile,
        )

        rhythm_cfg = ArrhythmiaConfig(rhythm=latent["rhythm"], base_hr_bpm=latent["hr_bpm"],
                                       duration_s=duration_s + 5.0, rng=self.rng)
        beats = self.rhythm_gen.generate(rhythm_cfg)

        beat_times, beat_records, blood_fraction_segments, seg_times = self._simulate_beats(
            beats, auto_state, latent, prof
        )

        n_internal = int(duration_s * FS_INTERNAL_HZ)
        t_internal = np.arange(n_internal) / FS_INTERNAL_HZ
        blood_fraction_full = self._assemble_continuous(seg_times, blood_fraction_segments,
                                                          t_internal, fill_value=latent.get(
                                                              "tissue_blood_fraction", 0.02))

        skin_params = SkinOpticalParams(
            melanin_fraction=latent["melanin_fraction"],
            spo2=0.98, venous_spo2=0.70,
            tissue_blood_fraction=0.02, pulsatile_fraction=0.006,
            hair_density=self.rng.uniform(0.0, 0.3),
            tattoo_optical_density=0.0,
            sweat_layer=0.0,
            contact_pressure=0.5,
        )
        optics_out = self.optics.generate_ppg_from_blood_volume(wavelength, skin_params, blood_fraction_full)
        clean_optical = optics_out["intensity"]

        contact_state = ContactState(mode=contact_mode)
        motion_event = MotionEvent(activity=activity, intensity_scale=1.0)
        accel = self.motion_model.accelerometer_signal(n_internal, motion_event)
        contact_out = self.contact_model.coupling_trace(n_internal, contact_state, motion_energy=accel)

        ambient_component = self.rng.normal(np.mean(clean_optical), 0.02 * np.std(clean_optical) + 1e-6, n_internal)
        coupled = clean_optical * contact_out["efficiency"] + (1 - contact_out["efficiency"]) * ambient_component

        motion_out = self.motion_model.couple_to_ppg(coupled, accel, motion_event)
        motioned = motion_out["ppg_with_motion"]

        noise_params = NoiseParams()
        noisy = self.noise_model.apply(motioned, FS_INTERNAL_HZ, noise_params,
                                        melanin_fraction=latent["melanin_fraction"], device_age_days=0.0)

        sensor_out = self.sensor_pipeline.run(noisy, FS_INTERNAL_HZ)
        final_ppg = sensor_out["raw_sensor_output"]

        meta = {
            "profile": prof.name,
            "activity": activity,
            "contact_mode": contact_mode,
            "wavelength": wavelength,
            "ac_dc_ratio_clean": optics_out["ac_dc_ratio"],
            "estimated_snr_db_after_motion": motion_out["estimated_snr_db"],
            "seed": self.seed,
            "notes": prof.notes,
        }

        return SimulationResult(
            ppg=final_ppg,
            fs_hz=fs_output_hz,
            clean_ppg_internal=clean_optical.astype(np.float32),
            fs_internal_hz=FS_INTERNAL_HZ,
            beat_times_s=np.array([b["t_start"] for b in beat_records if b["t_start"] < duration_s]),
            beat_types=[b["beat_type"] for b in beat_records if b["t_start"] < duration_s],
            rhythm_labels=[b["rhythm"] for b in beat_records if b["t_start"] < duration_s],
            hr_instantaneous_bpm=np.array([b["hr_bpm"] for b in beat_records if b["t_start"] < duration_s]),
            stroke_volume_ml=np.array([b["sv_ml"] for b in beat_records if b["t_start"] < duration_s]),
            ejection_fraction=np.array([b["ef"] for b in beat_records if b["t_start"] < duration_s]),
            ptt_s=np.array([b["ptt_s"] for b in beat_records if b["t_start"] < duration_s]),
            augmentation_index=np.array([b["aix"] for b in beat_records if b["t_start"] < duration_s]),
            pwv_m_s=np.array([b["pwv_m_s"] for b in beat_records if b["t_start"] < duration_s]),
            latent_physiology=latent,
            meta=meta,
        )

    # ------------------------------------------------------------------
    def _simulate_beats(self, beats, auto_state, latent, prof: DiseaseProfile):
        t_cursor = 0.0
        prev_ea = 1.1
        prev_p_end = 80.0
        beat_records = []
        blood_fraction_segments = []
        seg_times = []

        for beat in beats:
            local_hr = float(self.autonomic.interp_at(auto_state, np.array([t_cursor]), "hr_bpm")[0])
            local_resistance = float(self.autonomic.interp_at(auto_state, np.array([t_cursor]), "peripheral_resistance")[0])
            local_tone = float(self.autonomic.interp_at(auto_state, np.array([t_cursor]), "vascular_tone")[0])
            local_preload = float(self.autonomic.interp_at(auto_state, np.array([t_cursor]), "preload_state")[0])

            # combine rhythm-driven RR with autonomic HR modulation
            autonomic_scale = latent["hr_bpm"] / max(local_hr, 1e-3)
            rr = beat.rr_s * autonomic_scale
            rr = float(np.clip(rr, 0.18, 3.5))
            instantaneous_hr = 60.0 / rr

            cstate = CardiacState(
                contractility=latent["contractility"] * beat.contractility_scale,
                preload_state=float(np.clip(local_preload + latent["preload_bias"], 0.02, 1.5)),
                afterload_Ea=prev_ea,
                hr_bpm=instantaneous_hr,
                v0_ml=10.0,
                ees_nominal_mmHg_ml=2.5,
            )
            n_samples = max(int(FS_INTERNAL_HZ * rr), 4)
            lv = self.lv_model.beat(cstate, n_samples)
            flow = lv["flow_ml_s"] * beat.sv_scale
            if not beat.perfusion_ok:
                flow = flow * 0.0

            dt = lv["period_s"] / n_samples
            wk_params = params_from_physiology(latent["stiffness"] * (0.85 + 0.3 * local_tone),
                                                latent["resistance"] * local_resistance,
                                                mean_pressure_mmHg=prev_p_end)
            wk = self.windkessel.simulate(flow, dt, wk_params, p0_mmHg=prev_p_end)
            prev_p_end = float(wk["P_aortic_mmHg"][-1])
            prev_ea = wk["effective_arterial_elastance"] if np.isfinite(wk["effective_arterial_elastance"]) else prev_ea

            at_params = ArterialTreeParams(
                pwv0_m_s=4.0,
                age_years=latent["age_years"],
                reflection_coefficient=float(np.clip(0.25 + 0.20 * (latent["stiffness"] - 1.0)
                                                       + 0.15 * (latent["resistance"] - 1.0), 0.05, 0.85)),
            )
            tree = self.arterial_tree.propagate(wk["P_aortic_mmHg"], dt, latent["stiffness"],
                                                 latent["age_years"], wk["mean_pressure_mmHg"], at_params)

            blood_fraction = self.microvasc.pressure_to_blood_fraction(
                tree["radial_pressure_mmHg"], baseline_fraction=0.02,
                vascular_tone=local_tone, compliance_scale=1.0 / max(latent["stiffness"], 0.2),
            )

            t_axis = t_cursor + np.arange(n_samples) * dt
            blood_fraction_segments.append(blood_fraction)
            seg_times.append(t_axis)

            beat_records.append({
                "t_start": t_cursor, "beat_type": beat.beat_type, "rhythm": beat.label_rhythm,
                "hr_bpm": instantaneous_hr, "sv_ml": lv["sv_ml"] * beat.sv_scale, "ef": lv["ef"],
                "ptt_s": tree["ptt_s"], "aix": tree["augmentation_index"], "pwv_m_s": tree["pwv_m_s"],
            })

            t_cursor += rr

        return None, beat_records, blood_fraction_segments, seg_times

    @staticmethod
    def _assemble_continuous(seg_times, segments, t_query, fill_value: float) -> np.ndarray:
        if not segments:
            return np.full_like(t_query, fill_value, dtype=np.float32)
        all_t = np.concatenate(seg_times)
        all_v = np.concatenate(segments)
        order = np.argsort(all_t)
        all_t = all_t[order]
        all_v = all_v[order]
        return np.interp(t_query, all_t, all_v, left=fill_value, right=all_v[-1]).astype(np.float32)