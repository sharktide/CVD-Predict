"""
WristPPGSimulator: orchestrates the full pipeline.

Updated to wire together all new modules:
- Wrist anatomy (optics.WristAnatomy)
- Ambient light contamination (optics ambient_light_fraction)
- Skin temperature effects (optics, microvasculature)
- Closed-loop autonomic (autonomic.AutonomicSimulator with baroreflex)
- Cardiac arrest physiology (disease profiles with ischemia/SPO2/temp)
- Wrist-specific microvasculature (ischemia, reperfusion, venous pooling)
- Realistic motion (3-axis gravity, posture, strap dynamics)
- Wrist contact (ambient leakage, strap type, cardiac arrest recovery)

See evidence-base docstrings in each component module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .cardiac import CardiacState, LeftVentricleModel
from .windkessel import WindkesselModel, params_from_physiology
from .arterial_tree import ArterialTreeModel, ArterialTreeParams
from .autonomic import AutonomicSimulator
from .arrhythmia import RhythmGenerator, ArrhythmiaConfig
from .optics import SkinOpticalModel, SkinOpticalParams, WristAnatomy
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
    accel: np.ndarray
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
            physiological variable.
        """
        prof = PROFILES[profile] if isinstance(profile, str) else profile
        latent = prof.sample(self.rng)
        if latent_overrides:
            latent.update(latent_overrides)

        self.sensor_pipeline.params.fs_output_hz = fs_output_hz

        exercise_profile = exercise_profile or ("exercise" if activity in ("running", "lifting_weights") else "rest")

        # --- Wrist anatomy ---
        wrist_anatomy = WristAnatomy(
            radial_artery_depth_mm=self.rng.uniform(1.0, 3.0),
            subcutaneous_fat_mm=self.rng.uniform(2.0, 8.0),
            source_detector_distance_mm=self.rng.uniform(3.0, 7.0),
        )

        # --- Determine ambient light from contact_mode ---
        ambient_light = 0.0
        if contact_mode in ("loose", "rolling"):
            ambient_light = self.rng.uniform(0.1, 0.4)
        elif contact_mode == "partial_lift":
            ambient_light = self.rng.uniform(0.3, 0.7)

        # --- Skin temperature ---
        # Core 36.5, wrist 2-5C cooler depending on environment
        skin_temp = 36.5 - self.rng.uniform(2.0, 5.0)

        # --- Determine cardiac arrest state for microvasculature ---
        is_cardiac_arrest = latent.get("rhythm") in ("vf", "asystole", "agonal")
        arrest_start_s = self.rng.uniform(5.0, duration_s - 10.0) if is_cardiac_arrest else None

        # --- Determine ischemia parameters ---
        # SpO2 from disease profile
        target_spo2 = latent.get("spo2", 0.98)

        # --- Autonomic simulation ---
        auto_state = self.autonomic.simulate(
            duration_s=duration_s + 5.0,
            base_hr_bpm=latent["hr_bpm"],
            base_resistance=latent["resistance"],
            base_vascular_tone=latent["vascular_tone"],
            circadian_hour=circadian_hour,
            orthostatic_event_s=orthostatic_event_s,
            exercise_profile=exercise_profile,
            hrv_frac=latent.get("hrv_frac", 0.5),
            map_setpoint_mmHg=latent.get("map_setpoint_mmHg", 93.0),
        )

        # --- Rhythm generation ---
        rhythm_cfg = ArrhythmiaConfig(rhythm=latent["rhythm"], base_hr_bpm=latent["hr_bpm"],
                                       duration_s=duration_s + 5.0, rng=self.rng)
        beats = self.rhythm_gen.generate(rhythm_cfg)

        # --- Beat-by-beat simulation ---
        beat_times, beat_records, blood_fraction_segments, seg_times = self._simulate_beats(
            beats, auto_state, latent, prof, wrist_anatomy
        )

        # --- Assemble continuous blood fraction ---
        n_internal = int(duration_s * FS_INTERNAL_HZ)
        t_internal = np.arange(n_internal) / FS_INTERNAL_HZ
        blood_fraction_full = self._assemble_continuous(
            seg_times, blood_fraction_segments, t_internal,
            fill_value=latent.get("tissue_blood_fraction", 0.02)
        )

        # --- Compute overall perfusion scale for assembled waveform ---
        # During cardiac arrest, the assembled waveform retains interpolation
        # artifacts between tiny segments.  We apply a global perfusion_scale
        # to the assembled blood fraction to ensure the PPG goes flat.
        is_arrhythmic_profile = latent.get("rhythm") in ("vf", "asystole", "agonal")
        if is_arrhythmic_profile:
            blood_fraction_full = blood_fraction_full * 0.01 + latent.get("tissue_blood_fraction", 0.02) * 0.99

        # --- Optical model with wrist anatomy and ambient light ---
        skin_params = SkinOpticalParams(
            melanin_fraction=latent["melanin_fraction"],
            spo2=target_spo2,
            venous_spo2=0.70,
            tissue_blood_fraction=latent.get("tissue_blood_fraction", 0.025),
            pulsatile_fraction=latent.get("pulsatile_fraction", 0.008),
            hair_density=self.rng.uniform(0.0, 0.3),
            tattoo_optical_density=0.0,
            sweat_layer=0.0 if activity == "rest" else self.rng.uniform(0.0, 0.3),
            contact_pressure=0.5,
            body_temp_c=latent.get("body_temp_c", 36.5),
            wrist_anatomy=wrist_anatomy,
            ambient_light_fraction=ambient_light,
            skin_temperature_c=skin_temp,
        )
        optics_out = self.optics.generate_ppg_from_blood_volume(
            wavelength, skin_params, blood_fraction_full
        )
        clean_optical = optics_out["intensity"]

        # Scale clean optical signal so it's in the millivolt range
        # (real photodiode output is ~0.1-2V, our optics model outputs ~0.001)
        # Without this, sensor pipeline noise swamps the signal after AGC
        clean_optical = clean_optical * 500.0

        # --- Motion model (3-axis accelerometer) ---
        motion_event = MotionEvent(activity=activity, intensity_scale=1.0)
        accel = self.motion_model.accelerometer_signal(n_internal, motion_event)

        # --- Contact model (wrist-specific with ambient leakage) ---
        contact_state = ContactState(mode=contact_mode, posture="upright")
        contact_out = self.contact_model.coupling_trace(
            n_internal, contact_state, motion_energy=np.linalg.norm(accel, axis=1)
        )

        # --- Apply contact coupling with ambient light leakage ---
        # Ambient light is additive, not multiplicative
        ambient_level = float(np.mean(clean_optical)) * contact_out["ambient_leakage"]
        coupled = clean_optical * contact_out["efficiency"] + ambient_level

        # --- Apply motion artifacts ---
        motion_out = self.motion_model.couple_to_ppg(coupled, accel, motion_event)
        motioned = motion_out["ppg_with_motion"]

        # --- Add electronic/sensor noise ---
        noise_params = NoiseParams()
        noisy = self.noise_model.apply(
            motioned, FS_INTERNAL_HZ, noise_params,
            melanin_fraction=latent["melanin_fraction"], device_age_days=0.0
        )

        # --- Sensor pipeline (demodulation, filtering, downsampling) ---
        sensor_out = self.sensor_pipeline.run(noisy, FS_INTERNAL_HZ)
        final_ppg = sensor_out["raw_sensor_output"]

        # --- Resample accel to output rate ---
        n_out = len(final_ppg)
        t_out = np.linspace(0, t_internal[-1], n_out)
        accel_resampled = np.zeros((n_out, 3), dtype=np.float32)
        for axis in range(3):
            accel_resampled[:, axis] = np.interp(t_out, t_internal, accel[:, axis]).astype(np.float32)

        meta = {
            "profile": prof.name,
            "activity": activity,
            "contact_mode": contact_mode,
            "wavelength": wavelength,
            "ac_dc_ratio_clean": optics_out["ac_dc_ratio"],
            "estimated_snr_db_after_motion": motion_out["estimated_snr_db"],
            "seed": self.seed,
            "notes": prof.notes,
            "wrist_anatomy": {
                "radial_artery_depth_mm": wrist_anatomy.radial_artery_depth_mm,
                "subcutaneous_fat_mm": wrist_anatomy.subcutaneous_fat_mm,
                "source_detector_distance_mm": wrist_anatomy.source_detector_distance_mm,
            },
            "ambient_light_fraction": ambient_light,
            "skin_temperature_c": skin_temp,
            "spo2": target_spo2,
            "body_temp_c": latent.get("body_temp_c", 36.5),
        }

        return SimulationResult(
            ppg=final_ppg,
            accel=accel_resampled,
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
    def _simulate_beats(self, beats, auto_state, latent, prof: DiseaseProfile,
                         wrist_anatomy: WristAnatomy):
        """Simulate beat-by-beat hemodynamics with wrist-specific parameters."""
        t_cursor = 0.0
        prev_ea = 1.1
        prev_p_end = 80.0
        beat_records = []
        blood_fraction_segments = []
        seg_times = []

        # Compute baseline microvascular integrity from MAP
        map_target = latent.get("map_target_mmHg", 85.0)
        micro_integrity = self.microvasc.compute_microvascular_integrity(map_target)

        # Wrist-specific baseline (higher than finger due to shallow artery)
        baseline_bf = 0.025 * (1.0 + 0.5 * (1.0 - wrist_anatomy.subcutaneous_fat_mm / 8.0))

        for beat in beats:
            local_hr = float(self.autonomic.interp_at(auto_state, np.array([t_cursor]), "hr_bpm")[0])
            local_resistance = float(self.autonomic.interp_at(auto_state, np.array([t_cursor]), "peripheral_resistance")[0])
            local_tone = float(self.autonomic.interp_at(auto_state, np.array([t_cursor]), "vascular_tone")[0])
            local_preload = float(self.autonomic.interp_at(auto_state, np.array([t_cursor]), "preload_state")[0])

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
                reflection_coefficient=float(np.clip(
                    0.25 + 0.20 * (latent["stiffness"] - 1.0) + 0.15 * (latent["resistance"] - 1.0),
                    0.05, 0.85
                )),
            )
            tree = self.arterial_tree.propagate(
                wk["P_aortic_mmHg"], dt, latent["stiffness"],
                latent["age_years"], wk["mean_pressure_mmHg"], at_params
            )

            # Compute ischemia state for this beat
            is_arrhythmic = beat.beat_type in ("non_perfusion", "vf", "asystole", "agonal")
            ischemia_state = np.ones(n_samples) * (0.8 if is_arrhythmic else 0.0)
            ischemia_dur = np.ones(n_samples) * (t_cursor if is_arrhythmic else 0.0)

            # Perfusion scale: zero when heart isn't pumping
            perf_scale = 0.01 if not beat.perfusion_ok else 1.0

            # Venous pooling from posture
            posture = "upright"
            venous_pooling = self.microvasc.compute_venous_pooling(posture, t_cursor)

            blood_fraction = self.microvasc.pressure_to_blood_fraction(
                tree["radial_pressure_mmHg"],
                baseline_fraction=baseline_bf,
                vascular_tone=local_tone,
                compliance_scale=1.0 / max(latent["stiffness"], 0.2),
                ischemia_state=ischemia_state,
                ischemia_duration_s=ischemia_dur,
                venous_pooling_fraction=venous_pooling,
                microvascular_integrity=micro_integrity,
                perfusion_scale=perf_scale,
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
