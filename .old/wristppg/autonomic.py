"""
Continuous autonomic / systemic physiological state generator with
closed-loop baroreflex model.

This module implements a baroreflex-mediated cardiovascular control model
that couples heart rate, vascular tone, and peripheral resistance through
a negative feedback loop, replacing the open-loop stochastic oscillators
of the previous version. The baroreflex is the primary short-term
blood pressure regulatory mechanism and is essential for generating
realistic HRV, blood pressure variability, and hemodynamic responses
to perturbation.

Evidence base
-------------
- Baroreflex physiology: Mancia & Grassi, "The autonomic nervous system
  and hypertension", Eur Heart J Suppl 16:A1-A5 (2014); Seifert et al.,
  "Baroreflex: a novel target for therapeutic intervention in
  cardiovascular disease", J Am Heart Assoc (2023).
- DeBoer model of cardiovascular control: DeBoer, Karemaker & Strackee,
  "Hemodynamic fluctuations and baroreflex sensitivity in humans: a
  beat-to-beat model", Am J Physiol 253:H680-H689 (1987).
- Respiratory sinus arrhythmia (HF, 0.15-0.4 Hz): Task Force of
  ESC/NASPE, "Heart rate variability: standards of measurement,
  physiological interpretation and clinical use", Circulation
  93:1043-1065 (1996).
- Mayer waves (~0.1 Hz, LF band): Julien, "The enigma of Mayer waves:
  facts and models", Cardiovasc Res 70:12-21 (2006).
- VLF (<0.04 Hz) thermoregulatory/RAS: Task Force (1996); Kitney (1975).
- HRV loss in disease: reduced HRV is an independent predictor of
  mortality after MI and in heart failure: Kleiger et al., "Decreased
  heart rate variability and increased risk of mortality after MI",
  Circulation 76:732-741 (1987).

What is heuristic here
-----------------------
- The baroreflex gain is parameterized as a linear/saturating function
  rather than a full nonlinear reflex arc model. The gain is modulated
  by the disease profile's hrv_frac parameter, coupling disease state
  to HRV amplitude.
- Respiratory coupling is implemented as a phase-modulated oscillator
  (not a full respiratory pump model).
- Thermoregulatory VLF and circadian modulation remain as exogenous
  signals (not derived from core-temperature feedback).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class AutonomicState:
    hr_bpm: float
    hrv_frac: float          # fractional HR variability contribution
    peripheral_resistance: float   # multiplier, 1.0 = nominal
    vascular_tone: float           # 0 (fully dilated) - 1 (fully constricted)
    respiration_hz: float
    preload_state: float
    mean_arterial_pressure: float  # mmHg, used by baroreflex feedback


class AutonomicSimulator:
    """Generates a continuously evolving vector of autonomic/systemic
    state variables over the recording duration, incorporating a
    closed-loop baroreflex model for realistic cardiovascular coupling.

    The baroreflex provides negative feedback: when MAP rises above the
    setpoint, the reflex reduces HR and causes vasodilation; when MAP
    falls, HR increases and vasoconstriction occurs. The gain of this
    reflex is modulated by the hrv_frac parameter from disease profiles.
    """

    def __init__(self, rng: np.random.Generator, fs_state_hz: float = 4.0):
        self.rng = rng
        self.fs_state = fs_state_hz

    def _band_oscillator(self, t: np.ndarray, f_lo: float, f_hi: float,
                          n_components: int = 3, amp: float = 1.0,
                          hrv_scale: float = 1.0) -> np.ndarray:
        sig = np.zeros_like(t)
        for _ in range(n_components):
            f = self.rng.uniform(f_lo, f_hi)
            phase = self.rng.uniform(0, 2 * np.pi)
            drift = 1.0 + 0.3 * np.sin(2 * np.pi * self.rng.uniform(0.001, 0.01) * t + phase)
            sig += np.sin(2 * np.pi * f * t + phase) * drift
        return amp * hrv_scale * sig / n_components

    def simulate(self, duration_s: float, base_hr_bpm: float,
                 base_resistance: float = 1.0, base_vascular_tone: float = 0.5,
                 circadian_hour: float | None = None,
                 orthostatic_event_s: float | None = None,
                 exercise_profile: str = "rest",
                 hrv_frac: float = 0.15,
                 map_setpoint_mmHg: float = 93.0) -> dict:
        """Simulate the autonomic state with baroreflex feedback.

        Parameters
        ----------
        hrv_frac : float
            Disease-modulated HRV fraction. Scales HF/LF/VLF band
            amplitudes. Higher = more HRV (healthy), lower = reduced
            HRV (disease/cardiac arrest).
        map_setpoint_mmHg : float
            Baroreflex arterial pressure setpoint (mmHg).
        """
        n = int(duration_s * self.fs_state) + 1
        dt = 1.0 / self.fs_state
        t = np.arange(n) / self.fs_state

        # === HRV band amplitudes, scaled by hrv_frac ===
        # hrv_frac maps to physiological HRV:
        #   0.0-0.05: cardiac arrest / severe disease (near-zero HRV)
        #   0.05-0.15: moderate disease (reduced HRV)
        #   0.15-0.30: healthy (normal HRV)
        #   0.30-0.45: high parasympathetic tone (athlete / deep sleep)
        hf_amp = 0.3 + 0.7 * hrv_scale(hrv_frac)
        lf_amp = 0.2 + 0.6 * hrv_scale(hrv_frac)
        vlf_amp = 0.1 + 0.4 * hrv_scale(hrv_frac)

        # === Respiratory signal (HF band driver) ===
        resp_hz_center = self.rng.uniform(0.20, 0.28)
        hf = self._band_oscillator(t, 0.15, 0.40, n_components=1, amp=hf_amp, hrv_scale=1.0)
        resp_signal = np.sin(2 * np.pi * resp_hz_center * t)

        # === LF Mayer-wave band ===
        lf = self._band_oscillator(t, 0.06, 0.15, n_components=2, amp=lf_amp, hrv_scale=1.0)

        # === VLF thermoregulatory / RAS band ===
        vlf = self._band_oscillator(t, 0.01, 0.04, n_components=2, amp=vlf_amp, hrv_scale=1.0)

        # === Circadian modulation ===
        if circadian_hour is None:
            circadian_hour = 12.0
        hour_t = (circadian_hour + t / 3600.0) % 24.0
        circadian = -0.10 * np.cos(2 * np.pi * (hour_t - 5.0) / 24.0)

        # === Stress response ===
        stress_walk = np.cumsum(self.rng.normal(0, 0.002, n))
        stress = 1 / (1 + np.exp(-3 * (stress_walk - np.median(stress_walk))))
        stress = 0.5 + 0.5 * (stress - stress.mean()) / (stress.std() + 1e-6) * 0.3

        # === Exercise envelope ===
        exercise_gain = self._exercise_envelope(t, exercise_profile)

        # === Baroreflex closed-loop simulation ===
        # Initialize state variables for Euler integration
        hr_trace = np.full(n, base_hr_bpm, dtype=np.float32)
        resistance_trace = np.full(n, base_resistance, dtype=np.float32)
        tone_trace = np.full(n, base_vascular_tone, dtype=np.float32)
        map_trace = np.full(n, map_setpoint_mmHg, dtype=np.float32)
        preload_trace = np.full(n, 0.55, dtype=np.float32)

        # Baroreflex gain (scales with hrv_frac)
        # Higher hrv_frac = stronger reflex = more HRV
        baroreflex_gain_hr = 0.5 * hrv_scale(hrv_frac)      # beats/min per mmHg
        baroreflex_gain_resistance = 0.008 * hrv_scale(hrv_frac)  # per mmHg
        baroreflex_time_constant = 4.0  # seconds, reflex delay + effector time

        # Orthostatic transient
        orthostatic = np.zeros(n, dtype=np.float32)
        if orthostatic_event_s is not None:
            tau = 20.0
            step = np.clip((t - orthostatic_event_s) / 1.0, 0, 1)
            recovery = np.exp(-np.clip(t - orthostatic_event_s, 0, None) / tau)
            orthostatic = (step * (0.25 * recovery + 0.05 * (1 - recovery))).astype(np.float32)

        for i in range(1, n):
            # --- Compute current MAP from hemodynamic state ---
            # MAP ~ CO * SVR, where CO ~ HR * SV
            # We use a simplified model: MAP = base_MAP * (HR/base_HR) * (SVR/base_SVR) + perturbations
            hr_ratio = hr_trace[i-1] / max(base_hr_bpm, 1.0)
            svr_ratio = resistance_trace[i-1] / max(base_resistance, 1.0)
            # Add respiratory, LF, VLF contributions to MAP
            map_perturbation = (2.0 * hf[i] + 1.5 * lf[i] + 1.0 * vlf[i])
            current_map = map_setpoint_mmHg * hr_ratio * svr_ratio * (1.0 + 0.01 * map_perturbation)

            # Exercise increases MAP via CO
            current_map *= (1.0 + 0.15 * exercise_gain[i])

            # Orthostatic: standing drops venous return -> drops MAP initially
            current_map *= (1.0 - 0.20 * orthostatic[i])

            map_trace[i] = current_map

            # --- Baroreflex error signal ---
            map_error = current_map - map_setpoint_mmHg

            # Negative feedback: high MAP -> decrease HR, dilate vessels
            hr_correction = -baroreflex_gain_hr * map_error
            resistance_correction = -baroreflex_gain_resistance * map_error

            # === Apply corrections with physiological limits ===
            hr_bpm = base_hr_bpm * (1.0 + circadian[i] + 0.35 * exercise_gain[i]
                                      + 0.10 * (stress[i] - 0.5) + orthostatic[i])
            hr_bpm += hr_correction  # baroreflex correction
            hr_trace[i] = float(np.clip(hr_bpm, 25, 220))

            vascular_tone = base_vascular_tone + 0.25 * (stress[i] - 0.5) - 0.30 * exercise_gain[i] \
                + 0.15 * orthostatic[i] - 0.05 * vlf[i]
            vascular_tone = float(np.clip(vascular_tone, 0.0, 1.0))
            tone_trace[i] = vascular_tone

            resistance = base_resistance * (1.0 + 0.6 * (vascular_tone - 0.5) - 0.25 * exercise_gain[i])
            resistance += resistance_correction
            resistance_trace[i] = float(np.clip(resistance, 0.2, 4.0))

            # Preload: exercise increases venous return, standing decreases it
            preload = 0.55 + 0.20 * exercise_gain[i] - 0.30 * orthostatic[i] \
                + 0.05 * lf[i] - 0.10 * (stress[i] - 0.5)
            preload_trace[i] = float(np.clip(preload, 0.05, 1.4))

        respiration_hz = resp_hz_center * (1.0 + 0.5 * exercise_gain)

        return {
            "t_s": t,
            "hr_bpm": hr_trace,
            "peripheral_resistance": resistance_trace,
            "vascular_tone": tone_trace,
            "preload_state": preload_trace,
            "respiration_hz": respiration_hz,
            "resp_signal": resp_signal.astype(np.float32),
            "exercise_gain": exercise_gain.astype(np.float32),
            "stress": stress.astype(np.float32),
            "mean_arterial_pressure": map_trace,
            "fs_state": self.fs_state,
            "hrv_frac_used": hrv_frac,
        }

    def _exercise_envelope(self, t: np.ndarray, profile: str) -> np.ndarray:
        dur = t[-1] if len(t) else 1.0
        if profile == "rest":
            return np.zeros_like(t)
        if profile == "exercise":
            ramp = np.clip(t / max(dur * 0.15, 1e-3), 0, 1)
            decay = np.clip((dur - t) / max(dur * 0.30, 1e-3), 0, 1)
            return np.minimum(ramp, decay) * 0.9
        if profile == "recovery":
            return np.exp(-t / max(dur * 0.4, 1e-3)) * 0.7
        return np.zeros_like(t)

    @staticmethod
    def interp_at(state: dict, t_query_s: np.ndarray, key: str) -> np.ndarray:
        return np.interp(t_query_s, state["t_s"], state[key])


def hrv_scale(hrv_frac: float) -> float:
    """Map hrv_frac (0-1) to a multiplicative HRV amplitude scale.

    The mapping is nonlinear: near-zero hrv_frac (cardiac arrest) gives
    near-zero HRV amplitude, while normal hrv_frac (~0.15-0.30) gives
    full amplitude. This creates a realistic progression from pathological
    flatline HRV to healthy variability.
    """
    # Sigmoidal mapping: 50% at hrv_frac=0.10, ~100% at hrv_frac=0.25
    return float(1.0 / (1.0 + np.exp(-20.0 * (hrv_frac - 0.10))))
