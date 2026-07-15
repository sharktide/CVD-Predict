"""
Continuous autonomic / systemic physiological state generator.

Evidence base
-------------
- Respiratory sinus arrhythmia and its frequency band (0.15-0.4 Hz, "HF"):
  Task Force of ESC/NASPE, "Heart rate variability: standards of
  measurement, physiological interpretation and clinical use",
  Circulation 93:1043-1065 (1996).
- Mayer waves (~0.1 Hz, "LF" band) from baroreflex-sympathetic feedback
  loop delay: Julien, "The enigma of Mayer waves: facts and models",
  Cardiovasc Res 70:12-21 (2006).
- Very-low-frequency (<0.04 Hz) HRV component associated with
  thermoregulation and the renin-angiotensin system: Task Force (1996);
  Kitney, "An analysis of the thermal component of fluctuations in human
  heart rate", 1975.
- Circadian modulation of heart rate/BP (~24h, trough during sleep,
  morning surge): Degaute et al., "Quantitative analysis of the 24-hour
  blood pressure and heart rate patterns in young men", Hypertension
  18:199-210 (1991).
- Orthostatic response (heart rate/BP transient on standing, baroreflex
  buffering): Convertino, "Blood volume: its adaptation to endurance
  training", Med Sci Sports Exerc 23:1338-48 (1991); standard clinical
  orthostatic vitals literature.
- Vasoconstriction/dilation modulating peripheral resistance and pulse
  amplitude at the finger/wrist: Allen & Murray, "Age-related changes in
  peripheral pulse timing...", Physiol Meas 24:297-307 (2003).

What is heuristic here
-----------------------
- All bands are implemented as band-limited stochastic oscillators
  (each a sum of a few sinusoids with slowly randomized phase/frequency
  within its physiological band) rather than derived from a closed-loop
  baroreflex model (e.g., DeBoer or Ursino cardiovascular control
  models). This reproduces realistic *spectral content* but not a
  mechanistic closed feedback loop.
- Stress/exercise-recovery/hydration/blood-volume-shift effects are
  modeled as smooth exogenous multipliers on resistance/compliance/HR
  rather than derived from first-principles fluid balance equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AutonomicState:
    hr_bpm: float
    hrv_frac: float          # fractional HR variability contribution
    peripheral_resistance: float   # multiplier, 1.0 = nominal
    vascular_tone: float           # 0 (fully dilated) - 1 (fully constricted)
    respiration_hz: float
    preload_state: float


class AutonomicSimulator:
    """Generates a continuously evolving vector of autonomic/systemic
    state variables over the recording duration, later sampled
    beat-by-beat by the cardiac/vascular models.
    """

    def __init__(self, rng: np.random.Generator, fs_state_hz: float = 4.0):
        self.rng = rng
        self.fs_state = fs_state_hz

    def _band_oscillator(self, t: np.ndarray, f_lo: float, f_hi: float,
                          n_components: int = 3, amp: float = 1.0) -> np.ndarray:
        sig = np.zeros_like(t)
        for _ in range(n_components):
            f = self.rng.uniform(f_lo, f_hi)
            phase = self.rng.uniform(0, 2 * np.pi)
            # slow amplitude/frequency drift (nonstationarity, as observed
            # physiologically rather than a pure stationary sinusoid)
            drift = 1.0 + 0.3 * np.sin(2 * np.pi * self.rng.uniform(0.001, 0.01) * t + phase)
            sig += np.sin(2 * np.pi * f * t + phase) * drift
        return amp * sig / n_components

    def simulate(self, duration_s: float, base_hr_bpm: float,
                 base_resistance: float = 1.0, base_vascular_tone: float = 0.5,
                 circadian_hour: float | None = None,
                 orthostatic_event_s: float | None = None,
                 exercise_profile: str = "rest") -> dict:
        n = int(duration_s * self.fs_state) + 1
        t = np.arange(n) / self.fs_state

        # HF respiratory band
        resp_hz_center = self.rng.uniform(0.20, 0.28)
        hf = self._band_oscillator(t, 0.15, 0.40, n_components=1, amp=1.0)
        resp_signal = np.sin(2 * np.pi * resp_hz_center * t)

        # LF Mayer-wave band
        lf = self._band_oscillator(t, 0.06, 0.15, n_components=2, amp=1.0)

        # VLF thermoregulatory / RAS band
        vlf = self._band_oscillator(t, 0.01, 0.04, n_components=2, amp=1.0)

        # Circadian modulation (24h cosine), only meaningful for long
        # recordings; for short recordings this is nearly a constant
        # offset determined by time-of-day.
        if circadian_hour is None:
            circadian_hour = 12.0
        hour_t = (circadian_hour + t / 3600.0) % 24.0
        circadian = -0.10 * np.cos(2 * np.pi * (hour_t - 5.0) / 24.0)  # trough ~05:00, peak ~17:00

        # Stress response: slow random-walk-ish sympathetic tone in [0,1]
        stress_walk = np.cumsum(self.rng.normal(0, 0.002, n))
        stress = 1 / (1 + np.exp(-3 * (stress_walk - np.median(stress_walk))))
        stress = 0.5 + 0.5 * (stress - stress.mean()) / (stress.std() + 1e-6) * 0.3

        # Exercise / recovery envelope
        exercise_gain = self._exercise_envelope(t, exercise_profile)

        # Orthostatic transient: brief HR/resistance perturbation at a
        # given time (standing up), with baroreflex-buffered recovery
        # over ~30-60 s.
        orthostatic = np.zeros(n)
        if orthostatic_event_s is not None:
            tau = 20.0
            step = np.clip((t - orthostatic_event_s) / 1.0, 0, 1)
            recovery = np.exp(-np.clip(t - orthostatic_event_s, 0, None) / tau)
            orthostatic = step * (0.25 * recovery + 0.05 * (1 - recovery))

        hrv_composite = 0.5 * hf + 0.3 * lf + 0.2 * vlf
        hr_bpm = base_hr_bpm * (1.0 + 0.02 * hrv_composite + circadian + 0.35 * (exercise_gain - 0.0)
                                 + 0.10 * (stress - 0.5) + orthostatic)
        hr_bpm = np.clip(hr_bpm, 30, 220)

        vascular_tone = np.clip(base_vascular_tone + 0.25 * (stress - 0.5) - 0.30 * exercise_gain
                                 + 0.15 * orthostatic - 0.05 * vlf, 0.0, 1.0)
        peripheral_resistance = base_resistance * (1.0 + 0.6 * (vascular_tone - 0.5) - 0.25 * exercise_gain)
        peripheral_resistance = np.clip(peripheral_resistance, 0.2, 4.0)

        preload_state = np.clip(0.55 + 0.20 * exercise_gain - 0.30 * orthostatic
                                 + 0.05 * lf - 0.10 * (stress - 0.5), 0.05, 1.4)

        respiration_hz = resp_hz_center * (1.0 + 0.5 * exercise_gain)

        return {
            "t_s": t,
            "hr_bpm": hr_bpm.astype(np.float32),
            "peripheral_resistance": peripheral_resistance.astype(np.float32),
            "vascular_tone": vascular_tone.astype(np.float32),
            "preload_state": preload_state.astype(np.float32),
            "respiration_hz": respiration_hz.astype(np.float32),
            "resp_signal": resp_signal.astype(np.float32),
            "exercise_gain": exercise_gain.astype(np.float32),
            "stress": stress.astype(np.float32),
            "fs_state": self.fs_state,
        }

    def _exercise_envelope(self, t: np.ndarray, profile: str) -> np.ndarray:
        dur = t[-1] if len(t) else 1.0
        if profile == "rest":
            return np.zeros_like(t)
        if profile == "exercise":
            # ramp up, plateau, ramp down (recovery), smoothed
            ramp = np.clip(t / max(dur * 0.15, 1e-3), 0, 1)
            decay = np.clip((dur - t) / max(dur * 0.30, 1e-3), 0, 1)
            return np.minimum(ramp, decay) * 0.9
        if profile == "recovery":
            return np.exp(-t / max(dur * 0.4, 1e-3)) * 0.7
        return np.zeros_like(t)

    @staticmethod
    def interp_at(state: dict, t_query_s: np.ndarray, key: str) -> np.ndarray:
        return np.interp(t_query_s, state["t_s"], state[key])