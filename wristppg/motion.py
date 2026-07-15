"""
Motion artifact model: generates an accelerometer-like signal for a given
activity and couples it into the PPG through several physiologically
motivated mechanisms (not just additive noise).

Evidence base
-------------
- Wrist PPG motion artifacts are dominated by activity-synchronized
  quasi-periodic components (arm swing, footstrike) plus broadband
  components from sensor-skin decoupling: Zhang, Pi & Liu, "TROIKA...",
  IEEE TBME 62:522-31 (2015); Biagetti, Crippa, Falaschetti, Saggio &
  Turchetti, "Wrist PPG signal reconstruction... motion artifact
  identification", ICASSP (2018).
- Motion also modulates true perfusion, not just adds noise: muscle
  contraction transiently compresses local vasculature and venous
  pooling/blood displacement occurs with limb position changes relative
  to the heart (hydrostatic effect): Tamura et al. (2014), as cited in
  sensor_pipeline.py.
- Activity-specific frequency content: walking/running cadence
  ~1.5-2.5 Hz (Zhang et al. 2015); cycling pedal cadence ~1.0-1.7 Hz;
  typing/driving produce lower-amplitude, higher-frequency
  micro-vibration rather than a strong periodic component.

What is heuristic here
-----------------------
- The specific harmonic amplitude falloff, bone-vibration coupling
  strength, and per-activity parameter ranges are engineering
  approximations for producing plausible-looking, activity-differentiable
  artifacts, not fit to a labeled accelerometer/PPG dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

ACTIVITY_PROFILES = {
    "walking": dict(freq_range=(1.5, 2.0), n_harm=4, intensity=0.5, bursty=True),
    "running": dict(freq_range=(2.2, 3.0), n_harm=5, intensity=0.85, bursty=True),
    "cycling": dict(freq_range=(1.0, 1.7), n_harm=3, intensity=0.4, bursty=False),
    "typing": dict(freq_range=(3.0, 6.0), n_harm=2, intensity=0.12, bursty=True),
    "driving": dict(freq_range=(0.5, 1.5), n_harm=2, intensity=0.10, bursty=False),
    "sleep": dict(freq_range=(0.05, 0.2), n_harm=1, intensity=0.03, bursty=False),
    "lifting_weights": dict(freq_range=(0.3, 0.8), n_harm=2, intensity=0.7, bursty=True),
    "rest": dict(freq_range=(0.1, 0.3), n_harm=1, intensity=0.02, bursty=False),
}


@dataclass
class MotionEvent:
    activity: str = "rest"
    intensity_scale: float = 1.0  # multiplies the profile's base intensity


class MotionArtifactModel:
    def __init__(self, rng: np.random.Generator, fs_hz: float):
        self.rng = rng
        self.fs = fs_hz

    def accelerometer_signal(self, n_samples: int, event: MotionEvent) -> np.ndarray:
        profile = ACTIVITY_PROFILES.get(event.activity, ACTIVITY_PROFILES["rest"])
        t = np.arange(n_samples) / self.fs
        acc = np.zeros(n_samples)
        base_freq = self.rng.uniform(*profile["freq_range"])
        for h in range(1, profile["n_harm"] + 1):
            phase = self.rng.uniform(0, 2 * np.pi)
            acc += (profile["intensity"] / h) * np.sin(2 * np.pi * base_freq * h * t + phase)

        if profile["bursty"]:
            period = 1.0 / base_freq
            n_cycles = int(n_samples / self.fs / period) + 1
            envelope = np.zeros(n_samples)
            for i in range(n_cycles):
                center = i * period
                width = period * 0.25
                mask = np.exp(-0.5 * ((t - center) / width) ** 2)
                envelope += self.rng.exponential(1.0) * mask
            envelope = envelope / (envelope.max() + 1e-9)
            acc *= (0.4 + 0.6 * envelope)

        acc += self.rng.normal(0, 0.03 * profile["intensity"] + 1e-4, n_samples)  # sensor/bone micro-vibration
        return (acc * event.intensity_scale).astype(np.float32)

    def couple_to_ppg(self, ppg: np.ndarray, accel: np.ndarray, event: MotionEvent) -> dict:
        """Couple the accelerometer-like signal into the PPG via several
        mechanisms:
          - baseline wander (skin deformation / sensor-skin gap changes)
          - amplitude modulation (muscle compression / venous pooling
            transiently altering local perfusion)
          - local morphology distortion (bone vibration adding
            high-frequency ripple during contact transients)
          - stochastic dropout (brief total decoupling during high-g events)
        """
        n = len(ppg)
        profile = ACTIVITY_PROFILES.get(event.activity, ACTIVITY_PROFILES["rest"])
        accel_env = gaussian_filter1d(np.abs(accel), sigma=max(self.fs * 0.05, 1))

        baseline_wander = gaussian_filter1d(accel, sigma=max(self.fs * 0.2, 1)) * 0.5 * np.std(ppg)
        perfusion_mod = 1.0 + 0.20 * profile["intensity"] * np.tanh(accel_env * 2.0) \
            - 0.10 * profile["intensity"] * accel_env  # net: compression can raise or lower local flow transiently
        bone_vibration = accel * 0.15 * np.std(ppg) * (accel_env > 0.3 * (accel_env.max() + 1e-9))

        dropout_mask = np.ones(n)
        if profile["intensity"] > 0.5:
            n_events = self.rng.poisson(profile["intensity"] * n / self.fs / 8.0)
            for _ in range(int(n_events)):
                idx = self.rng.integers(0, n)
                length = int(self.rng.uniform(0.05, 0.3) * self.fs)
                dropout_mask[idx:idx + length] *= self.rng.uniform(0.1, 0.5)

        artifacted = (ppg * perfusion_mod + baseline_wander + bone_vibration) * dropout_mask

        snr_est = float(10 * np.log10((np.var(ppg) + 1e-12) / (np.var(artifacted - ppg) + 1e-12)))

        return {
            "ppg_with_motion": artifacted.astype(np.float32),
            "accelerometer": accel.astype(np.float32),
            "estimated_snr_db": snr_est,
        }