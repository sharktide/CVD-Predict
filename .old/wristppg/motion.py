"""
Motion artifact model: generates 3-axis wrist accelerometer signals with
realistic dynamics (gravity vector, posture, gait biomechanics, strap
coupling) and couples them into the PPG through physiologically motivated
mechanisms.

Evidence base
-------------
- Wrist PPG motion artifacts: Zhang, Pi & Liu, "TROIKA...", IEEE TBME
  62:522-31 (2015); Biagetti et al., "Wrist PPG signal reconstruction
  and motion artifact identification", ICASSP (2018).
- Wrist accelerometer characteristics: Apple Watch Series 4-9 measure
  ±2g/±4g/±8g/±16g at 100 Hz; typical SNR ~10-12 bits effective;
  gravity vector always present at ~9.81 m/s^2.
- Gait biomechanics: walking cadence 1.5-2.5 Hz (Zhao, "Wearable
  inertial measurement units...", Sensors 19:3351 (2019)); running
  2.0-3.5 Hz; vertical impact transient at heel strike.
- Arm swing during locomotion: 0.3-0.7 m/s peak velocity for walking,
  with characteristic asymmetric pattern (dominant arm swings less):
  Perry, "Gait Analysis: Normal and Pathological Function" (2010).
- Motion artifact coupling to PPG: perfusion modulation from muscle
  compression and venous pooling (Tamura et al., 2014); baseline wander
  from skin-sensor gap changes; stochastic dropout during high-g events
  (Zhang et al., 2015).
- Wrist strap dynamics: loose strap allows relative motion between
  sensor and skin; contact pressure varies with wrist posture and
  movement amplitude: "Characterization of PPG sensor contact", IEEE
  EMBS (2017).

What is heuristic here
-----------------------
- Activity-specific frequency profiles are parameterized from published
  cadence ranges but not fit to a labeled wrist IMU dataset.
- Strap dynamics are modeled as a second-order mechanical system with
  tunable damping, rather than derived from actual watch-strap mass/
  stiffness measurements.
- Posture effects use a simplified tilt model rather than full IMU
  fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

ACTIVITY_PROFILES = {
    "walking": dict(
        freq_range=(1.5, 2.2), n_harm=5, intensity=0.55,
        bursty=True, gravity_oscillation=True,
        arm_swing_amplitude=0.4, vertical_impact=0.3,
    ),
    "running": dict(
        freq_range=(2.2, 3.2), n_harm=6, intensity=0.90,
        bursty=True, gravity_oscillation=True,
        arm_swing_amplitude=0.6, vertical_impact=0.5,
    ),
    "cycling": dict(
        freq_range=(1.0, 1.7), n_harm=4, intensity=0.35,
        bursty=False, gravity_oscillation=False,
        arm_swing_amplitude=0.1, vertical_impact=0.05,
    ),
    "typing": dict(
        freq_range=(3.0, 6.0), n_harm=3, intensity=0.10,
        bursty=True, gravity_oscillation=False,
        arm_swing_amplitude=0.02, vertical_impact=0.02,
    ),
    "driving": dict(
        freq_range=(0.5, 1.5), n_harm=2, intensity=0.08,
        bursty=False, gravity_oscillation=False,
        arm_swing_amplitude=0.03, vertical_impact=0.03,
    ),
    "sleep": dict(
        freq_range=(0.05, 0.2), n_harm=1, intensity=0.02,
        bursty=False, gravity_oscillation=False,
        arm_swing_amplitude=0.01, vertical_impact=0.01,
    ),
    "lifting_weights": dict(
        freq_range=(0.3, 0.8), n_harm=3, intensity=0.75,
        bursty=True, gravity_oscillation=False,
        arm_swing_amplitude=0.2, vertical_impact=0.4,
    ),
    "rest": dict(
        freq_range=(0.1, 0.3), n_harm=1, intensity=0.02,
        bursty=False, gravity_oscillation=False,
        arm_swing_amplitude=0.01, vertical_impact=0.01,
    ),
    "seizure": dict(
        freq_range=(3.0, 8.0), n_harm=4, intensity=0.85,
        bursty=True, gravity_oscillation=True,
        arm_swing_amplitude=0.5, vertical_impact=0.6,
    ),
    "cpr_chest_compressions": dict(
        freq_range=(1.5, 2.5), n_harm=3, intensity=0.70,
        bursty=True, gravity_oscillation=True,
        arm_swing_amplitude=0.3, vertical_impact=0.5,
    ),
}


@dataclass
class MotionEvent:
    activity: str = "rest"
    intensity_scale: float = 1.0
    posture: str = "upright"  # upright, supine, prone, left_lateral, right_lateral
    strap_tightness: float = 0.7  # 0 (very loose) - 1 (very tight)


@dataclass
class StrapDynamics:
    """Second-order mechanical model of watch-strap motion relative to wrist."""
    mass_kg: float = 0.05        # watch mass (~50g for Apple Watch)
    spring_n_m: float = 50.0     # strap stiffness (N/m)
    damping_ratio: float = 0.3   # underdamped (typical for fabric/fluoroelastomer strap)
    natural_freq_hz: float = 0.0  # computed from mass/spring


class MotionArtifactModel:
    def __init__(self, rng: np.random.Generator, fs_hz: float):
        self.rng = rng
        self.fs = fs_hz

    def accelerometer_signal(self, n_samples: int, event: MotionEvent) -> np.ndarray:
        """Generate 3-axis wrist accelerometer signal with realistic dynamics.

        Returns ndarray of shape (n_samples, 3) with axes [x, y, z].
        - x: primary motion axis (arm swing direction / anteroposterior)
        - y: secondary axis (mediolateral / lateral)
        - z: vertical (includes gravity component, always ~9.81 m/s^2)
        """
        profile = ACTIVITY_PROFILES.get(event.activity, ACTIVITY_PROFILES["rest"])
        t = np.arange(n_samples) / self.fs
        base_freq = self.rng.uniform(*profile["freq_range"])

        # === Gravity vector ===
        # Always present; orientation depends on posture
        gravity_magnitude = 9.81  # m/s^2
        gravity_orientation = self._posture_to_gravity(event.posture)

        # === Generate 3-axis accelerometer ===
        acc = np.zeros((n_samples, 3), dtype=np.float64)

        # Gravity component (constant + slow oscillation from posture changes)
        for axis in range(3):
            acc[:, axis] = gravity_magnitude * gravity_orientation[axis]

        # === Activity-specific motion ===
        # Primary motion axis (x = arm swing)
        motion_primary = np.zeros(n_samples)
        for h in range(1, profile["n_harm"] + 1):
            phase = self.rng.uniform(0, 2 * np.pi)
            # Harmonic amplitude falloff: 1/h with slight randomization
            amp = profile["intensity"] * (1.0 / (h ** 0.8))
            motion_primary += amp * np.sin(2 * np.pi * base_freq * h * t + phase)

        # Bursty envelope for gait-like activities (footstrike timing)
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
            motion_primary *= (0.4 + 0.6 * envelope)

        # Add broadband noise (sensor noise + microvibrations)
        motion_primary += self.rng.normal(0, 0.02 * profile["intensity"] + 1e-4, n_samples)
        motion_primary *= event.intensity_scale

        # === Decompose into 3 axes with realistic correlations ===
        acc[:, 0] += motion_primary  # x: primary arm swing
        acc[:, 1] += 0.3 * motion_primary + self.rng.normal(0, 0.015 * profile["intensity"] + 1e-4, n_samples)
        # z: vertical component includes impact transients for gait
        acc[:, 2] += 0.15 * motion_primary + self.rng.normal(0, 0.01 * profile["intensity"] + 1e-4, n_samples)

        # Add vertical impact transients (heel strike for walking/running)
        if profile["vertical_impact"] > 0.1:
            impact = self._generate_impact_transients(t, base_freq, profile["vertical_impact"])
            acc[:, 2] += impact * event.intensity_scale

        # Add gravity oscillation from wrist tilt during arm swing
        if profile["gravity_oscillation"]:
            tilt_angle = profile["arm_swing_amplitude"] * np.sin(2 * np.pi * base_freq * t)
            acc[:, 2] += gravity_magnitude * 0.1 * np.sin(tilt_angle)
            acc[:, 0] += gravity_magnitude * 0.05 * np.cos(tilt_angle)

        # === Apply strap dynamics (second-order mechanical filter) ===
        acc = self._apply_strap_dynamics(acc, event.strap_tightness, profile)

        return acc.astype(np.float32)

    def _posture_to_gravity(self, posture: str) -> tuple:
        """Convert posture to gravity vector components [x, y, z]."""
        orientations = {
            "upright": (0.0, 0.0, 1.0),
            "supine": (0.0, 1.0, 0.0),
            "prone": (0.0, -1.0, 0.0),
            "left_lateral": (1.0, 0.0, 0.0),
            "right_lateral": (-1.0, 0.0, 0.0),
        }
        return orientations.get(posture, (0.0, 0.0, 1.0))

    def _generate_impact_transients(self, t: np.ndarray, cadence_hz: float,
                                     amplitude: float) -> np.ndarray:
        """Generate heel-strike-like impact transients on the z-axis."""
        impact = np.zeros_like(t)
        period = 1.0 / cadence_hz
        n_cycles = int(t[-1] / period) + 1
        for i in range(n_cycles):
            center = i * period
            # Sharp spike (heel strike) followed by dampened oscillation
            dt = t - center
            mask = (dt > 0) & (dt < 0.15)
            impact[mask] += amplitude * np.exp(-30 * dt[mask]) * np.sin(2 * np.pi * 40 * dt[mask])
        return impact

    def _apply_strap_dynamics(self, acc: np.ndarray, strap_tightness: float,
                               profile: dict) -> np.ndarray:
        """Apply second-order mechanical filter representing watch-strap dynamics.

        Loose strap (low tightness) allows more relative motion between
        sensor and skin, amplifying low-frequency components and adding
        resonance peaks. Tight strap (high tightness) faithfully transmits
        motion.
        """
        if strap_tightness > 0.85:
            return acc  # tight strap: negligible dynamics

        # Damping ratio decreases with loose strap (more oscillatory)
        damping = 0.3 + 0.6 * strap_tightness
        # Natural frequency of strap system (Hz)
        natural_freq = 2.0 + 8.0 * strap_tightness  # 2-10 Hz

        # Apply as a simple IIR lowpass with resonance at natural_freq
        # For very loose strap, add significant low-frequency resonance
        resonance_gain = (1.0 - strap_tightness) * 0.5  # 0-0.5x amplification at resonance
        w0 = 2 * np.pi * natural_freq
        alpha = resonance_gain * np.exp(-damping * np.arange(len(acc)) / self.fs)

        for axis in range(3):
            # Simple second-order resonant filter
            filtered = gaussian_filter1d(acc[:, axis], sigma=max(self.fs / (natural_freq * 2), 1))
            acc[:, axis] = acc[:, axis] * strap_tightness + filtered * (1 - strap_tightness)
            # Add resonance oscillation
            acc[:, axis] += alpha * np.sin(w0 * np.arange(len(acc)) / self.fs + self.rng.uniform(0, 2*np.pi))

        return acc

    def couple_to_ppg(self, ppg: np.ndarray, accel: np.ndarray, event: MotionEvent) -> dict:
        """Couple the accelerometer signal into the PPG via several
        physiologically motivated mechanisms:
          1. Baseline wander (skin deformation / sensor-skin gap changes)
          2. Perfusion modulation (muscle compression / venous pooling
             transiently altering local perfusion)
          3. High-frequency ripple (bone vibration during contact transients)
          4. Stochastic dropout (brief total decoupling during high-g events)
          5. Motion伪影 artifact via optical path modulation
        """
        n = len(ppg)
        profile = ACTIVITY_PROFILES.get(event.activity, ACTIVITY_PROFILES["rest"])

        # Use magnitude for coupling
        if accel.ndim == 2:
            accel_mag = np.sqrt(np.sum(accel[:, :2] ** 2, axis=-1))  # horizontal magnitude
            accel_vertical = accel[:, 2] if accel.shape[1] > 2 else accel_mag
        else:
            accel_mag = accel
            accel_vertical = accel

        accel_env = gaussian_filter1d(np.abs(accel_mag), sigma=max(self.fs * 0.05, 1))

        # 1. Baseline wander: low-frequency offset from skin deformation
        #    Amplitude proportional to motion intensity and inversely to strap tightness
        strap_factor = 1.0 + 2.0 * (1.0 - event.strap_tightness)
        baseline_wander = gaussian_filter1d(accel_mag, sigma=max(self.fs * 0.3, 1)) \
            * 0.4 * np.std(ppg) * strap_factor

        # 2. Perfusion modulation: muscle contraction compresses local vasculature
        #    and limb position changes alter hydrostatic pressure
        perfusion_mod = 1.0 + 0.25 * profile["intensity"] * np.tanh(accel_env * 2.0) \
            - 0.12 * profile["intensity"] * accel_env
        # Add hydrostatic effect from vertical motion (arm above/below heart)
        hydrostatic = 0.08 * np.sin(np.arctan2(accel_vertical, accel_mag + 1e-6))
        perfusion_mod += hydrostatic

        # 3. High-frequency ripple: bone vibration during contact transients
        bone_vibration = accel_mag * 0.12 * np.std(ppg) * (accel_env > 0.3 * (accel_env.max() + 1e-9))

        # 4. Stochastic dropout: brief total decoupling during high-g events
        dropout_mask = np.ones(n)
        dropout_intensity = profile["intensity"] * strap_factor
        if dropout_intensity > 0.3:
            n_events = self.rng.poisson(dropout_intensity * n / self.fs / 6.0)
            for _ in range(int(n_events)):
                idx = self.rng.integers(0, n)
                length = int(self.rng.uniform(0.03, 0.25) * self.fs)
                dropout_mask[idx:min(idx + length, n)] *= self.rng.uniform(0.05, 0.4)

        # 5. Optical path modulation: arm motion changes source-detector distance
        path_modulation = 1.0 + 0.03 * np.sin(2 * np.pi * self.rng.uniform(0.5, 2.0)
                                                 * np.arange(n) / self.fs) * profile["intensity"]

        artifacted = (ppg * perfusion_mod * path_modulation + baseline_wander + bone_vibration) \
            * dropout_mask

        snr_est = float(10 * np.log10((np.var(ppg) + 1e-12) / (np.var(artifacted - ppg) + 1e-12)))

        return {
            "ppg_with_motion": artifacted.astype(np.float32),
            "accelerometer": accel.astype(np.float32),
            "estimated_snr_db": snr_est,
            "perfusion_modulation": perfusion_mod.astype(np.float32),
            "dropout_events": int((dropout_mask < 0.9).sum()),
        }
