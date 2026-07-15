"""
Optical coupling / sensor-skin contact model.

Rather than a flat multiplicative "quality" scalar, contact state is
represented as a physical coupling efficiency that modulates the optical
path (feeding back into optics.SkinOpticalParams.contact_pressure and an
explicit air-gap attenuation term), consistent with the qualitative
force-vs-signal relationship reported in:

- Teng & Zhang, "The effect of contacting force on photoplethysmographic
  signals", Physiol Meas 25:1323-35 (2004): PPG amplitude rises with
  contact force up to an optimum, then falls as local vessels are
  compressed/occluded.
- Fallow, Tarumi & Tanaka, "Influence of skin type and wavelength on
  light wave reflectance rate...", J Clin Monit Comput 27:313-7 (2013),
  for general coupling/reflectance behavior at the skin interface.

What is heuristic here
-----------------------
- Loose-strap, partial-lift, and rolling-contact dynamics are modeled as
  a time-varying coupling-efficiency trace built from a small library of
  parametric events (step changes, slow drifts, oscillatory rolling
  contact during motion) rather than measured from real strap-tension
  sensors, which are not available in consumer devices/public datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class ContactState:
    mode: str = "good"  # good, loose, tight, partial_lift, hair, sweat, water, rolling


class ContactModel:
    def __init__(self, rng: np.random.Generator, fs_hz: float):
        self.rng = rng
        self.fs = fs_hz

    def coupling_trace(self, n_samples: int, state: ContactState,
                        motion_energy: np.ndarray | None = None) -> dict:
        """Returns a dict with:
          - efficiency: 0-1 optical coupling efficiency over time
          - pressure: 0-1 effective contact pressure over time (feeds optics)
        """
        t = np.arange(n_samples) / self.fs
        efficiency = np.ones(n_samples)
        pressure = np.full(n_samples, 0.5)

        if state.mode == "good":
            efficiency *= self.rng.uniform(0.92, 1.0)
            pressure[:] = self.rng.uniform(0.45, 0.6)

        elif state.mode == "loose":
            # low, fluctuating coupling; more sensitive to motion-induced lift
            efficiency *= self.rng.uniform(0.35, 0.6)
            pressure[:] = self.rng.uniform(0.15, 0.3)
            drift = 0.15 * np.sin(2 * np.pi * self.rng.uniform(0.01, 0.05) * t)
            efficiency = np.clip(efficiency + drift, 0.05, 1.0)

        elif state.mode == "tight":
            pressure[:] = self.rng.uniform(0.85, 0.98)
            efficiency *= self.rng.uniform(0.6, 0.85)  # venous engorgement/arterial compression reduces AC

        elif state.mode == "partial_lift":
            n_events = self.rng.integers(1, 4)
            for _ in range(n_events):
                start = self.rng.integers(0, n_samples)
                length = int(self.rng.uniform(0.3, 3.0) * self.fs)
                end = min(start + length, n_samples)
                efficiency[start:end] *= self.rng.uniform(0.05, 0.3)

        elif state.mode == "hair":
            efficiency *= self.rng.uniform(0.5, 0.75)

        elif state.mode == "sweat":
            efficiency *= self.rng.uniform(0.7, 0.95)
            efficiency += 0.05 * np.sin(2 * np.pi * self.rng.uniform(0.02, 0.08) * t)

        elif state.mode == "water":
            efficiency *= self.rng.uniform(0.1, 0.4)  # water on optical window scatters/reflects strongly

        elif state.mode == "rolling":
            base = self.rng.uniform(0.6, 0.85)
            if motion_energy is not None and len(motion_energy) == n_samples:
                me = motion_energy / (np.max(np.abs(motion_energy)) + 1e-9)
                efficiency = base * (1.0 - 0.4 * np.abs(me))
            else:
                efficiency = base + 0.2 * np.sin(2 * np.pi * self.rng.uniform(0.5, 2.0) * t)
            pressure[:] = self.rng.uniform(0.3, 0.7)

        efficiency = gaussian_filter1d(np.clip(efficiency, 0.02, 1.0), sigma=max(self.fs * 0.05, 1))
        pressure = np.clip(pressure, 0.0, 1.0)

        return {"efficiency": efficiency.astype(np.float32), "pressure": pressure.astype(np.float32)}