"""
Wrist-specific optical coupling / sensor-skin contact model.

Models the time-varying coupling between a wrist-worn PPG sensor and
the skin, incorporating:
- Wrist anatomy (radial artery depth, fat pad thickness)
- Strap tension and its variation with movement
- Skin-sensor interface dynamics (air gaps, moisture, hair)
- Posture-dependent contact changes

Evidence base
-------------
- Teng & Zhang (2004): PPG amplitude vs contact force relationship.
- Fallow et al. (2013): skin type and wavelength effects on reflectance.
- Wang et al., "Investigation of the effects of contacting pressure on
  PPG", Biomed Signal Process Control 45:247-254 (2018).
- Wrist-watch contact: Apple Watch uses photodiode array with LED at
  ~5mm source-detector distance; contact quality depends on strap type
  (Milanese loop vs Sport band vs Leather).

What is heuristic here
-----------------------
- Strap dynamics are modeled as parametric events rather than derived
  from mechanical measurements.
- Air-gap attenuation uses a simple exponential model rather than
  Fresnel reflection calculations.
- Moisture effects are scalar multipliers, not spectral.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class ContactState:
    mode: str = "good"  # good, loose, tight, partial_lift, hair, sweat, water, rolling
    strap_type: str = "sport"  # sport, milanese, leather, none
    posture: str = "upright"  # upright, supine, prone


class ContactModel:
    def __init__(self, rng: np.random.Generator, fs_hz: float):
        self.rng = rng
        self.fs = fs_hz

    def coupling_trace(self, n_samples: int, state: ContactState,
                        motion_energy: np.ndarray | None = None) -> dict:
        """Returns a dict with:
          - efficiency: 0-1 optical coupling efficiency over time
          - pressure: 0-1 effective contact pressure over time (feeds optics)
          - ambient_leakage: 0-1 fraction of ambient light reaching sensor
        """
        t = np.arange(n_samples) / self.fs
        efficiency = np.ones(n_samples)
        pressure = np.full(n_samples, 0.5)
        ambient_leakage = np.zeros(n_samples)

        # Strap type affects baseline coupling
        strap_base = {"sport": 0.95, "milanese": 0.85, "leather": 0.75, "none": 0.30}
        base_efficiency = strap_base.get(state.strap_type, 0.90)

        if state.mode == "good":
            efficiency *= self.rng.uniform(0.90, 1.0) * base_efficiency
            pressure[:] = self.rng.uniform(0.45, 0.60)
            ambient_leakage[:] = 0.02  # minimal ambient in good contact

        elif state.mode == "loose":
            efficiency *= self.rng.uniform(0.30, 0.55) * base_efficiency
            pressure[:] = self.rng.uniform(0.12, 0.28)
            ambient_leakage[:] = self.rng.uniform(0.15, 0.40)  # significant ambient
            # Fluctuating coupling from strap sliding
            drift = 0.15 * np.sin(2 * np.pi * self.rng.uniform(0.01, 0.05) * t)
            efficiency = np.clip(efficiency + drift, 0.05, 1.0)
            # Add strap bounce frequency (2-5 Hz for loose sport band)
            bounce_freq = self.rng.uniform(2.0, 5.0)
            bounce = 0.08 * np.sin(2 * np.pi * bounce_freq * t)
            efficiency = np.clip(efficiency + bounce, 0.05, 1.0)

        elif state.mode == "tight":
            pressure[:] = self.rng.uniform(0.82, 0.95)
            efficiency *= self.rng.uniform(0.55, 0.80) * base_efficiency
            ambient_leakage[:] = 0.01  # very tight = minimal ambient
            # Excessive pressure occludes veins, reducing venous return
            if pressure[0] > 0.90:
                efficiency *= 0.85  # venous engorgement reduces AC

        elif state.mode == "partial_lift":
            n_events = self.rng.integers(1, 5)
            for _ in range(n_events):
                start = self.rng.integers(0, n_samples)
                length = int(self.rng.uniform(0.2, 2.5) * self.fs)
                end = min(start + length, n_samples)
                efficiency[start:end] *= self.rng.uniform(0.03, 0.25)
                ambient_leakage[start:end] = self.rng.uniform(0.3, 0.7)

        elif state.mode == "hair":
            efficiency *= self.rng.uniform(0.45, 0.70) * base_efficiency
            ambient_leakage[:] = 0.05

        elif state.mode == "sweat":
            efficiency *= self.rng.uniform(0.65, 0.90)
            efficiency += 0.05 * np.sin(2 * np.pi * self.rng.uniform(0.02, 0.08) * t)
            ambient_leakage[:] = 0.03  # sweat can fill air gaps, slightly reducing ambient

        elif state.mode == "water":
            efficiency *= self.rng.uniform(0.08, 0.35)  # water scatters strongly
            ambient_leakage[:] = self.rng.uniform(0.1, 0.3)

        elif state.mode == "rolling":
            base = self.rng.uniform(0.55, 0.80) * base_efficiency
            if motion_energy is not None and len(motion_energy) == n_samples:
                me = motion_energy / (np.max(np.abs(motion_energy)) + 1e-9)
                efficiency = base * (1.0 - 0.4 * np.abs(me))
                ambient_leakage = 0.05 + 0.15 * np.abs(me)
            else:
                efficiency = base + 0.15 * np.sin(2 * np.pi * self.rng.uniform(0.5, 2.0) * t)
                ambient_leakage[:] = 0.08
            pressure[:] = self.rng.uniform(0.25, 0.65)

        elif state.mode == "cardiac_arrest_recovery":
            # Post-resuscitation: patient may be on a stretcher, sensor
            # partially dislodged, sweat/fluids present
            efficiency *= self.rng.uniform(0.40, 0.70)
            pressure[:] = self.rng.uniform(0.20, 0.50)
            ambient_leakage[:] = self.rng.uniform(0.10, 0.30)
            # Occasional total lift events (patient moving on stretcher)
            n_lifts = self.rng.integers(0, 3)
            for _ in range(n_lifts):
                start = self.rng.integers(0, n_samples)
                length = int(self.rng.uniform(0.5, 3.0) * self.fs)
                end = min(start + length, n_samples)
                efficiency[start:end] *= 0.05

        # Smooth the coupling trace (physically realistic temporal dynamics)
        efficiency = gaussian_filter1d(np.clip(efficiency, 0.02, 1.0), sigma=max(self.fs * 0.05, 1))
        pressure = np.clip(pressure, 0.0, 1.0)
        ambient_leakage = gaussian_filter1d(np.clip(ambient_leakage, 0.0, 1.0), sigma=max(self.fs * 0.1, 1))

        return {
            "efficiency": efficiency.astype(np.float32),
            "pressure": pressure.astype(np.float32),
            "ambient_leakage": ambient_leakage.astype(np.float32),
        }
