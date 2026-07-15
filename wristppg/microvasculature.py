"""
Wrist microvascular bed: converts local (radial artery) blood pressure
into instantaneous arterial blood volume fraction, which the optical
model then converts into detected light intensity.

Evidence base
-------------
- Arterial wall compliance relates pressure to volume nonlinearly
  (volume increases steeply at low distending pressure, saturates at
  high pressure as collagen fibers recruit): Langewouters, Wesseling &
  Goedhard, "The static elastic properties of 45 human thoracic and 20
  abdominal aortas...", J Biomech 17:425-435 (1984) — we reuse the
  qualitative sigmoidal pressure-volume shape from this work, not its
  exact per-subject coefficients.
- Vasoconstriction/dilation shifts the operating point on this P-V curve
  and changes the microvascular bed's baseline blood content, which is
  the physiological basis for finger/wrist PPG amplitude changing with
  sympathetic tone: Allen & Murray (2003), as cited in optics.py.

What is heuristic here
-----------------------
- The sigmoidal P-V relationship parameters (steepness, saturation
  pressure) are set to plausible values reproducing a realistic AC/DC
  ratio (~1-3%) rather than fit to a specific compliance dataset for the
  radial artery / wrist microvasculature specifically (most published
  P-V curves are for larger central arteries).
"""

from __future__ import annotations

import numpy as np


class MicrovascularBedModel:
    def pressure_to_blood_fraction(self, pressure_mmHg: np.ndarray,
                                    baseline_fraction: float = 0.02,
                                    vascular_tone: float = 0.5,
                                    compliance_scale: float = 1.0) -> np.ndarray:
        """Sigmoidal (saturating) pressure-volume relationship.

        Vasoconstriction (high tone) both lowers baseline blood content
        and flattens the pressure-volume curve (stiffer, less
        distensible arterioles under sympathetic drive).
        """
        p = np.asarray(pressure_mmHg, dtype=np.float64)
        p0 = 90.0  # operating-point pressure for half-maximal distension
        steepness = (0.06 * compliance_scale) * (1.0 - 0.5 * vascular_tone)
        steepness = max(steepness, 1e-4)
        sigmoid = 1.0 / (1.0 + np.exp(-steepness * (p - p0)))
        sigmoid = (sigmoid - sigmoid.min()) / (sigmoid.max() - sigmoid.min() + 1e-9)

        tone_baseline_scale = (1.0 - 0.5 * vascular_tone)
        amplitude = baseline_fraction * 0.35 * tone_baseline_scale  # pulsatile swing
        fraction = baseline_fraction * tone_baseline_scale + amplitude * (sigmoid - 0.5)
        return np.clip(fraction, 1e-4, 0.5).astype(np.float32)