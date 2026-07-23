"""
Arterial tree wave propagation from aorta to radial/wrist microvasculature.

Evidence base
-------------
- Pulse wave velocity (PWV) from wall properties (Moens-Korteweg):
  PWV = sqrt(E*h / (2*r*rho)), Moens (1878), Korteweg (1878); modern
  treatment in Nichols, O'Rourke & Vlachopoulos, "McDonald's Blood Flow
  in Arteries", 6th ed. (2011), Ch. 4.
- Exponential pressure dependence of PWV (arteries stiffen when
  distended): Hughes, Babbs, Geddes & Bourland, "Elastic modulus is not
  constant... exponential pressure-elastic modulus relation", Med Biol
  Eng Comput 17:638-641 (1979): PWV(P) = PWV0 * exp(alpha * (P - P0)).
- Age-related arterial stiffening: McEniery et al., "Normal vascular
  aging: differential effects on wave reflection and aortic pulse wave
  velocity", J Am Coll Cardiol 46:1753-60 (2005).
- Wave reflection / augmentation index formalism (forward + backward
  wave superposition, reflection coefficient from impedance mismatch at
  branch points and periphery): Westerhof, Sipkema, van den Bos & Elzinga,
  "Forward and backward waves in the arterial system", Cardiovasc Res
  6:648-656 (1972); Nichols & O'Rourke (as above), Ch. 8. Augmentation
  index (AIx) definition: Kelly, Hayward, Avolio & O'Rourke, Circulation
  80:1652-1659 (1989).
- Pulse (pressure) amplification from aorta to peripheral arteries, more
  pronounced in younger/more compliant vasculature: Nichols & O'Rourke
  Ch. 8; London & Pannier, Nephrol Dial Transplant 25:3815-23 (2010).

What is heuristic here
-----------------------
- Rather than solving the full nonlinear 1-D Navier-Stokes network
  (Womersley/Olufsen-style PDE over a branching tree), we use a reduced
  "effective path + single dominant reflection site" model: a forward
  wave travels a lumped aorta->radial path length at a pressure- and
  stiffness-dependent PWV, and a single reflected wave (representing the
  net effect of peripheral impedance mismatches, dominated by the
  lower-body reflection site) arrives with a reflection coefficient and
  round-trip delay from a second lumped path length. This reproduces the
  qualitative and approximately quantitative behavior described in the
  augmentation-index literature (AIx increasing with age/stiffness,
  earlier reflected-wave return time with stiffer vessels) without the
  computational cost/complexity of a full distributed network solve, and
  is explicitly a simplification of a much richer PDE-based reality.
- Path lengths (aorta-to-reflection-site, aorta-to-wrist) are fixed
  population-average anthropometric estimates, not subject-specific.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


AORTA_TO_REFLECTION_SITE_M = 0.45   # ~ effective distance to major reflection site (iliac/renal bifurcation region)
AORTA_TO_WRIST_M = 0.75             # ~ aorta -> subclavian -> brachial -> radial -> wrist


@dataclass
class ArterialTreeParams:
    pwv0_m_s: float = 5.0           # baseline (low-pressure) pulse wave velocity
    pressure_alpha: float = 0.017   # Hughes exponential pressure coefficient (1/mmHg)
    age_years: float = 40.0
    reflection_coefficient: float = 0.5  # 0 (no reflection) to ~0.8 (severe PAD/stiffness)


def pwv_from_state(stiffness: float, age_years: float, mean_pressure_mmHg: float,
                    params: ArterialTreeParams) -> float:
    """Moens-Korteweg + Hughes exponential pressure law + age term.

    Age contributes an approximately linear PWV increase of ~0.04-0.09
    m/s per year above 30, consistent with the population regression in
    McEniery et al. (2005) (their carotid-femoral PWV vs age slope,
    applied here as a general stiffening proxy since we do not model a
    distinct carotid-femoral segment).
    """
    age_term = max(age_years - 30.0, 0.0) * 0.06
    pwv0 = params.pwv0_m_s * max(stiffness, 0.2) + age_term
    pwv = pwv0 * np.exp(params.pressure_alpha * (mean_pressure_mmHg - 93.0))
    return float(np.clip(pwv, 3.0, 15.0))


class ArterialTreeModel:
    """Propagates the aortic pressure waveform to the wrist, producing a
    radial-artery pressure waveform with realistic PTT, wave reflection,
    and pulse amplification.
    """

    def compute_ptt_s(self, pwv_m_s: float, path_length_m: float = AORTA_TO_WRIST_M) -> float:
        return path_length_m / pwv_m_s

    def propagate(self, p_aortic_mmHg: np.ndarray, dt_s: float,
                   stiffness: float, age_years: float, mean_pressure_mmHg: float,
                   params: ArterialTreeParams) -> dict:
        pwv = pwv_from_state(stiffness, age_years, mean_pressure_mmHg, params)
        ptt_wrist_s = self.compute_ptt_s(pwv, AORTA_TO_WRIST_M)
        t_reflect_s = self.compute_ptt_s(pwv, AORTA_TO_REFLECTION_SITE_M)

        n = len(p_aortic_mmHg)
        shift_wrist = int(round(ptt_wrist_s / dt_s))
        shift_reflect_roundtrip = int(round(2 * t_reflect_s / dt_s))

        forward = np.roll(p_aortic_mmHg, shift_wrist)
        forward[:shift_wrist] = p_aortic_mmHg[0]

        # Reflected component: attenuated, delayed copy of the pulsatile
        # (AC) part of the forward wave, per Westerhof forward/backward
        # wave decomposition (net single-reflection-site approximation).
        ac = p_aortic_mmHg - np.mean(p_aortic_mmHg)
        refl = params.reflection_coefficient * ac
        refl = np.roll(refl, shift_wrist + shift_reflect_roundtrip)
        refl[:shift_wrist + shift_reflect_roundtrip] = 0.0

        # Pulse (pressure) amplification toward the periphery: more
        # pronounced with compliant (young, low-stiffness) vessels; the
        # amplification factor here (1.1-1.4x) spans the physiological
        # range reported for brachial/radial vs central pulse pressure
        # (Nichols & O'Rourke, Ch. 8).
        amp_factor = 1.10 + 0.30 * np.clip(1.0 / max(stiffness, 0.2), 0.0, 1.0)
        radial = np.mean(forward) + (forward - np.mean(forward)) * amp_factor + refl

        aix = self._augmentation_index(radial)

        return {
            "radial_pressure_mmHg": radial.astype(np.float32),
            "pwv_m_s": pwv,
            "ptt_s": ptt_wrist_s,
            "reflection_return_time_s": t_reflect_s,
            "augmentation_index": aix,
            "amplification_factor": float(amp_factor),
        }

    @staticmethod
    def _augmentation_index(waveform: np.ndarray) -> float:
        """AIx = (P2 - P1) / PP, P1 = first systolic shoulder (inflection),
        P2 = systolic peak, PP = pulse pressure. Kelly et al. (1989).
        Simplified inflection detection via the first local
        derivative sign change after the initial upstroke.
        """
        pp = np.max(waveform) - np.min(waveform)
        if pp < 1e-6:
            return 0.0
        d1 = np.gradient(waveform)
        peak_idx = int(np.argmax(waveform))
        p2 = waveform[peak_idx]
        # search for inflection (local max of derivative before global peak)
        inflect_idx = peak_idx
        for i in range(1, peak_idx):
            if d1[i - 1] > 0 and d1[i] <= 0:
                inflect_idx = i
        p1 = waveform[inflect_idx] if inflect_idx != peak_idx else waveform[max(peak_idx // 2, 0)]
        return float((p2 - p1) / pp)