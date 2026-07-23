"""
Cardiac mechanics: time-varying elastance model of the left ventricle.

Evidence base
-------------
- Time-varying elastance concept: Suga & Sagawa, "Instantaneous
  pressure-volume relationships and their ratio in the excised, supported
  canine left ventricle", Circ Res 35:117-126 (1974).
- Normalized double-Hill elastance activation function e(t) used here:
  Stergiopulos, Meister & Westerhof, "Determinants of stroke volume and
  systolic/diastolic arterial pressure", Am J Physiol 270:H2050-9 (1996).
- Single-beat ventriculo-arterial coupling estimate of end-systolic volume
  (Ees vs effective arterial elastance Ea): Sunagawa, Maughan, Burkhoff &
  Sagawa, "Left ventricular interaction with arterial load studied in
  isolated canine ventricle", Am J Physiol 245:H773-80 (1983); Kelly et al.
  Circulation 86:513-521 (1992) for the clinical single-beat method.
- Frank-Starling relationship (preload dependence of stroke work/volume):
  Frank (1895) / Starling (1918) classical curves, modern formulation in
  Klabunde, "Cardiovascular Physiology Concepts", 2nd ed., Ch. 4.

What is heuristic here
-----------------------
- The mapping from "preload state" (0-1, driven by autonomic/venous-return
  model) to end-diastolic volume (EDV) via a saturating exponential is a
  reasonable but not literature-calibrated curve fit; real EDV depends on
  venous return curves, atrial contribution, and filling time that are
  only crudely captured.
- Beat-to-beat contractility variability is modeled as a bounded stochastic
  process, not derived from a specific autonomic transfer function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


def elastance_activation(t_norm: np.ndarray, t_es_frac: float = 0.30,
                          m1: float = 1.9, m2: float = 21.9) -> np.ndarray:
    """Normalized double-Hill elastance function e(t) in [0, 1].

    Stergiopulos et al. (1996), Eq. 1-2. ``t_norm`` is time normalized by
    cycle length (0 to 1). ``t_es_frac`` sets the fraction of the cycle at
    which peak elastance (end systole) occurs.
    """
    tau = 0.2 + 0.15 * t_es_frac  # Stergiopulos scaling of the peak time
    g1 = (t_norm / tau) ** m1
    g2 = (t_norm / tau) ** m2
    h = (g1 / (1 + g1)) * (1 / (1 + g2))
    return h / (np.max(h) + 1e-12)


@dataclass
class CardiacState:
    """Continuous-time cardiac mechanical state (beat-to-beat)."""
    contractility: float = 1.0     # Ees multiplier (1.0 = nominal healthy)
    preload_state: float = 0.5     # 0-1, venous return / filling adequacy
    afterload_Ea: float = 1.1      # effective arterial elastance (mmHg/mL), set by Windkessel
    hr_bpm: float = 72.0
    v0_ml: float = 10.0            # unstressed LV volume
    ees_nominal_mmHg_ml: float = 2.5  # nominal end-systolic elastance (healthy)


class LeftVentricleModel:
    """Beat-by-beat LV pressure/volume/flow generator via time-varying
    elastance and single-beat ventriculo-arterial coupling.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def edv_from_preload(self, preload_state: float, base_edv_ml: float = 120.0) -> float:
        """Frank-Starling filling: EDV rises with preload but saturates
        (ventricular compliance falls at high filling pressure)."""
        preload_state = np.clip(preload_state, 0.0, 1.5)
        return base_edv_ml * (1.0 - np.exp(-2.2 * preload_state)) / (1.0 - np.exp(-2.2 * 1.0))

    def single_beat_esv(self, ees: float, ea: float, edv: float, v0: float) -> float:
        """Sunagawa/Kelly single-beat coupling estimate of end-systolic volume.

        At the operating point that maximizes stroke work for a linear
        ESPVR/EA framework: Ves = (Ees*V0 + Ea*EDV) / (Ees + Ea).
        """
        ees = max(ees, 1e-3)
        ea = max(ea, 1e-3)
        ves = (ees * v0 + ea * edv) / (ees + ea)
        return float(np.clip(ves, v0 + 1.0, edv - 1.0))

    def beat(self, state: CardiacState, n_samples: int) -> dict:
        """Generate one cardiac cycle's LV pressure, volume, and aortic
        (ejected) flow, sampled at ``n_samples`` points over the cycle.

        Returns dict with keys: t_norm, lv_pressure, lv_volume, flow_ml_s,
        edv, esv, sv, ef, ejection_mask.
        """
        period_s = 60.0 / state.hr_bpm
        t_norm = np.linspace(0.0, 1.0, n_samples, endpoint=False)

        ees = state.ees_nominal_mmHg_ml * state.contractility
        edv = self.edv_from_preload(state.preload_state)
        esv = self.single_beat_esv(ees, state.afterload_Ea, edv, state.v0_ml)
        sv = edv - esv
        ef = sv / edv if edv > 0 else 0.0

        e_t = elastance_activation(t_norm)
        peak_idx = int(np.argmax(e_t))

        # Volume trajectory: isovolumic phases + ejection (systole) +
        # filling (diastole). Ejection window approximated as the rising
        # edge of e(t) up to its peak plus a short decay (aortic valve
        # closes near dicrotic notch, ~ when e(t) falls below ~0.6 of peak
        # post-peak). Filling occupies the remainder of diastole with an
        # RC-like exponential approach to EDV, consistent with rapid
        # filling + diastasis + atrial kick lumped together.
        post_peak_thresh = 0.55
        closing_idx = peak_idx
        for i in range(peak_idx, n_samples):
            if e_t[i] < post_peak_thresh * e_t[peak_idx]:
                closing_idx = i
                break
        else:
            closing_idx = min(peak_idx + n_samples // 8, n_samples - 1)

        volume = np.empty(n_samples)
        # Isovolumic contraction: hold at EDV until elastance starts rising
        # appreciably (small activation threshold).
        onset_idx = int(np.searchsorted(e_t[:peak_idx + 1] > 0.02, True)) if peak_idx > 0 else 0
        volume[:onset_idx] = edv
        # Ejection: volume falls from EDV to ESV, shaped by e(t) rise
        # (higher elastance -> faster volume decline for a given pressure
        # gradient); use normalized e(t) as the ejection progress curve.
        if closing_idx > onset_idx:
            prog = e_t[onset_idx:closing_idx + 1]
            prog = (prog - prog.min()) / (prog.max() - prog.min() + 1e-9)
            volume[onset_idx:closing_idx + 1] = edv - prog * (edv - esv)
        # Isovolumic relaxation + filling: exponential approach back to EDV
        tail_len = n_samples - (closing_idx + 1)
        if tail_len > 0:
            tau_fill = 0.30  # fraction-of-cycle filling time constant (heuristic)
            tt = np.linspace(0, 1, tail_len)
            volume[closing_idx + 1:] = esv + (edv - esv) * (1 - np.exp(-tt / tau_fill))

        lv_pressure = e_t * (volume - state.v0_ml)
        lv_pressure = np.clip(lv_pressure, 0, None)

        # Aortic flow = -dV/dt during ejection only (valve open), zero
        # otherwise (mitral inflow is not delivered to the arterial tree).
        dv = np.gradient(volume, period_s / n_samples)
        flow = np.zeros(n_samples)
        eject_mask = np.zeros(n_samples, dtype=bool)
        eject_mask[onset_idx:closing_idx + 1] = True
        flow[eject_mask] = np.clip(-dv[eject_mask], 0, None)

        # Real aortic/mitral valves open and close over a short but finite
        # interval (a few ms), not instantaneously; a hard boolean mask
        # here creates an unphysical step discontinuity in flow that, when
        # differentiated for the Windkessel inertance term, produces huge
        # spurious dQ/dt spikes. Smoothing the valve transition over a
        # couple of samples removes this artifact while leaving the
        # overall ejection profile (set by the elastance activation
        # function) essentially unchanged.
        smooth_sigma = max(n_samples / 100.0, 0.8)
        flow = gaussian_filter1d(flow, sigma=smooth_sigma)
        flow = np.clip(flow, 0, None)

        return {
            "t_norm": t_norm,
            "period_s": period_s,
            "lv_pressure_mmHg": lv_pressure.astype(np.float32),
            "lv_volume_ml": volume.astype(np.float32),
            "flow_ml_s": flow.astype(np.float32),
            "edv_ml": edv,
            "esv_ml": esv,
            "sv_ml": sv,
            "ef": float(np.clip(ef, 0.05, 0.85)),
            "ejection_mask": eject_mask,
        }