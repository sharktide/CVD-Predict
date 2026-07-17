"""
Wrist microvascular bed: converts arterial blood pressure into
instantaneous arterial blood volume fraction, which the optical
model then converts into detected light intensity.

Wrist-specific features:
- Radial artery depth 1-3mm with surrounding tissue compliance
- Wrist microvasculature has less collateral circulation than fingers
- Ischemia/reperfusion dynamics during cardiac arrest/CPR
- Venous pooling in dependent wrist positions
- Vasodilation collapse in severe hypoperfusion

Evidence base
-------------
- Arterial wall compliance (sigmoidal P-V): Langewouters, Wesseling &
  Goedhard (1984).
- Vasoconstriction shifts P-V operating point: Allen & Murray (2003).
- Ischemia-reperfusion hyperemic response: Kalliokoski et al., "Dynamic
  PET imaging of muscle blood flow during exercise", J Physiol 543:1003
  (2002); Halliwill, "Forearm ischemia reactive hyperemia", Clin Auton
  Res 11:29-35 (2001).
- Microvascular collapse in shock: Trzeciak et al., "Microcirculatory
  dysfunction in sepsis", Crit Care Med 35:981-989 (2007).
- Venous pooling: Monson et al., "Venous pooling in dependent limbs",
  J Appl Physiol 97:1109-1118 (2004).

What is heuristic here
-----------------------
- Ischemia/reperfusion model uses exponential time constants rather
  than detailed ion-channel-level modeling.
- Microvascular collapse uses a simple threshold model rather than
  endothelial dysfunction dynamics.
- Venous pooling is a gravitational scalar model.
"""

from __future__ import annotations

import numpy as np


class MicrovascularBedModel:
    """Microvasculature: pressure → blood volume fraction for wrist.

    Key differences from generic model:
    - Wrist radial artery is shallow, so baseline blood fraction is
      higher (~0.025) and pulsatile fraction is larger (~0.008)
    - Ischemia threshold is lower (wrist has lower perfusion at rest
      than fingertips)
    - Collateral circulation is poor, so ischemic events are more severe
    - Microvascular collapse in shock happens at higher MAP than
      central vasculature
    """

    def pressure_to_blood_fraction(
        self,
        pressure_mmHg: np.ndarray,
        baseline_fraction: float = 0.025,
        vascular_tone: float = 0.5,
        compliance_scale: float = 1.0,
        ischemia_state: np.ndarray | None = None,
        ischemia_duration_s: np.ndarray | None = None,
        venous_pooling_fraction: float = 0.0,
        microvascular_integrity: float = 1.0,
        perfusion_scale: float = 1.0,
    ) -> np.ndarray:
        """Convert arterial pressure waveform to blood volume fraction.

        Parameters
        ----------
        pressure_mmHg : (N,) array
            Arterial pressure waveform (mmHg).
        baseline_fraction : float
            Baseline tissue blood volume fraction (wrist ~0.025).
        vascular_tone : float
            0 = maximally vasodilated, 1 = maximally constricted.
        compliance_scale : float
            Multiplier on arterial compliance (1 = normal).
        ischemia_state : (N,) array or None
            0-1 ischemic state at each sample (0 = normal, 1 = fully
            ischemic).
        ischemia_duration_s : (N,) array or None
            Duration of current ischemic episode in seconds.
        venous_pooling_fraction : float
            0-1 fraction of blood volume shifted to venous pool due
            to gravity/dependency.
        microvascular_integrity : float
            0-1 factor representing microvascular health (1 = healthy,
            0 = completely collapsed as in severe sepsis).
        perfusion_scale : float
            0-1 factor for pulsatile amplitude. Set to ~0 when
            perfusion_ok=False (cardiac arrest) so PPG goes flat.
        """
        p = np.asarray(pressure_mmHg, dtype=np.float64)
        N = len(p)

        # --- Sigmoidal pressure-volume relationship ---
        # Wrist operates at lower MAP (~70-90 mmHg) than central aorta
        p0 = 80.0  # half-maximal distension pressure (wrist radial artery)
        steepness = (0.06 * compliance_scale) * (1.0 - 0.5 * vascular_tone)
        steepness = max(steepness, 1e-4)

        sigmoid = 1.0 / (1.0 + np.exp(-steepness * (p - p0)))

        # Pulsatile amplitude proportional to actual pulse pressure
        # (NOT normalized away — this is what makes arrest PPG flat)
        pulse_pressure = float(np.ptp(p))  # max - min
        normal_pulse_pressure = 40.0       # normal resting pulse pressure (mmHg)
        pp_scale = np.clip(pulse_pressure / normal_pulse_pressure, 0.0, 3.0)

        # --- Tone-dependent baseline and amplitude ---
        tone_baseline_scale = 1.0 - 0.5 * vascular_tone
        pulsatile_amplitude = baseline_fraction * 0.40 * tone_baseline_scale * pp_scale * perfusion_scale

        # Center sigmoid so it oscillates around 0
        sig_range = sigmoid.max() - sigmoid.min()
        sigmoid_centered = (sigmoid - np.mean(sigmoid)) / (sig_range + 1e-9)

        fraction = baseline_fraction * tone_baseline_scale + pulsatile_amplitude * sigmoid_centered

        # --- Microvascular integrity (shock collapse) ---
        # In sepsis/shock, microvasculature collapses progressively
        fraction *= microvascular_integrity

        # --- Ischemia effect ---
        # Prolonged ischemia (no arterial inflow) causes:
        # 1. Gradual depletion of oxygen stores (exponential decay)
        # 2. Reactive hyperemia upon reperfusion (overshoot)
        if ischemia_state is not None:
            isch = np.asarray(ischemia_state, dtype=np.float64)
            isch_dur = np.asarray(ischemia_duration_s, dtype=np.float64) if ischemia_duration_s is not None else np.zeros(N)

            # Ischemia depletes blood fraction (exponential decay with tau~30s for wrist)
            tau_depletion = 30.0  # seconds, faster than finger (wrist has less reserve)
            depletion = np.exp(-np.clip(isch_dur, 0, 120) / tau_depletion)

            # Ischemic vasodilation: tone drops, but perfusion is compromised
            ischemic_dilation = 0.3 * isch  # partial vasodilation during ischemia

            fraction *= (depletion * (1.0 - 0.3 * isch) + ischemic_dilation * 0.1)

            # Reactive hyperemia: after ischemia ends, blood flow overshoots
            # by ~50-100% for ~10-30 seconds (wrist: shorter than finger)
            reperfusion_mask = (isch[:-1] > 0.5) & (isch[1:] < 0.5)
            reperfusion_indices = np.where(reperfusion_mask)[0]
            for idx in reperfusion_indices:
                end = min(idx + int(20.0 * 128), N)  # 20 second hyperemic response
                t_hyp = np.arange(end - idx) / 128.0
                tau_hyp = 8.0  # time constant of hyperemic decay (wrist)
                overshoot = 0.8 * np.exp(-t_hyp / tau_hyp)  # 80% overshoot peak
                fraction[idx:end] *= (1.0 + overshoot)

        # --- Venous pooling ---
        # Dependent wrist position causes venous engorgement, reducing
        # arterial pulsatile amplitude
        if venous_pooling_fraction > 0:
            # Venous pooling increases baseline but reduces pulsatility
            pooling_boost = venous_pooling_fraction * 0.01  # increase DC
            pulsatile_reduction = 1.0 - 0.5 * venous_pooling_fraction  # reduce AC
            fraction = fraction * pulsatile_reduction + pooling_boost

        return np.clip(fraction, 1e-4, 0.5).astype(np.float32)

    def compute_microvascular_integrity(self, map_mmHg: float) -> float:
        """Compute microvascular integrity from MAP.

        In shock, microvascular perfusion fails progressively:
        - MAP > 65 mmHg: integrity ~1.0 (autoregulation maintained)
        - MAP 50-65: progressive collapse (autoregulation failing)
        - MAP < 50: severe collapse (no autoregulation)
        """
        if map_mmHg >= 65.0:
            return 1.0
        elif map_mmHg >= 50.0:
            return 0.3 + 0.7 * ((map_mmHg - 50.0) / 15.0)
        else:
            return max(0.05, 0.3 * (map_mmHg / 50.0))

    def compute_venous_pooling(self, posture: str, time_vertical_s: float) -> float:
        """Compute venous pooling fraction from posture history.

        Dependent wrist (arm hanging below heart) causes venous pooling.
        Returns 0-1 fraction.
        """
        if posture == "prone":
            return 0.3  # arm below heart in prone
        elif posture == "upright":
            return 0.1  # arm at side = mild dependency
        elif posture == "supine":
            return 0.0  # arm at heart level
        else:
            # Time-dependent pooling in prolonged dependency
            return min(0.4, 0.05 + 0.01 * time_vertical_s / 60.0)
