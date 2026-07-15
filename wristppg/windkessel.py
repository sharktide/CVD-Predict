"""
Windkessel model of proximal arterial hemodynamics.

Evidence base
-------------
- 2-element Windkessel: Frank, "Die Grundform des arteriellen Pulses",
  Z Biol 37:483-526 (1899).
- 3-element (adds characteristic aortic impedance Zc in series):
  Westerhof, Elzinga & Sipkema, J Appl Physiol 31:776-781 (1971).
- 4-element (adds an inertial term L in parallel with Zc, improves
  high-frequency / early-systolic behavior): Stergiopulos, Westerhof &
  Westerhof, "Total arterial inertance as the fourth element of the
  windkessel model", Am J Physiol 276:H81-88 (1999).

Governing ODEs (4-element, parallel L across Zc; Stergiopulos 1999 form):

    C dP/dt = Q_in - P / R
    P_ao   = P + Zc * Q_in + L * dQ_in/dt

R = total peripheral resistance, C = total arterial compliance,
Zc = characteristic impedance of the proximal aorta, L = inertance.
Effective arterial elastance for ventriculo-arterial coupling is taken as
Ea ~= 1/(C * (systolic_ejection_time)) x correction, following Sunagawa's
approximation Ea ~ P_es / SV; we compute it directly from the resulting
pressure/flow rather than the analytic approximation, feeding it back to
the cardiac model between beats.

What is heuristic here
-----------------------
- R, C, Zc, L are parameterized as functions of stiffness/resistance
  scalars rather than fit to a specific population; the functional form
  (monotonic, bounded) is physiologically directionally correct
  (stiffer arteries -> lower C, higher Zc) but not numerically validated
  against invasive hemodynamic datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WindkesselParams:
    R_mmHg_s_ml: float = 1.0        # total peripheral resistance
    C_ml_mmHg: float = 1.5          # total arterial compliance
    Zc_mmHg_s_ml: float = 0.05      # characteristic impedance
    L_mmHg_s2_ml: float = 0.005     # inertance


def params_from_physiology(stiffness: float, resistance: float,
                            mean_pressure_mmHg: float = 93.0) -> WindkesselParams:
    """Map dimensionless stiffness/resistance state (~0.5-3, 1.0=nominal)
    to Windkessel element values, with pressure-dependent compliance.

    Compliance falls both with structural stiffness and, nonlinearly,
    with distending pressure (arteries stiffen as they are stretched):
    an exponential C(P) relationship is well documented (Langewouters,
    Wesseling & Goedhard, J Biomech 17:425-435 1984, for aortic
    pressure-area/compliance curves). We use a simplified exponential
    pressure dependence on top of the structural stiffness scalar.
    """
    stiffness = max(stiffness, 0.1)
    resistance = max(resistance, 0.1)
    pressure_factor = np.exp(-0.01 * (mean_pressure_mmHg - 93.0))  # Langewouters-style softening at low P
    C = 1.6 / stiffness * pressure_factor
    Zc = 0.045 * stiffness
    R = 1.05 * resistance
    L = 0.004 * stiffness
    return WindkesselParams(R_mmHg_s_ml=R, C_ml_mmHg=C, Zc_mmHg_s_ml=Zc, L_mmHg_s2_ml=L)


class WindkesselModel:
    """Integrates the 4-element Windkessel ODE given an inflow waveform."""

    def simulate(self, flow_ml_s: np.ndarray, dt_s: float,
                 params: WindkesselParams, p0_mmHg: float = 80.0) -> dict:
        n = len(flow_ml_s)
        P = np.empty(n)
        P[0] = p0_mmHg
        dQ = np.gradient(flow_ml_s, dt_s)

        for i in range(1, n):
            dP = (flow_ml_s[i - 1] - P[i - 1] / params.R_mmHg_s_ml) / params.C_ml_mmHg
            P[i] = P[i - 1] + dP * dt_s

        P_ao = P + params.Zc_mmHg_s_ml * flow_ml_s + params.L_mmHg_s2_ml * dQ

        sv = float(np.trapezoid(flow_ml_s, dx=dt_s))
        ea = (np.max(P_ao) - np.min(P_ao)) / max(sv, 1e-6)  # Sunagawa Ea ~ Pes_amplitude/SV approx

        return {
            "P_windkessel_mmHg": P.astype(np.float32),
            "P_aortic_mmHg": P_ao.astype(np.float32),
            "effective_arterial_elastance": float(ea),
            "mean_pressure_mmHg": float(np.mean(P_ao)),
            "sys_pressure_mmHg": float(np.max(P_ao)),
            "dia_pressure_mmHg": float(np.min(P_ao)),
        }