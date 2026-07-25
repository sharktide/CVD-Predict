"""
Disease progression models for the OHCA Predictor simulator.

Implements stochastic differential equation (SDE) based models for 20 disease
types that can contribute to out-of-hospital cardiac arrest. Each model
simulates realistic pathophysiological progression using Euler-Maruyama
discretization of dS = mu(S,t)*dt + sigma(S,t)*dW.

Safety-critical medical simulation — all parameters derived from clinical
literature and validated against known disease trajectories.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from ohca_predictor.config import OHCAConfig, get_config


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DiseaseType(Enum):
    """All disease types that can contribute to OHCA."""

    ACS_UNSTABLE_ANGINA = "acs_unstable_angina"
    ACS_NSTEMI = "acs_nstemi"
    ACS_STEMI = "acs_stemi"
    DCM = "dcm"
    HCM = "hcm"
    ARVC = "arvc"
    BRUGADA = "brugada"
    LQTS = "lqts"
    SQTS = "sqts"
    CPVT = "cpvt"
    AF = "af"
    VT = "vt"
    VF = "vf"
    COMPLETE_HEART_BLOCK = "complete_heart_block"
    HYPERKALEMIA = "hyperkalemia"
    HYPOKALEMIA = "hypokalemia"
    PE = "pe"
    RESPIRATORY_FAILURE = "respiratory_failure"
    SEPSIS = "sepsis"
    DRUG_TOXICITY = "drug_toxicity"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DiseaseState:
    """Represents the current state of a disease in the simulation."""

    disease_type: DiseaseType
    onset_time_hours: float
    severity: float  # [0.0, 1.0]
    progression_rate: float
    stage: int
    complications: List[str] = field(default_factory=list)
    is_active: bool = True
    probability_of_collapse: float = 0.0
    latent_pathway: np.ndarray = field(default_factory=lambda: np.zeros(64))


@dataclass
class DiseaseEvent:
    """Records a discrete event during disease progression."""

    disease_type: DiseaseType
    time_hours: float
    event_type: str  # "onset", "progression", "complication", "resolution"
    severity_change: float
    description: str


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class DiseaseModel(ABC):
    """
    Abstract base class for all disease progression models.

    Each concrete model implements pathophysiology via SDEs:
        dS = mu(S, t) * dt + sigma(S, t) * dW

    where W is a standard Wiener process.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        patient_state: Dict[str, Any],
        config: OHCAConfig,
    ) -> None:
        self.rng = rng
        self.patient_state = patient_state
        self.config = config
        self.disease_config = config.disease
        self.events: List[DiseaseEvent] = []
        self._active_diseases: Dict[DiseaseType, DiseaseState] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @abstractmethod
    def update(
        self,
        dt: float,
        current_time: float,
        patient_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Advance the model by *dt* hours and return modified patient_state."""

    @abstractmethod
    def compute_collapse_probability(
        self, patient_state: Dict[str, Any]
    ) -> float:
        """Return probability [0,1] of imminent cardiac arrest."""

    # ------------------------------------------------------------------
    # SDE helpers
    # ------------------------------------------------------------------

    def euler_maruyama_step(
        self,
        value: float,
        mu: float,
        sigma: float,
        dt: float,
    ) -> float:
        """Single Euler-Maruyama step: value + mu*dt + sigma*sqrt(dt)*N(0,1)."""
        noise = self.rng.standard_normal()
        return value + mu * dt + sigma * np.sqrt(dt) * noise

    def sde_step_clamped(
        self,
        value: float,
        mu: float,
        sigma: float,
        dt: float,
        lo: float = 0.0,
        hi: float = 1.0,
    ) -> float:
        """Euler-Maruyama step with hard bounds [lo, hi]."""
        new_val = self.euler_maruyama_step(value, mu, sigma, dt)
        return float(np.clip(new_val, lo, hi))

    def wiener_increment(self, dt: float) -> float:
        """Standard Wiener-process increment scaled by sqrt(dt)."""
        return float(self.rng.standard_normal() * np.sqrt(dt))

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    def _record_event(
        self,
        disease_type: DiseaseType,
        time_hours: float,
        event_type: str,
        severity_change: float,
        description: str,
    ) -> None:
        self.events.append(
            DiseaseEvent(
                disease_type=disease_type,
                time_hours=time_hours,
                event_type=event_type,
                severity_change=severity_change,
                description=description,
            )
        )

    # ------------------------------------------------------------------
    # Interaction bookkeeping
    # ------------------------------------------------------------------

    def set_active_diseases(
        self, diseases: Dict[DiseaseType, DiseaseState]
    ) -> None:
        """Provide the model with a snapshot of all currently active diseases."""
        self._active_diseases = diseases

    def _has_comorbidity(self, dtype: DiseaseType) -> bool:
        return dtype in self._active_diseases and self._active_diseases[dtype].is_active

    # ------------------------------------------------------------------
    # Latent pathway helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_latent(dim: int = 64) -> np.ndarray:
        return np.zeros(dim, dtype=np.float64)

    @staticmethod
    def _update_latent(
        latent: np.ndarray, mu: np.ndarray, sigma: np.ndarray, dt: float, rng: np.random.Generator
    ) -> np.ndarray:
        noise = rng.standard_normal(size=latent.shape)
        return latent + mu * dt + sigma * np.sqrt(dt) * noise


# =========================================================================
#  A.  ACS FAMILY
# =========================================================================

class UnstableAnginaModel(DiseaseModel):
    """
    Partial coronary occlusion with ischemic burden progression.

    Clinical trajectory:
    - Ischemic burden rises from 0 -> 0.6 over hours.
    - Troponin shows a gradual (but sub-diagnostic) rise.
    - HR increases, HRV decreases.
    - 5-15% risk of progression to STEMI.
    - 2-5% risk of VT/VF.
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.ischemic_burden: float = 0.0
        self.troponin: float = 0.0
        self.hr: float = float(patient_state.get("heart_rate", 72.0))
        self.hrv: float = float(patient_state.get("hrv_sdnn", 40.0))
        self.stage: int = 0
        self.severity: float = 0.05
        self.latent = self._init_latent()
        self._progression_to_stemi_risk: float = 0.0
        self._vt_vf_risk: float = 0.0

    def _mu_ischemic(self, s: float) -> float:
        return 0.02 * (1.0 - s / 0.6)

    def _sigma_ischemic(self, s: float) -> float:
        return 0.005 * (1.0 + s)

    def update(
        self, dt: float, current_time: float, patient_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.severity >= 0.95:
            return patient_state

        mu_i = self._mu_ischemic(self.ischemic_burden)
        sig_i = self._sigma_ischemic(self.ischemic_burden)
        self.ischemic_burden = self.sde_step_clamped(
            self.ischemic_burden, mu_i, sig_i, dt, 0.0, 1.0
        )

        tau_trop = 18.0
        self.troponin += (0.6 - self.troponin) * (dt / tau_trop) + 0.002 * self.wiener_increment(dt)
        self.troponin = float(np.clip(self.troponin, 0.0, 5.0))

        self.hr += 0.3 * self.ischemic_burden * dt + 0.05 * self.wiener_increment(dt)
        self.hr = float(np.clip(self.hr, 55.0, 180.0))
        self.hrv -= 0.1 * self.ischemic_burden * dt + 0.02 * self.wiener_increment(dt)
        self.hrv = float(np.clip(self.hrv, 5.0, 80.0))

        self.stage = min(3, int(self.ischemic_burden / 0.2))
        self.severity = float(np.clip(self.ischemic_burden / 0.6, 0.0, 1.0))

        self._progression_to_stemi_risk = min(0.15, 0.05 + 0.10 * self.ischemic_burden)
        self._vt_vf_risk = min(0.05, 0.02 + 0.03 * self.ischemic_burden)

        if self.stage >= 1 and "ischemic_pain" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("ischemic_pain")
            self._record_event(DiseaseType.ACS_UNSTABLE_ANGINA, current_time, "complication", 0.05, "Ischemic chest pain at rest")

        if self.rng.random() < self._progression_to_stemi_risk * dt:
            patient_state.setdefault("complications", []).append("progression_to_stemi")
            self._record_event(DiseaseType.ACS_UNSTABLE_ANGINA, current_time, "complication", 0.20, "Progression toward STEMI")

        if self.rng.random() < self._vt_vf_risk * dt:
            patient_state.setdefault("complications", []).append("vt_risk_elevated")
            self._record_event(DiseaseType.ACS_UNSTABLE_ANGINA, current_time, "complication", 0.15, "VT/VF risk elevated")

        patient_state["ischemic_burden"] = self.ischemic_burden
        patient_state["troponin"] = self.troponin
        patient_state["heart_rate"] = self.hr
        patient_state["hrv_sdnn"] = self.hrv
        patient_state["st_depression"] = float(np.clip(0.5 * self.ischemic_burden, 0.0, 3.0))
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(
            self.latent,
            mu=np.full(64, mu_i * 0.01),
            sigma=np.full(64, sig_i * 0.005),
            dt=dt,
            rng=self.rng,
        )
        patient_state["latent_pathway"] = self.latent

        return patient_state

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.005
        risk = base + 0.02 * self.ischemic_burden + 0.01 * max(0, self.troponin - 0.3)
        if self._has_comorbidity(DiseaseType.DCM):
            risk *= 1.4
        if self._has_comorbidity(DiseaseType.HYPERKALEMIA):
            risk *= 1.3
        return float(np.clip(risk, 0.0, 1.0))


class NSTEMIModel(DiseaseModel):
    """
    Subendocardial ischemia — NSTEMI.

    - Troponin rises from 0 -> 50 ng/L over 6-12 h.
    - ST-segment depression in contiguous leads.
    - Risk of progression to transmural infarction.
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.ischemic_burden: float = 0.1
        self.troponin: float = 0.0
        self.st_depression: float = 0.0
        self.hr: float = float(patient_state.get("heart_rate", 78.0))
        self.hrv: float = float(patient_state.get("hrv_sdnn", 35.0))
        self.stage: int = 0
        self.severity: float = 0.1
        self.latent = self._init_latent()
        self._onset_achieved = False

    def _mu_ischemic(self, s: float) -> float:
        return 0.05 * (1.0 - s / 0.85)

    def _sigma_ischemic(self, s: float) -> float:
        return 0.008 * (1.0 + 0.5 * s)

    def update(
        self, dt: float, current_time: float, patient_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.severity >= 0.98:
            return patient_state

        mu_i = self._mu_ischemic(self.ischemic_burden)
        sig_i = self._sigma_ischemic(self.ischemic_burden)
        self.ischemic_burden = self.sde_step_clamped(
            self.ischemic_burden, mu_i, sig_i, dt, 0.0, 1.0
        )

        k_trop = 0.25
        self.troponin += k_trop * (50.0 - self.troponin) * dt + 0.3 * self.wiener_increment(dt)
        self.troponin = float(np.clip(self.troponin, 0.0, 200.0))

        self.st_depression = float(np.clip(
            1.5 * self.ischemic_burden + 0.05 * self.wiener_increment(dt),
            0.0, 4.0,
        ))

        self.hr += 0.4 * self.ischemic_burden * dt + 0.08 * self.wiener_increment(dt)
        self.hr = float(np.clip(self.hr, 60.0, 180.0))
        self.hrv -= 0.15 * self.ischemic_burden * dt + 0.03 * self.wiener_increment(dt)
        self.hrv = float(np.clip(self.hrv, 5.0, 70.0))

        self.stage = min(4, int(self.troponin / 12.5))
        self.severity = float(np.clip(self.ischemic_burden * 0.7 + (self.troponin / 50.0) * 0.3, 0.0, 1.0))

        if not self._onset_achieved and self.troponin > 5.0:
            self._onset_achieved = True
            self._record_event(DiseaseType.ACS_NSTEMI, current_time, "onset", 0.10, "Troponin above diagnostic threshold")

        if self.stage >= 2 and "dynamic_st_changes" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("dynamic_st_changes")
            self._record_event(DiseaseType.ACS_NSTEMI, current_time, "complication", 0.05, "Dynamic ST depression evolution")

        if self.rng.random() < 0.01 * dt:
            patient_state.setdefault("complications", []).append("af_risk_elevated")
            self._record_event(DiseaseType.ACS_NSTEMI, current_time, "complication", 0.03, "New-onset AF risk elevated")

        patient_state["ischemic_burden"] = self.ischemic_burden
        patient_state["troponin"] = self.troponin
        patient_state["st_depression"] = self.st_depression
        patient_state["heart_rate"] = self.hr
        patient_state["hrv_sdnn"] = self.hrv
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(
            self.latent,
            mu=np.full(64, mu_i * 0.015),
            sigma=np.full(64, sig_i * 0.008),
            dt=dt,
            rng=self.rng,
        )
        patient_state["latent_pathway"] = self.latent

        return patient_state

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.01
        risk = base + 0.03 * self.ischemic_burden + 0.002 * self.troponin / 50.0
        if self.troponin > 30.0:
            risk += 0.03
        if self._has_comorbidity(DiseaseType.DCM):
            risk *= 1.5
        if self._has_comorbidity(DiseaseType.HYPERKALEMIA):
            risk *= 1.3
        return float(np.clip(risk, 0.0, 1.0))


class STEMIModel(DiseaseModel):
    """
    Complete coronary occlusion — STEMI.

    - Ischemic burden 0.4 -> 1.0 over 30-60 min.
    - Rapid troponin rise.
    - ST elevation, Killip class progression.
    - VT/VF 10-15%, OHCA 8-12% in 24 h.
    - EF reduction.
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.ischemic_burden: float = 0.4
        self.troponin: float = 2.0
        self.st_elevation: float = 1.0
        self.hr: float = float(patient_state.get("heart_rate", 90.0))
        self.bp_systolic: float = float(patient_state.get("bp_systolic", 110.0))
        self.ef: float = float(patient_state.get("ef", 0.55))
        self.killip: int = 1
        self.stage: int = 0
        self.severity: float = 0.3
        self.latent = self._init_latent()
        self._vt_vf_risk: float = 0.0
        self._ohca_risk_24h: float = 0.0

    def _mu_ischemic(self, s: float) -> float:
        return 0.15 * (1.0 - s)

    def _sigma_ischemic(self, s: float) -> float:
        return 0.02 * (1.0 + s)

    def update(
        self, dt: float, current_time: float, patient_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.severity >= 0.99:
            return patient_state

        mu_i = self._mu_ischemic(self.ischemic_burden)
        sig_i = self._sigma_ischemic(self.ischemic_burden)
        self.ischemic_burden = self.sde_step_clamped(
            self.ischemic_burden, mu_i, sig_i, dt, 0.0, 1.0
        )

        k_trop = 0.4
        self.troponin += k_trop * (200.0 - self.troponin) * dt + 1.0 * self.wiener_increment(dt)
        self.troponin = float(np.clip(self.troponin, 0.0, 1000.0))

        self.st_elevation = float(np.clip(
            3.0 * self.ischemic_burden + 0.1 * self.wiener_increment(dt),
            0.0, 8.0,
        ))

        self.hr += 0.8 * self.ischemic_burden * dt + 0.15 * self.wiener_increment(dt)
        self.hr = float(np.clip(self.hr, 50.0, 200.0))
        self.bp_systolic -= 0.5 * self.ischemic_burden * dt + 0.1 * self.wiener_increment(dt)
        self.bp_systolic = float(np.clip(self.bp_systolic, 60.0, 180.0))

        self.ef -= 0.02 * self.ischemic_burden * dt + 0.005 * self.wiener_increment(dt)
        self.ef = float(np.clip(self.ef, 0.05, 0.70))

        if self.bp_systolic < 90:
            self.killip = max(self.killip, 3)
        elif self.bp_systolic < 100:
            self.killip = max(self.killip, 2)
        if self.ef < 0.30:
            self.killip = max(self.killip, 3)

        self.stage = min(5, int(self.ischemic_burden * 5))
        self.severity = float(np.clip(
            self.ischemic_burden * 0.5
            + max(0, self.troponin - 50) / 950 * 0.2
            + (5 - self.killip) / 5 * 0.1
            + max(0, 0.55 - self.ef) / 0.5 * 0.2,
            0.0, 1.0,
        ))

        self._vt_vf_risk = min(0.15, 0.10 + 0.05 * self.ischemic_burden)
        self._ohca_risk_24h = min(0.12, 0.08 + 0.04 * self.ischemic_burden)

        if self.rng.random() < self._vt_vf_risk * dt:
            patient_state.setdefault("complications", []).append("vt_vf_event")
            self._record_event(DiseaseType.ACS_STEMI, current_time, "complication", 0.25, "VT/VF episode during STEMI")

        if self.rng.random() < self._ohca_risk_24h * dt / 24.0:
            patient_state["ohca_event"] = True
            self._record_event(DiseaseType.ACS_STEMI, current_time, "complication", 0.50, "OHCA event within 24h window")

        if self.killip >= 3 and "cardiogenic_shock" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("cardiogenic_shock")
            self._record_event(DiseaseType.ACS_STEMI, current_time, "complication", 0.30, f"Killip class {self.killip} — cardiogenic shock")

        patient_state["ischemic_burden"] = self.ischemic_burden
        patient_state["troponin"] = self.troponin
        patient_state["st_elevation"] = self.st_elevation
        patient_state["heart_rate"] = self.hr
        patient_state["bp_systolic"] = self.bp_systolic
        patient_state["ef"] = self.ef
        patient_state["killip_class"] = self.killip
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(
            self.latent,
            mu=np.full(64, mu_i * 0.02),
            sigma=np.full(64, sig_i * 0.015),
            dt=dt,
            rng=self.rng,
        )
        patient_state["latent_pathway"] = self.latent

        return patient_state

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.05
        risk = (
            base
            + 0.10 * self.ischemic_burden
            + 0.005 * max(0, self.troponin - 50) / 100.0
            + 0.05 * (1.0 - self.ef)
            + 0.03 * (self.killip - 1)
        )
        if self.bp_systolic < 80:
            risk += 0.10
        if self._has_comorbidity(DiseaseType.HYPERKALEMIA):
            risk *= 1.5
        if self._has_comorbidity(DiseaseType.RESPIRATORY_FAILURE):
            risk *= 1.4
        return float(np.clip(risk, 0.0, 1.0))


# =========================================================================
#  B.  HEART FAILURE FAMILY
# =========================================================================

class DCMModel(DiseaseModel):
    """
    Dilated cardiomyopathy — progressive EF reduction.

    - EF drops from 0.50 -> 0.15 over weeks-months.
    - Neurohormonal activation (elevated catecholamines, RAAS).
    - BNP elevation.
    - OHCA risk dramatically increases when EF < 0.25.
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.ef: float = float(patient_state.get("ef", 0.50))
        self.bnp: float = 50.0
        self.neurohormonal: float = 0.2
        self.hr: float = float(patient_state.get("heart_rate", 75.0))
        self.stage: int = 1
        self.severity: float = 0.2
        self.latent = self._init_latent()
        self._ef_critical: bool = False

    def _mu_ef(self, ef: float) -> float:
        return -0.005 * (ef - 0.15)

    def _sigma_ef(self, ef: float) -> float:
        return 0.003 * max(0.1, ef)

    def update(
        self, dt: float, current_time: float, patient_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        if self.severity >= 0.98:
            return patient_state

        mu_e = self._mu_ef(self.ef)
        sig_e = self._sigma_ef(self.ef)
        self.ef = self.sde_step_clamped(self.ef, mu_e, sig_e, dt, 0.05, 0.70)

        self.bnp += 15.0 * (0.50 - self.ef) * dt + 2.0 * self.wiener_increment(dt)
        self.bnp = float(np.clip(self.bnp, 10.0, 2000.0))

        self.neurohormonal += 0.01 * (0.50 - self.ef) * dt + 0.002 * self.wiener_increment(dt)
        self.neurohormonal = float(np.clip(self.neurohormonal, 0.0, 1.0))

        self.hr += 0.2 * (1.0 - self.ef) * dt + 0.05 * self.wiener_increment(dt)
        self.hr = float(np.clip(self.hr, 55.0, 160.0))

        self.stage = min(5, max(1, int((0.50 - self.ef) / 0.08)))
        self.severity = float(np.clip((0.50 - self.ef) / 0.45, 0.0, 1.0))

        self._ef_critical = self.ef < 0.25
        if self._ef_critical and "critical_ef" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("critical_ef")
            self._record_event(DiseaseType.DCM, current_time, "complication", 0.20, f"EF critically low: {self.ef:.2f}")

        if self.ef < 0.30 and "vt_dcm_risk" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("vt_dcm_risk")
            self._record_event(DiseaseType.DCM, current_time, "complication", 0.10, "VT risk elevated due to low EF")

        patient_state["ef"] = self.ef
        patient_state["bnp"] = self.bnp
        patient_state["neurohormonal_activation"] = self.neurohormonal
        patient_state["heart_rate"] = self.hr
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(
            self.latent,
            mu=np.full(64, mu_e * 0.01),
            sigma=np.full(64, sig_e * 0.008),
            dt=dt,
            rng=self.rng,
        )
        patient_state["latent_pathway"] = self.latent

        return patient_state

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.005
        risk = base
        if self.ef < 0.25:
            risk += 0.15 + 0.20 * (0.25 - self.ef) / 0.20
        elif self.ef < 0.35:
            risk += 0.05 + 0.10 * (0.35 - self.ef) / 0.10
        risk += 0.02 * self.neurohormonal
        if self._has_comorbidity(DiseaseType.ACS_STEMI):
            risk *= 1.6
        if self._has_comorbidity(DiseaseType.HYPERKALEMIA):
            risk *= 1.4
        return float(np.clip(risk, 0.0, 1.0))


class HCMModel(DiseaseModel):
    """
    Hypertrophic cardiomyopathy.

    - Asymmetric septal hypertrophy (wall thickness 15-35 mm).
    - Increased myocardial irritability.
    - LVOT obstruction under certain conditions.
    - Sudden cardiac death risk.
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.wall_thickness: float = float(patient_state.get("septal_wall_thickness", 18.0))
        self.lvot_gradient: float = 20.0
        self.myocardial_irritability: float = 0.3
        self.hr: float = float(patient_state.get("heart_rate", 72.0))
        self.stage: int = 0
        self.severity: float = 0.15
        self.latent = self._init_latent()
        self._lvot_obstruction: bool = False

    def _mu_wall(self, w: float) -> float:
        return 0.002 * (25.0 - w)

    def _sigma_wall(self, w: float) -> float:
        return 0.05

    def update(
        self, dt: float, current_time: float, patient_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        mu_w = self._mu_wall(self.wall_thickness)
        sig_w = self._sigma_wall(self.wall_thickness)
        self.wall_thickness = self.sde_step_clamped(
            self.wall_thickness, mu_w, sig_w, dt, 12.0, 40.0
        )

        self.lvot_gradient += 0.5 * max(0, self.wall_thickness - 20) * dt + 1.0 * self.wiener_increment(dt)
        self.lvot_gradient = float(np.clip(self.lvot_gradient, 0.0, 200.0))
        self._lvot_obstruction = self.lvot_gradient > 30.0

        self.myocardial_irritability += 0.005 * (self.wall_thickness - 15) / 20.0 * dt + 0.002 * self.wiener_increment(dt)
        self.myocardial_irritability = float(np.clip(self.myocardial_irritability, 0.1, 1.0))

        if self._lvot_obstruction:
            self.hr += 0.3 * dt + 0.1 * self.wiener_increment(dt)
        else:
            self.hr += 0.05 * self.wiener_increment(dt)
        self.hr = float(np.clip(self.hr, 55.0, 180.0))

        self.stage = min(4, int((self.wall_thickness - 15) / 5))
        self.severity = float(np.clip(
            (self.wall_thickness - 15) / 20.0 * 0.4
            + self.myocardial_irritability * 0.3
            + min(1.0, self.lvot_gradient / 100.0) * 0.3,
            0.0, 1.0,
        ))

        if self._lvot_obstruction and "lvot_obstruction" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("lvot_obstruction")
            self._record_event(DiseaseType.HCM, current_time, "complication", 0.15, f"LVOT gradient {self.lvot_gradient:.0f} mmHg")

        if self.myocardial_irritability > 0.6 and "hcm_arrhythmia_risk" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("hcm_arrhythmia_risk")
            self._record_event(DiseaseType.HCM, current_time, "complication", 0.10, "High myocardial irritability — arrhythmia risk")

        patient_state["septal_wall_thickness"] = self.wall_thickness
        patient_state["lvot_gradient"] = self.lvot_gradient
        patient_state["myocardial_irritability"] = self.myocardial_irritability
        patient_state["heart_rate"] = self.hr
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(
            self.latent,
            mu=np.full(64, mu_w * 0.005),
            sigma=np.full(64, sig_w * 0.01),
            dt=dt,
            rng=self.rng,
        )
        patient_state["latent_pathway"] = self.latent

        return patient_state

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.01
        risk = base
        risk += 0.03 * max(0, (self.wall_thickness - 20) / 15.0)
        risk += 0.02 * max(0, self.lvot_gradient - 30) / 100.0
        risk += 0.02 * max(0, self.myocardial_irritability - 0.5)
        if self._has_comorbidity(DiseaseType.AF):
            risk += 0.02
        if self._has_comorbidity(DiseaseType.HYPERKALEMIA):
            risk *= 1.3
        return float(np.clip(risk, 0.0, 1.0))


class ARVCModel(DiseaseModel):
    """
    Arrhythmogenic right ventricular cardiomyopathy.

    - Fibrofatty replacement of RV myocardium.
    - RV dilation.
    - LBBB-morphology VT.
    - Exercise-triggered arrhythmias.
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.rv_dilation: float = 0.2
        self.fibrofatty_fraction: float = 0.1
        self.rv_ef: float = 0.55
        self.hr: float = float(patient_state.get("heart_rate", 72.0))
        self.exercise_load: float = 0.0
        self.stage: int = 0
        self.severity: float = 0.1
        self.latent = self._init_latent()
        self._vt_triggered: bool = False

    def _mu_fibro(self, f: float) -> float:
        return 0.003 * (0.6 - f)

    def _sigma_fibro(self, f: float) -> float:
        return 0.004 * (1.0 + f)

    def update(
        self, dt: float, current_time: float, patient_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        mu_f = self._mu_fibro(self.fibrofatty_fraction)
        sig_f = self._sigma_fibro(self.fibrofatty_fraction)
        self.fibrofatty_fraction = self.sde_step_clamped(
            self.fibrofatty_fraction, mu_f, sig_f, dt, 0.0, 0.8
        )

        self.rv_dilation += 0.005 * self.fibrofatty_fraction * dt + 0.001 * self.wiener_increment(dt)
        self.rv_dilation = float(np.clip(self.rv_dilation, 0.0, 1.0))

        self.rv_ef -= 0.003 * self.fibrofatty_fraction * dt + 0.001 * self.wiener_increment(dt)
        self.rv_ef = float(np.clip(self.rv_ef, 0.10, 0.65))

        self.hr += 0.1 * self.fibrofatty_fraction * dt + 0.05 * self.wiener_increment(dt)
        self.hr = float(np.clip(self.hr, 55.0, 180.0))

        self.exercise_load = float(np.clip(
            patient_state.get("exercise_level", 0.0), 0.0, 1.0
        ))

        self.stage = min(4, int(self.fibrofatty_fraction / 0.15))
        self.severity = float(np.clip(
            self.fibrofatty_fraction * 0.5
            + (1.0 - self.rv_ef / 0.55) * 0.3
            + self.rv_dilation * 0.2,
            0.0, 1.0,
        ))

        vt_threshold = 0.4 - 0.2 * self.exercise_load
        if self.fibrofatty_fraction > vt_threshold and self.rng.random() < 0.02 * dt:
            patient_state.setdefault("complications", []).append("lbbb_vt")
            self._vt_triggered = True
            self._record_event(DiseaseType.ARVC, current_time, "complication", 0.20, "LBBB-morphology VT episode")

        if self.rv_dilation > 0.5 and "rv_dilation_significant" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("rv_dilation_significant")
            self._record_event(DiseaseType.ARVC, current_time, "complication", 0.10, "Significant RV dilation")

        patient_state["rv_dilation"] = self.rv_dilation
        patient_state["fibrofatty_fraction"] = self.fibrofatty_fraction
        patient_state["rv_ef"] = self.rv_ef
        patient_state["heart_rate"] = self.hr
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(
            self.latent,
            mu=np.full(64, mu_f * 0.01),
            sigma=np.full(64, sig_f * 0.008),
            dt=dt,
            rng=self.rng,
        )
        patient_state["latent_pathway"] = self.latent

        return patient_state

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.008
        risk = base
        risk += 0.04 * self.fibrofatty_fraction
        risk += 0.02 * max(0, 1.0 - self.rv_ef / 0.55)
        risk += 0.03 * self.exercise_load
        if self._vt_triggered:
            risk += 0.05
        if self._has_comorbidity(DiseaseType.AF):
            risk += 0.01
        return float(np.clip(risk, 0.0, 1.0))


# =========================================================================
#  C.  CHANNELOPATHIES
# =========================================================================

class BrugadaModel(DiseaseModel):
    """
    Brugada syndrome — sodium channel dysfunction.
    Coved ST elevation V1-V2, highest risk during sleep/rest, fever-triggered, VF risk 1-2%/year.
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.st_elevation_v1v2: float = 0.5
        self.na_channel_dysfunction: float = 0.4
        self.fever: float = float(patient_state.get("temperature", 36.8))
        self.sleep_state: float = 0.0
        self.stage: int = 0
        self.severity: float = 0.15
        self.latent = self._init_latent()
        self._vf_risk_annual: float = 0.015

    def _mu_na(self, na: float) -> float:
        return 0.002 * (0.6 - na)

    def _sigma_na(self, na: float) -> float:
        return 0.005 * (1.0 + na)

    def update(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        if self.severity >= 0.95:
            return patient_state

        mu_na = self._mu_na(self.na_channel_dysfunction)
        sig_na = self._sigma_na(self.na_channel_dysfunction)
        self.na_channel_dysfunction = self.sde_step_clamped(self.na_channel_dysfunction, mu_na, sig_na, dt, 0.1, 0.9)

        self.fever = float(patient_state.get("temperature", self.fever))
        fever_factor = max(0, (self.fever - 37.5) / 2.0)

        self.sleep_state = float(np.clip(patient_state.get("sleep_state", 0.0), 0.0, 1.0))
        sleep_factor = self.sleep_state * 0.5

        self.st_elevation_v1v2 = float(np.clip(
            1.5 * self.na_channel_dysfunction + 0.8 * fever_factor + 0.3 * sleep_factor
            + 0.05 * self.wiener_increment(dt), 0.0, 5.0,
        ))

        self.stage = min(3, int(self.na_channel_dysfunction / 0.2))
        self.severity = float(np.clip(
            self.na_channel_dysfunction * 0.5 + fever_factor * 0.3
            + min(1.0, self.st_elevation_v1v2 / 3.0) * 0.2, 0.0, 1.0,
        ))

        self._vf_risk_annual = min(0.05, 0.01 + 0.02 * self.na_channel_dysfunction + 0.02 * fever_factor)

        if fever_factor > 0.3 and self.rng.random() < 0.01 * dt * fever_factor:
            patient_state.setdefault("complications", []).append("brugada_fever_vf")
            self._record_event(DiseaseType.BRUGADA, current_time, "complication", 0.30, "Fever-triggered VF risk")

        if self.st_elevation_v1v2 > 2.0 and "coved_st_elevation" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("coved_st_elevation")
            self._record_event(DiseaseType.BRUGADA, current_time, "complication", 0.10, "Coved ST elevation V1-V2")

        patient_state["st_elevation_v1v2"] = self.st_elevation_v1v2
        patient_state["na_channel_dysfunction"] = self.na_channel_dysfunction
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(self.latent, mu=np.full(64, mu_na * 0.01), sigma=np.full(64, sig_na * 0.008), dt=dt, rng=self.rng)
        patient_state["latent_pathway"] = self.latent
        return patient_state

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.005
        risk = base + 0.02 * self.na_channel_dysfunction + 0.01 * min(1.0, self.st_elevation_v1v2 / 3.0)
        if self.fever > 38.0:
            risk += 0.02
        if self._has_comorbidity(DiseaseType.HYPERKALEMIA):
            risk *= 1.4
        if self._has_comorbidity(DiseaseType.DRUG_TOXICITY):
            risk *= 1.5
        return float(np.clip(risk, 0.0, 1.0))


class LQTSModel(DiseaseModel):
    """
    Long QT syndrome. QTc > 470ms(male)/>480ms(female), Torsades de Pointes risk.
    Subtypes: LQT1(loud noise/exercise), LQT2(stress/auditory), LQT3(sleep/rest).
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.subtype: int = int(patient_state.get("lqts_subtype", self.rng.choice([1, 2, 3])))
        self.qtc: float = float(patient_state.get("qtc", 490.0))
        self.ikr_dysfunction: float = 0.5
        self.trigger_stimulus: float = 0.0
        self.hr: float = float(patient_state.get("heart_rate", 72.0))
        self.stage: int = 0
        self.severity: float = 0.15
        self.latent = self._init_latent()
        self._tdp_risk: float = 0.0

    def _mu_qtc(self, qtc: float) -> float:
        return 0.01 * (520.0 - qtc)

    def _sigma_qtc(self, qtc: float) -> float:
        return 1.0

    def update(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        if self.severity >= 0.95:
            return patient_state

        mu_q = self._mu_qtc(self.qtc)
        sig_q = self._sigma_qtc(self.qtc)
        self.qtc = self.sde_step_clamped(self.qtc, mu_q, sig_q, dt, 400.0, 700.0)

        self.ikr_dysfunction += 0.003 * (self.qtc - 470) / 200.0 * dt + 0.001 * self.wiener_increment(dt)
        self.ikr_dysfunction = float(np.clip(self.ikr_dysfunction, 0.1, 0.9))

        self.trigger_stimulus = self._compute_trigger(patient_state)

        self.hr += 0.05 * self.wiener_increment(dt)
        self.hr = float(np.clip(self.hr, 55.0, 150.0))

        self.stage = min(4, int((self.qtc - 470) / 30))
        self.severity = float(np.clip(
            (self.qtc - 470) / 200.0 * 0.5 + self.ikr_dysfunction * 0.3
            + min(1.0, self.trigger_stimulus) * 0.2, 0.0, 1.0,
        ))

        self._tdp_risk = min(0.08, 0.01 + 0.03 * max(0, self.qtc - 480) / 200.0 + 0.02 * self.trigger_stimulus)

        if self.rng.random() < self._tdp_risk * dt:
            patient_state.setdefault("complications", []).append("torsades_de_pointes")
            self._record_event(DiseaseType.LQTS, current_time, "complication", 0.35, "Torsades de Pointes episode")

        if self.qtc > 550 and "severe_qt_prolongation" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("severe_qt_prolongation")
            self._record_event(DiseaseType.LQTS, current_time, "complication", 0.15, f"QTc critically prolonged: {self.qtc:.0f} ms")

        patient_state["qtc"] = self.qtc
        patient_state["ikr_dysfunction"] = self.ikr_dysfunction
        patient_state["heart_rate"] = self.hr
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(self.latent, mu=np.full(64, mu_q * 0.005), sigma=np.full(64, sig_q * 0.003), dt=dt, rng=self.rng)
        patient_state["latent_pathway"] = self.latent
        return patient_state

    def _compute_trigger(self, patient_state: Dict[str, Any]) -> float:
        exercise = patient_state.get("exercise_level", 0.0)
        stress = patient_state.get("stress_level", 0.0)
        noise = patient_state.get("auditory_stimulus", 0.0)
        sleep = patient_state.get("sleep_state", 0.0)
        if self.subtype == 1:
            return float(np.clip(0.6 * exercise + 0.3 * noise, 0.0, 1.0))
        elif self.subtype == 2:
            return float(np.clip(0.5 * stress + 0.3 * noise + 0.2 * exercise, 0.0, 1.0))
        else:
            return float(np.clip(0.7 * sleep + 0.1 * stress, 0.0, 1.0))

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.005
        risk = base + 0.03 * max(0, self.qtc - 480) / 200.0 + 0.02 * self.trigger_stimulus + 0.01 * self.ikr_dysfunction
        if self._has_comorbidity(DiseaseType.HYPERKALEMIA):
            risk *= 1.5
        if self._has_comorbidity(DiseaseType.DRUG_TOXICITY):
            risk *= 1.6
        return float(np.clip(risk, 0.0, 1.0))


class SQTSModel(DiseaseModel):
    """
    Short QT syndrome. QTc < 340ms, tall peaked T-waves, VF risk ~1%/year.
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.qtc: float = float(patient_state.get("qtc", 310.0))
        self.iks_gain: float = 0.6
        self.t_wave_amplitude: float = 1.5
        self.stage: int = 0
        self.severity: float = 0.1
        self.latent = self._init_latent()
        self._vf_risk_annual: float = 0.01

    def _mu_qtc(self, qtc: float) -> float:
        return 0.01 * (300.0 - qtc)

    def _sigma_qtc(self, qtc: float) -> float:
        return 0.5

    def update(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        mu_q = self._mu_qtc(self.qtc)
        sig_q = self._sigma_qtc(self.qtc)
        self.qtc = self.sde_step_clamped(self.qtc, mu_q, sig_q, dt, 200.0, 380.0)

        self.iks_gain += 0.002 * (0.34 - self.qtc / 1000.0) * dt + 0.001 * self.wiener_increment(dt)
        self.iks_gain = float(np.clip(self.iks_gain, 0.2, 0.9))

        self.t_wave_amplitude = float(np.clip(
            2.0 - 0.003 * self.qtc + 0.1 * self.wiener_increment(dt), 0.5, 4.0,
        ))

        self.stage = min(3, int((340.0 - self.qtc) / 15))
        self.severity = float(np.clip(
            (340.0 - self.qtc) / 140.0 * 0.5 + self.iks_gain * 0.3
            + min(1.0, self.t_wave_amplitude / 3.0) * 0.2, 0.0, 1.0,
        ))

        self._vf_risk_annual = min(0.04, 0.01 + 0.015 * max(0, 340 - self.qtc) / 140.0)

        if self.rng.random() < self._vf_risk_annual * dt / (24.0 * 365.0) * 100:
            patient_state.setdefault("complications", []).append("sqts_vf")
            self._record_event(DiseaseType.SQTS, current_time, "complication", 0.30, "VF in short QT syndrome")

        patient_state["qtc"] = self.qtc
        patient_state["t_wave_amplitude"] = self.t_wave_amplitude
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(self.latent, mu=np.full(64, mu_q * 0.003), sigma=np.full(64, sig_q * 0.002), dt=dt, rng=self.rng)
        patient_state["latent_pathway"] = self.latent
        return patient_state

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.003
        risk = base + 0.015 * max(0, 340 - self.qtc) / 140.0
        if self._has_comorbidity(DiseaseType.HYPERKALEMIA):
            risk *= 1.3
        return float(np.clip(risk, 0.0, 1.0))


class CPVTModel(DiseaseModel):
    """
    Catecholaminergic polymorphic ventricular tachycardia.
    Abnormal RyR2 calcium handling, bidirectional VT during exercise, normal resting ECG.
    30% OHCA by age 30 if untreated.
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.ryr2_dysfunction: float = 0.4
        self.calcium_leak: float = 0.2
        self.exercise_load: float = 0.0
        self.hr: float = float(patient_state.get("heart_rate", 72.0))
        self.stage: int = 0
        self.severity: float = 0.15
        self.latent = self._init_latent()
        self._bidirectional_vt_risk: float = 0.0

    def _mu_ryr2(self, r: float) -> float:
        return 0.003 * (0.7 - r)

    def _sigma_ryr2(self, r: float) -> float:
        return 0.005 * (1.0 + r)

    def update(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        if self.severity >= 0.95:
            return patient_state

        mu_r = self._mu_ryr2(self.ryr2_dysfunction)
        sig_r = self._sigma_ryr2(self.ryr2_dysfunction)
        self.ryr2_dysfunction = self.sde_step_clamped(self.ryr2_dysfunction, mu_r, sig_r, dt, 0.1, 0.95)

        self.calcium_leak += 0.005 * self.ryr2_dysfunction * dt + 0.002 * self.wiener_increment(dt)
        self.calcium_leak = float(np.clip(self.calcium_leak, 0.0, 1.0))

        self.exercise_load = float(np.clip(patient_state.get("exercise_level", 0.0), 0.0, 1.0))
        hr_from_exercise = 72.0 + 100.0 * self.exercise_load
        self.hr += 0.5 * (hr_from_exercise - self.hr) * dt + 0.1 * self.wiener_increment(dt)
        self.hr = float(np.clip(self.hr, 55.0, 200.0))

        self.stage = min(4, int(self.ryr2_dysfunction / 0.2))
        self.severity = float(np.clip(
            self.ryr2_dysfunction * 0.4 + self.calcium_leak * 0.3 + self.exercise_load * 0.3, 0.0, 1.0,
        ))

        self._bidirectional_vt_risk = min(0.15, 0.05 * self.ryr2_dysfunction * (1.0 + 2.0 * self.exercise_load))

        if self.rng.random() < self._bidirectional_vt_risk * dt:
            patient_state.setdefault("complications", []).append("bidirectional_vt")
            self._record_event(DiseaseType.CPVT, current_time, "complication", 0.30, "Bidirectional VT during exercise")

        if self.ryr2_dysfunction > 0.7 and self.exercise_load > 0.5 and "cpvt_high_risk" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("cpvt_high_risk")
            self._record_event(DiseaseType.CPVT, current_time, "complication", 0.20, "High-risk CPVT: severe RyR2 dysfunction with exercise")

        patient_state["ryr2_dysfunction"] = self.ryr2_dysfunction
        patient_state["calcium_leak"] = self.calcium_leak
        patient_state["heart_rate"] = self.hr
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(self.latent, mu=np.full(64, mu_r * 0.01), sigma=np.full(64, sig_r * 0.008), dt=dt, rng=self.rng)
        patient_state["latent_pathway"] = self.latent
        return patient_state

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.008
        risk = base + 0.04 * self.ryr2_dysfunction * (1.0 + self.exercise_load) + 0.02 * self.calcium_leak
        if self._has_comorbidity(DiseaseType.HYPERKALEMIA):
            risk *= 1.4
        if self._has_comorbidity(DiseaseType.DRUG_TOXICITY):
            risk *= 1.3
        return float(np.clip(risk, 0.0, 1.0))


# =========================================================================
#  D.  TACHYARRHYTHMIAS
# =========================================================================

class AFModel(DiseaseModel):
    """
    Atrial fibrillation. Chaotic atrial activity, irregularly irregular ventricular response.
    RVR vs controlled, stroke risk, tachycardia-mediated cardiomyopathy over time.
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.af_burden: float = 0.3
        self.ventricular_rate: float = float(patient_state.get("heart_rate", 110.0))
        self.irregularity: float = 0.5
        self.stroke_risk_chads: float = float(patient_state.get("chads2_score", 1.0))
        self.ef: float = float(patient_state.get("ef", 0.55))
        self.stage: int = 0
        self.severity: float = 0.15
        self.latent = self._init_latent()
        self._rvr: bool = True

    def _mu_burden(self, b: float) -> float:
        return 0.005 * (0.7 - b)

    def _sigma_burden(self, b: float) -> float:
        return 0.01 * (1.0 + b)

    def update(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        if self.severity >= 0.95:
            return patient_state

        mu_b = self._mu_burden(self.af_burden)
        sig_b = self._sigma_burden(self.af_burden)
        self.af_burden = self.sde_step_clamped(self.af_burden, mu_b, sig_b, dt, 0.0, 1.0)

        rate_target = 80.0 + 80.0 * self.af_burden
        self.ventricular_rate += 0.5 * (rate_target - self.ventricular_rate) * dt + 1.0 * self.wiener_increment(dt)
        self.ventricular_rate = float(np.clip(self.ventricular_rate, 50.0, 220.0))
        self._rvr = self.ventricular_rate > 100.0

        self.irregularity = float(np.clip(0.3 + 0.5 * self.af_burden + 0.1 * self.wiener_increment(dt), 0.0, 1.0))

        if self._rvr and self.af_burden > 0.5:
            self.ef -= 0.001 * self.af_burden * dt + 0.0005 * self.wiener_increment(dt)
            self.ef = float(np.clip(self.ef, 0.10, 0.70))

        self.stage = min(4, int(self.af_burden / 0.25))
        self.severity = float(np.clip(
            self.af_burden * 0.4 + min(1.0, max(0, self.ventricular_rate - 80) / 120.0) * 0.3
            + max(0, 1.0 - self.ef / 0.55) * 0.3, 0.0, 1.0,
        ))

        if self._rvr and "af_rvr" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("af_rvr")
            self._record_event(DiseaseType.AF, current_time, "complication", 0.08, f"RVR: HR {self.ventricular_rate:.0f}")

        if self.af_burden > 0.7 and "tachycardia_mediomyopathy_risk" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("tachycardia_mediomyopathy_risk")
            self._record_event(DiseaseType.AF, current_time, "complication", 0.10, "Tachycardia-mediated cardiomyopathy risk")

        if self.stroke_risk_chads >= 2 and self.rng.random() < 0.001 * dt:
            patient_state.setdefault("complications", []).append("stroke_risk_high")
            self._record_event(DiseaseType.AF, current_time, "complication", 0.15, f"High stroke risk (CHADS2={self.stroke_risk_chads:.0f})")

        patient_state["af_burden"] = self.af_burden
        patient_state["ventricular_rate"] = self.ventricular_rate
        patient_state["heart_rate"] = self.ventricular_rate
        patient_state["irregularity"] = self.irregularity
        patient_state["ef"] = self.ef
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(self.latent, mu=np.full(64, mu_b * 0.01), sigma=np.full(64, sig_b * 0.008), dt=dt, rng=self.rng)
        patient_state["latent_pathway"] = self.latent
        return patient_state

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.003
        risk = base
        if self._rvr:
            risk += 0.01 * min(1.0, max(0, self.ventricular_rate - 100) / 120.0)
        risk += 0.02 * max(0, 1.0 - self.ef / 0.55)
        if self._has_comorbidity(DiseaseType.DCM):
            risk *= 1.5
        if self._has_comorbidity(DiseaseType.HYPERKALEMIA):
            risk *= 1.3
        return float(np.clip(risk, 0.0, 1.0))


class VTModel(DiseaseModel):
    """
    Ventricular tachycardia. Reentrant circuit, monomorphic/polymorphic,
    hemodynamic compromise, pre-arrest precursor to VF.
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.vt_burden: float = 0.0
        self.vt_rate: float = 180.0
        self.is_polymorphic: bool = False
        self.bp_systolic: float = float(patient_state.get("bp_systolic", 110.0))
        self.ef: float = float(patient_state.get("ef", 0.50))
        self.stage: int = 0
        self.severity: float = 0.2
        self.latent = self._init_latent()
        self._vf_conversion_risk: float = 0.0

    def _mu_burden(self, b: float) -> float:
        return 0.03 * (1.0 - b) * (1.0 - b)

    def _sigma_burden(self, b: float) -> float:
        return 0.02 * (1.0 + 2.0 * b)

    def update(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        if self.severity >= 0.99:
            return patient_state

        mu_b = self._mu_burden(self.vt_burden)
        sig_b = self._sigma_burden(self.vt_burden)
        self.vt_burden = self.sde_step_clamped(self.vt_burden, mu_b, sig_b, dt, 0.0, 1.0)

        self.vt_rate += 2.0 * self.vt_burden * dt + 1.0 * self.wiener_increment(dt)
        self.vt_rate = float(np.clip(self.vt_rate, 120.0, 280.0))

        if self.vt_burden > 0.3:
            self.is_polymorphic = self.rng.random() < 0.3 * self.vt_burden

        compromise = min(1.0, self.vt_burden * 0.8 + (280.0 - self.vt_rate) / 160.0 * 0.2)
        self.bp_systolic = float(np.clip(110.0 - 50.0 * compromise + 5.0 * self.wiener_increment(dt), 50.0, 140.0))

        self.ef -= 0.005 * self.vt_burden * dt + 0.002 * self.wiener_increment(dt)
        self.ef = float(np.clip(self.ef, 0.05, 0.65))

        self.stage = min(5, int(self.vt_burden * 5))
        self.severity = float(np.clip(
            self.vt_burden * 0.5 + min(1.0, max(0, 110.0 - self.bp_systolic) / 60.0) * 0.3
            + (0.5 - min(0.5, self.ef)) * 0.4, 0.0, 1.0,
        ))

        self._vf_conversion_risk = min(0.10, 0.02 + 0.06 * self.vt_burden + 0.02 * int(self.is_polymorphic))

        if self.rng.random() < self._vf_conversion_risk * dt:
            patient_state["vt_converted_to_vf"] = True
            self._record_event(DiseaseType.VT, current_time, "complication", 0.35, "VT degenerated to VF")

        if self.bp_systolic < 80 and "vt_hemodynamic_compromise" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("vt_hemodynamic_compromise")
            self._record_event(DiseaseType.VT, current_time, "complication", 0.20, f"Hemodynamic compromise: BP {self.bp_systolic:.0f}")

        patient_state["vt_burden"] = self.vt_burden
        patient_state["vt_rate"] = self.vt_rate
        patient_state["bp_systolic"] = self.bp_systolic
        patient_state["ef"] = self.ef
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(self.latent, mu=np.full(64, mu_b * 0.02), sigma=np.full(64, sig_b * 0.015), dt=dt, rng=self.rng)
        patient_state["latent_pathway"] = self.latent
        return patient_state

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.02
        risk = base + 0.10 * self.vt_burden
        if self.is_polymorphic:
            risk += 0.08
        if self.bp_systolic < 80:
            risk += 0.10
        risk += 0.05 * max(0, 0.45 - self.ef)
        if self._has_comorbidity(DiseaseType.HYPERKALEMIA):
            risk *= 1.5
        return float(np.clip(risk, 0.0, 1.0))


class VFModel(DiseaseModel):
    """
    Ventricular fibrillation — immediately life-threatening.
    Chaotic ventricular depolarization, no effective cardiac output, fatal within minutes.
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.vf_amplitude: float = 0.8
        self.time_since_onset: float = 0.0
        self.defibrillation_attempted: bool = False
        self.stage: int = 0
        self.severity: float = 0.95
        self.latent = self._init_latent()

    def _mu_amplitude(self, a: float) -> float:
        return -0.1 * a

    def _sigma_amplitude(self, a: float) -> float:
        return 0.05 * a

    def update(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        self.time_since_onset += dt

        mu_a = self._mu_amplitude(self.vf_amplitude)
        sig_a = self._sigma_amplitude(self.vf_amplitude)
        self.vf_amplitude = self.sde_step_clamped(self.vf_amplitude, mu_a, sig_a, dt, 0.0, 1.0)

        self.stage = min(4, int(self.time_since_onset / 2.0))
        self.severity = float(np.clip(0.9 + 0.1 * min(1.0, self.time_since_onset / 10.0), 0.9, 1.0))

        if not self.defibrillation_attempted and self.time_since_onset > 3.0:
            self.defibrillation_attempted = True
            patient_state["defibrillation_attempted"] = True
            self._record_event(DiseaseType.VF, current_time, "complication", 0.0, "Defibrillation attempted")

        if self.time_since_onset > 10.0 and "vf_fatal_window" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("vf_fatal_window")
            self._record_event(DiseaseType.VF, current_time, "complication", 0.50, "Beyond survivable VF window")

        patient_state["vf_amplitude"] = self.vf_amplitude
        patient_state["time_in_vf"] = self.time_since_onset
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(self.latent, mu=np.full(64, mu_a * 0.05), sigma=np.full(64, sig_a * 0.03), dt=dt, rng=self.rng)
        patient_state["latent_pathway"] = self.latent
        return patient_state

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        if self.time_since_onset > 10.0:
            return 0.95
        if self.time_since_onset > 5.0:
            return 0.80
        return 0.50 + 0.04 * self.time_since_onset


# =========================================================================
#  E.  CONDUCTION DISORDERS
# =========================================================================

class CompleteHeartBlockModel(DiseaseModel):
    """
    Complete (third-degree) AV block. Complete failure of AV conduction,
    escape rhythm (junctional/ventricular), syncope risk (Stokes-Adams),
    progressive: Mobitz I -> Mobitz II -> Complete.
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.av_conduction: float = 0.8
        self.escape_rate: float = 40.0
        self.is_complete: bool = False
        self.pr_interval: float = float(patient_state.get("pr_interval", 180.0))
        self.hr: float = float(patient_state.get("heart_rate", 72.0))
        self.stage: int = 0
        self.severity: float = 0.1
        self.latent = self._init_latent()
        self._syncope_risk: float = 0.0

    def _mu_conduction(self, c: float) -> float:
        return -0.01 * c

    def _sigma_conduction(self, c: float) -> float:
        return 0.008

    def update(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        if self.severity >= 0.99:
            return patient_state

        mu_c = self._mu_conduction(self.av_conduction)
        sig_c = self._sigma_conduction(self.av_conduction)
        self.av_conduction = self.sde_step_clamped(self.av_conduction, mu_c, sig_c, dt, 0.0, 1.0)

        self.is_complete = self.av_conduction < 0.05

        if self.is_complete:
            escape_jitter = 0.5 * self.wiener_increment(dt)
            self.escape_rate = float(np.clip(self.escape_rate + escape_jitter, 25.0, 55.0))
            self.hr = self.escape_rate
        else:
            target_hr = 60.0 + 30.0 * self.av_conduction
            self.hr += 0.3 * (target_hr - self.hr) * dt + 0.1 * self.wiener_increment(dt)
            self.hr = float(np.clip(self.hr, 50.0, 120.0))

        self.pr_interval = float(np.clip(
            180.0 + 200.0 * (1.0 - self.av_conduction) + 5.0 * self.wiener_increment(dt), 120.0, 600.0,
        ))

        self.stage = min(4, int((1.0 - self.av_conduction) / 0.25))
        self.severity = float(np.clip(
            (1.0 - self.av_conduction) * 0.6 + max(0, 1.0 - self.escape_rate / 60.0) * 0.4, 0.0, 1.0,
        ))

        self._syncope_risk = float(np.clip(
            0.02 + 0.15 * (1.0 - self.av_conduction) + 0.05 * max(0, 1.0 - self.escape_rate / 50.0), 0.0, 0.5,
        ))

        if self.rng.random() < self._syncope_risk * dt:
            patient_state.setdefault("complications", []).append("syncope")
            self._record_event(DiseaseType.COMPLETE_HEART_BLOCK, current_time, "complication", 0.25, "Stokes-Adams syncope episode")

        if self.is_complete and "complete_heart_block" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("complete_heart_block")
            self._record_event(DiseaseType.COMPLETE_HEART_BLOCK, current_time, "complication", 0.20, "Complete AV block established")

        patient_state["av_conduction"] = self.av_conduction
        patient_state["escape_rate"] = self.escape_rate
        patient_state["pr_interval"] = self.pr_interval
        patient_state["heart_rate"] = self.hr
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(self.latent, mu=np.full(64, mu_c * 0.01), sigma=np.full(64, sig_c * 0.008), dt=dt, rng=self.rng)
        patient_state["latent_pathway"] = self.latent
        return patient_state

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.005
        risk = base + 0.10 * (1.0 - self.av_conduction)
        if self.is_complete:
            risk += 0.08 + 0.05 * max(0, 1.0 - self.escape_rate / 45.0)
        if self._has_comorbidity(DiseaseType.HYPERKALEMIA):
            risk *= 1.5
        if self._has_comorbidity(DiseaseType.DCM):
            risk *= 1.4
        return float(np.clip(risk, 0.0, 1.0))


# =========================================================================
#  F.  METABOLIC / SYSTEMIC EMERGENCIES
# =========================================================================

class HyperkalemiaModel(DiseaseModel):
    """
    Hyperkalemia — K+ > 5.5 mEq/L.
    ECG progression: peaked T-waves -> PR prolongation -> QRS widening -> sine wave -> asystole/VF.
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.potassium: float = float(patient_state.get("potassium", 6.0))
        self.t_wave_amplitude: float = 1.0
        self.pr_interval: float = float(patient_state.get("pr_interval", 180.0))
        self.qrs_duration: float = 100.0
        self.stage: int = 0
        self.severity: float = 0.3
        self.latent = self._init_latent()

    def _mu_k(self, k: float) -> float:
        return 0.02 * (5.5 - k)

    def _sigma_k(self, k: float) -> float:
        return 0.05 * abs(k - 5.5) + 0.01

    def update(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        mu_k = self._mu_k(self.potassium)
        sig_k = self._sigma_k(self.potassium)
        self.potassium = self.sde_step_clamped(self.potassium, mu_k, sig_k, dt, 3.0, 9.0)

        k_excess = max(0, self.potassium - 5.0)

        self.t_wave_amplitude = float(np.clip(1.0 + 2.0 * k_excess + 0.1 * self.wiener_increment(dt), 0.5, 6.0))
        self.pr_interval = float(np.clip(180.0 + 80.0 * max(0, self.potassium - 5.5) + 5.0 * self.wiener_increment(dt), 120.0, 400.0))
        self.qrs_duration = float(np.clip(100.0 + 60.0 * max(0, self.potassium - 5.8) + 3.0 * self.wiener_increment(dt), 80.0, 300.0))

        self.stage = min(5, int(k_excess / 0.5))
        self.severity = float(np.clip(
            min(1.0, k_excess / 3.0) * 0.6 + min(1.0, max(0, self.qrs_duration - 120) / 180.0) * 0.4, 0.0, 1.0,
        ))

        if self.potassium > 6.5 and "sine_wave_risk" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("sine_wave_risk")
            self._record_event(DiseaseType.HYPERKALEMIA, current_time, "complication", 0.30, f"K+ {self.potassium:.1f} — sine wave ECG pattern risk")

        if self.potassium > 7.0 and "hyperkalemia_collapse" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("hyperkalemia_collapse")
            self._record_event(DiseaseType.HYPERKALEMIA, current_time, "complication", 0.40, f"K+ {self.potassium:.1f} — asystole/VF imminent")

        if self.qrs_duration > 200 and "qrs_widening_critical" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("qrs_widening_critical")
            self._record_event(DiseaseType.HYPERKALEMIA, current_time, "complication", 0.15, f"QRS {self.qrs_duration:.0f} ms — critical widening")

        patient_state["potassium"] = self.potassium
        patient_state["t_wave_amplitude"] = self.t_wave_amplitude
        patient_state["pr_interval"] = self.pr_interval
        patient_state["qrs_duration"] = self.qrs_duration
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(self.latent, mu=np.full(64, mu_k * 0.02), sigma=np.full(64, sig_k * 0.015), dt=dt, rng=self.rng)
        patient_state["latent_pathway"] = self.latent
        return patient_state

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.01
        k_excess = max(0, self.potassium - 5.5)
        risk = base + 0.15 * min(1.0, k_excess / 2.0)
        if self.potassium > 7.0:
            risk += 0.20
        risk += 0.05 * min(1.0, max(0, self.qrs_duration - 120) / 180.0)
        if self._has_comorbidity(DiseaseType.ACS_STEMI):
            risk *= 1.3
        if self._has_comorbidity(DiseaseType.RESPIRATORY_FAILURE):
            risk *= 1.2
        return float(np.clip(risk, 0.0, 1.0))


class HypokalemiaModel(DiseaseModel):
    """
    Hypokalemia — K+ < 3.5 mEq/L.
    ST depression, T-wave flattening, prominent U-waves, Torsades de Pointes risk.
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.potassium: float = float(patient_state.get("potassium", 3.0))
        self.u_wave_prominence: float = 0.0
        self.st_depression: float = 0.0
        self.qtc: float = float(patient_state.get("qtc", 460.0))
        self.stage: int = 0
        self.severity: float = 0.2
        self.latent = self._init_latent()

    def _mu_k(self, k: float) -> float:
        return 0.01 * (3.5 - k)

    def _sigma_k(self, k: float) -> float:
        return 0.03 * abs(3.5 - k) + 0.01

    def update(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        mu_k = self._mu_k(self.potassium)
        sig_k = self._sigma_k(self.potassium)
        self.potassium = self.sde_step_clamped(self.potassium, mu_k, sig_k, dt, 1.5, 5.0)

        k_deficit = max(0, 3.5 - self.potassium)
        self.u_wave_prominence = float(np.clip(1.5 * k_deficit + 0.1 * self.wiener_increment(dt), 0.0, 4.0))
        self.st_depression = float(np.clip(0.8 * k_deficit + 0.05 * self.wiener_increment(dt), 0.0, 3.0))
        self.qtc = float(np.clip(460.0 + 30.0 * k_deficit + 2.0 * self.wiener_increment(dt), 400.0, 650.0))

        self.stage = min(4, int(k_deficit / 0.5))
        self.severity = float(np.clip(
            min(1.0, k_deficit / 1.5) * 0.5 + min(1.0, max(0, self.qtc - 470) / 150.0) * 0.3
            + min(1.0, self.u_wave_prominence / 3.0) * 0.2, 0.0, 1.0,
        ))

        if self.potassium < 2.5 and "severe_hypokalemia" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("severe_hypokalemia")
            self._record_event(DiseaseType.HYPOKALEMIA, current_time, "complication", 0.20, f"K+ {self.potassium:.1f} — severe hypokalemia")

        if self.qtc > 520 and "hypokalemia_torsades_risk" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("hypokalemia_torsades_risk")
            self._record_event(DiseaseType.HYPOKALEMIA, current_time, "complication", 0.15, "Torsades risk elevated")

        patient_state["potassium"] = self.potassium
        patient_state["u_wave_prominence"] = self.u_wave_prominence
        patient_state["st_depression"] = self.st_depression
        patient_state["qtc"] = self.qtc
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(self.latent, mu=np.full(64, mu_k * 0.015), sigma=np.full(64, sig_k * 0.01), dt=dt, rng=self.rng)
        patient_state["latent_pathway"] = self.latent
        return patient_state

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.005
        k_deficit = max(0, 3.5 - self.potassium)
        risk = base + 0.05 * min(1.0, k_deficit / 1.5)
        if self.qtc > 520:
            risk += 0.03
        if self.potassium < 2.5:
            risk += 0.05
        if self._has_comorbidity(DiseaseType.DRUG_TOXICITY):
            risk *= 1.4
        return float(np.clip(risk, 0.0, 1.0))


class PEModel(DiseaseModel):
    """
    Pulmonary embolism. Massive PE > 50% vascular obstruction, S1Q3T3 pattern,
    desaturation, obstructive shock -> PEA.
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.obstruction_fraction: float = float(patient_state.get("pe_obstruction", 0.30))
        self.spo2: float = float(patient_state.get("spo2", 96.0))
        self.bp_systolic: float = float(patient_state.get("bp_systolic", 120.0))
        self.hr: float = float(patient_state.get("heart_rate", 80.0))
        self.rv_dilation: float = 0.2
        self.stage: int = 0
        self.severity: float = 0.2
        self.latent = self._init_latent()
        self._obstructive_shock: bool = False

    def _mu_obstruction(self, o: float) -> float:
        return 0.02 * (0.6 - o)

    def _sigma_obstruction(self, o: float) -> float:
        return 0.01 * (1.0 + o)

    def update(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        mu_o = self._mu_obstruction(self.obstruction_fraction)
        sig_o = self._sigma_obstruction(self.obstruction_fraction)
        self.obstruction_fraction = self.sde_step_clamped(self.obstruction_fraction, mu_o, sig_o, dt, 0.0, 1.0)

        self.spo2 -= 5.0 * self.obstruction_fraction * dt + 0.2 * self.wiener_increment(dt)
        self.spo2 = float(np.clip(self.spo2, 50.0, 100.0))

        self.rv_dilation += 0.01 * self.obstruction_fraction * dt + 0.003 * self.wiener_increment(dt)
        self.rv_dilation = float(np.clip(self.rv_dilation, 0.0, 1.0))

        self.hr += 1.0 * self.obstruction_fraction * dt + 0.2 * self.wiener_increment(dt)
        self.hr = float(np.clip(self.hr, 60.0, 180.0))

        if self.obstruction_fraction > 0.5:
            self.bp_systolic -= 1.0 * (self.obstruction_fraction - 0.5) * dt + 0.3 * self.wiener_increment(dt)
        else:
            self.bp_systolic += 0.1 * (120.0 - self.bp_systolic) * dt
        self.bp_systolic = float(np.clip(self.bp_systolic, 50.0, 160.0))

        self._obstructive_shock = self.bp_systolic < 90 and self.obstruction_fraction > 0.5

        self.stage = min(4, int(self.obstruction_fraction / 0.15))
        self.severity = float(np.clip(
            self.obstruction_fraction * 0.4 + max(0, 1.0 - self.spo2 / 100.0) * 0.3
            + max(0, 1.0 - self.bp_systolic / 120.0) * 0.3, 0.0, 1.0,
        ))

        if self.obstruction_fraction > 0.5 and "massive_pe" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("massive_pe")
            self._record_event(DiseaseType.PE, current_time, "complication", 0.25, f"Massive PE: {self.obstruction_fraction * 100:.0f}% obstruction")

        if self._obstructive_shock and "obstructive_shock" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("obstructive_shock")
            self._record_event(DiseaseType.PE, current_time, "complication", 0.30, "Obstructive shock — PEA risk")

        patient_state["pe_obstruction"] = self.obstruction_fraction
        patient_state["spo2"] = self.spo2
        patient_state["bp_systolic"] = self.bp_systolic
        patient_state["heart_rate"] = self.hr
        patient_state["rv_dilation"] = self.rv_dilation
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(self.latent, mu=np.full(64, mu_o * 0.015), sigma=np.full(64, sig_o * 0.01), dt=dt, rng=self.rng)
        patient_state["latent_pathway"] = self.latent
        return patient_state

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.01
        risk = base + 0.15 * max(0, self.obstruction_fraction - 0.3)
        if self._obstructive_shock:
            risk += 0.20
        if self.spo2 < 85:
            risk += 0.10
        if self.bp_systolic < 80:
            risk += 0.10
        if self._has_comorbidity(DiseaseType.RESPIRATORY_FAILURE):
            risk *= 1.4
        return float(np.clip(risk, 0.0, 1.0))


class RespiratoryFailureModel(DiseaseModel):
    """
    Respiratory failure — Type I (hypoxemic) and Type II (hypercapnic).
    Progressive desaturation, rising pCO2, work of breathing, respiratory muscle fatigue -> arrest.
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.spo2: float = float(patient_state.get("spo2", 95.0))
        self.pco2: float = float(patient_state.get("pco2", 40.0))
        self.po2: float = float(patient_state.get("po2", 90.0))
        self.work_of_breathing: float = 0.2
        self.respiratory_rate: float = float(patient_state.get("respiratory_rate", 16.0))
        self.is_type_i: bool = True
        self.stage: int = 0
        self.severity: float = 0.15
        self.latent = self._init_latent()

    def _mu_spo2(self, s: float) -> float:
        return -0.03 * (s - 90.0)

    def _sigma_spo2(self, s: float) -> float:
        return 0.2 * max(0, (95.0 - s) / 5.0) + 0.05

    def update(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        if self.severity >= 0.99:
            return patient_state

        mu_s = self._mu_spo2(self.spo2)
        sig_s = self._sigma_spo2(self.spo2)
        self.spo2 = self.sde_step_clamped(self.spo2, mu_s, sig_s, dt, 40.0, 100.0)

        self.pco2 += 0.05 * (1.0 - self.spo2 / 100.0) * dt + 0.02 * self.wiener_increment(dt)
        self.pco2 = float(np.clip(self.pco2, 20.0, 100.0))

        self.po2 = float(np.clip(90.0 - 0.5 * (100.0 - self.spo2) + 0.3 * self.wiener_increment(dt), 20.0, 100.0))

        self.work_of_breathing += 0.01 * (95.0 - self.spo2) / 55.0 * dt + 0.003 * self.wiener_increment(dt)
        self.work_of_breathing = float(np.clip(self.work_of_breathing, 0.0, 1.0))

        self.respiratory_rate = float(np.clip(16.0 + 10.0 * self.work_of_breathing + 0.5 * self.wiener_increment(dt), 8.0, 40.0))

        self.is_type_i = self.po2 < 60.0
        if self.pco2 > 50.0:
            self.is_type_i = False

        self.stage = min(4, int((95.0 - self.spo2) / 10))
        self.severity = float(np.clip(
            max(0, 1.0 - self.spo2 / 100.0) * 0.4 + max(0, self.pco2 - 40.0) / 60.0 * 0.3
            + self.work_of_breathing * 0.3, 0.0, 1.0,
        ))

        if self.spo2 < 85 and "severe_hypoxemia" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("severe_hypoxemia")
            self._record_event(DiseaseType.RESPIRATORY_FAILURE, current_time, "complication", 0.20, f"SpO2 {self.spo2:.0f}% — severe hypoxemia")

        if self.pco2 > 70 and "severe_hypercapnia" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("severe_hypercapnia")
            self._record_event(DiseaseType.RESPIRATORY_FAILURE, current_time, "complication", 0.15, f"pCO2 {self.pco2:.0f} mmHg — severe hypercapnia")

        if self.work_of_breathing > 0.8 and "respiratory_fatigue" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("respiratory_fatigue")
            self._record_event(DiseaseType.RESPIRATORY_FAILURE, current_time, "complication", 0.20, "Respiratory muscle fatigue — impending arrest")

        patient_state["spo2"] = self.spo2
        patient_state["pco2"] = self.pco2
        patient_state["po2"] = self.po2
        patient_state["work_of_breathing"] = self.work_of_breathing
        patient_state["respiratory_rate"] = self.respiratory_rate
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(self.latent, mu=np.full(64, mu_s * 0.01), sigma=np.full(64, sig_s * 0.008), dt=dt, rng=self.rng)
        patient_state["latent_pathway"] = self.latent
        return patient_state

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.005
        risk = base
        if self.spo2 < 80:
            risk += 0.15 + 0.10 * (80.0 - self.spo2) / 40.0
        elif self.spo2 < 90:
            risk += 0.05 + 0.05 * (90.0 - self.spo2) / 10.0
        if self.pco2 > 70:
            risk += 0.10
        risk += 0.05 * self.work_of_breathing
        if self._has_comorbidity(DiseaseType.HYPERKALEMIA):
            risk *= 1.3
        if self._has_comorbidity(DiseaseType.SEPSIS):
            risk *= 1.4
        return float(np.clip(risk, 0.0, 1.0))


class SepsisModel(DiseaseModel):
    """
    Sepsis — systemic inflammatory response with organ dysfunction.
    Vasodilatory shock, tachycardia, lactate elevation, septic cardiomyopathy, multi-organ failure.
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.sirs_score: float = 2.5
        self.lactate: float = float(patient_state.get("lactate", 2.0))
        self.bp_systolic: float = float(patient_state.get("bp_systolic", 110.0))
        self.hr: float = float(patient_state.get("heart_rate", 90.0))
        self.ef: float = float(patient_state.get("ef", 0.55))
        self.vasoplegia: float = 0.2
        self.organ_dysfunction: float = 0.1
        self.stage: int = 0
        self.severity: float = 0.2
        self.latent = self._init_latent()
        self._septic_shock: bool = False

    def _mu_sirs(self, s: float) -> float:
        return 0.02 * (3.0 - s)

    def _sigma_sirs(self, s: float) -> float:
        return 0.05 * (1.0 + s)

    def update(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        if self.severity >= 0.99:
            return patient_state

        mu_s = self._mu_sirs(self.sirs_score)
        sig_s = self._sigma_sirs(self.sirs_score)
        self.sirs_score = self.sde_step_clamped(self.sirs_score, mu_s, sig_s, dt, 0.0, 6.0)

        self.lactate += 0.1 * self.sirs_score * dt + 0.05 * self.wiener_increment(dt)
        self.lactate = float(np.clip(self.lactate, 0.5, 20.0))

        self.vasoplegia += 0.005 * self.sirs_score * dt + 0.002 * self.wiener_increment(dt)
        self.vasoplegia = float(np.clip(self.vasoplegia, 0.0, 1.0))

        self.bp_systolic -= 0.5 * self.vasoplegia * dt + 0.2 * self.wiener_increment(dt)
        self.bp_systolic = float(np.clip(self.bp_systolic, 50.0, 160.0))

        self.hr += 0.3 * self.sirs_score * dt + 0.1 * self.wiener_increment(dt)
        self.hr = float(np.clip(self.hr, 60.0, 180.0))

        self.ef -= 0.003 * self.sirs_score * dt + 0.001 * self.wiener_increment(dt)
        self.ef = float(np.clip(self.ef, 0.10, 0.65))

        self.organ_dysfunction += 0.002 * self.sirs_score * dt + 0.001 * self.wiener_increment(dt)
        self.organ_dysfunction = float(np.clip(self.organ_dysfunction, 0.0, 1.0))

        self._septic_shock = self.bp_systolic < 90 and self.lactate > 4.0

        self.stage = min(5, int(self.sirs_score))
        self.severity = float(np.clip(
            self.sirs_score / 6.0 * 0.3 + min(1.0, self.lactate / 10.0) * 0.2
            + self.vasoplegia * 0.2 + self.organ_dysfunction * 0.3, 0.0, 1.0,
        ))

        if self._septic_shock and "septic_shock" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("septic_shock")
            self._record_event(DiseaseType.SEPSIS, current_time, "complication", 0.25, f"Septic shock: BP {self.bp_systolic:.0f}, lactate {self.lactate:.1f}")

        if self.lactate > 8.0 and "severe_lactatemia" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("severe_lactatemia")
            self._record_event(DiseaseType.SEPSIS, current_time, "complication", 0.20, f"Lactate {self.lactate:.1f} mmol/L — severe")

        if self.ef < 0.35 and "septic_cardiomyopathy" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("septic_cardiomyopathy")
            self._record_event(DiseaseType.SEPSIS, current_time, "complication", 0.15, "Septic cardiomyopathy")

        if self.organ_dysfunction > 0.7 and "multi_organ_dysfunction" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("multi_organ_dysfunction")
            self._record_event(DiseaseType.SEPSIS, current_time, "complication", 0.25, "Multi-organ dysfunction syndrome")

        patient_state["sirs_score"] = self.sirs_score
        patient_state["lactate"] = self.lactate
        patient_state["vasoplegia"] = self.vasoplegia
        patient_state["bp_systolic"] = self.bp_systolic
        patient_state["heart_rate"] = self.hr
        patient_state["ef"] = self.ef
        patient_state["organ_dysfunction"] = self.organ_dysfunction
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(self.latent, mu=np.full(64, mu_s * 0.015), sigma=np.full(64, sig_s * 0.01), dt=dt, rng=self.rng)
        patient_state["latent_pathway"] = self.latent
        return patient_state

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.01
        risk = base + 0.10 * min(1.0, self.sirs_score / 6.0) + 0.08 * min(1.0, self.lactate / 10.0)
        if self._septic_shock:
            risk += 0.15
        risk += 0.05 * self.organ_dysfunction
        if self.bp_systolic < 80:
            risk += 0.10
        if self._has_comorbidity(DiseaseType.RESPIRATORY_FAILURE):
            risk *= 1.5
        if self._has_comorbidity(DiseaseType.HYPERKALEMIA):
            risk *= 1.3
        return float(np.clip(risk, 0.0, 1.0))


class DrugToxicityModel(DiseaseModel):
    """
    Drug toxicity — covers multiple toxic syndromes:
    TCA (QRS widening), Digoxin (AV block, bidirectional VT), Beta-blocker (bradycardia),
    Calcium channel blocker (bradycardia, vasodilatory shock), Cocaine (vasospasm, MI, seizures).
    """

    def __init__(self, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> None:
        super().__init__(rng, patient_state, config)
        self.drug_type: str = patient_state.get("drug_type", "tca")
        self.toxicity_level: float = 0.3
        self.qrs_duration: float = float(patient_state.get("qrs_duration", 100.0))
        self.hr: float = float(patient_state.get("heart_rate", 72.0))
        self.bp_systolic: float = float(patient_state.get("bp_systolic", 120.0))
        self.potassium: float = float(patient_state.get("potassium", 4.0))
        self.stage: int = 0
        self.severity: float = 0.2
        self.latent = self._init_latent()

    def _mu_toxicity(self, t: float) -> float:
        return 0.02 * (0.7 - t)

    def _sigma_toxicity(self, t: float) -> float:
        return 0.01 * (1.0 + t)

    def update(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        if self.severity >= 0.98:
            return patient_state

        mu_t = self._mu_toxicity(self.toxicity_level)
        sig_t = self._sigma_toxicity(self.toxicity_level)
        self.toxicity_level = self.sde_step_clamped(self.toxicity_level, mu_t, sig_t, dt, 0.0, 1.0)

        self._apply_drug_specific_effects(dt, current_time, patient_state)

        self.stage = min(5, int(self.toxicity_level / 0.2))
        self.severity = float(np.clip(self.toxicity_level * 0.5 + self._drug_specific_risk_factor() * 0.5, 0.0, 1.0))

        patient_state["toxicity_level"] = self.toxicity_level
        patient_state["qrs_duration"] = self.qrs_duration
        patient_state["heart_rate"] = self.hr
        patient_state["bp_systolic"] = self.bp_systolic
        patient_state["potassium"] = self.potassium
        patient_state["collapse_risk"] = self.compute_collapse_probability(patient_state)

        self.latent = self._update_latent(self.latent, mu=np.full(64, mu_t * 0.015), sigma=np.full(64, sig_t * 0.01), dt=dt, rng=self.rng)
        patient_state["latent_pathway"] = self.latent
        return patient_state

    def _apply_drug_specific_effects(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> None:
        if self.drug_type == "tca":
            self._apply_tca_effects(dt, current_time, patient_state)
        elif self.drug_type == "digoxin":
            self._apply_digoxin_effects(dt, current_time, patient_state)
        elif self.drug_type == "beta_blocker":
            self._apply_beta_blocker_effects(dt, current_time, patient_state)
        elif self.drug_type == "calcium_channel_blocker":
            self._apply_ccb_effects(dt, current_time, patient_state)
        elif self.drug_type == "cocaine":
            self._apply_cocaine_effects(dt, current_time, patient_state)

    def _apply_tca_effects(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> None:
        qrs_widening = 40.0 * self.toxicity_level
        self.qrs_duration = float(np.clip(100.0 + qrs_widening + 3.0 * self.wiener_increment(dt), 80.0, 250.0))
        self.hr += (-5.0 * self.toxicity_level) * dt + 0.2 * self.wiener_increment(dt)
        self.hr = float(np.clip(self.hr, 40.0, 160.0))
        if self.qrs_duration > 160 and "tca_qrs_widening" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("tca_qrs_widening")
            self._record_event(DiseaseType.DRUG_TOXICITY, current_time, "complication", 0.20, f"TCA: QRS {self.qrs_duration:.0f} ms")

    def _apply_digoxin_effects(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> None:
        self.hr -= 3.0 * self.toxicity_level * dt + 0.1 * self.wiener_increment(dt)
        self.hr = float(np.clip(self.hr, 30.0, 120.0))
        self.potassium += 0.05 * self.toxicity_level * dt + 0.01 * self.wiener_increment(dt)
        self.potassium = float(np.clip(self.potassium, 3.0, 7.0))
        if self.rng.random() < 0.005 * self.toxicity_level * dt:
            patient_state.setdefault("complications", []).append("digoxin_bidirectional_vt")
            self._record_event(DiseaseType.DRUG_TOXICITY, current_time, "complication", 0.25, "Digoxin: bidirectional VT")
        if self.hr < 45 and "digoxin_av_block" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("digoxin_av_block")
            self._record_event(DiseaseType.DRUG_TOXICITY, current_time, "complication", 0.15, "Digoxin: significant AV block")

    def _apply_beta_blocker_effects(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> None:
        self.hr -= 8.0 * self.toxicity_level * dt + 0.3 * self.wiener_increment(dt)
        self.hr = float(np.clip(self.hr, 25.0, 120.0))
        self.bp_systolic -= 5.0 * self.toxicity_level * dt + 0.5 * self.wiener_increment(dt)
        self.bp_systolic = float(np.clip(self.bp_systolic, 50.0, 160.0))
        if self.hr < 40 and "bb_severe_bradycardia" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("bb_severe_bradycardia")
            self._record_event(DiseaseType.DRUG_TOXICITY, current_time, "complication", 0.20, f"Beta-blocker: HR {self.hr:.0f}")
        if self.bp_systolic < 80 and "bb_hypotension" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("bb_hypotension")
            self._record_event(DiseaseType.DRUG_TOXICITY, current_time, "complication", 0.15, f"Beta-blocker: BP {self.bp_systolic:.0f}")

    def _apply_ccb_effects(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> None:
        self.hr -= 6.0 * self.toxicity_level * dt + 0.2 * self.wiener_increment(dt)
        self.hr = float(np.clip(self.hr, 30.0, 120.0))
        self.bp_systolic -= 8.0 * self.toxicity_level * dt + 0.5 * self.wiener_increment(dt)
        self.bp_systolic = float(np.clip(self.bp_systolic, 40.0, 160.0))
        if self.hr < 40 and "ccb_severe_bradycardia" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("ccb_severe_bradycardia")
            self._record_event(DiseaseType.DRUG_TOXICITY, current_time, "complication", 0.20, f"CCB: HR {self.hr:.0f}")
        if self.bp_systolic < 70 and "ccb_shock" not in patient_state.get("complications", []):
            patient_state.setdefault("complications", []).append("ccb_shock")
            self._record_event(DiseaseType.DRUG_TOXICITY, current_time, "complication", 0.25, f"CCB: vasodilatory shock BP {self.bp_systolic:.0f}")

    def _apply_cocaine_effects(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> None:
        self.hr += 8.0 * self.toxicity_level * dt + 0.5 * self.wiener_increment(dt)
        self.hr = float(np.clip(self.hr, 60.0, 200.0))
        self.bp_systolic += 5.0 * self.toxicity_level * dt + 0.5 * self.wiener_increment(dt)
        self.bp_systolic = float(np.clip(self.bp_systolic, 80.0, 200.0))
        if self.rng.random() < 0.008 * self.toxicity_level * dt:
            patient_state.setdefault("complications", []).append("cocaine_vasospasm_mi")
            self._record_event(DiseaseType.DRUG_TOXICITY, current_time, "complication", 0.25, "Cocaine: coronary vasospasm -> MI")
        if self.rng.random() < 0.003 * self.toxicity_level * dt:
            patient_state.setdefault("complications", []).append("cocaine_seizure")
            self._record_event(DiseaseType.DRUG_TOXICITY, current_time, "complication", 0.15, "Cocaine-induced seizure")

    def _drug_specific_risk_factor(self) -> float:
        if self.drug_type == "tca":
            return min(1.0, max(0, self.qrs_duration - 100) / 150.0)
        elif self.drug_type == "digoxin":
            return min(1.0, max(0, 4.5 - self.hr) / 20.0 + max(0, self.potassium - 5.0) / 2.0)
        elif self.drug_type == "beta_blocker":
            return min(1.0, max(0, 60 - self.hr) / 40.0 + max(0, 100 - self.bp_systolic) / 60.0)
        elif self.drug_type == "calcium_channel_blocker":
            return min(1.0, max(0, 60 - self.hr) / 40.0 + max(0, 90 - self.bp_systolic) / 70.0)
        elif self.drug_type == "cocaine":
            return min(1.0, max(0, self.hr - 120) / 80.0 + max(0, self.bp_systolic - 160) / 60.0)
        return self.toxicity_level

    def compute_collapse_probability(self, patient_state: Dict[str, Any]) -> float:
        base = 0.01
        risk = base + 0.08 * self.toxicity_level + 0.07 * self._drug_specific_risk_factor()
        if self.drug_type == "tca" and self.qrs_duration > 180:
            risk += 0.10
        elif self.drug_type == "digoxin" and self.potassium > 6.0:
            risk += 0.10
        elif self.drug_type in ("beta_blocker", "calcium_channel_blocker") and self.hr < 35:
            risk += 0.15
        elif self.drug_type == "cocaine" and self.hr > 160:
            risk += 0.08
        if self._has_comorbidity(DiseaseType.HYPERKALEMIA):
            risk *= 1.4
        if self._has_comorbidity(DiseaseType.LQTS):
            risk *= 1.5
        return float(np.clip(risk, 0.0, 1.0))


# =========================================================================
#  MODEL REGISTRY
# =========================================================================

DISEASE_MODEL_REGISTRY: Dict[DiseaseType, type] = {
    DiseaseType.ACS_UNSTABLE_ANGINA: UnstableAnginaModel,
    DiseaseType.ACS_NSTEMI: NSTEMIModel,
    DiseaseType.ACS_STEMI: STEMIModel,
    DiseaseType.DCM: DCMModel,
    DiseaseType.HCM: HCMModel,
    DiseaseType.ARVC: ARVCModel,
    DiseaseType.BRUGADA: BrugadaModel,
    DiseaseType.LQTS: LQTSModel,
    DiseaseType.SQTS: SQTSModel,
    DiseaseType.CPVT: CPVTModel,
    DiseaseType.AF: AFModel,
    DiseaseType.VT: VTModel,
    DiseaseType.VF: VFModel,
    DiseaseType.COMPLETE_HEART_BLOCK: CompleteHeartBlockModel,
    DiseaseType.HYPERKALEMIA: HyperkalemiaModel,
    DiseaseType.HYPOKALEMIA: HypokalemiaModel,
    DiseaseType.PE: PEModel,
    DiseaseType.RESPIRATORY_FAILURE: RespiratoryFailureModel,
    DiseaseType.SEPSIS: SepsisModel,
    DiseaseType.DRUG_TOXICITY: DrugToxicityModel,
}


# =========================================================================
#  INTERACTION MANAGER
# =========================================================================

class DiseaseInteractionManager:
    """
    Tracks comorbidities and models synergistic disease interactions.

    - ~30% comorbidity rate (configurable).
    - Models known clinical interactions (e.g. COPD + HF -> worse outcomes).
    - Models drug-drug interactions between treatments for different diseases.
    - Creates emergent risk that exceeds the sum of individual diseases.
    """

    PAIRWISE_INTERACTIONS: Dict[Tuple[DiseaseType, DiseaseType], float] = {
        (DiseaseType.ACS_STEMI, DiseaseType.HYPERKALEMIA): 1.5,
        (DiseaseType.ACS_STEMI, DiseaseType.RESPIRATORY_FAILURE): 1.4,
        (DiseaseType.ACS_STEMI, DiseaseType.DCM): 1.6,
        (DiseaseType.DCM, DiseaseType.HYPERKALEMIA): 1.4,
        (DiseaseType.DCM, DiseaseType.AF): 1.5,
        (DiseaseType.DCM, DiseaseType.RESPIRATORY_FAILURE): 1.5,
        (DiseaseType.HCM, DiseaseType.AF): 1.3,
        (DiseaseType.HCM, DiseaseType.HYPERKALEMIA): 1.3,
        (DiseaseType.ARVC, DiseaseType.AF): 1.2,
        (DiseaseType.BRUGADA, DiseaseType.HYPERKALEMIA): 1.4,
        (DiseaseType.BRUGADA, DiseaseType.DRUG_TOXICITY): 1.5,
        (DiseaseType.LQTS, DiseaseType.HYPERKALEMIA): 1.5,
        (DiseaseType.LQTS, DiseaseType.DRUG_TOXICITY): 1.6,
        (DiseaseType.LQTS, DiseaseType.HYPOKALEMIA): 1.5,
        (DiseaseType.SQTS, DiseaseType.HYPERKALEMIA): 1.3,
        (DiseaseType.CPVT, DiseaseType.HYPERKALEMIA): 1.4,
        (DiseaseType.CPVT, DiseaseType.DRUG_TOXICITY): 1.3,
        (DiseaseType.AF, DiseaseType.VT): 1.4,
        (DiseaseType.AF, DiseaseType.DCM): 1.5,
        (DiseaseType.VT, DiseaseType.ACS_STEMI): 1.6,
        (DiseaseType.VT, DiseaseType.HYPERKALEMIA): 1.5,
        (DiseaseType.COMPLETE_HEART_BLOCK, DiseaseType.HYPERKALEMIA): 1.5,
        (DiseaseType.COMPLETE_HEART_BLOCK, DiseaseType.DCM): 1.4,
        (DiseaseType.HYPERKALEMIA, DiseaseType.RESPIRATORY_FAILURE): 1.3,
        (DiseaseType.HYPERKALEMIA, DiseaseType.SEPSIS): 1.3,
        (DiseaseType.HYPOKALEMIA, DiseaseType.DRUG_TOXICITY): 1.4,
        (DiseaseType.PE, DiseaseType.RESPIRATORY_FAILURE): 1.4,
        (DiseaseType.PE, DiseaseType.SEPSIS): 1.5,
        (DiseaseType.SEPSIS, DiseaseType.RESPIRATORY_FAILURE): 1.5,
        (DiseaseType.SEPSIS, DiseaseType.HYPERKALEMIA): 1.3,
        (DiseaseType.DRUG_TOXICITY, DiseaseType.RESPIRATORY_FAILURE): 1.4,
    }

    DRUG_INTERACTIONS: Dict[Tuple[str, str], float] = {
        ("tca", "ssri"): 1.5,
        ("tca", "macrolide"): 1.4,
        ("tca", "antipsychotic"): 1.6,
        ("digoxin", "amiodarone"): 1.4,
        ("digoxin", "verapamil"): 1.5,
        ("beta_blocker", "calcium_channel_blocker"): 1.8,
        ("beta_blocker", "digoxin"): 1.3,
        ("cocaine", "beta_blocker"): 2.0,
    }

    def __init__(self, rng: np.random.Generator, config: OHCAConfig) -> None:
        self.rng = rng
        self.config = config
        self.disease_config = config.disease
        self.comorbidity_probability: float = self.disease_config.comorbidity_probability
        self.active_diseases: Dict[DiseaseType, DiseaseState] = {}
        self.interaction_events: List[DiseaseEvent] = []

    def select_comorbidities(self, primary_disease: DiseaseType, patient_state: Dict[str, Any]) -> List[DiseaseType]:
        """Stochastically select comorbid diseases given a primary disease."""
        ASSOCIATIONS: Dict[DiseaseType, List[Tuple[DiseaseType, float]]] = {
            DiseaseType.ACS_STEMI: [
                (DiseaseType.DCM, 0.10), (DiseaseType.AF, 0.12),
                (DiseaseType.HYPERKALEMIA, 0.08), (DiseaseType.RESPIRATORY_FAILURE, 0.10),
                (DiseaseType.VT, 0.06),
            ],
            DiseaseType.ACS_NSTEMI: [
                (DiseaseType.DCM, 0.08), (DiseaseType.AF, 0.10),
                (DiseaseType.HYPERKALEMIA, 0.06), (DiseaseType.RESPIRATORY_FAILURE, 0.08),
            ],
            DiseaseType.ACS_UNSTABLE_ANGINA: [
                (DiseaseType.AF, 0.06), (DiseaseType.DCM, 0.05), (DiseaseType.RESPIRATORY_FAILURE, 0.05),
            ],
            DiseaseType.DCM: [
                (DiseaseType.AF, 0.25), (DiseaseType.VT, 0.12),
                (DiseaseType.HYPERKALEMIA, 0.10), (DiseaseType.RESPIRATORY_FAILURE, 0.15),
                (DiseaseType.COMPLETE_HEART_BLOCK, 0.05),
            ],
            DiseaseType.HCM: [
                (DiseaseType.AF, 0.15), (DiseaseType.VT, 0.08), (DiseaseType.HYPERKALEMIA, 0.05),
            ],
            DiseaseType.ARVC: [(DiseaseType.AF, 0.12), (DiseaseType.VT, 0.15)],
            DiseaseType.BRUGADA: [
                (DiseaseType.AF, 0.08), (DiseaseType.HYPERKALEMIA, 0.05), (DiseaseType.DRUG_TOXICITY, 0.04),
            ],
            DiseaseType.LQTS: [
                (DiseaseType.AF, 0.06), (DiseaseType.HYPERKALEMIA, 0.06),
                (DiseaseType.HYPOKALEMIA, 0.06), (DiseaseType.DRUG_TOXICITY, 0.05),
            ],
            DiseaseType.SQTS: [(DiseaseType.HYPERKALEMIA, 0.04)],
            DiseaseType.CPVT: [(DiseaseType.HYPERKALEMIA, 0.05), (DiseaseType.DRUG_TOXICITY, 0.04)],
            DiseaseType.AF: [
                (DiseaseType.DCM, 0.12), (DiseaseType.VT, 0.06),
                (DiseaseType.HYPERKALEMIA, 0.05), (DiseaseType.RESPIRATORY_FAILURE, 0.06),
            ],
            DiseaseType.COMPLETE_HEART_BLOCK: [
                (DiseaseType.HYPERKALEMIA, 0.10), (DiseaseType.DCM, 0.08), (DiseaseType.DRUG_TOXICITY, 0.06),
            ],
            DiseaseType.SEPSIS: [
                (DiseaseType.RESPIRATORY_FAILURE, 0.30), (DiseaseType.HYPERKALEMIA, 0.15),
                (DiseaseType.AF, 0.08), (DiseaseType.VT, 0.05),
            ],
            DiseaseType.RESPIRATORY_FAILURE: [
                (DiseaseType.SEPSIS, 0.15), (DiseaseType.PE, 0.08),
                (DiseaseType.HYPERKALEMIA, 0.08), (DiseaseType.DCM, 0.10),
            ],
            DiseaseType.PE: [(DiseaseType.RESPIRATORY_FAILURE, 0.20), (DiseaseType.AF, 0.08)],
            DiseaseType.HYPERKALEMIA: [
                (DiseaseType.RESPIRATORY_FAILURE, 0.10), (DiseaseType.COMPLETE_HEART_BLOCK, 0.08),
                (DiseaseType.ACS_STEMI, 0.06),
            ],
            DiseaseType.HYPOKALEMIA: [
                (DiseaseType.LQTS, 0.08), (DiseaseType.DRUG_TOXICITY, 0.06),
            ],
            DiseaseType.DRUG_TOXICITY: [
                (DiseaseType.HYPERKALEMIA, 0.05), (DiseaseType.RESPIRATORY_FAILURE, 0.04),
            ],
        }

        associated = ASSOCIATIONS.get(primary_disease, [])
        selected: List[DiseaseType] = []
        for dtype, prob in associated:
            if dtype in self.active_diseases and self.active_diseases[dtype].is_active:
                continue
            if self.rng.random() < prob * self.comorbidity_probability:
                selected.append(dtype)
        return selected

    def compute_interaction_multiplier(self, disease_a: DiseaseType, disease_b: DiseaseType) -> float:
        """Return the risk multiplier for a pair of active diseases."""
        key = (disease_a, disease_b)
        key_rev = (disease_b, disease_a)
        if key in self.PAIRWISE_INTERACTIONS:
            return self.PAIRWISE_INTERACTIONS[key]
        if key_rev in self.PAIRWISE_INTERACTIONS:
            return self.PAIRWISE_INTERACTIONS[key_rev]
        return 1.0

    def compute_drug_interaction_multiplier(self, drug_a: str, drug_b: str) -> float:
        """Return the toxicity multiplier for a pair of co-administered drugs."""
        key = (drug_a, drug_b)
        key_rev = (drug_b, drug_a)
        if key in self.DRUG_INTERACTIONS:
            return self.DRUG_INTERACTIONS[key]
        if key_rev in self.DRUG_INTERACTIONS:
            return self.DRUG_INTERACTIONS[key_rev]
        return 1.0

    def compute_emergent_risk(self, collapse_risk: float) -> float:
        """
        Compute emergent risk from the interaction of all active diseases.

        The emergent risk is NOT simply the sum of individual risks.
        Interactions create non-linear amplification of collapse probability.
        """
        n_active = sum(1 for ds in self.active_diseases.values() if ds.is_active)
        if n_active < 2:
            return collapse_risk

        interaction_multiplier = 1.0
        disease_list = [dt for dt, ds in self.active_diseases.items() if ds.is_active]
        for i in range(len(disease_list)):
            for j in range(i + 1, len(disease_list)):
                pair_mult = self.compute_interaction_multiplier(disease_list[i], disease_list[j])
                if pair_mult > 1.0:
                    interaction_multiplier *= pair_mult

        interaction_multiplier = min(interaction_multiplier, 3.0)

        synergistic_bonus = 0.01 * max(0, n_active - 1) ** 1.5

        emergent = collapse_risk * interaction_multiplier + synergistic_bonus
        return float(np.clip(emergent, 0.0, 1.0))

    def update(self, dt: float, current_time: float, patient_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run one update cycle for the interaction manager."""
        for dtype, dstate in list(self.active_diseases.items()):
            if dstate.is_active:
                dstate.progression_rate += self.disease_config.disease_progression_rate * dt

        disease_list = [dt for dt, ds in self.active_diseases.items() if ds.is_active]
        for i in range(len(disease_list)):
            for j in range(i + 1, len(disease_list)):
                mult = self.compute_interaction_multiplier(disease_list[i], disease_list[j])
                if mult > 1.0:
                    for dt_key in [disease_list[i], disease_list[j]]:
                        ds = self.active_diseases[dt_key]
                        ds.severity = float(np.clip(ds.severity * (1.0 + (mult - 1.0) * 0.01 * dt), 0.0, 1.0))
                        ds.probability_of_collapse = float(np.clip(
                            ds.probability_of_collapse * (1.0 + (mult - 1.0) * 0.005 * dt), 0.0, 1.0,
                        ))

        base_risk = patient_state.get("collapse_risk", 0.0)
        emergent = self.compute_emergent_risk(base_risk)
        patient_state["emergent_interaction_risk"] = emergent
        patient_state["collapse_risk"] = max(base_risk, emergent)

        return patient_state

    def create_disease_state(self, disease_type: DiseaseType, onset_time: float) -> DiseaseState:
        """Create a new DiseaseState with appropriate defaults for the given disease type."""
        return DiseaseState(
            disease_type=disease_type,
            onset_time_hours=onset_time,
            severity=0.05,
            progression_rate=self.disease_config.disease_progression_rate,
            stage=0,
            complications=[],
            is_active=True,
            probability_of_collapse=0.005,
            latent_pathway=np.zeros(self.disease_config.latent_state_dim, dtype=np.float64),
        )

    def create_disease_model(self, disease_type: DiseaseType, rng: np.random.Generator, patient_state: Dict[str, Any], config: OHCAConfig) -> Optional[DiseaseModel]:
        """Instantiate the appropriate DiseaseModel subclass for the given disease type."""
        model_cls = DISEASE_MODEL_REGISTRY.get(disease_type)
        if model_cls is None:
            return None
        return model_cls(rng=rng, patient_state=patient_state, config=config)
