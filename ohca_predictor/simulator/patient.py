"""
Virtual patient for the OHCA Predictor simulator module.

Provides a ``VirtualPatient`` class that represents continuous human
physiology over a 24-72 hour simulation window, backed by stochastic
physiological sub-models and pharmacokinetic models for common medications.

Usage::

    import numpy as np
    from ohca_predictor.config import get_config
    from ohca_predictor.simulator.patient import VirtualPatient

    rng = np.random.default_rng(seed=42)
    config = get_config()
    patient = VirtualPatient(rng=rng, config=config)

    for step in range(2880):          # 24 h at 30-s steps
        patient.update_physiology(dt=1/120, circadian_phase=..., activity_state=...)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Callable

import numpy as np
from scipy.special import expit  # logistic sigmoid

from ohca_predictor.config import OHCAConfig, SimulationConfig, DiseaseConfig
from ohca_predictor.simulator.physiology import (
    CircadianModel,
    ActivityModel,
    ANSModel,
    RespiratoryModel,
    ThermoregulationModel,
    HydrationModel,
)


# ===========================================================================
# Medication data-class and pharmacokinetic helpers
# ===========================================================================

def _hill_function(
    concentration: float,
    ec50: float,
    emax: float = 1.0,
    gamma: float = 1.0,
) -> float:
    """Hill (Emax) concentration-effect curve.

    E(C) = Emax * C^gamma / (EC50^gamma + C^gamma)

    Args:
        concentration: Plasma concentration (ng mL^-1 or mg L^-1).
        ec50: Concentration producing 50 % effect.
        emax: Maximum effect.
        gamma: Hill coefficient (steepness).
    """
    c = max(concentration, 0.0)
    denom = ec50 ** gamma + c ** gamma
    if denom < 1e-15:
        return 0.0
    return emax * c ** gamma / denom


def _make_hill(ec50: float, emax: float = 1.0, gamma: float = 1.0) -> Callable[[float], float]:
    """Factory that returns a Hill function with fixed EC50/emax/gamma."""
    def curve(c: float) -> float:
        return _hill_function(c, ec50, emax, gamma)
    return curve


@dataclass
class Medication:
    """Pharmacokinetic model for a single medication.

    Uses a one-compartment first-order absorption model::

        dAbs/dt = -ka * Abs
        dC/dt   =  (ka * Abs / Vd) - ke * C

    Closed-form solution (single bolus)::

        C(t) = (Dose * ka) / (Vd * (ka - ke)) * (e^{-ke*t} - e^{-ka*t})
    """

    name: str
    drug_class: str
    dose_mg: float
    half_life_hours: float
    plasma_concentration: float = 0.0
    absorption_rate: float = 1.0
    elimination_rate: float = 0.1
    effect_concentration_curve: Callable[[float], float] = field(
        default_factory=lambda: _make_hill(ec50=50.0)
    )
    time_since_dose: float = 0.0
    volume_of_distribution: float = 1.0
    bioavailability: float = 1.0
    is_active: bool = True

    def __post_init__(self) -> None:
        if self.half_life_hours > 0 and self.elimination_rate <= 0:
            self.elimination_rate = np.log(2) / self.half_life_hours

    def update(self, dt: float) -> float:
        """Advance PK state by *dt* hours using one-compartment model.

        Returns:
            Current plasma concentration.
        """
        self.time_since_dose += dt
        if self.time_since_dose < 1e-12 or self.dose_mg <= 0:
            return self.plasma_concentration

        ka = self.absorption_rate
        ke = self.elimination_rate
        vd = max(self.volume_of_distribution, 0.01)
        t = self.time_since_dose

        if abs(ka - ke) < 1e-9:
            # Limiting case when ka ~ ke
            self.plasma_concentration = (
                self.dose_mg * self.bioavailability * ka * t * np.exp(-ke * t) / vd
            )
        else:
            self.plasma_concentration = (
                self.dose_mg
                * self.bioavailability
                * ka
                / (vd * (ka - ke))
                * (np.exp(-ke * t) - np.exp(-ka * t))
            )

        self.plasma_concentration = max(0.0, self.plasma_concentration)
        return self.plasma_concentration

    def administer(self, dose_mg: float) -> None:
        """Reset the absorption timer with a new dose."""
        self.dose_mg = dose_mg
        self.time_since_dose = 0.0

    @property
    def effect(self) -> float:
        """Current pharmacodynamic effect [0, 1] from Hill curve."""
        return self.effect_concentration_curve(self.plasma_concentration)


# ---------------------------------------------------------------------------
# Pre-defined drug parameter sets
# ---------------------------------------------------------------------------

def _create_metoprolol(dose_mg: float = 50.0) -> Medication:
    """Metoprolol (beta-blocker): reduces HR and BP."""
    return Medication(
        name="metoprolol",
        drug_class="beta_blocker",
        dose_mg=dose_mg,
        half_life_hours=4.0,
        absorption_rate=1.5,
        elimination_rate=np.log(2) / 4.0,
        volume_of_distribution=3.5,
        bioavailability=0.5,
        effect_concentration_curve=_make_hill(ec50=50.0, emax=0.3, gamma=1.5),
    )


def _create_lisinopril(dose_mg: float = 10.0) -> Medication:
    """Lisinopril (ACE inhibitor): reduces BP."""
    return Medication(
        name="lisinopril",
        drug_class="ace_inhibitor",
        dose_mg=dose_mg,
        half_life_hours=12.0,
        absorption_rate=1.0,
        elimination_rate=np.log(2) / 12.0,
        volume_of_distribution=3.5,
        bioavailability=0.25,
        effect_concentration_curve=_make_hill(ec50=10.0, emax=0.25, gamma=1.2),
    )


def _create_amiodarone(dose_mg: float = 150.0) -> Medication:
    """Amiodarone (class III antiarrhythmic): prolongs QT, suppresses arrhythmias."""
    return Medication(
        name="amiodarone",
        drug_class="class_III_antiarrhythmic",
        dose_mg=dose_mg,
        half_life_hours=50.0,
        absorption_rate=0.3,
        elimination_rate=np.log(2) / 50.0,
        volume_of_distribution=60.0,
        bioavailability=0.5,
        effect_concentration_curve=_make_hill(ec50=1000.0, emax=0.4, gamma=1.0),
    )


def _create_furosemide(dose_mg: float = 40.0) -> Medication:
    """Furosemide (loop diuretic): reduces volume, promotes K+ loss."""
    return Medication(
        name="furosemide",
        drug_class="loop_diuretic",
        dose_mg=dose_mg,
        half_life_hours=2.0,
        absorption_rate=2.0,
        elimination_rate=np.log(2) / 2.0,
        volume_of_distribution=0.15,
        bioavailability=0.5,
        effect_concentration_curve=_make_hill(ec50=1000.0, emax=0.5, gamma=1.3),
    )


def _create_kcl(dose_mg: float = 40.0) -> Medication:
    """Potassium chloride supplement."""
    return Medication(
        name="kcl",
        drug_class="electrolyte_supplement",
        dose_mg=dose_mg,
        half_life_hours=1.0,
        absorption_rate=3.0,
        elimination_rate=np.log(2) / 1.0,
        volume_of_distribution=0.3,
        bioavailability=0.8,
        effect_concentration_curve=_make_hill(ec50=20.0, emax=0.5, gamma=1.0),
    )


def _create_aspirin(dose_mg: float = 325.0) -> Medication:
    """Aspirin (antiplatelet, NSAID)."""
    return Medication(
        name="aspirin",
        drug_class="antiplatelet",
        dose_mg=dose_mg,
        half_life_hours=2.5,
        absorption_rate=2.5,
        elimination_rate=np.log(2) / 2.5,
        volume_of_distribution=0.15,
        bioavailability=0.68,
        effect_concentration_curve=_make_hill(ec50=100.0, emax=0.35, gamma=1.0),
    )


def _create_heparin(dose_mg: float = 5000.0) -> Medication:
    """Heparin (anticoagulant): IV only, short half-life."""
    return Medication(
        name="heparin",
        drug_class="anticoagulant",
        dose_mg=dose_mg,
        half_life_hours=1.5,
        absorption_rate=10.0,
        elimination_rate=np.log(2) / 1.5,
        volume_of_distribution=0.06,
        bioavailability=1.0,
        effect_concentration_curve=_make_hill(ec50=0.5, emax=0.8, gamma=2.0),
    )


def _create_epinephrine(dose_mg: float = 1.0) -> Medication:
    """Epinephrine (sympathomimetic): IV, ultra-short half-life."""
    return Medication(
        name="epinephrine",
        drug_class="sympathomimetic",
        dose_mg=dose_mg,
        half_life_hours=0.033,
        absorption_rate=50.0,
        elimination_rate=np.log(2) / 0.033,
        volume_of_distribution=0.5,
        bioavailability=1.0,
        effect_concentration_curve=_make_hill(ec50=0.01, emax=0.9, gamma=2.0),
    )


def _create_digoxin(dose_mg: float = 0.25) -> Medication:
    """Digoxin (cardiac glycoside): positive inotrope."""
    return Medication(
        name="digoxin",
        drug_class="cardiac_glycoside",
        dose_mg=dose_mg,
        half_life_hours=40.0,
        absorption_rate=0.5,
        elimination_rate=np.log(2) / 40.0,
        volume_of_distribution=7.0,
        bioavailability=0.7,
        effect_concentration_curve=_make_hill(ec50=1.5, emax=0.3, gamma=1.0),
    )


def _create_sotalol(dose_mg: float = 80.0) -> Medication:
    """Sotalol (class III antiarrhythmic / beta-blocker)."""
    return Medication(
        name="sotalol",
        drug_class="class_III_antiarrhythmic",
        dose_mg=dose_mg,
        half_life_hours=12.0,
        absorption_rate=1.2,
        elimination_rate=np.log(2) / 12.0,
        volume_of_distribution=2.0,
        bioavailability=0.9,
        effect_concentration_curve=_make_hill(ec50=200.0, emax=0.35, gamma=1.2),
    )


DRUG_FACTORIES: dict[str, Callable[[], Medication]] = {
    "metoprolol": _create_metoprolol,
    "lisinopril": _create_lisinopril,
    "amiodarone": _create_amiodarone,
    "furosemide": _create_furosemide,
    "kcl": _create_kcl,
    "aspirin": _create_aspirin,
    "heparin": _create_heparin,
    "epinephrine": _create_epinephrine,
    "digoxin": _create_digoxin,
    "sotalol": _create_sotalol,
}


# ===========================================================================
# Virtual Patient
# ===========================================================================

class VirtualPatient:
    """Complete virtual patient with continuous physiology over 24-72 h.

    Demographics are drawn once from population distributions and remain
    immutable.  Latent physiological state variables evolve continuously
    via stochastic differential equations (SDEs) driven by circadian,
    activity, autonomic, respiratory, thermal, and hydration sub-models.

    The ``probability_of_ohca`` attribute is derived from the current
    physiological state and is **never** used as a direct label.
    """

    ETHNICITIES = ["white", "black", "hispanic", "asian", "other"]
    ETHNICITY_PROBS = np.array([0.578, 0.136, 0.187, 0.060, 0.039])
    OCCUPATIONS = [
        "sedentary",
        "light_manual",
        "heavy_manual",
        "athletic",
        "desk_professional",
    ]
    OCCUPATION_PROBS = np.array([0.15, 0.10, 0.10, 0.05, 0.60])
    OHCA_MECHANISMS = ["VF", "VT", "PEA", "asystole"]

    # ------------------------------------------------------------------ init

    def __init__(self, rng: np.random.Generator, config: OHCAConfig) -> None:
        self.rng = rng
        self.config = config
        self.sim_config: SimulationConfig = config.simulation
        self.disease_config: DiseaseConfig = config.disease

        # ---- Static demographics (immutable after creation) ---------------
        self.patient_id: str = str(uuid.UUID(bytes=self.rng.bytes(16)))
        self.age: float = float(np.clip(rng.normal(55.0, 18.0), 18.0, 100.0))
        self.sex: str = "male" if rng.random() < 0.49 else "female"

        # Height / weight / BMI
        if self.sex == "male":
            self.height_cm: float = float(rng.normal(175.0, 8.0))
        else:
            self.height_cm: float = float(rng.normal(162.0, 7.0))
        self.bmi: float = float(np.clip(rng.normal(27.0, 6.0), 14.0, 55.0))
        self.weight_kg: float = self.bmi * (self.height_cm / 100.0) ** 2

        # Ethnicity
        self.ethnicity: str = str(
            rng.choice(self.ETHNICITIES, p=self.ETHNICITY_PROBS)
        )

        # Occupation
        self.occupation: str = str(
            rng.choice(self.OCCUPATIONS, p=self.OCCUPATION_PROBS)
        )

        # Smoking: age-dependent probability
        if self.age < 25:
            p_current = 0.12
        elif self.age < 45:
            p_current = 0.18
        elif self.age < 65:
            p_current = 0.14
        else:
            p_current = 0.07
        p_former = max(0.0, min(0.5, 0.3 - p_current))
        p_never = max(0.0, 1.0 - p_current - p_former)
        self.smoking_status: str = str(
            rng.choice(["never", "former", "current"], p=[p_never, p_former, p_current])
        )

        # Alcohol
        self.alcohol_use: str = str(
            rng.choice(["none", "light", "moderate", "heavy"], p=[0.20, 0.35, 0.30, 0.15])
        )

        # Fitness: inversely correlated with age and BMI
        raw_fitness = 1.0 - 0.5 * ((self.age - 18.0) / 82.0) - 0.3 * ((self.bmi - 18.0) / 37.0)
        self.fitness_level: float = float(np.clip(raw_fitness + rng.normal(0, 0.08), 0.0, 1.0))

        # Sleep quality
        self.sleep_quality: float = float(np.clip(rng.normal(0.65, 0.15), 0.0, 1.0))

        # ---- Sub-models --------------------------------------------------
        self.circadian_model = CircadianModel(
            rng=rng,
            period_hours=float(np.clip(rng.normal(24.0, 0.5), 23.5, 24.5)),
        )
        self.activity_model = ActivityModel(rng=rng, fitness_level=self.fitness_level)
        self.ans_model = ANSModel(rng=rng)
        self.respiratory_model = RespiratoryModel(rng=rng)
        self.thermoregulation_model = ThermoregulationModel(rng=rng)
        self.hydration_model = HydrationModel(rng=rng)

        # ---- Latent physiological state variables -------------------------
        self.cardiac_output_lpm: float = 5.0
        self.ejection_fraction: float = 0.60
        self.heart_rate_bpm: float = 72.0
        self.heart_rate_variability_ms: float = 40.0
        self.lf_hf_ratio: float = 1.5
        self.systolic_bp_mmhg: float = 120.0
        self.diastolic_bp_mmhg: float = 80.0
        self.mean_arterial_pressure_mmhg: float = 93.0
        self.respiratory_rate_bpm: float = 14.0
        self.spo2_percent: float = 98.0
        self.skin_temperature_c: float = 33.0
        self.core_temperature_c: float = 37.0
        self.blood_potassium_mmol: float = 4.0
        self.blood_sodium_mmol: float = 140.0
        self.blood_glucose_mgdl: float = 90.0
        self.blood_calcium_mmol: float = 2.3
        self.blood_ph: float = 7.4
        self.lactate_mmol: float = 1.0
        self.troponin_ng_l: float = 5.0
        self.bnp_pg_ml: float = 30.0
        self.hydration_level: float = 0.75
        self.metabolic_rate: float = 1.0
        self.autonomic_tone_sympathetic: float = 0.3
        self.autonomic_tone_parasympathetic: float = 0.7
        self.respiratory_coupling_strength: float = 0.5
        self.myocardial_irritability: float = 0.1
        self.ischemic_burden: float = 0.0
        self.inflammatory_state: float = 0.05
        self.autonomic_nervous_system_balance: float = -0.4

        # ---- Disease state -----------------------------------------------
        self.active_diseases: List[str] = []
        self.disease_history: List[str] = []
        self.probability_of_ohca: float = 0.001
        self.time_to_ohca_hours: Optional[float] = None
        self.ohca_mechanism: Optional[str] = None

        # ---- Medications --------------------------------------------------
        self.medications: List[Medication] = []

        # ---- Time tracking -----------------------------------------------
        self.simulation_time_hours: float = 0.0
        self.simulation_time_seconds: float = 0.0

        # Latent cascade for OHCA (replaces direct formula)
        self._cascade_stage = 0          # 0=stable, 1=subclinical, 2=preclinical, 3=imminent
        self._cascade_timer = 0.0        # hours spent in current stage
        self._cascade_thresholds = [     # stochastic crossing thresholds per stage
            np.random.uniform(0.3, 0.7),  # stage 0->1: must exceed
            np.random.uniform(0.4, 0.8),  # stage 1->2: must exceed
            np.random.uniform(0.5, 0.9),  # stage 2->3: must exceed
        ]
        self._cascade_noise = np.random.normal(0, 0.05)  # per-patient noise offset
        self._cascade_memory = np.zeros(5)  # rolling buffer of recent risk scores
        self._electrical_instability = 0.0   # latent: ventricular excitability
        self._mechanical_stretch = 0.0       # latent: myocardial wall stress
        self._metabolic_strain = 0.0         # latent: cellular energy deficit

        # Anatomical variability (affects sensor generation)
        self.chest_impedance = np.random.uniform(0.6, 1.4)  # body composition affects ECG
        self.ecg_axis_deviation = np.random.uniform(-30, 30)  # degrees
        self.skin_melanin_index = np.random.uniform(0.2, 0.8)  # affects PPG SNR
        self.adipose_thickness = np.random.uniform(0.5, 3.0)  # cm, affects signal attenuation
        self.sensor_contact_quality = np.random.uniform(0.6, 1.0)  # loose bands, etc.

        # Medication adherence (stochastic non-compliance)
        self.med_adherence = {}  # drug_name -> adherence_state
        self._adherence_update_interval = np.random.uniform(12, 48)  # hours between adherence checks
        self._adherence_timer = 0.0
        self._recently_stopped = []  # drugs abruptly stopped (rebound risk)

        # ---- Initialise sub-models and derived quantities -----------------
        self._initialise_physiology()

    def _initialise_physiology(self) -> None:
        """Set initial physiological state consistent with demographics."""
        # Sub-model initial states
        self.circadian_model.initial_state(self.rng)
        self.activity_model.initial_state(self.rng)
        self.ans_model.initial_state(self.rng)
        self.respiratory_model.initial_state(self.rng)
        self.thermoregulation_model.initial_state(self.rng)
        self.hydration_model.initial_state(self.rng)

        # Age-adjusted ejection fraction (declines ~0.5 %/year after 40)
        age_decline = max(0.0, (self.age - 40.0) * 0.005)
        self.ejection_fraction = float(np.clip(
            0.65 - age_decline + self.rng.normal(0, 0.03), 0.15, 0.75
        ))

        # HR baseline adjusts for fitness
        self.heart_rate_bpm = float(np.clip(
            72.0 - 10.0 * self.fitness_level + self.rng.normal(0, 3), 45.0, 100.0
        ))

        # BP baseline adjusts for age and sex
        bp_age_effect = max(0.0, (self.age - 30.0) * 0.4)
        sex_offset = 5.0 if self.sex == "male" else 0.0
        self.systolic_bp_mmhg = float(np.clip(
            118.0 + bp_age_effect + sex_offset + self.rng.normal(0, 5), 90.0, 180.0
        ))
        self.diastolic_bp_mmhg = float(np.clip(
            76.0 + bp_age_effect * 0.3 + self.rng.normal(0, 4), 50.0, 110.0
        ))
        self.mean_arterial_pressure_mmhg = (
            self.diastolic_bp_mmhg
            + (self.systolic_bp_mmhg - self.diastolic_bp_mmhg) / 3.0
        )

        # Cardiac output = HR * stroke volume (approx)
        sv = 70.0 * (1.0 + 0.3 * self.fitness_level)  # mL
        self.cardiac_output_lpm = float(np.clip(
            self.heart_rate_bpm * sv / 1000.0, 3.0, 8.0
        ))

        # HRV: inversely correlated with age and sympathodominance
        self.heart_rate_variability_ms = float(np.clip(
            60.0 - 0.3 * self.age + 10.0 * self.fitness_level
            + self.rng.normal(0, 5),
            5.0,
            150.0,
        ))
        self.lf_hf_ratio = float(np.clip(
            1.5 + 0.01 * self.age - 0.5 * self.fitness_level
            + self.rng.normal(0, 0.2),
            0.3,
            5.0,
        ))

        # Synchronise hydration model state
        self.hydration_model.hydration_level = self.hydration_level
        self.hydration_model.blood_potassium = self.blood_potassium_mmol
        self.hydration_model.blood_sodium = self.blood_sodium_mmol
        self.hydration_model.blood_calcium = self.blood_calcium_mmol
        self.hydration_model.blood_glucose = self.blood_glucose_mgdl
        self.hydration_model.blood_ph = self.blood_ph
        self.hydration_model.lactate = self.lactate_mmol
        self.hydration_model.troponin = self.troponin_ng_l
        self.hydration_model.bnp = self.bnp_pg_ml

        # Synchronise respiratory model
        self.respiratory_model.resp_rate = self.respiratory_rate_bpm

        # Synchronise thermoregulation model
        self.thermoregulation_model.core_temp = self.core_temperature_c
        self.thermoregulation_model.skin_temp = self.skin_temperature_c

        # Synchronise ANS model
        self.ans_model.sympathetic_tone = self.autonomic_tone_sympathetic
        self.ans_model.parasympathetic_tone = self.autonomic_tone_parasympathetic

        # Synchronise activity model
        self.activity_model.fitness_level = self.fitness_level

        # Compute initial derived quantities
        self.compute_derived_quantities()

    # ======================================================================
    # Main update
    # ======================================================================

    def update_physiology(
        self,
        dt: float,
        circadian_phase: float,
        activity_state: str,
    ) -> None:
        """Advance all physiological state by *dt* hours.

        This is the main integration step.  It:

        1. Advances the circadian oscillator.
        2. Updates the activity Markov chain.
        3. Applies circadian modulation to key variables.
        4. Applies activity-dependent effects.
        5. Applies medication pharmacokinetics.
        6. Evolves latent state variables via SDEs.
        7. Updates the autonomic nervous system.
        8. Updates respiratory dynamics.
        9. Updates thermoregulation.
        10. Updates hydration / electrolytes.
        11. Computes derived quantities (MAP, P(OHCA)).
        12. Clamps all variables to physiological ranges.

        Args:
            dt: Time step in hours.
            circadian_phase: Current circadian phase (hours since midnight).
            activity_state: Current activity state name.
        """
        self.simulation_time_hours += dt
        self.simulation_time_seconds += dt * 3600.0

        # ---- 1. Circadian oscillator ------------------------------------
        self.circadian_model.update(dt, circadian_phase)

        # ---- 2. Activity Markov chain ------------------------------------
        self.activity_model.update(dt, circadian_phase)

        # ---- 3. Circadian modulation -------------------------------------
        self.apply_circadian_modulation(circadian_phase)

        # ---- 4. Activity effects -----------------------------------------
        self.apply_activity_effects(activity_state)

        # ---- 5. Medications ----------------------------------------------
        self.apply_medications(dt)
        self.update_medication_adherence(dt)

        # ---- 6. SDE evolution of latent state ----------------------------
        self._evolve_latent_state(dt, activity_state)

        # ---- 7. Autonomic nervous system ---------------------------------
        circ_alertness = self.circadian_model.circadian_alertness
        activity_level = self.activity_model.activity_level
        self.mean_arterial_pressure_mmhg = (
            self.diastolic_bp_mmhg
            + (self.systolic_bp_mmhg - self.diastolic_bp_mmhg) / 3.0
        )
        symp_drive, para_drive = self.ans_model.update(
            dt=dt,
            activity_level=activity_level,
            mean_arterial_pressure=self.mean_arterial_pressure_mmhg,
            resp_rate_bpm=self.respiratory_rate_bpm,
            circadian_alertness=circ_alertness,
        )
        self.autonomic_tone_sympathetic = self.ans_model.sympathetic_tone
        self.autonomic_tone_parasympathetic = self.ans_model.parasympathetic_tone
        self.autonomic_nervous_system_balance = (
            self.autonomic_tone_sympathetic - self.autonomic_tone_parasympathetic
        )

        # ---- 8. Respiratory model ----------------------------------------
        self.respiratory_model.update(
            dt=dt,
            metabolic_rate=self.metabolic_rate,
            blood_ph=self.blood_ph,
            activity_level=activity_level,
            spo2_percent=self.spo2_percent,
            circadian_alertness=circ_alertness,
            time_seconds=self.simulation_time_seconds,
        )
        self.respiratory_rate_bpm = self.respiratory_model.resp_rate
        self.respiratory_coupling_strength = float(np.clip(
            0.3 + 0.2 * self.autonomic_tone_parasympathetic, 0.0, 1.0
        ))

        # ---- 9. Thermoregulation -----------------------------------------
        self.thermoregulation_model.update(
            dt=dt,
            metabolic_rate=self.metabolic_rate,
            sympathetic_tone=self.autonomic_tone_sympathetic,
            inflammatory_state=self.inflammatory_state,
            circadian_alertness=circ_alertness,
        )
        self.core_temperature_c = self.thermoregulation_model.core_temp
        self.skin_temperature_c = self.thermoregulation_model.skin_temp

        # ---- 10. Hydration / electrolytes --------------------------------
        self.hydration_model.update(
            dt=dt,
            metabolic_rate=self.metabolic_rate,
            activity_level=activity_level,
            sympathetic_tone=self.autonomic_tone_sympathetic,
            inflammatory_state=self.inflammatory_state,
            ischemic_burden=self.ischemic_burden,
            myocardial_irritability=self.myocardial_irritability,
            ejection_fraction=self.ejection_fraction,
        )

        # ---- 11. Derived quantities --------------------------------------
        self.compute_derived_quantities()

        # ---- 12. Clamp ---------------------------------------------------
        self.clamp_state()

    # ======================================================================
    # Circadian modulation
    # ======================================================================

    def apply_circadian_modulation(self, circadian_phase: float) -> None:
        """Modulate HR, BP, temperature, and hormones by circadian phase.

        The circadian signal from ``CircadianModel`` is used to impose
        physiological oscillations consistent with known diurnal patterns:
        - HR and BP peak during biological day.
        - Core temperature follows a ~0.5 deg C rhythm.
        - Metabolic rate dips during sleep.
        """
        alertness = self.circadian_model.circadian_alertness
        melatonin = self.circadian_model.melatonin_level

        # HR: circadian variation of ~5 bpm
        hr_circadian_offset = 5.0 * (alertness - 0.5)
        self.heart_rate_bpm += hr_circadian_offset * 0.05

        # BP: systolic dips ~10 mmHg at night (non-dipping pattern if <5%)
        bp_circadian_offset = 10.0 * (alertness - 0.5)
        self.systolic_bp_mmhg += bp_circadian_offset * 0.03
        self.diastolic_bp_mmhg += bp_circadian_offset * 0.02

        # Metabolic rate: lower during sleep
        self.metabolic_rate = float(np.clip(
            1.0 + 0.2 * (alertness - 0.5) + self.rng.normal(0, 0.02),
            0.8,
            2.5,
        ))

        # Inflammatory state: slight circadian rhythm (higher at night)
        self.inflammatory_state += 0.001 * melatonin * (1.0 - alertness)

    # ======================================================================
    # Activity effects
    # ======================================================================

    def apply_activity_effects(self, activity_state: str) -> None:
        """Modify physiology based on current activity level.

        Higher activity increases HR, CO, metabolic rate, and respiratory
        drive while transiently reducing HRV.
        """
        activity_multipliers = {
            "sleeping": 0.7,
            "resting": 1.0,
            "light_activity": 1.3,
            "moderate_activity": 1.7,
            "vigorous_activity": 2.2,
        }
        multiplier = activity_multipliers.get(activity_state, 1.0)

        # HR response
        hr_target = 72.0 * multiplier - 10.0 * self.fitness_level * max(0.0, multiplier - 1.0)
        hr_correction = (hr_target - self.heart_rate_bpm) * 0.1
        self.heart_rate_bpm += hr_correction

        # CO response
        co_target = 5.0 * multiplier
        self.cardiac_output_lpm += (co_target - self.cardiac_output_lpm) * 0.05

        # HRV: decreases with activity
        activity_level = self.activity_model.activity_level
        hrv_target = 40.0 - 8.0 * activity_level + 10.0 * self.fitness_level
        self.heart_rate_variability_ms += (
            hrv_target - self.heart_rate_variability_ms
        ) * 0.05

        # SpO2: slight decrease with vigorous activity
        if activity_state == "vigorous_activity":
            self.spo2_percent -= 0.02
        elif activity_state == "sleeping":
            self.spo2_percent += 0.01

        # Sweat from thermoregulation (activity-driven heat)
        self.thermoregulation_model.sweat_rate = 0.01 * activity_level

        # Ischemic burden: vigorous activity in susceptible patients
        if activity_state == "vigorous_activity" and self.ejection_fraction < 0.4:
            self.ischemic_burden += 0.002

    # ======================================================================
    # Medications
    # ======================================================================

    def apply_medications(self, dt: float) -> None:
        """Apply pharmacokinetic and pharmacodynamic effects of medications.

        Each medication's PK state is updated, then its pharmacodynamic
        effect is applied to the relevant physiological variables.
        """
        for med in self.medications:
            med.update(dt)
            eff = med.effect
            if eff < 1e-6:
                continue

            if med.drug_class == "beta_blocker":
                self.heart_rate_bpm -= eff * 8.0
                self.systolic_bp_mmhg -= eff * 5.0
                self.diastolic_bp_mmhg -= eff * 3.0
                self.myocardial_irritability -= eff * 0.1

            elif med.drug_class == "ace_inhibitor":
                self.systolic_bp_mmhg -= eff * 8.0
                self.diastolic_bp_mmhg -= eff * 5.0

            elif med.drug_class == "class_III_antiarrhythmic":
                self.myocardial_irritability -= eff * 0.2
                self.lf_hf_ratio -= eff * 0.3

            elif med.drug_class == "loop_diuretic":
                self.hydration_level -= eff * 0.005 * dt
                self.blood_potassium_mmol -= eff * 0.02 * dt
                self.systolic_bp_mmhg -= eff * 3.0

            elif med.drug_class == "electrolyte_supplement":
                self.blood_potassium_mmol += eff * 0.1 * dt

            elif med.drug_class == "antiplatelet":
                self.ischemic_burden -= eff * 0.01 * dt

            elif med.drug_class == "anticoagulant":
                self.ischemic_burden -= eff * 0.005 * dt

            elif med.drug_class == "sympathomimetic":
                self.heart_rate_bpm += eff * 15.0
                self.systolic_bp_mmhg += eff * 10.0
                self.autonomic_tone_sympathetic += eff * 0.2
                self.myocardial_irritability += eff * 0.15

            elif med.drug_class == "cardiac_glycoside":
                self.heart_rate_bpm -= eff * 5.0
                self.ejection_fraction += eff * 0.05

    def update_medication_adherence(self, dt_hours: float) -> None:
        """Simulate stochastic medication non-adherence and abrupt cessations."""
        self._adherence_timer += dt_hours
        if self._adherence_timer < self._adherence_update_interval:
            return
        self._adherence_timer = 0.0
        self._adherence_update_interval = np.random.uniform(12, 48)
        
        for med in self.medications:
            if not med.is_active:
                continue
            name = med.name
            if name not in self.med_adherence:
                self.med_adherence[name] = np.random.uniform(0.85, 1.0)  # baseline adherence
            
            # Random dose skipping (5% chance per check)
            if np.random.random() < 0.05:
                med.dose_mg *= np.random.uniform(0.0, 0.5)  # missed or partial dose
            
            # Abrupt cessation events (1% chance, much rarer)
            if np.random.random() < 0.01 and name not in self._recently_stopped:
                med.is_active = False
                self._recently_stopped.append(name)
                # Rebound effects based on drug class
                if 'beta' in name.lower() or 'metoprolol' in name.lower():
                    self.autonomic_tone_sympathetic = min(1.0, self.autonomic_tone_sympathetic + 0.4)
                    self.heart_rate_bpm += 25  # rebound tachycardia
                    self.systolic_bp_mmhg += 30  # hypertensive crisis
                elif 'lisinopril' in name.lower() or 'ace' in name.lower():
                    self.systolic_bp_mmhg += 25
                    self.diastolic_bp_mmhg += 15
                elif 'furosemide' in name.lower():
                    self.hydration_level = min(1.0, self.hydration_level + 0.3)
                    self.blood_potassium_mmol += 1.5  # rebound hyperkalemia
            
            # Drug reactivation after a "realization" period (24-72h)
            for stopped_name in self._recently_stopped[:]:
                if np.random.random() < 0.02:  # ~50h average to restart
                    self._recently_stopped.remove(stopped_name)
                    for m in self.medications:
                        if m.name == stopped_name:
                            m.is_active = True

    # ======================================================================
    # Latent state SDE evolution
    # ======================================================================

    def _evolve_latent_state(self, dt: float, activity_state: str) -> None:
        """Evolve all latent physiological variables via SDEs.

        Each variable follows:

            dX = drift(X) * dt + sigma * sqrt(dt) * N(0, 1)

        Drift terms represent homeostatic regulation: exponential return
        toward a setpoint modulated by the current physiological context.
        """
        rng = self.rng
        sqrt_dt = np.sqrt(dt)

        # Cardiac output: homeostatic return to 5.0 L/min
        co_setpoint = 5.0 + 0.5 * self.activity_model.activity_level
        drift_co = 0.3 * (co_setpoint - self.cardiac_output_lpm)
        self.cardiac_output_lpm += drift_co * dt + 0.1 * sqrt_dt * rng.standard_normal()

        # Ejection fraction: slow drift with ischemia/medication effects
        ef_setpoint = 0.60 - 0.2 * self.ischemic_burden - 0.1 * max(0.0, self.age - 60.0) / 40.0
        drift_ef = 0.05 * (ef_setpoint - self.ejection_fraction)
        self.ejection_fraction += drift_ef * dt + 0.005 * sqrt_dt * rng.standard_normal()

        # Heart rate: driven by ANS
        hr_baseline = 72.0 - 10.0 * self.fitness_level
        hr_setpoint = hr_baseline + 20.0 * self.autonomic_tone_sympathetic - 25.0 * self.autonomic_tone_parasympathetic
        drift_hr = 0.5 * (hr_setpoint - self.heart_rate_bpm)
        self.heart_rate_bpm += drift_hr * dt + 1.5 * sqrt_dt * rng.standard_normal()

        # HRV: driven by parasympathetic tone
        hrv_setpoint = 5.0 + 100.0 * self.autonomic_tone_parasympathetic
        drift_hrv = 0.2 * (hrv_setpoint - self.heart_rate_variability_ms)
        self.heart_rate_variability_ms += drift_hrv * dt + 2.0 * sqrt_dt * rng.standard_normal()

        # LF/HF ratio: sympathetic / parasympathetic
        lf_hf_setpoint = max(0.3, self.autonomic_tone_sympathetic / max(0.01, self.autonomic_tone_parasympathetic))
        drift_lf = 0.3 * (lf_hf_setpoint - self.lf_hf_ratio)
        self.lf_hf_ratio += drift_lf * dt + 0.1 * sqrt_dt * rng.standard_normal()

        # Blood pressure: driven by CO and peripheral resistance
        svr = 80.0  # systemic vascular resistance (Wood units approximation)
        bp_setpoint_sys = 0.1 * self.cardiac_output_lpm * svr * (1.0 + 0.005 * self.age)
        drift_sys = 0.3 * (bp_setpoint_sys - self.systolic_bp_mmhg)
        self.systolic_bp_mmhg += drift_sys * dt + 2.0 * sqrt_dt * rng.standard_normal()

        bp_setpoint_dia = 0.06 * self.cardiac_output_lpm * svr * 0.8
        drift_dia = 0.3 * (bp_setpoint_dia - self.diastolic_bp_mmhg)
        self.diastolic_bp_mmhg += drift_dia * dt + 1.5 * sqrt_dt * rng.standard_normal()

        # MAP is derived
        self.mean_arterial_pressure_mmhg = (
            self.diastolic_bp_mmhg
            + (self.systolic_bp_mmhg - self.diastolic_bp_mmhg) / 3.0
        )

        # SpO2: driven by respiratory model
        spo2_setpoint = self.respiratory_model.spo2_from_po2
        drift_spo2 = 0.5 * (spo2_setpoint - self.spo2_percent)
        self.spo2_percent += drift_spo2 * dt + 0.3 * sqrt_dt * rng.standard_normal()

        # Potassium: driven by hydration model
        drift_k = 0.2 * (self.hydration_model.blood_potassium - self.blood_potassium_mmol)
        self.blood_potassium_mmol += drift_k * dt + self.hydration_model.sigma_potassium * sqrt_dt * rng.standard_normal()

        # Sodium
        drift_na = 0.2 * (self.hydration_model.blood_sodium - self.blood_sodium_mmol)
        self.blood_sodium_mmol += drift_na * dt + self.hydration_model.sigma_sodium * sqrt_dt * rng.standard_normal()

        # Calcium
        drift_ca = 0.2 * (self.hydration_model.blood_calcium - self.blood_calcium_mmol)
        self.blood_calcium_mmol += drift_ca * dt + self.hydration_model.sigma_calcium * sqrt_dt * rng.standard_normal()

        # Glucose
        drift_glu = 0.2 * (self.hydration_model.blood_glucose - self.blood_glucose_mgdl)
        self.blood_glucose_mgdl += drift_glu * dt + self.hydration_model.sigma_glucose * sqrt_dt * rng.standard_normal()

        # pH
        drift_pH = 0.15 * (self.hydration_model.blood_ph - self.blood_ph)
        self.blood_ph += drift_pH * dt + self.hydration_model.sigma_ph * sqrt_dt * rng.standard_normal()

        # Lactate
        drift_lac = 0.2 * (self.hydration_model.lactate - self.lactate_mmol)
        self.lactate_mmol += drift_lac * dt + self.hydration_model.sigma_lactate * sqrt_dt * rng.standard_normal()

        # Troponin
        drift_trop = 0.1 * (self.hydration_model.troponin - self.troponin_ng_l)
        self.troponin_ng_l += drift_trop * dt + self.hydration_model.sigma_troponin * sqrt_dt * rng.standard_normal()

        # BNP
        drift_bnp = 0.1 * (self.hydration_model.bnp - self.bnp_pg_ml)
        self.bnp_pg_ml += drift_bnp * dt + self.hydration_model.sigma_bnp * sqrt_dt * rng.standard_normal()

        # Hydration level
        drift_hyd = 0.3 * (self.hydration_model.hydration_level - self.hydration_level)
        self.hydration_level += drift_hyd * dt + 0.01 * sqrt_dt * rng.standard_normal()

        # Myocardial irritability: driven by electrolytes, ischemia, drugs
        irritability_setpoint = 0.1
        irritability_setpoint += 0.2 * max(0.0, self.blood_potassium_mmol - 5.5)
        irritability_setpoint += 0.1 * max(0.0, 3.5 - self.blood_potassium_mmol)
        irritability_setpoint += 0.1 * max(0.0, self.blood_calcium_mmol - 3.0)
        irritability_setpoint += 0.3 * self.ischemic_burden
        drift_irr = 0.2 * (irritability_setpoint - self.myocardial_irritability)
        self.myocardial_irritability += drift_irr * dt + 0.02 * sqrt_dt * rng.standard_normal()

        # Ischemic burden: slow accumulation and resolution
        if self.ejection_fraction < 0.4:
            self.ischemic_burden += 0.001 * dt
        else:
            self.ischemic_burden -= 0.0005 * dt
        # Ischemia from inflammation
        self.ischemic_burden += 0.0003 * self.inflammatory_state * dt

        # Inflammatory state: slow resolution
        self.inflammatory_state -= 0.001 * self.inflammatory_state * dt

    # ======================================================================
    # Derived quantities
    # ======================================================================

    def compute_derived_quantities(self) -> None:
        """Compute derived quantities from current physiological state.

        MAP, probability of OHCA, and cardiac output relationships are
        updated here.  ``probability_of_ohca`` is a continuous risk score
        derived from multiple state variables and is **never** used as a
        binary label.
        """
        # MAP (recalculate to stay consistent)
        self.mean_arterial_pressure_mmhg = (
            self.diastolic_bp_mmhg
            + (self.systolic_bp_mmhg - self.diastolic_bp_mmhg) / 3.0
        )

        # CO = HR * SV / 1000  (SV estimated from EF and body size)
        sv_estimated = 70.0 * (self.ejection_fraction / 0.60)
        co_from_hrv = self.heart_rate_bpm * sv_estimated / 1000.0
        self.cardiac_output_lpm = 0.5 * self.cardiac_output_lpm + 0.5 * co_from_hrv

        # Latent cascade: nonlinear multi-variable interaction
        # Electrical instability: K+ gradient + ischemia + irritability interact
        k_dep = abs(self.blood_potassium_mmol - 4.0) / 2.0
        self._electrical_instability = 0.9 * self._electrical_instability + 0.1 * (
            1.0 / (1.0 + np.exp(-5.0 * (self.myocardial_irritability + k_dep + self.ischemic_burden - 0.6)))
        )

        # Mechanical stretch: EF decline + BP + volume overload
        ef_deficit = max(0, 0.5 - self.ejection_fraction)
        bp_strain = max(0, self.systolic_bp_mmhg - 160) / 200.0
        self._mechanical_stretch = 0.9 * self._mechanical_stretch + 0.1 * (
            1.0 / (1.0 + np.exp(-4.0 * (ef_deficit + bp_strain + self.bnp_pg_ml / 1000.0 - 0.4)))
        )

        # Metabolic strain: lactate + pH + troponin with hysteresis lag
        lactate_strain = max(0, self.lactate_mmol - 2.0) / 8.0
        acidosis_strain = max(0, 7.4 - self.blood_ph) / 0.4
        self._metabolic_strain = 0.92 * self._metabolic_strain + 0.08 * (
            1.0 / (1.0 + np.exp(-6.0 * (lactate_strain + acidosis_strain + self.troponin_ng_l / 500.0 - 0.3)))
        )

        # Multi-variable nonlinear interaction (NOT a linear combination)
        # The key: all three must be elevated simultaneously for stage progression
        interaction_score = (
            self._electrical_instability * self._mechanical_stretch * 3.0
            + self._electrical_instability * self._metabolic_strain * 2.5
            + self._mechanical_stretch * self._metabolic_strain * 2.0
            + self._electrical_instability + self._mechanical_stretch + self._metabolic_strain
        ) / 7.0 + self._cascade_noise

        # Update rolling memory buffer
        self._cascade_memory = np.roll(self._cascade_memory, 1)
        self._cascade_memory[0] = interaction_score

        # Markov stage transitions with hysteresis
        stage_score = np.mean(self._cascade_memory[:3])  # smoothed over 3 timesteps
        if self._cascade_stage < 3:
            if stage_score > self._cascade_thresholds[self._cascade_stage]:
                self._cascade_stage += 1
                self._cascade_timer = 0.0
        elif self._cascade_stage > 0:
            if stage_score < self._cascade_thresholds[self._cascade_stage - 1] * 0.7:
                self._cascade_stage -= 1
                self._cascade_timer = 0.0

        self._cascade_timer += 1.0 / 3600.0  # assume dt=1s

        # Probability from stage (NOT a clean function)
        base_probs = [0.001, 0.02, 0.15, 0.70]
        stage_prob = base_probs[self._cascade_stage]
        # Add temporal noise within stage
        time_decay = 1.0 + 0.3 * np.sin(self._cascade_timer * 2.0 * np.pi)
        self.probability_of_ohca = np.clip(stage_prob * time_decay + np.random.normal(0, 0.01), 0, 1)

    # ======================================================================
    # Clamping
    # ======================================================================

    def clamp_state(self) -> None:
        """Keep all physiological variables within valid ranges."""
        self.cardiac_output_lpm = float(np.clip(self.cardiac_output_lpm, 3.0, 8.0))
        self.ejection_fraction = float(np.clip(self.ejection_fraction, 0.15, 0.75))
        self.heart_rate_bpm = float(np.clip(self.heart_rate_bpm, 35.0, 220.0))
        self.heart_rate_variability_ms = float(np.clip(self.heart_rate_variability_ms, 5.0, 150.0))
        self.lf_hf_ratio = float(np.clip(self.lf_hf_ratio, 0.3, 5.0))
        self.systolic_bp_mmhg = float(np.clip(self.systolic_bp_mmhg, 80.0, 200.0))
        self.diastolic_bp_mmhg = float(np.clip(self.diastolic_bp_mmhg, 40.0, 130.0))
        self.mean_arterial_pressure_mmhg = float(np.clip(self.mean_arterial_pressure_mmhg, 55.0, 155.0))
        self.respiratory_rate_bpm = float(np.clip(self.respiratory_rate_bpm, 8.0, 40.0))
        self.spo2_percent = float(np.clip(self.spo2_percent, 70.0, 100.0))
        self.skin_temperature_c = float(np.clip(self.skin_temperature_c, 34.0, 38.5))
        self.core_temperature_c = float(np.clip(self.core_temperature_c, 36.0, 41.0))
        self.blood_potassium_mmol = float(np.clip(self.blood_potassium_mmol, 3.0, 7.0))
        self.blood_sodium_mmol = float(np.clip(self.blood_sodium_mmol, 125.0, 160.0))
        self.blood_glucose_mgdl = float(np.clip(self.blood_glucose_mgdl, 40.0, 500.0))
        self.blood_calcium_mmol = float(np.clip(self.blood_calcium_mmol, 1.5, 3.5))
        self.blood_ph = float(np.clip(self.blood_ph, 7.0, 7.8))
        self.lactate_mmol = float(np.clip(self.lactate_mmol, 0.5, 15.0))
        self.troponin_ng_l = float(np.clip(self.troponin_ng_l, 0.0, 500.0))
        self.bnp_pg_ml = float(np.clip(self.bnp_pg_ml, 5.0, 5000.0))
        self.hydration_level = float(np.clip(self.hydration_level, 0.0, 1.0))
        self.metabolic_rate = float(np.clip(self.metabolic_rate, 0.8, 2.5))
        self.autonomic_tone_sympathetic = float(np.clip(self.autonomic_tone_sympathetic, 0.0, 1.0))
        self.autonomic_tone_parasympathetic = float(np.clip(self.autonomic_tone_parasympathetic, 0.0, 1.0))
        self.respiratory_coupling_strength = float(np.clip(self.respiratory_coupling_strength, 0.0, 1.0))
        self.myocardial_irritability = float(np.clip(self.myocardial_irritability, 0.0, 1.0))
        self.ischemic_burden = float(np.clip(self.ischemic_burden, 0.0, 1.0))
        self.inflammatory_state = float(np.clip(self.inflammatory_state, 0.0, 1.0))
        self.autonomic_nervous_system_balance = float(np.clip(self.autonomic_nervous_system_balance, -1.0, 1.0))
        self.probability_of_ohca = float(np.clip(self.probability_of_ohca, 0.0, 1.0))
        # Cascade latent variables
        self._electrical_instability = float(np.clip(getattr(self, '_electrical_instability', 0.0), 0.0, 1.0))
        self._mechanical_stretch = float(np.clip(getattr(self, '_mechanical_stretch', 0.0), 0.0, 1.0))
        self._metabolic_strain = float(np.clip(getattr(self, '_metabolic_strain', 0.0), 0.0, 1.0))
