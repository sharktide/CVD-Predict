"""
Population generator for the OHCA Predictor simulator module.

Creates a diverse virtual population with realistic comorbidity distributions,
medication assignments, disease assignments, and hard-negative cohorts that
share signal characteristics with true OHCA patients.

Usage::

    import numpy as np
    from ohca_predictor.config import get_config
    from ohca_predictor.simulator.population import PopulationGenerator

    rng = np.random.default_rng(seed=42)
    config = get_config()
    generator = PopulationGenerator(config, rng)
    patients = generator.generate_population(n_patients=1000)
"""

from __future__ import annotations

from typing import List

import numpy as np

from ohca_predictor.config import OHCAConfig, SimulationConfig
from ohca_predictor.simulator.patient import VirtualPatient
from ohca_predictor.simulator.diseases import DiseaseType, DiseaseInteractionManager


# ---------------------------------------------------------------------------
# Comorbidity list with base prevalences
# ---------------------------------------------------------------------------

COMORBIDITY_BASE_PREVALENCE = {
    "hypertension": 0.45,
    "diabetes": 0.15,
    "coronary_artery_disease": 0.10,
    "heart_failure": 0.05,
    "copd": 0.08,
    "chronic_kidney_disease": 0.08,
    "atrial_fibrillation": 0.05,
    "obesity": 0.30,
    "sleep_apnea": 0.15,
    "depression": 0.20,
    "anxiety": 0.15,
    "anemia": 0.05,
    "osteoarthritis": 0.15,
}


# ---------------------------------------------------------------------------
# Medication mapping
# ---------------------------------------------------------------------------

MEDICATION_MAP = {
    "hypertension": [
        "ace_inhibitor", "arb", "beta_blocker", "ccb", "diuretic",
    ],
    "diabetes": [
        "metformin", "insulin", "sulfonylurea",
    ],
    "heart_failure": [
        "ace_inhibitor", "beta_blocker", "diuretic", "aldosterone_antagonist",
    ],
    "atrial_fibrillation": [
        "anticoagulant", "rate_control",
    ],
    "coronary_artery_disease": [
        "aspirin", "statin", "beta_blocker", "ace_inhibitor",
    ],
}


# ---------------------------------------------------------------------------
# Disease type mapping for OHCA risk
# ---------------------------------------------------------------------------

DISEASE_ASSIGNMENT_PROBS = {
    "coronary_artery_disease": [
        (DiseaseType.ACS_UNSTABLE_ANGINA, 0.15),
        (DiseaseType.ACS_NSTEMI, 0.10),
        (DiseaseType.ACS_STEMI, 0.08),
    ],
    "heart_failure": [
        (DiseaseType.DCM, 0.12),
        (DiseaseType.HCM, 0.05),
    ],
    "atrial_fibrillation": [
        (DiseaseType.AF, 0.15),
        (DiseaseType.VT, 0.05),
    ],
    "hypertension": [
        (DiseaseType.HCM, 0.02),
    ],
    "diabetes": [
        (DiseaseType.HYPERKALEMIA, 0.05),
    ],
    "chronic_kidney_disease": [
        (DiseaseType.HYPERKALEMIA, 0.08),
        (DiseaseType.RESPIRATORY_FAILURE, 0.03),
    ],
}


class PopulationGenerator:
    """Creates a diverse virtual population with realistic distributions.

    Generates ``VirtualPatient`` instances with demographic attributes
    sampled from specified distributions, comorbidities assigned based on
    age/BMI/sex, medications matched to comorbidities, and disease states
    chosen from clinically relevant ontologies.

    Args:
        config: Root configuration.
        rng: Numpy random generator for reproducibility.
    """

    def __init__(self, config: OHCAConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.sim_config: SimulationConfig = config.simulation

    # ======================================================================
    # Main generation method
    # ======================================================================

    def generate_population(self, n_patients: int) -> List[VirtualPatient]:
        """Generate *n_patients* virtual patients with realistic demographics.

        Steps:
            1. Create ``VirtualPatient`` instances.
            2. Override demographics with specified distributions.
            3. Assign comorbidities with age/BMI/sex modulation.
            4. Assign medications based on comorbidities.
            5. Assign diseases for OHCA risk.
            6. Ensure ~30% have 2+ concurrent disease processes.
            7. Create hard-negative cohort with ECG/sensor abnormalities.

        Args:
            n_patients: Number of patients to generate.

        Returns:
            List of fully configured ``VirtualPatient`` instances.
        """
        patients: List[VirtualPatient] = []

        for _ in range(n_patients):
            patient_rng = np.random.default_rng(
                self.rng.integers(0, 2**63)
            )

            patient = VirtualPatient(rng=patient_rng, config=self.config)

            self._override_demographics(patient)

            comorbidities = self._assign_comorbidities(
                patient.age, patient.sex, patient.bmi,
                patient.smoking_status, patient_rng,
            )
            patient._comorbidities = comorbidities

            medications = self._assign_medications(comorbidities, patient_rng)
            patient._medication_names = medications

            diseases = self._assign_diseases(
                patient.age, patient.sex, comorbidities, patient_rng,
            )
            patient.active_diseases = [d.value for d in diseases]

            if self._is_hard_negative(patient):
                patient._is_hard_negative = True
            else:
                patient._is_hard_negative = False

            patients.append(patient)

        self._ensure_polydisease_patients(patients)

        # Assign OHCA events to high-risk patients
        self.assign_ohca_events(patients, ohca_rate=0.40)

        return patients

    # ======================================================================
    # Demographic override methods
    # ======================================================================

    def _override_demographics(self, patient: VirtualPatient) -> None:
        """Override patient demographics with specified distributions."""
        patient.age = self._assign_age(self.rng)
        patient.sex = self._assign_sex(self.rng)
        patient.ethnicity = self._assign_ethnicity(self.rng)

        bmi_category = self._assign_bmi_category(self.rng)
        patient.bmi = self._sample_bmi_from_category(bmi_category, self.rng)

        if patient.sex == "male":
            patient.height_cm = float(
                np.clip(self.rng.normal(175.0, 8.0), 155.0, 200.0)
            )
        else:
            patient.height_cm = float(
                np.clip(self.rng.normal(162.0, 7.0), 145.0, 185.0)
            )
        patient.weight_kg = patient.bmi * (patient.height_cm / 100.0) ** 2

        patient.occupation = self._assign_occupation(
            patient_rng := self.rng, patient.age,
            self._compute_fitness(patient.age, patient.bmi, "sedentary"),
        )

        patient.smoking_status = self._assign_smoking(patient_rng, patient.age)
        patient.alcohol_use = self._assign_alcohol(patient_rng, patient.age, patient.sex)

        patient.fitness_level = self._compute_fitness(
            patient.age, patient.bmi, patient.occupation,
        )
        patient.sleep_quality = float(
            np.clip(patient_rng.normal(0.65, 0.15), 0.0, 1.0)
        )

    def _assign_age(self, rng: np.random.Generator) -> float:
        """Sample age with elderly overrepresentation.

        Distribution:
            - Young adults [18, 40]: 30%
            - Middle-aged [40, 65]: 35%
            - Elderly [65, 80]: 25%
            - Very elderly [80, 100]: 10%
        """
        bucket = rng.random()
        if bucket < 0.30:
            return float(rng.uniform(18.0, 40.0))
        elif bucket < 0.65:
            return float(rng.uniform(40.0, 65.0))
        elif bucket < 0.90:
            return float(rng.uniform(65.0, 80.0))
        else:
            return float(rng.uniform(80.0, 100.0))

    def _assign_sex(self, rng: np.random.Generator) -> str:
        """Assign sex: Male 49%, Female 51%."""
        return "male" if rng.random() < 0.49 else "female"

    def _assign_ethnicity(self, rng: np.random.Generator) -> str:
        """Assign ethnicity per US Census distribution."""
        ethnicities = ["white", "hispanic", "black", "asian", "other"]
        probs = np.array([0.578, 0.187, 0.136, 0.060, 0.039])
        return str(rng.choice(ethnicities, p=probs))

    def _assign_bmi_category(self, rng: np.random.Generator) -> str:
        """Assign BMI category with specified distribution."""
        categories = [
            "underweight", "normal", "overweight", "obese", "morbid_obese",
        ]
        probs = np.array([0.02, 0.35, 0.35, 0.25, 0.03])
        return str(rng.choice(categories, p=probs))

    def _sample_bmi_from_category(
        self, category: str, rng: np.random.Generator,
    ) -> float:
        """Sample a BMI value within the specified category range."""
        if category == "underweight":
            return float(np.clip(rng.normal(17.5, 1.0), 14.0, 18.49))
        elif category == "normal":
            return float(np.clip(rng.normal(22.5, 2.0), 18.5, 24.99))
        elif category == "overweight":
            return float(np.clip(rng.normal(27.5, 1.5), 25.0, 29.99))
        elif category == "obese":
            return float(np.clip(rng.normal(34.0, 2.5), 30.0, 39.99))
        else:
            return float(np.clip(rng.normal(44.0, 3.0), 40.0, 60.0))

    def _assign_occupation(
        self, rng: np.random.Generator, age: float, fitness: float,
    ) -> str:
        """Assign occupation influenced by age and fitness."""
        occupations = [
            "sedentary", "light_manual", "heavy_manual",
            "athletic", "desk_professional",
        ]
        base_probs = np.array([0.15, 0.10, 0.10, 0.05, 0.60])

        if age < 30:
            base_probs[3] *= 2.0
            base_probs[0] *= 0.5
        elif age > 65:
            base_probs[3] *= 0.2
            base_probs[2] *= 0.3
            base_probs[0] *= 1.5

        if fitness > 0.7:
            base_probs[3] *= 2.0
            base_probs[2] *= 1.5
        elif fitness < 0.3:
            base_probs[3] *= 0.3
            base_probs[2] *= 0.5
            base_probs[0] *= 1.5

        probs = base_probs / base_probs.sum()
        return str(rng.choice(occupations, p=probs))

    def _assign_smoking(self, rng: np.random.Generator, age: float) -> str:
        """Assign smoking status with age-dependent probabilities."""
        if age < 25:
            p_current = 0.12
        elif age < 45:
            p_current = 0.18
        elif age < 65:
            p_current = 0.14
        else:
            p_current = 0.07

        p_former = max(0.0, min(0.5, 0.3 - p_current))
        p_never = max(0.0, 1.0 - p_current - p_former)

        return str(rng.choice(["never", "former", "current"], p=[p_never, p_former, p_current]))

    def _assign_alcohol(
        self, rng: np.random.Generator, age: float, sex: str,
    ) -> str:
        """Assign alcohol use influenced by age and sex."""
        base_probs = np.array([0.20, 0.35, 0.30, 0.15])
        if sex == "male":
            base_probs[3] *= 1.3
            base_probs[1] *= 0.9
        else:
            base_probs[3] *= 0.7
            base_probs[1] *= 1.1

        if age < 25:
            base_probs[3] *= 1.5
            base_probs[0] *= 0.8
        elif age > 65:
            base_probs[3] *= 0.5
            base_probs[2] *= 0.7
            base_probs[0] *= 1.3

        probs = base_probs / base_probs.sum()
        return str(rng.choice(["none", "light", "moderate", "heavy"], p=probs))

    def _compute_fitness(
        self, age: float, bmi: float, occupation: str,
    ) -> float:
        """Compute fitness level from age, BMI, and occupation."""
        raw = 1.0
        raw -= 0.4 * ((age - 18.0) / 82.0)
        raw -= 0.3 * ((bmi - 18.0) / 37.0)

        occ_mod = {
            "athletic": 0.15,
            "heavy_manual": 0.10,
            "light_manual": 0.05,
            "desk_professional": 0.0,
            "sedentary": -0.05,
        }
        raw += occ_mod.get(occupation, 0.0)

        return float(np.clip(raw, 0.0, 1.0))

    # ======================================================================
    # Comorbidity assignment
    # ======================================================================

    def _assign_comorbidities(
        self,
        age: float,
        sex: str,
        bmi: float,
        smoking: str,
        rng: np.random.Generator,
    ) -> List[str]:
        """Assign comorbidities with age/BMI/sex-dependent probabilities.

        Returns:
            List of comorbidity names.
        """
        comorbidities: List[str] = []

        p_hypertension = 0.45
        if age < 40:
            p_hypertension *= 0.3
        elif age < 55:
            p_hypertension *= 0.7
        elif age < 70:
            p_hypertension *= 1.2
        else:
            p_hypertension *= 1.5
        if sex == "male":
            p_hypertension *= 1.1
        if rng.random() < p_hypertension:
            comorbidities.append("hypertension")

        p_diabetes = 0.15
        if age < 40:
            p_diabetes *= 0.4
        elif age < 55:
            p_diabetes *= 0.8
        elif age < 70:
            p_diabetes *= 1.3
        else:
            p_diabetes *= 1.2
        if bmi > 30:
            p_diabetes *= 2.0
        elif bmi > 25:
            p_diabetes *= 1.3
        if rng.random() < p_diabetes:
            comorbidities.append("diabetes")

        p_cad = 0.10
        if age < 40:
            p_cad *= 0.2
        elif age < 55:
            p_cad *= 0.6
        elif age < 70:
            p_cad *= 1.3
        else:
            p_cad *= 1.8
        if sex == "male":
            p_cad *= 1.4
        if rng.random() < p_cad:
            comorbidities.append("coronary_artery_disease")

        p_hf = 0.05
        if age < 50:
            p_hf *= 0.3
        elif age < 65:
            p_hf *= 0.8
        elif age < 80:
            p_hf *= 1.8
        else:
            p_hf *= 2.5
        if "coronary_artery_disease" in comorbidities:
            p_hf *= 2.0
        if "hypertension" in comorbidities:
            p_hf *= 1.5
        if rng.random() < p_hf:
            comorbidities.append("heart_failure")

        p_copd = 0.08
        if smoking == "current":
            p_copd *= 3.0
        elif smoking == "former":
            p_copd *= 1.8
        if age > 50:
            p_copd *= 1.3
        if rng.random() < p_copd:
            comorbidities.append("copd")

        p_ckd = 0.08
        if age < 40:
            p_ckd *= 0.3
        elif age < 55:
            p_ckd *= 0.7
        elif age < 70:
            p_ckd *= 1.2
        else:
            p_ckd *= 1.8
        if "diabetes" in comorbidities:
            p_ckd *= 2.0
        if "hypertension" in comorbidities:
            p_ckd *= 1.5
        if rng.random() < p_ckd:
            comorbidities.append("chronic_kidney_disease")

        p_af = 0.05
        if age < 50:
            p_af *= 0.2
        elif age < 65:
            p_af *= 0.7
        elif age < 80:
            p_af *= 1.8
        else:
            p_af *= 2.5
        if "heart_failure" in comorbidities:
            p_af *= 2.0
        if "coronary_artery_disease" in comorbidities:
            p_af *= 1.5
        if rng.random() < p_af:
            comorbidities.append("atrial_fibrillation")

        p_obesity = 0.30
        if bmi > 30:
            p_obesity = 1.0
        elif bmi > 25:
            p_obesity = 0.5
        elif bmi < 18.5:
            p_obesity = 0.0
        if rng.random() < p_obesity:
            comorbidities.append("obesity")

        p_sa = 0.15
        if bmi < 25:
            p_sa *= 0.3
        elif bmi < 30:
            p_sa *= 0.8
        elif bmi < 35:
            p_sa *= 1.5
        else:
            p_sa *= 2.5
        if sex == "male":
            p_sa *= 1.5
        if age > 50:
            p_sa *= 1.3
        if rng.random() < p_sa:
            comorbidities.append("sleep_apnea")

        p_depression = 0.20
        if age < 30:
            p_depression *= 1.2
        elif age > 65:
            p_depression *= 0.7
        if sex == "female":
            p_depression *= 1.3
        if rng.random() < p_depression:
            comorbidities.append("depression")

        p_anxiety = 0.15
        if age < 40:
            p_anxiety *= 1.3
        elif age > 65:
            p_anxiety *= 0.6
        if sex == "female":
            p_anxiety *= 1.2
        if rng.random() < p_anxiety:
            comorbidities.append("anxiety")

        p_anemia = 0.05
        if age > 65:
            p_anemia *= 1.8
        if sex == "female":
            p_anemia *= 1.4
        if "chronic_kidney_disease" in comorbidities:
            p_anemia *= 2.5
        if rng.random() < p_anemia:
            comorbidities.append("anemia")

        p_oa = 0.15
        if age < 40:
            p_oa *= 0.2
        elif age < 55:
            p_oa *= 0.6
        elif age < 70:
            p_oa *= 1.3
        else:
            p_oa *= 2.0
        if bmi > 30:
            p_oa *= 1.5
        if rng.random() < p_oa:
            comorbidities.append("osteoarthritis")

        return comorbidities

    # ======================================================================
    # Medication assignment
    # ======================================================================

    def _assign_medications(
        self, comorbidities: List[str], rng: np.random.Generator,
    ) -> List[str]:
        """Assign medications based on active comorbidities.

        Returns:
            List of medication class names.
        """
        medications: List[str] = []

        for comorbidity in comorbidities:
            if comorbidity in MEDICATION_MAP:
                available = MEDICATION_MAP[comorbidity]
                n_to_add = max(1, len(available) // 2 + rng.integers(0, 2))
                n_to_add = min(n_to_add, len(available))
                chosen = rng.choice(
                    available, size=n_to_add, replace=False,
                )
                medications.extend(chosen.tolist())

        medications = list(dict.fromkeys(medications))

        if rng.random() < 0.20:
            extra_pool = [
                "statin", "aspirin", "ace_inhibitor", "beta_blocker",
                "diuretic", "metformin", "anticoagulant",
            ]
            extras = rng.choice(
                extra_pool,
                size=min(3, len(extra_pool)),
                replace=False,
            )
            medications.extend(extras.tolist())
            medications = list(dict.fromkeys(medications))

        return medications

    # ======================================================================
    # Disease assignment
    # ======================================================================

    def _assign_diseases(
        self,
        age: float,
        sex: str,
        comorbidities: List[str],
        rng: np.random.Generator,
    ) -> List[DiseaseType]:
        """Assign disease types for OHCA risk from comorbidities.

        Also assigns sex-specific channelopathies:
            - HCM: higher in males
            - LQTS: higher detection in females
            - ARVC: higher in males
            - Brugada: 8:1 male predominance

        Returns:
            List of DiseaseType enum values.
        """
        diseases: List[DiseaseType] = []

        for comorbidity in comorbidities:
            if comorbidity in DISEASE_ASSIGNMENT_PROBS:
                for dtype, prob in DISEASE_ASSIGNMENT_PROBS[comorbidity]:
                    if rng.random() < prob:
                        diseases.append(dtype)

        p_hcm = 0.01
        if sex == "male":
            p_hcm *= 2.0
        if age < 50:
            p_hcm *= 1.2
        if "hypertension" in comorbidities:
            p_hcm *= 1.5
        if rng.random() < p_hcm:
            diseases.append(DiseaseType.HCM)

        p_lqts = 0.008
        if sex == "female":
            p_lqts *= 2.5
        if age < 40:
            p_lqts *= 1.5
        if rng.random() < p_lqts:
            diseases.append(DiseaseType.LQTS)

        p_arvc = 0.008
        if sex == "male":
            p_arvc *= 3.0
        if age < 50:
            p_arvc *= 1.3
        if rng.random() < p_arvc:
            diseases.append(DiseaseType.ARVC)

        p_brugada = 0.005
        if sex == "male":
            p_brugada *= 8.0
        if rng.random() < p_brugada:
            diseases.append(DiseaseType.BRUGADA)

        p_sqts = 0.005
        if sex == "male":
            p_sqts *= 1.5
        if rng.random() < p_sqts:
            diseases.append(DiseaseType.SQTS)

        p_cpvt = 0.005
        if age < 35:
            p_cpvt *= 2.0
        if rng.random() < p_cpvt:
            diseases.append(DiseaseType.CPVT)

        p_vt = 0.01
        if "coronary_artery_disease" in comorbidities:
            p_vt *= 3.0
        if "heart_failure" in comorbidities:
            p_vt *= 2.5
        if age > 60:
            p_vt *= 1.5
        if rng.random() < p_vt:
            diseases.append(DiseaseType.VT)

        p_pe = 0.01
        if age > 60:
            p_pe *= 1.5
        if "obesity" in comorbidities:
            p_pe *= 1.8
        if "atrial_fibrillation" in comorbidities:
            p_pe *= 1.3
        if rng.random() < p_pe:
            diseases.append(DiseaseType.PE)

        p_sepsis = 0.005
        if age > 65:
            p_sepsis *= 1.5
        if "diabetes" in comorbidities:
            p_sepsis *= 1.3
        if "chronic_kidney_disease" in comorbidities:
            p_sepsis *= 1.3
        if rng.random() < p_sepsis:
            diseases.append(DiseaseType.SEPSIS)

        return diseases

    # ======================================================================
    # Polydisease enforcement
    # ======================================================================

    def _ensure_polydisease_patients(
        self, patients: List[VirtualPatient],
    ) -> None:
        """Ensure at least 30% of patients have 2+ disease processes.

        If fewer than 30% have 2+ diseases, randomly upgrade some patients
        by adding appropriate diseases.
        """
        n_total = len(patients)
        target_n = int(0.30 * n_total)

        current_n = sum(
            1 for p in patients if len(getattr(p, "active_diseases", [])) >= 2
        )

        if current_n >= target_n:
            return

        candidates = [
            p for p in patients
            if len(getattr(p, "active_diseases", [])) < 2
        ]

        n_needed = target_n - current_n
        rng = self.rng

        for patient in candidates[:n_needed]:
            comorbidities = getattr(patient, "_comorbidities", [])
            extra_diseases = self._assign_diseases(
                patient.age, patient.sex, comorbidities, rng,
            )
            existing = set(patient.active_diseases)
            for d in extra_diseases:
                if d.value not in existing:
                    patient.active_diseases.append(d.value)
                    existing.add(d.value)
                    break

    # ======================================================================
    # OHCA event assignment
    # ======================================================================

    # High-risk diseases that can cause OHCA within the observation window
    HIGH_RISK_DISEASES = {
        "acs_stemi", "acs_nstemi", "acs_unstable_angina",
        "vt", "vf", "lqts", "cpvt", "brugada", "arvc",
        "hcm", "dcm", "complete_heart_block", "pe",
        "sepsis", "hyperkalemia", "respiratory_failure",
    }

    def assign_ohca_events(
        self,
        patients: List[VirtualPatient],
        ohca_rate: float = 0.20,
        observation_hours: float = 0.12,
    ) -> None:
        """Assign OHCA event times for high-risk patients.

        ~20% of patients with high-risk diseases will have OHCA within
        the observation window. The time_to_ohca_hours is set so that
        disease progression is visible in the signals BEFORE the event.

        For OHCA prediction to work, the event must occur WITHIN the
        patient's observation window, and the label is 1.0 when the
        event is within the pre_arrest_window (4h by default) of the
        current timepoint.

        Args:
            patients: List of VirtualPatient objects (modified in-place).
            ohca_rate: Fraction of patients to assign OHCA events.
            observation_hours: Maximum observation window in hours.
        """
        rng = self.rng
        pre_arrest_window = 4.0  # hours - label is 1 if OHCA within this window

        for patient in patients:
            active = set(getattr(patient, "active_diseases", []))
            has_high_risk = bool(active & self.HIGH_RISK_DISEASES)

            # Probability of OHCA assignment
            if has_high_risk:
                p_ohca = ohca_rate * 2.5  # Higher rate for high-risk
            else:
                p_ohca = ohca_rate * 0.25  # Lower rate for low-risk

            if rng.random() < p_ohca:
                # Set OHCA time WITHIN the observation window
                # Must be > 0 and <= observation_hours
                # This ensures _derive_label returns 1.0 for timepoints near 0
                time_to_ohca = float(rng.uniform(0.02, max(0.03, observation_hours * 0.8)))
                patient.time_to_ohca_hours = time_to_ohca

                # For high-risk diseases, make disease progression more aggressive
                if has_high_risk:
                    patient._ohca_severity = 1.0
                else:
                    # Generic deterioration: set up for VT/VF
                    patient._ohca_severity = rng.uniform(0.5, 1.0)
                    patient._progressive_deterioration = True
            else:
                patient.time_to_ohca_hours = None
                patient._ohca_severity = 0.0

    # ======================================================================
    # Hard negative identification
    # ======================================================================

    def _is_hard_negative(self, patient: VirtualPatient) -> bool:
        """Determine if a patient qualifies as a hard negative.

        Hard negatives have ECG/sensor abnormalities but are NOT at imminent
        OHCA risk.  Categories:

        - Bundle Branch Block (LBBB/RBBB): ~2%
        - Pacemaker: ~3% elderly
        - Old MI: ~5% of population >60
        - Chronic PVCs: ~5%
        - AF (well-controlled): ~5% >65
        - Athlete's Heart: ~2% young
        - LVH: ~10% hypertensive
        - Sleep Apnea: ~15% obese
        - COPD: ~8% smokers
        - Diabetes: ~15% >50
        - Obesity: ~30%
        - Chronic Ischemia: ~5% >60
        """
        age = patient.age
        bmi = patient.bmi
        smoking = patient.smoking_status
        comorbidities = getattr(patient, "_comorbidities", [])

        rng = self.rng

        if rng.random() < 0.02:
            patient._hard_negative_type = "bundle_branch_block"
            return True

        if age > 65 and rng.random() < 0.03:
            patient._hard_negative_type = "pacemaker"
            return True

        if age > 60 and rng.random() < 0.05:
            patient._hard_negative_type = "old_mi"
            return True

        if rng.random() < 0.05:
            patient._hard_negative_type = "chronic_pvcs"
            return True

        if age > 65 and rng.random() < 0.05:
            patient._hard_negative_type = "well_controlled_af"
            return True

        if age < 35 and patient.fitness_level > 0.7 and rng.random() < 0.02:
            patient._hard_negative_type = "athlete_heart"
            return True

        if "hypertension" in comorbidities and rng.random() < 0.10:
            patient._hard_negative_type = "lvh"
            return True

        if bmi > 30 and rng.random() < 0.15:
            patient._hard_negative_type = "sleep_apnea"
            return True

        if smoking in ("current", "former") and rng.random() < 0.08:
            patient._hard_negative_type = "copd"
            return True

        if age > 50 and "diabetes" in comorbidities and rng.random() < 0.15:
            patient._hard_negative_type = "diabetes"
            return True

        if bmi > 30 and rng.random() < 0.30:
            patient._hard_negative_type = "obesity"
            return True

        if age > 60 and rng.random() < 0.05:
            patient._hard_negative_type = "chronic_ischemia"
            return True

        return False
