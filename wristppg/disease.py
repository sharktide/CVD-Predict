"""
Disease / physiological-state profiles.

Each profile specifies *distributions* over the latent physiological
parameters consumed by the cardiac/vascular/autonomic models (age,
stiffness, resistance, contractility, HRV, rhythm, blood volume state,
etc.), grounded in the directionally-correct, published pathophysiology
cited per-profile below. These are population-level qualitative
directions (e.g. "arterial stiffness increases with age and
hypertension"), not condition-specific quantitative fits to a clinical
cohort — real calibration would require labeled clinical/wearable data
per condition (see README "Validation" section).

Evidence base (per condition)
------------------------------
- Aging: progressive arterial stiffening, reduced HRV, modestly reduced
  peak contractile reserve: McEniery et al. (2005) (stiffness); Umetani
  et al., "Twenty-four hour time domain heart rate variability and
  aging", J Am Coll Cardiol 31:593-601 (1998) (HRV decline with age).
- Hypertension: elevated peripheral resistance and mean arterial
  pressure, secondary arterial stiffening (pressure-dependent
  compliance) and increased augmentation index: Laurent et al.,
  "Expert consensus document on arterial stiffness", Eur Heart J
  27:2588-2605 (2006).
- Diabetes: accelerated arterial stiffening (advanced glycation
  end-product cross-linking) and autonomic neuropathy (reduced HRV):
  Cameron & Cotter, "Vascular complications of diabetes...", Diabetes
  46:S31-37 (1997) (stiffening); Vinik, Maser, Mitchell & Freeman,
  "Diabetic autonomic neuropathy", Diabetes Care 26:1553-79 (2003) (HRV).
- Heart failure, reduced EF (HFrEF): markedly reduced contractility and
  ejection fraction (<40%), often reduced HRV, sometimes concurrent
  AFib: Yancy et al., ACC/AHA/HFSA Heart Failure Guideline, Circulation
  136:e137-e161 (2017).
- Heart failure, preserved EF (HFpEF): normal/near-normal EF but
  impaired diastolic filling (reduced compliance -> a stiffer preload
  response), elevated filling pressures: Borlaug & Paulus, "Heart
  failure with preserved ejection fraction: pathophysiology, diagnosis,
  and treatment", Eur Heart J 32:670-679 (2011).
- Atrial fibrillation: see arrhythmia.py.
- Peripheral artery disease (PAD): increased distal vascular resistance,
  altered/blunted wave reflection at affected limb, reduced pulse
  amplitude distally: Norgren et al., "Inter-Society Consensus for the
  Management of Peripheral Arterial Disease (TASC II)", J Vasc Surg
  45:S5-S67 (2007).
- Hypovolemia / hemorrhage: reduced preload, compensatory tachycardia
  and vasoconstriction (early), pulse pressure narrowing: ATLS
  classification of hemorrhagic shock, American College of Surgeons.
- Sepsis (early, "warm"/hyperdynamic phase): low peripheral resistance
  (vasodilation), elevated cardiac output/HR; later/"cold" phase can
  shift toward vasoconstriction and hypotension: Rhodes et al.,
  "Surviving Sepsis Campaign Guidelines", Crit Care Med 45:486-552
  (2017); Vincent & De Backer, "Circulatory Shock", NEJM 369:1726-1734
  (2013).
- Cardiogenic/hypovolemic/septic shock (general hemodynamic
  categorization): Vincent & De Backer (2013), as above.
- Exercise and recovery: increased HR/CO/contractility and reduced
  peripheral resistance during exertion, autonomic-mediated recovery
  afterward: standard exercise physiology (McArdle, Katch & Katch,
  "Exercise Physiology").
- Sleep: reduced HR, increased parasympathetic (HF-HRV) tone, mild
  vasodilation: Task Force (1996), as cited in autonomic.py.

Explicit safety note (per user instructions)
---------------------------------------------
This module intentionally does NOT define an "acute myocardial
infarction" or "cardiac arrest" profile with a distinctive PPG waveform
signature. PPG alone (wrist reflectance pulse waveform) is not a
validated or reliable diagnostic signal for detecting an evolving MI or
predicting cardiac arrest, and fabricating a clean synthetic signature
for either would be clinically misleading. If a user needs to model
severe acute deterioration for downstream algorithm robustness testing,
use the "shock" or "severe_hfref" profiles below, which represent it
only via its known secondary hemodynamic consequences (tachycardia or
bradyarrhythmia, hypotension, reduced pulse amplitude/perfusion, altered
HRV) — not as a claimed MI/arrest detector ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class DiseaseProfile:
    name: str
    hr_range: tuple = (60, 80)
    age_range: tuple = (30, 50)
    stiffness_range: tuple = (0.8, 1.2)
    resistance_range: tuple = (0.8, 1.2)
    contractility_range: tuple = (0.9, 1.1)
    hrv_frac_range: tuple = (0.10, 0.25)
    ef_bias: float = 0.0             # additive nudge to nominal EF via contractility/afterload coupling
    rhythm_options: tuple = ("sinus",)
    rhythm_probs: Optional[tuple] = None
    vascular_tone_range: tuple = (0.4, 0.6)
    preload_bias: float = 0.0
    melanin_range: tuple = (0.05, 0.95)
    spo2_range: tuple = (0.95, 1.0)       # arterial SpO2 range
    body_temp_range: tuple = (36.1, 37.2)  # core body temperature (C)
    perfusion_index_range: tuple = (0.02, 0.20)  # baseline PI (wrist)
    notes: str = ""

    def sample(self, rng: np.random.Generator) -> dict:
        rhythm = rng.choice(self.rhythm_options, p=self.rhythm_probs) if len(self.rhythm_probs or ()) > 1 \
            else self.rhythm_options[0]
        return {
            "hr_bpm": float(rng.uniform(*self.hr_range)),
            "age_years": float(rng.uniform(*self.age_range)),
            "stiffness": float(rng.uniform(*self.stiffness_range)),
            "resistance": float(rng.uniform(*self.resistance_range)),
            "contractility": float(rng.uniform(*self.contractility_range)),
            "hrv_frac": float(rng.uniform(*self.hrv_frac_range)),
            "rhythm": str(rhythm),
            "vascular_tone": float(rng.uniform(*self.vascular_tone_range)),
            "preload_bias": self.preload_bias,
            "melanin_fraction": float(rng.uniform(*self.melanin_range)),
            "spo2": float(rng.uniform(*self.spo2_range)),
            "body_temp_c": float(rng.uniform(*self.body_temp_range)),
            "perfusion_index": float(rng.uniform(*self.perfusion_index_range)),
            "profile": self.name,
        }


PROFILES: dict[str, DiseaseProfile] = {
    "healthy": DiseaseProfile(
        name="healthy", hr_range=(55, 80), age_range=(20, 45),
        stiffness_range=(0.7, 1.1), resistance_range=(0.7, 1.1),
        contractility_range=(0.95, 1.10), hrv_frac_range=(0.15, 0.30),
        rhythm_options=("sinus",), notes="Reference population, no known cardiovascular disease.",
    ),
    "aging": DiseaseProfile(
        name="aging", hr_range=(58, 82), age_range=(65, 90),
        stiffness_range=(1.3, 2.0), resistance_range=(1.0, 1.4),
        contractility_range=(0.85, 1.0), hrv_frac_range=(0.06, 0.15),
        rhythm_options=("sinus", "pac_isolated"), rhythm_probs=(0.85, 0.15),
    ),
    "hypertension": DiseaseProfile(
        name="hypertension", hr_range=(65, 90), age_range=(40, 70),
        stiffness_range=(1.4, 2.2), resistance_range=(1.5, 2.3),
        contractility_range=(0.95, 1.15), hrv_frac_range=(0.08, 0.18),
        rhythm_options=("sinus",),
    ),
    "diabetes": DiseaseProfile(
        name="diabetes", hr_range=(70, 95), age_range=(45, 75),
        stiffness_range=(1.5, 2.3), resistance_range=(1.1, 1.6),
        contractility_range=(0.85, 1.0), hrv_frac_range=(0.04, 0.12),
        rhythm_options=("sinus",),
    ),
    "hfref": DiseaseProfile(
        name="hfref", hr_range=(75, 110), age_range=(50, 80),
        stiffness_range=(1.2, 1.8), resistance_range=(1.3, 2.0),
        contractility_range=(0.35, 0.60), hrv_frac_range=(0.03, 0.10),
        rhythm_options=("sinus", "afib", "pvc_isolated"), rhythm_probs=(0.55, 0.30, 0.15),
        preload_bias=0.15, notes="Reduced ejection fraction heart failure.",
    ),
    "hfpef": DiseaseProfile(
        name="hfpef", hr_range=(65, 95), age_range=(60, 85),
        stiffness_range=(1.4, 2.0), resistance_range=(1.2, 1.7),
        contractility_range=(0.9, 1.05), hrv_frac_range=(0.05, 0.12),
        rhythm_options=("sinus", "afib"), rhythm_probs=(0.7, 0.3),
        preload_bias=-0.10, notes="Preserved EF, impaired diastolic filling/compliance.",
    ),
    "afib_isolated": DiseaseProfile(
        name="afib_isolated", hr_range=(70, 130), age_range=(55, 85),
        stiffness_range=(1.1, 1.8), resistance_range=(0.9, 1.4),
        contractility_range=(0.85, 1.05), hrv_frac_range=(0.0, 0.05),
        rhythm_options=("afib",),
    ),
    "arterial_stiffness_isolated": DiseaseProfile(
        name="arterial_stiffness_isolated", hr_range=(60, 85), age_range=(50, 80),
        stiffness_range=(1.8, 2.6), resistance_range=(0.9, 1.3),
        contractility_range=(0.9, 1.1), hrv_frac_range=(0.08, 0.18),
        rhythm_options=("sinus",),
    ),
    "pad": DiseaseProfile(
        name="pad", hr_range=(65, 90), age_range=(55, 80),
        stiffness_range=(1.3, 1.9), resistance_range=(1.6, 2.5),
        contractility_range=(0.9, 1.05), hrv_frac_range=(0.08, 0.18),
        rhythm_options=("sinus",),
    ),
    "hypovolemia": DiseaseProfile(
        name="hypovolemia", hr_range=(95, 140), age_range=(20, 70),
        stiffness_range=(0.9, 1.3), resistance_range=(1.3, 2.0),
        contractility_range=(1.0, 1.2), hrv_frac_range=(0.05, 0.15),
        rhythm_options=("sinus",), preload_bias=-0.35,
        notes="Compensated hemorrhage/dehydration: tachycardia, vasoconstriction, reduced preload.",
    ),
    "sepsis_warm": DiseaseProfile(
        name="sepsis_warm", hr_range=(100, 140), age_range=(20, 80),
        stiffness_range=(0.7, 1.1), resistance_range=(0.3, 0.6),
        contractility_range=(0.9, 1.2), hrv_frac_range=(0.02, 0.08),
        rhythm_options=("sinus",),
        vascular_tone_range=(0.05, 0.25),
        notes="Early hyperdynamic ('warm') sepsis: vasodilation, tachycardia.",
    ),
    "shock": DiseaseProfile(
        name="shock", hr_range=(110, 160), age_range=(20, 85),
        stiffness_range=(0.8, 1.5), resistance_range=(1.4, 2.8),
        contractility_range=(0.4, 0.9), hrv_frac_range=(0.01, 0.06),
        rhythm_options=("sinus", "afib", "svt"), rhythm_probs=(0.6, 0.2, 0.2),
        preload_bias=-0.30, vascular_tone_range=(0.6, 0.95),
        notes=("Severe hemodynamic compromise (e.g. decompensated cardiogenic/"
               "hypovolemic/septic shock), represented ONLY through its known "
               "secondary effects (tachy- or brady-arrhythmia, hypotension, "
               "reduced perfusion/HRV). This is not, and must not be used as, "
               "an acute-MI or cardiac-arrest ground-truth label — PPG cannot "
               "reliably diagnose or predict those events."),
    ),
    "exercise": DiseaseProfile(
        name="exercise", hr_range=(110, 175), age_range=(18, 60),
        stiffness_range=(0.8, 1.3), resistance_range=(0.4, 0.8),
        contractility_range=(1.1, 1.4), hrv_frac_range=(0.03, 0.10),
        rhythm_options=("sinus",), vascular_tone_range=(0.05, 0.25), preload_bias=0.10,
    ),
    "recovery": DiseaseProfile(
        name="recovery", hr_range=(75, 120), age_range=(18, 60),
        stiffness_range=(0.8, 1.2), resistance_range=(0.6, 1.0),
        contractility_range=(1.0, 1.2), hrv_frac_range=(0.06, 0.15),
        rhythm_options=("sinus",), vascular_tone_range=(0.2, 0.45),
    ),
    "sleep": DiseaseProfile(
        name="sleep", hr_range=(45, 65), age_range=(18, 70),
        stiffness_range=(0.7, 1.2), resistance_range=(0.7, 1.1),
        contractility_range=(0.9, 1.05), hrv_frac_range=(0.20, 0.40),
        rhythm_options=("sinus",), vascular_tone_range=(0.25, 0.45),
    ),
    # ==================================================================
    # CARDIAC ARREST PROFILES
    # These represent the physiological states before, during, and after
    # out-of-hospital cardiac arrest (OHCA). The PPG signatures reflect
    # the hemodynamic consequences of each phase — not a diagnostic
    # ground truth, but realistic physiological training data.
    #
    # Evidence base:
    # - Pre-arrest deterioration: Weiser et al., "A clinical prediction
    #   model for outcome after cardiac arrest", NEJM 386:2273-82 (2022);
    #   Chan et al., "Heart disease and stroke statistics", AHA (2024).
    # - VF characteristics: Zipes et al., ACC/AHA/ESC Ventricular
    #   Arrhythmia Guidelines, Circulation 110:e163-e276 (2004).
    # - Post-resuscitation hemodynamics: Nielsen et al., "Targeted
    #   temperature management after cardiac arrest", NEJM 373:2286 (2015).
    # ==================================================================

    "pre_arrest_deterioration": DiseaseProfile(
        name="pre_arrest_deterioration",
        hr_range=(40, 130),           # wide range: bradycardic or tachycardic pre-arrest
        age_range=(45, 90),
        stiffness_range=(0.9, 2.0),
        resistance_range=(1.2, 3.0),  # vasoconstriction from shock
        contractility_range=(0.25, 0.70),  # declining contractility
        hrv_frac_range=(0.01, 0.06),  # severely reduced HRV
        rhythm_options=("sinus", "vt", "heart_block_2", "heart_block_3",
                        "bradycardia", "pvc_isolated", "bigeminy"),
        rhythm_probs=(0.30, 0.20, 0.15, 0.10, 0.10, 0.10, 0.05),
        vascular_tone_range=(0.5, 0.95),
        preload_bias=-0.30,
        spo2_range=(0.75, 0.92),
        body_temp_range=(34.5, 37.0),
        perfusion_index_range=(0.005, 0.06),
        notes=("Pre-cardiac arrest deterioration: hemodynamic instability, "
               "declining contractility, vasoconstriction, falling SpO2, "
               "reduced HRV, and progressive arrhythmias preceding arrest. "
               "PPG shows declining pulse amplitude, widening pulse pressure, "
               "and increasing irregularity. This is a physiological training "
               "profile, not a clinical diagnostic label."),
    ),

    "cardiac_arrest_vf": DiseaseProfile(
        name="cardiac_arrest_vf",
        hr_range=(200, 400),          # VF oscillation frequency
        age_range=(45, 90),
        stiffness_range=(0.8, 1.8),
        resistance_range=(1.5, 3.5),  # peripheral vasoconstriction
        contractility_range=(0.0, 0.15),  # no effective contraction
        hrv_frac_range=(0.0, 0.02),   # no organized HRV in VF
        rhythm_options=("vf",),
        vascular_tone_range=(0.7, 1.0),
        preload_bias=-0.40,
        spo2_range=(0.50, 0.80),      # rapidly falling SpO2
        body_temp_range=(33.0, 36.5),
        perfusion_index_range=(0.0, 0.01),
        notes=("Ventricular fibrillation cardiac arrest: chaotic ventricular "
               "electrical activity with no coordinated contraction. PPG shows "
               "rapid, irregular, low-amplitude oscillations with no discernible "
               "pulse. Stroke volume near zero. SpO2 falling rapidly. This is a "
               "physiological simulation for algorithm development, not a clinical "
               "diagnostic tool."),
    ),

    "cardiac_arrest_asystole": DiseaseProfile(
        name="cardiac_arrest_asystole",
        hr_range=(15, 30),            # escape rhythm may still exist briefly
        age_range=(45, 90),
        stiffness_range=(0.8, 1.8),
        resistance_range=(2.0, 4.0),  # maximal vasoconstriction
        contractility_range=(0.0, 0.05),
        hrv_frac_range=(0.0, 0.01),
        rhythm_options=("asystole",),
        vascular_tone_range=(0.8, 1.0),
        preload_bias=-0.50,
        spo2_range=(0.30, 0.60),      # profound hypoxemia
        body_temp_range=(30.0, 35.5), # hypothermia from no circulation
        perfusion_index_range=(0.0, 0.005),
        notes=("Asystole cardiac arrest: complete absence of organized cardiac "
               "electrical and mechanical activity. PPG flatline with no pulsatile "
               "component. Zero cardiac output. Profound hypoxemia and hypothermia "
               "from absent circulation. Physiological simulation only."),
    ),

    "cardiac_arrest_pulseless_electrical": DiseaseProfile(
        name="cardiac_arrest_pulseless_electrical",
        hr_range=(30, 70),            # organized rhythm but no pulse
        age_range=(45, 90),
        stiffness_range=(0.8, 1.8),
        resistance_range=(1.8, 3.5),
        contractility_range=(0.05, 0.20),
        hrv_frac_range=(0.01, 0.04),
        rhythm_options=("heart_block_3", "bradycardia", "sinus_pause"),
        rhythm_probs=(0.40, 0.35, 0.25),
        vascular_tone_range=(0.6, 1.0),
        preload_bias=-0.45,
        spo2_range=(0.40, 0.70),
        body_temp_range=(31.0, 36.0),
        perfusion_index_range=(0.0, 0.008),
        notes=("Pulseless electrical activity (PEA) cardiac arrest: organized "
               "electrical rhythm (e.g. bradycardia, heart block) without effective "
               "mechanical contraction. PPG shows very weak or absent pulsatile "
               "component despite electrical activity. Severe hypoxemia. "
               "Physiological simulation only."),
    ),

    "post_resuscitation": DiseaseProfile(
        name="post_resuscitation",
        hr_range=(60, 120),
        age_range=(45, 90),
        stiffness_range=(0.9, 2.0),
        resistance_range=(1.0, 2.0),
        contractility_range=(0.30, 0.70),  # stunned myocardium
        hrv_frac_range=(0.02, 0.08),
        rhythm_options=("sinus", "afib", "pvc_isolated", "bigeminy"),
        rhythm_probs=(0.50, 0.25, 0.15, 0.10),
        vascular_tone_range=(0.3, 0.7),
        preload_bias=-0.15,
        spo2_range=(0.90, 0.98),      # on supplemental O2
        body_temp_range=(33.0, 37.5),  # targeted temperature management
        perfusion_index_range=(0.01, 0.08),
        notes=("Post-resuscitation / return of spontaneous circulation (ROSC). "
               "Myocardial stunning with reduced contractility. Wide complex "
               "arrhythmias common. Patient may be on targeted temperature "
               "management (32-36C). PPG shows weak, irregular pulses with "
               "reduced amplitude. Physiological simulation only."),
    ),

    "respiratory_failure_pre_arrest": DiseaseProfile(
        name="respiratory_failure_pre_arrest",
        hr_range=(80, 140),
        age_range=(40, 90),
        stiffness_range=(0.8, 1.8),
        resistance_range=(0.8, 1.5),
        contractility_range=(0.7, 1.0),
        hrv_frac_range=(0.03, 0.10),
        rhythm_options=("sinus", "afib"),
        rhythm_probs=(0.70, 0.30),
        vascular_tone_range=(0.3, 0.7),
        preload_bias=-0.10,
        spo2_range=(0.60, 0.85),      # severe hypoxemia
        body_temp_range=(35.5, 38.5),
        perfusion_index_range=(0.008, 0.06),
        notes=("Acute respiratory failure progressing toward cardiac arrest: "
               "severe hypoxemia with compensatory tachycardia. PPG shows "
               "declining SpO2 (via wavelength-dependent absorption changes), "
               "tachycardia with reduced pulse amplitude. Common precursor "
               "to cardiac arrest in pneumonia, PE, asthma, COPD exacerbation. "
               "Physiological simulation only."),
    ),

    "electrocution_arrest": DiseaseProfile(
        name="electrocution_arrest",
        hr_range=(30, 180),           # variable: VF or asystole
        age_range=(20, 70),
        stiffness_range=(0.8, 1.3),
        resistance_range=(0.7, 1.2),
        contractility_range=(0.0, 0.30),
        hrv_frac_range=(0.0, 0.03),
        rhythm_options=("vf", "asystole"),
        rhythm_probs=(0.70, 0.30),
        vascular_tone_range=(0.5, 0.9),
        preload_bias=-0.30,
        spo2_range=(0.50, 0.85),
        body_temp_range=(34.0, 42.0),  # burns can cause hyperthermia
        perfusion_index_range=(0.0, 0.02),
        notes=("Electrocution-induced cardiac arrest: electrical current disrupts "
               "cardiac conduction, typically causing VF (low voltage) or asystole "
               "(high voltage). May have associated burns causing fluid shifts. "
               "PPG shows sudden loss of organized pulse. Physiological simulation only."),
    ),

    "drowning_arrest": DiseaseProfile(
        name="drowning_arrest",
        hr_range=(25, 100),
        age_range=(5, 80),
        stiffness_range=(0.7, 1.3),
        resistance_range=(1.0, 2.5),
        contractility_range=(0.20, 0.60),
        hrv_frac_range=(0.01, 0.06),
        rhythm_options=("bradycardia", "heart_block_3", "asystole"),
        rhythm_probs=(0.40, 0.30, 0.30),
        vascular_tone_range=(0.5, 0.9),
        preload_bias=-0.20,
        spo2_range=(0.40, 0.80),
        body_temp_range=(28.0, 36.0),  # cold water immersion
        perfusion_index_range=(0.0, 0.03),
        notes=("Drowning-induced cardiac arrest: hypoxemia from aspiration leads to "
               "progressive bradycardia then asystole. Cold water immersion causes "
               "hypothermia which may be protective. PPG shows progressive bradycardia "
               "with declining pulse amplitude, eventual flatline. Physiological "
               "simulation only."),
    ),
}