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
    notes: str = ""

    def sample(self, rng: np.random.Generator) -> dict:
        rhythm = rng.choice(self.rhythm_options, p=self.rhythm_probs) if len(self.rhythm_options) > 1 \
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
}