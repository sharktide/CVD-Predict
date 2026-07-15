"""
wristppg — a physiologically-grounded wrist PPG simulator for research /
data-augmentation use.

See README.md in the package root for a full description of the pipeline,
the evidence base for each component, and an explicit list of assumptions
that are heuristic rather than validated against primary literature or
public wrist-PPG datasets.

This package is NOT a substitute for real clinical or wearable data. It
does not model, and must never be used to claim it models, acute
myocardial infarction or cardiac arrest signatures from PPG alone.
"""

from .simulator import WristPPGSimulator
from .disease import DiseaseProfile, PROFILES

__all__ = ["WristPPGSimulator", "DiseaseProfile", "PROFILES"]
__version__ = "0.1.0"