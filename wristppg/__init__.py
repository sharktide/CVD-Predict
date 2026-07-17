"""
wristppg — a physiologically-grounded wrist PPG simulator for research /
data-augmentation use.

See README.md in the package root for a full description of the pipeline,
the evidence base for each component, and an explicit list of assumptions
that are heuristic rather than validated against primary literature or
public wrist-PPG datasets.
"""

from .simulator import WristPPGSimulator, SimulationResult
from .disease import DiseaseProfile, PROFILES
from .motion import MotionArtifactModel, MotionEvent

__all__ = ["WristPPGSimulator", "SimulationResult", "DiseaseProfile", "PROFILES",
           "MotionArtifactModel", "MotionEvent"]
__version__ = "0.2.0"