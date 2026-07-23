"""
wristppg — a physiologically-grounded wrist PPG simulator for research /
data-augmentation use.

v0.3.0: Major overhaul for cardiac arrest realism with wrist anatomy,
ambient light contamination, baroreflex autonomic model, cardiac arrest
disease profiles, and realistic wrist motion artifacts.
"""

from .simulator import WristPPGSimulator, SimulationResult
from .disease import DiseaseProfile, PROFILES
from .motion import MotionArtifactModel, MotionEvent
from .autonomic import AutonomicSimulator
from .microvasculature import MicrovascularBedModel
from .optics import SkinOpticalModel, SkinOpticalParams, WristAnatomy
from .contact import ContactModel, ContactState
from .arrhythmia import RhythmGenerator, ArrhythmiaConfig

__all__ = [
    "WristPPGSimulator", "SimulationResult",
    "DiseaseProfile", "PROFILES",
    "MotionArtifactModel", "MotionEvent",
    "AutonomicSimulator",
    "MicrovascularBedModel",
    "SkinOpticalModel", "SkinOpticalParams", "WristAnatomy",
    "ContactModel", "ContactState",
    "RhythmGenerator", "ArrhythmiaConfig",
]
__version__ = "0.3.0"
