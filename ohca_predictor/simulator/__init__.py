"""OHCA Predictor simulator module.

Provides virtual patient generation and continuous physiological simulation
for out-of-hospital cardiac arrest prediction model training and evaluation.

Core exports:
    VirtualPatient  - Complete patient with demographics and latent state.
    CircadianModel  - Kronauer-type SCN oscillator with melatonin feedback.
    ActivityModel   - Time-inhomogeneous Markov chain for physical activity.
    ANSModel        - Autonomic nervous system with RSA and baroreflex.
    Medication      - Pharmacokinetic model for a single drug.
"""

from ohca_predictor.simulator.patient import (
    VirtualPatient,
    Medication,
)
from ohca_predictor.simulator.physiology import (
    CircadianModel,
    ActivityModel,
    ANSModel,
    RespiratoryModel,
    ThermoregulationModel,
    HydrationModel,
)

__all__ = [
    "VirtualPatient",
    "Medication",
    "CircadianModel",
    "ActivityModel",
    "ANSModel",
    "RespiratoryModel",
    "ThermoregulationModel",
    "HydrationModel",
]
