"""
OHCA Predictor: Multimodal Wearable AI System for Predicting
Out-of-Hospital Cardiac Arrest (OHCA).

Safety-critical medical ML system for forecasting OHCA up to 4 hours in
advance by processing multi-modal wearable sensor streams.
"""

__version__ = "1.0.0"

from ohca_predictor.config import (
    OHCAConfig,
    ReproducibilityConfig,
    SimulationConfig,
    DiseaseConfig,
    SensorConfig,
    ModelConfig,
    TrainingConfig,
    EvaluationConfig,
    LLMConfig,
    get_config,
    override_config,
)

__all__ = [
    "__version__",
    "OHCAConfig",
    "ReproducibilityConfig",
    "SimulationConfig",
    "DiseaseConfig",
    "SensorConfig",
    "ModelConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "LLMConfig",
    "get_config",
    "override_config",
]
