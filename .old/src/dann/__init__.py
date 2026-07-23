"""
DANN (Domain-Adversarial Neural Network) module for cardiac arrest prediction.
"""

from .model import (
    DANN_CardiacArrest,
    SharedFeatureEncoder,
    DomainClassifier,
    TemporalPredictor,
    GradientReversalLayer,
    FocalLoss,
    DANNLoss,
    build_dann_model,
)

from .train import (
    CardiacArrestDataset,
    create_data_loaders,
    train_epoch,
    validate,
    main,
)

__all__ = [
    "DANN_CardiacArrest",
    "SharedFeatureEncoder",
    "DomainClassifier",
    "TemporalPredictor",
    "GradientReversalLayer",
    "FocalLoss",
    "DANNLoss",
    "build_dann_model",
    "CardiacArrestDataset",
    "create_data_loaders",
    "train_epoch",
    "validate",
    "main",
]
