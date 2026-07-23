"""Central configuration loader for the CVD risk prediction pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

# ---------------------------------------------------------------------------
# Project root (one level above src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path_str: str) -> Path:
    """Resolve a config path string relative to PROJECT_ROOT."""
    p = Path(path_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    path = _resolve(path)
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Config accessors
# ---------------------------------------------------------------------------

def get_paths_config() -> Dict[str, Any]:
    return load_yaml("configs/paths.yaml")


def get_model_config() -> Dict[str, Any]:
    return load_yaml("configs/model.yaml")


def get_training_config() -> Dict[str, Any]:
    return load_yaml("configs/training.yaml")


def get_eval_config() -> Dict[str, Any]:
    return load_yaml("configs/eval.yaml")


# ---------------------------------------------------------------------------
# Convenience resolved path helpers
# ---------------------------------------------------------------------------

def resolved_paths() -> Dict[str, Path]:
    """Return fully resolved directory paths from paths.yaml."""
    raw = get_paths_config()
    return {key: _resolve(val) for key, val in raw.items()}
