"""
Centralized configuration system for the OHCA Predictor project.

Safety-critical medical ML system - all parameters validated at initialization.
Uses Python dataclasses for compile-time safety and a singleton pattern for
global access with override support.

Usage:
    from ohca_predictor.config import get_config, override_config

    config = get_config()
    custom_config = override_config(
        training=TrainingConfig(batch_size=64),
        reproducibility=ReproducibilityConfig(seed=123)
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import copy


# Module-level singleton instance
_config: Optional["OHCAConfig"] = None


def get_config() -> "OHCAConfig":
    """
    Get the singleton configuration instance.

    Returns the cached OHCAConfig if it exists, otherwise creates and caches
    a new default instance.

    Returns:
        OHCAConfig: The singleton configuration instance.
    """
    global _config
    if _config is None:
        _config = OHCAConfig()
    return _config


def override_config(**kwargs: Any) -> "OHCAConfig":
    """
    Create a new OHCAConfig with specified overrides.

    Returns a new configuration instance with the given overrides applied
    on top of the current singleton (or default if not initialized). The
    singleton itself is NOT modified.

    Args:
        **kwargs: Keyword arguments where keys are sub-config field names
                  (e.g., training=TrainingConfig(batch_size=64)).

    Returns:
        OHCAConfig: A new configuration instance with overrides applied.
    """
    global _config
    if _config is None:
        _config = OHCAConfig()
    new_config = copy.deepcopy(_config)
    for key, value in kwargs.items():
        if not hasattr(new_config, key):
            raise ValueError(f"Unknown config field: {key}")
        setattr(new_config, key, value)
    return new_config


@dataclass
class ReproducibilityConfig:
    """
    Reproducibility settings for deterministic training and evaluation.

    All derived seeds are computed deterministically from the master seed
    to ensure reproducibility across runs.
    """

    seed: int = 42
    numpy_seed: int = field(init=False)
    tensorflow_seed: int = field(init=False)
    python_hash_seed: int = field(init=False)
    deterministic: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    op_deterministic: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, got {type(self.seed).__name__}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")
        self.numpy_seed = self.seed + 1000
        self.tensorflow_seed = self.seed + 2000
        self.python_hash_seed = self.seed + 3000


@dataclass
class SimulationConfig:
    """
    Simulation and synthetic data generation settings.

    Controls sampling rates, variable window durations, population parameters,
    and device-to-signal mapping. All signals in a window share the same
    wall-clock duration; the model handles variable-length inputs natively.

    Device mapping:
        - Polar H10 chest strap: ECG (single-lead, 130 Hz) + Accelerometer (tri-axis, 52 Hz)
        - Chest sensor (separate): Gyroscope (tri-axis, 52 Hz)
        - Wrist sensor: PPG (50 Hz) + SpO2 (10 Hz) + Temperature (1 Hz)
        - Derived: Respiration from ECG (ECG-derived respiration, 25 Hz)
    """

    sampling_rates: Dict[str, float] = field(
        default_factory=lambda: {
            "ecg_hz": 130,
            "accelerometer_hz": 52,
            "gyroscope_hz": 52,
            "ppg_hz": 50,
            "spo2_hz": 10,
            "temperature_hz": 1,
            "respiration_hz": 25,
        }
    )
    device_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "ecg": "polar_h10_chest",
            "accelerometer": "polar_h10_chest",
            "gyroscope": "chest_sensor",
            "ppg": "wrist_sensor",
            "spo2": "wrist_sensor",
            "temperature": "wrist_sensor",
            "respiration": "derived_from_ecg",
        }
    )
    min_window_duration_hours: float = 0.05
    max_window_duration_hours: float = 48.0
    pre_arrest_window_hours: float = 4.0
    population_size: int = 10000
    prevalence_ohca: float = 0.001

    def __post_init__(self) -> None:
        if self.min_window_duration_hours <= 0:
            raise ValueError(
                f"min_window_duration_hours must be positive, got {self.min_window_duration_hours}"
            )
        if self.max_window_duration_hours <= 0:
            raise ValueError(
                f"max_window_duration_hours must be positive, got {self.max_window_duration_hours}"
            )
        if self.min_window_duration_hours > self.max_window_duration_hours:
            raise ValueError(
                f"min_window_duration_hours ({self.min_window_duration_hours}) must be <= "
                f"max_window_duration_hours ({self.max_window_duration_hours})"
            )
        if self.pre_arrest_window_hours <= 0:
            raise ValueError(
                f"pre_arrest_window_hours must be positive, got {self.pre_arrest_window_hours}"
            )
        if self.population_size <= 0:
            raise ValueError(
                f"population_size must be positive, got {self.population_size}"
            )
        if not (0 <= self.prevalence_ohca <= 1):
            raise ValueError(
                f"prevalence_ohca must be in [0, 1], got {self.prevalence_ohca}"
            )
        for key, rate in self.sampling_rates.items():
            if not isinstance(rate, (int, float)) or rate <= 0:
                raise ValueError(
                    f"sampling_rate for '{key}' must be positive, got {rate}"
                )


@dataclass
class DiseaseConfig:
    """
    Disease simulation and progression settings.

    Defines the supported disease types, comorbidity probabilities,
    and latent space configuration for the disease model.
    """

    disease_types: List[str] = field(
        default_factory=lambda: [
            "acs_unstable_angina",
            "acs_nstemi",
            "acs_stemi",
            "dcm",
            "hcm",
            "arvc",
            "brugada",
            "lqts",
            "sqts",
            "cpvt",
            "af",
            "vt",
            "vf",
            "complete_heart_block",
            "hyperkalemia",
            "hypokalemia",
            "pe",
            "respiratory_failure",
            "sepsis",
            "drug_toxicity",
        ]
    )
    comorbidity_probability: float = 0.3
    disease_progression_rate: float = 0.1
    latent_state_dim: int = 64

    def __post_init__(self) -> None:
        if not self.disease_types:
            raise ValueError("disease_types must not be empty")
        for dt in self.disease_types:
            if not isinstance(dt, str) or not dt.strip():
                raise ValueError(f"disease_type must be a non-empty string, got {dt!r}")
        if not (0 <= self.comorbidity_probability <= 1):
            raise ValueError(
                f"comorbidity_probability must be in [0, 1], got {self.comorbidity_probability}"
            )
        if not (0 <= self.disease_progression_rate <= 1):
            raise ValueError(
                f"disease_progression_rate must be in [0, 1], got {self.disease_progression_rate}"
            )
        if self.latent_state_dim <= 0:
            raise ValueError(
                f"latent_state_dim must be positive, got {self.latent_state_dim}"
            )


@dataclass
class SensorConfig:
    """
    Sensor hardware simulation settings.

    Models realistic sensor behavior including ADC quantization, clock drift,
    noise, communication latency, electrode effects, and power constraints.
    """

    adc_resolution_bits: int = 12
    adc_vref: float = 3.3
    clock_drift_ppm: float = 5.0
    timestamp_jitter_ms: float = 2.0
    packet_loss_rate: float = 0.02
    bluetooth_latency_ms_mean: float = 50.0
    bluetooth_latency_ms_std: float = 15.0
    skin_impedance_range: Tuple[float, float] = (1.0, 100.0)
    electrode_drying_rate: float = 0.01
    battery_voltage_range: Tuple[float, float] = (3.0, 4.2)
    sensor_saturation_threshold: float = 0.95

    def __post_init__(self) -> None:
        if self.adc_resolution_bits <= 0:
            raise ValueError(
                f"adc_resolution_bits must be positive, got {self.adc_resolution_bits}"
            )
        if self.adc_vref <= 0:
            raise ValueError(f"adc_vref must be positive, got {self.adc_vref}")
        if self.clock_drift_ppm < 0:
            raise ValueError(
                f"clock_drift_ppm must be non-negative, got {self.clock_drift_ppm}"
            )
        if self.timestamp_jitter_ms < 0:
            raise ValueError(
                f"timestamp_jitter_ms must be non-negative, got {self.timestamp_jitter_ms}"
            )
        if not (0 <= self.packet_loss_rate <= 1):
            raise ValueError(
                f"packet_loss_rate must be in [0, 1], got {self.packet_loss_rate}"
            )
        if self.bluetooth_latency_ms_mean < 0:
            raise ValueError(
                f"bluetooth_latency_ms_mean must be non-negative, got {self.bluetooth_latency_ms_mean}"
            )
        if self.bluetooth_latency_ms_std < 0:
            raise ValueError(
                f"bluetooth_latency_ms_std must be non-negative, got {self.bluetooth_latency_ms_std}"
            )
        if len(self.skin_impedance_range) != 2:
            raise ValueError(
                f"skin_impedance_range must be a 2-tuple, got {len(self.skin_impedance_range)} elements"
            )
        if self.skin_impedance_range[0] < 0 or self.skin_impedance_range[1] < 0:
            raise ValueError("skin_impedance_range values must be non-negative")
        if self.skin_impedance_range[0] > self.skin_impedance_range[1]:
            raise ValueError(
                f"skin_impedance_range[0] ({self.skin_impedance_range[0]}) must be <= "
                f"skin_impedance_range[1] ({self.skin_impedance_range[1]})"
            )
        if not (0 <= self.electrode_drying_rate <= 1):
            raise ValueError(
                f"electrode_drying_rate must be in [0, 1], got {self.electrode_drying_rate}"
            )
        if len(self.battery_voltage_range) != 2:
            raise ValueError(
                f"battery_voltage_range must be a 2-tuple, got {len(self.battery_voltage_range)} elements"
            )
        if self.battery_voltage_range[0] < 0 or self.battery_voltage_range[1] < 0:
            raise ValueError("battery_voltage_range values must be non-negative")
        if self.battery_voltage_range[0] > self.battery_voltage_range[1]:
            raise ValueError(
                f"battery_voltage_range[0] ({self.battery_voltage_range[0]}) must be <= "
                f"battery_voltage_range[1] ({self.battery_voltage_range[1]})"
            )
        if not (0 < self.sensor_saturation_threshold <= 1):
            raise ValueError(
                f"sensor_saturation_threshold must be in (0, 1], got {self.sensor_saturation_threshold}"
            )


@dataclass
class ModelConfig:
    """
    Transformer model architecture settings.

    Defines all hyperparameters for the multi-modal transformer encoder,
    attention mechanism, and output heads. Supports variable-length inputs;
    tokenizers adaptively downsample any duration to a consistent token count.
    """

    model_dim: int = 256
    num_attention_heads: int = 8
    num_encoder_layers: int = 6
    feedforward_dim: int = 1024
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    static_embedding_dim: int = 64
    max_positional_encoding: int = 8192
    num_survival_bins: int = 12
    uncertainty_samples: int = 50
    tokens_per_modality: int = 512

    def __post_init__(self) -> None:
        if self.model_dim <= 0:
            raise ValueError(f"model_dim must be positive, got {self.model_dim}")
        if self.num_attention_heads <= 0:
            raise ValueError(
                f"num_attention_heads must be positive, got {self.num_attention_heads}"
            )
        if self.model_dim % self.num_attention_heads != 0:
            raise ValueError(
                f"model_dim ({self.model_dim}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_encoder_layers <= 0:
            raise ValueError(
                f"num_encoder_layers must be positive, got {self.num_encoder_layers}"
            )
        if self.feedforward_dim <= 0:
            raise ValueError(
                f"feedforward_dim must be positive, got {self.feedforward_dim}"
            )
        if not (0 <= self.dropout_rate <= 1):
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {self.dropout_rate}"
            )
        if not (0 <= self.attention_dropout_rate <= 1):
            raise ValueError(
                f"attention_dropout_rate must be in [0, 1], got {self.attention_dropout_rate}"
            )
        if self.static_embedding_dim <= 0:
            raise ValueError(
                f"static_embedding_dim must be positive, got {self.static_embedding_dim}"
            )
        if self.max_positional_encoding <= 0:
            raise ValueError(
                f"max_positional_encoding must be positive, got {self.max_positional_encoding}"
            )
        if self.num_survival_bins <= 0:
            raise ValueError(
                f"num_survival_bins must be positive, got {self.num_survival_bins}"
            )
        if self.uncertainty_samples <= 0:
            raise ValueError(
                f"uncertainty_samples must be positive, got {self.uncertainty_samples}"
            )
        if self.tokens_per_modality <= 0:
            raise ValueError(
                f"tokens_per_modality must be positive, got {self.tokens_per_modality}"
            )


@dataclass
class TrainingConfig:
    """
    Training hyperparameters and optimization settings.

    Controls batch size, learning rate scheduling, loss weights, gradient
    handling, and early stopping behavior.
    """

    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    min_learning_rate: float = 1e-7
    mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 4
    focal_loss_gamma: float = 2.0
    false_negative_weight: float = 10.0
    false_positive_weight: float = 1.0
    survival_loss_weight: float = 0.3
    auxiliary_loss_weight: float = 0.1
    contrastive_loss_weight: float = 0.2
    reconstruction_loss_weight: float = 0.15
    early_stopping_patience: int = 15
    checkpoint_save_best_only: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {self.warmup_steps}"
            )
        if self.min_learning_rate < 0:
            raise ValueError(
                f"min_learning_rate must be non-negative, got {self.min_learning_rate}"
            )
        if self.min_learning_rate >= self.learning_rate:
            raise ValueError(
                f"min_learning_rate ({self.min_learning_rate}) must be < "
                f"learning_rate ({self.learning_rate})"
            )
        if self.gradient_clip_norm <= 0:
            raise ValueError(
                f"gradient_clip_norm must be positive, got {self.gradient_clip_norm}"
            )
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}"
            )
        if self.focal_loss_gamma < 0:
            raise ValueError(
                f"focal_loss_gamma must be non-negative, got {self.focal_loss_gamma}"
            )
        if self.false_negative_weight < 0:
            raise ValueError(
                f"false_negative_weight must be non-negative, got {self.false_negative_weight}"
            )
        if self.false_positive_weight < 0:
            raise ValueError(
                f"false_positive_weight must be non-negative, got {self.false_positive_weight}"
            )
        if not (0 <= self.survival_loss_weight <= 1):
            raise ValueError(
                f"survival_loss_weight must be in [0, 1], got {self.survival_loss_weight}"
            )
        if not (0 <= self.auxiliary_loss_weight <= 1):
            raise ValueError(
                f"auxiliary_loss_weight must be in [0, 1], got {self.auxiliary_loss_weight}"
            )
        if not (0 <= self.contrastive_loss_weight <= 1):
            raise ValueError(
                f"contrastive_loss_weight must be in [0, 1], got {self.contrastive_loss_weight}"
            )
        if not (0 <= self.reconstruction_loss_weight <= 1):
            raise ValueError(
                f"reconstruction_loss_weight must be in [0, 1], got {self.reconstruction_loss_weight}"
            )
        if self.early_stopping_patience <= 0:
            raise ValueError(
                f"early_stopping_patience must be positive, got {self.early_stopping_patience}"
            )


@dataclass
class EvaluationConfig:
    """
    Evaluation and clinical validation settings.

    Defines bootstrap resampling parameters, confidence levels,
    clinical alert thresholds, calibration settings, and OOD detection.
    """

    bootstrap_iterations: int = 1000
    confidence_level: float = 0.95
    clinical_thresholds: List[float] = field(
        default_factory=lambda: [0.10, 0.15, 0.20, 0.25, 0.30]
    )
    primary_threshold: float = 0.20
    calibration_bins: int = 10
    ood_detection_method: str = "mahalanobis"

    def __post_init__(self) -> None:
        if self.bootstrap_iterations <= 0:
            raise ValueError(
                f"bootstrap_iterations must be positive, got {self.bootstrap_iterations}"
            )
        if not (0 < self.confidence_level < 1):
            raise ValueError(
                f"confidence_level must be in (0, 1), got {self.confidence_level}"
            )
        if not self.clinical_thresholds:
            raise ValueError("clinical_thresholds must not be empty")
        for t in self.clinical_thresholds:
            if not (0 <= t <= 1):
                raise ValueError(
                    f"clinical_thresholds must contain values in [0, 1], got {t}"
                )
        if self.primary_threshold not in self.clinical_thresholds:
            raise ValueError(
                f"primary_threshold ({self.primary_threshold}) must be one of "
                f"clinical_thresholds ({self.clinical_thresholds})"
            )
        if self.calibration_bins <= 0:
            raise ValueError(
                f"calibration_bins must be positive, got {self.calibration_bins}"
            )
        valid_methods = {"mahalanobis", "energy", "ensemble"}
        if self.ood_detection_method not in valid_methods:
            raise ValueError(
                f"ood_detection_method must be one of {valid_methods}, "
                f"got {self.ood_detection_method!r}"
            )


@dataclass
class LLMConfig:
    """
    Large Language Model integration settings.

    Configures the LLM endpoint for clinical report generation and
    natural language explanations of model predictions.
    """

    endpoint_url: str = "https://opencode.ai/zen/v1/chat/completions"
    model_id: str = "nemotron-3-ultra-free"
    max_tokens: int = 1024
    temperature: float = 0.3
    timeout_seconds: int = 30
    max_retries: int = 3

    def __post_init__(self) -> None:
        if not self.endpoint_url:
            raise ValueError("endpoint_url must not be empty")
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")
        if not (0 <= self.temperature <= 2):
            raise ValueError(
                f"temperature must be in [0, 2], got {self.temperature}"
            )
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {self.timeout_seconds}"
            )
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got {self.max_retries}"
            )


@dataclass
class OHCAConfig:
    """
    Root configuration for the OHCA Predictor system.

    Aggregates all sub-configurations into a single validated configuration
    object. Supports deep copy and field-level overrides via override_config().
    """

    reproducibility: ReproducibilityConfig = field(
        default_factory=ReproducibilityConfig
    )
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    disease: DiseaseConfig = field(default_factory=DiseaseConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def override(self, **kwargs: Any) -> "OHCAConfig":
        """
        Create a new OHCAConfig with specified field overrides.

        Args:
            **kwargs: Field names and their new values (e.g., batch_size=64
                      will set training.batch_size = 64).

        Returns:
            OHCAConfig: A new configuration instance with overrides applied.
        """
        new_config = copy.deepcopy(self)
        for key, value in kwargs.items():
            parts = key.split(".")
            if len(parts) == 1:
                if not hasattr(new_config, parts[0]):
                    raise ValueError(f"Unknown config field: {parts[0]}")
                setattr(new_config, parts[0], value)
            elif len(parts) == 2:
                config_name, field_name = parts
                if not hasattr(new_config, config_name):
                    raise ValueError(f"Unknown config section: {config_name}")
                sub_config = getattr(new_config, config_name)
                if not hasattr(sub_config, field_name):
                    raise ValueError(
                        f"Unknown field {field_name} in config section {config_name}"
                    )
                setattr(sub_config, field_name, value)
            else:
                raise ValueError(
                    f"Override path must have at most 2 levels (section.field), got {key!r}"
                )
        return new_config
