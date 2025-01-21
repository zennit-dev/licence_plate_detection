from pydantic import BaseModel

from .environment import MonitoringConfig, PathConfig
from .model import PredictionConfig, StorageConfig
from .training import MetricsConfig, ModelConfig, TrainingConfig


class EnvironmentConfig(BaseModel):
    """Combined environment configuration."""

    path: PathConfig = PathConfig()
    monitoring: MonitoringConfig = MonitoringConfig()


class ModelSettingsConfig(BaseModel):
    """Combined model settings configuration."""

    prediction: PredictionConfig = PredictionConfig()
    storage: StorageConfig = StorageConfig()


class TrainingSettingsConfig(BaseModel):
    """Combined training settings configuration."""

    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    metrics: MetricsConfig = MetricsConfig()


__all__ = ["EnvironmentConfig", "ModelSettingsConfig", "TrainingSettingsConfig"]
