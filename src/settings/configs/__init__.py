from pydantic import BaseModel

from .environment import LoggingConfig, PathConfig
from .model import ArchitectureConfig, HyperparametersConfig, LayerConfig, StorageConfig
from .optimizers import AdamConfig, RMSPropConfig, SDGConfig
from .training import (
    DataAugmentationConfig,
    DataConfig,
    EarlyStoppingConfig,
    MetricsConfig,
    PredictionConfig,
    TrainingConfig,
)


class EnvironmentConfig(BaseModel):
    """Combined environment configuration."""

    path: PathConfig = PathConfig()
    logging: LoggingConfig = LoggingConfig()


class ModelSettingsConfig(BaseModel):
    """Combined model settings configuration."""

    storage: StorageConfig = StorageConfig()
    architecture: ArchitectureConfig = ArchitectureConfig()
    hyperparameters: HyperparametersConfig = HyperparametersConfig()
    layer: LayerConfig = LayerConfig()


class TrainingSettingsConfig(BaseModel):
    """Combined training settings configuration."""

    training: TrainingConfig = TrainingConfig()
    metrics: MetricsConfig = MetricsConfig()
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    data_augmentation: DataAugmentationConfig = DataAugmentationConfig()
    data: DataConfig = DataConfig()
    prediction: PredictionConfig = PredictionConfig()


class OptimizerSettingsConfig(BaseModel):
    """Combined optimizer settings configuration."""

    adam: AdamConfig = AdamConfig()
    sgd: SDGConfig = SDGConfig()
    rmsprop: RMSPropConfig = RMSPropConfig()


__all__ = [
    "EnvironmentConfig",
    "ModelSettingsConfig",
    "TrainingSettingsConfig",
    "OptimizerSettingsConfig",
]
