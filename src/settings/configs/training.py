from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Training configuration settings."""

    batch_size: int = Field(32, title="Batch size", description="Number of samples per batch")
    epochs: int = Field(10, title="Epochs", description="Number of training epochs")
    learning_rate: float = Field(
        0.001, title="Learning rate", description="Learning rate for optimizer"
    )
    validation_split: float = Field(
        0.2, title="Validation split", description="Validation split ratio"
    )
    early_stopping_patience: int = Field(
        10, title="Early stopping patience", description="Early stopping patience"
    )
    verbose: int = Field(1, title="Verbose", description="Verbosity mode (0, 1, or 2)")


class MetricsConfig(BaseModel):
    """Metrics configuration settings."""

    min_accuracy: float = Field(
        0.8, title="Minimum accuracy", description="Minimum accuracy threshold"
    )
    min_precision: float = Field(
        0.8, title="Minimum precision", description="Minimum precision threshold"
    )
    min_recall: float = Field(0.8, title="Minimum recall", description="Minimum recall threshold")
    min_f1_score: float = Field(
        0.8, title="Minimum F1 score", description="Minimum F1 score threshold"
    )


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration settings."""

    enabled: bool = Field(True, title="Enabled", description="Enable early stopping")
    monitor: str = Field(
        "val_loss", title="Monitor", description="Quantity to be monitored for early stopping"
    )
    patience: int = Field(
        10, title="Patience", description="Number of epochs with no improvement after stopping"
    )
    verbose: int = Field(0, title="Verbose", description="Verbosity mode (0, 1, or 2)")
    restore_best_weights: bool = Field(
        True, title="Restore best weights", description="Restore best weights after stopping"
    )


class PredictionConfig(BaseModel):
    """Prediction configuration settings."""

    batch_processing: bool = Field(
        False, title="Batch processing", description="Enable batch processing for predictions"
    )
    timeout_ms: int = Field(
        5000, title="Timeout", description="Timeout in milliseconds for prediction requests"
    )
    return_probabilities: bool = Field(
        False, title="Return probabilities", description="Return class probabilities"
    )
    enable_preprocessing: bool = Field(
        False, title="Enable preprocessing", description="Enable data preprocessing"
    )
    enable_postprocessing: bool = Field(
        False, title="Enable postprocessing", description="Enable data postprocessing"
    )
    log_predictions: bool = Field(
        False, title="Log predictions", description="Log predictions to file"
    )


class DataAugmentationConfig(BaseModel):
    """Data augmentation configuration settings."""

    horizontal_flip: bool = Field(
        True, title="Horizontal flip", description="Randomly flip images horizontally"
    )
    vertical_flip: bool = Field(
        False, title="Vertical flip", description="Randomly flip images vertically"
    )
    rotation_range: float = Field(
        0.0, title="Rotation range", description="Randomly rotate images within a range"
    )


class DataConfig(BaseModel):
    """Data configuration settings."""

    data_dir: str = Field(
        default="data", title="Data directory", description="Data directory for training"
    )
    log_dir: str = Field(
        default="logs", title="Log directory", description="Log directory for logging"
    )
    preprocessing_workers: int = Field(
        default=4, title="Preprocessing workers", description="Number of workers for preprocessing"
    )
