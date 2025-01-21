from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Training configuration settings."""

    hidden_layers: list[int] = Field(
        [64, 32], title="Hidden layers", description="Number of neurons in hidden layers"
    )
    activation: str = Field(
        "relu", title="Activation function", description="Activation function for hidden layers"
    )
    optimizer: str = Field("adam", title="Optimizer", description="Optimizer for training")
    dropout_rate: float = Field(
        0.2, title="Dropout rate", description="Dropout rate for regularization"
    )
    input_shape: list[int] = Field(
        [224, 224, 3], title="Input shape", description="Input shape for the model"
    )
    output_classes: int = Field(2, title="Output classes", description="Number of output classes")


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
    early_stopping: bool = Field(True, title="Early stopping", description="Enable early stopping")


class MetricsConfig(BaseModel):
    """Metrics configuration settings."""

    min_accuracy: float = Field(
        0.8, title="Minimum accuracy", description="Minimum accuracy threshold"
    )
    min_precision: float = Field(
        0.8, title="Minimum precision", description="Minimum precision threshold"
    )
    min_recall: float = Field(0.8, title="Minimum recall", description="Minimum recall threshold")
    min_f1: float = Field(0.8, title="Minimum F1 score", description="Minimum F1 score threshold")
