from typing import Dict, List, Tuple

from pydantic import BaseModel, Field


class StorageConfig(BaseModel):
    """Base paths configuration."""

    storage_dir: str = Field(
        default="storage", title="Storage directory", description="Storage directory for data"
    )
    cache_dir: str = Field(
        default="cache", title="Cache directory", description="Cache directory for data"
    )
    format: str = Field(
        default="saved_model", title="Model format", description="Model format for saving"
    )
    version: str = Field(
        default="v1", title="Model version", description="Model version for saving"
    )


class ArchitectureConfig(BaseModel):
    """Model architecture configuration settings."""

    optimizer: str = Field(default="adam", title="Optimizer", description="Optimizer to use")
    loss: str = Field(default="categorical_crossentropy", title="Loss", description="Loss function")
    metrics: List[str] = Field(
        default=["accuracy"], title="Metrics", description="Metrics to evaluate model performance"
    )


class HyperparametersConfig(BaseModel):
    """Model hyperparameters configuration settings."""

    hidden_layers: tuple[int, int, int] = Field(
        default=(512, 256, 128), title="Hidden layers", description="Number of neurons per layer"
    )
    activation: str = Field(default="relu", title="Activation", description="Activation function")
    dropout_rate: float = Field(
        default=0.3, title="Dropout", description="Dropout rate for regularization"
    )
    input_shape: tuple[int, int, int] = Field(
        default=(32, 32, 3), title="Input shape", description="Input shape for model"
    )


class LayerConfig(BaseModel):
    """Layer configuration settings."""

    class ConvBlockConfig(BaseModel):
        """Convolutional block configuration settings."""

        filters: List[int] = Field(
            default=[64, 64],
            title="Filters",
            description="List of filter sizes for each conv layer in block",
        )
        kernel_size: Tuple[int, int] = Field(
            default=(3, 3), title="Kernel size", description="Size of the convolutional kernel"
        )
        pool_size: Tuple[int, int] = Field(
            default=(2, 2), title="Pool size", description="Size of the pooling window"
        )
        dropout_rate: float = Field(
            default=0.2, title="Dropout", description="Dropout rate for this block"
        )

    class DenseLayerConfig(BaseModel):
        """Dense layer configuration settings."""

        units: int = Field(
            default=512, title="Units", description="Number of neurons in dense layer"
        )
        dropout_rate: float = Field(
            default=0.3, title="Dropout", description="Dropout rate for regularization"
        )

    conv_blocks: Dict[str, ConvBlockConfig] = Field(
        default_factory=dict,
        title="Convolutional blocks",
        description="Dictionary of convolutional block configurations",
    )

    dense_layers: Dict[str, DenseLayerConfig] = Field(
        default_factory=dict,
        title="Dense layers",
        description="Dictionary of dense layer configurations",
    )
