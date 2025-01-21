from pydantic import BaseModel, Field


class PredictionConfig(BaseModel):
    """Model configuration settings."""

    batch_processing: bool = Field(default=True)
    cache_predictions: bool = Field(default=False)
    timeout_ms: int = Field(default=5000)
    return_probabilities: bool = Field(default=True)
    enable_preprocessing: bool = Field(default=True)
    enable_postprocessing: bool = Field(default=True)
    log_predictions: bool = Field(default=False)


class StorageConfig(BaseModel):
    """Storage configuration settings."""

    format: str = Field(default="saved_model")
    version: str = Field(default="v1")
