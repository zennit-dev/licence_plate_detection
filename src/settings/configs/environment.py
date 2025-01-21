from pydantic import BaseModel, Field


class PathConfig(BaseModel):
    """Base paths configuration."""

    base_dir: str = Field(default=".")
    data_dir: str = Field(default="data")
    model_dir: str = Field(default="models")
    log_dir: str = Field(default="logs")
    cache_dir: str = Field(default=".cache")
    temp_dir: str = Field(default="/tmp/ml-tensorflow")


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""

    logger_name: str = Field(default="ml-tensorflow")
    log_level: str = Field(default="INFO")
    enable_file_logging: bool = Field(default=True)
    enable_mlflow: bool = Field(default=False)
    enable_wandb: bool = Field(default=False)
