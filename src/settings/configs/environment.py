from pydantic import BaseModel, Field


class PathConfig(BaseModel):
    """System configuration settings."""

    base_dir: str = Field(
        default=".", title="Base directory", description="Base directory for project"
    )
    temp_dir: str = Field(
        default="tmp", title="Temporary directory", description="Temporary directory for data"
    )


class LoggingConfig(BaseModel):
    """Logging configuration settings."""

    logger_name: str = Field(default="app", title="Logger name", description="Logger name for data")
    log_level: str = Field(default="INFO", title="Log level", description="Log level for data")
    console_logging: bool = Field(
        default=True, title="Console logging", description="Console logging for data"
    )
    file_logging: bool = Field(
        default=True, title="File logging", description="File logging for data"
    )
