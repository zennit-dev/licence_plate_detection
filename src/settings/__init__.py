"""Settings package initialization."""

from src.settings.config import config as _load_config
from src.settings.logger import Logging

config = _load_config()
logger = Logging()

__all__ = ["config", "logger"]
