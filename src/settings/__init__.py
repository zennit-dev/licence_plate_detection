"""Settings package initialization."""

from .config import config as _load_config
from .logger import Logging

config = _load_config()
logger = Logging()

__all__ = ["config", "logger"]
