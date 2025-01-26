"""Logging configuration for the ML project."""

import logging
import sys
from pathlib import Path

from src.settings.config import config as _load_config

config = _load_config()

# Define custom log level
SUCCESS = 25  # Between INFO(20) and WARNING(30)
logging.addLevelName(SUCCESS, "SUCCESS")

# ANSI color codes
COLOR_INFO = "\033[94m"
COLOR_SUCCESS = "\033[92m"
COLOR_WARNING = "\033[93m"
COLOR_ERROR = "\033[91m"
COLOR_CRITICAL = "\033[41m"
COLOR_RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    """Custom formatter adding colors to log levels."""

    COLORS = {
        "DEBUG": "",
        "INFO": COLOR_INFO,
        "SUCCESS": COLOR_SUCCESS,
        "WARNING": COLOR_WARNING,
        "ERROR": COLOR_ERROR,
        "CRITICAL": COLOR_CRITICAL,
    }

    def __init__(self, fmt: str, use_color: bool = True) -> None:
        """Initialize formatter with color option."""
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        # Format the record first
        formatted = super().format(record)

        # Add color to the entire message if enabled
        if self.use_color and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            formatted = f"{color}{formatted}{COLOR_RESET}"

        return formatted


def setup_logging() -> bool:
    """
    Configure logging for both file and console output.

    Returns:
        bool: True if logging was configured successfully, False otherwise
    """
    try:
        # Create logs directory if it doesn't exist
        log_dir = Path(config.training.data.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create formatters
        console_formatter = ColorFormatter(
            "%(asctime)s - %(levelname)s - %(message)s", use_color=True
        )
        file_formatter = ColorFormatter(
            "%(asctime)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s", use_color=False
        )

        # Configure the app logger first
        app_logger = logging.getLogger(config.environment.logging.logger_name)

        # Set the base logging level based on config
        log_level = getattr(logging, config.environment.logging.log_level.upper(), logging.INFO)
        app_logger.setLevel(log_level)

        # Remove any existing handlers to avoid duplicates
        app_logger.handlers.clear()

        # Console handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        app_logger.addHandler(console_handler)

        # Add file handler if enabled
        if config.environment.logging.file_logging:
            file_handler = logging.FileHandler(log_dir / "app.log", mode="a")
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(log_level)
            app_logger.addHandler(file_handler)

        # Prevent propagation to root logger
        app_logger.propagate = False
        return True
    except Exception as e:
        print(f"Failed to configure logging: {e}")
        return False


class Logging:
    """Logging class with Laravel-like interface."""

    def __init__(self) -> None:
        """Initialize the Logging class."""
        setup_logging()

    @staticmethod
    def info(message: str) -> None:
        """Log an informational message."""
        logger = logging.getLogger(config.environment.logging.logger_name)
        logger.info(message)

    @staticmethod
    def success(message: str) -> None:
        """Log a success message."""
        logger = logging.getLogger(config.environment.logging.logger_name)
        logger.log(SUCCESS, message)

    @staticmethod
    def warning(message: str) -> None:
        """Log a warning message."""
        logger = logging.getLogger(config.environment.logging.logger_name)
        logger.warning(message)

    @staticmethod
    def error(message: str) -> None:
        """Log an error message."""
        logger = logging.getLogger(config.environment.logging.logger_name)
        logger.error(message)

    @staticmethod
    def debug(message: str) -> None:
        """Log a debug message."""
        logger = logging.getLogger(config.environment.logging.logger_name)
        logger.debug(message)

    @staticmethod
    def critical(message: str) -> None:
        """Log a critical message."""
        logger = logging.getLogger(config.environment.logging.logger_name)
        logger.critical(message)
