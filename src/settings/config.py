"""Configuration management system with environment variable support."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.settings.configs import EnvironmentConfig, ModelSettingsConfig, TrainingSettingsConfig


def _load_all_configs() -> Dict[str, Any]:
    """Load all configuration files."""
    config_dir = Path("configs")
    config_files = {
        "model": config_dir / "model.yml",
        "training": config_dir / "training.yml",
        "environment": config_dir / "environment.yml",
    }

    cfg = {}
    for name, path in config_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            cfg[name] = yaml.safe_load(f) or {}

    return cfg


class Config:
    """Main configuration class with Laravel-like interface."""

    def __init__(self) -> None:
        self._config = _load_all_configs()

        # Initialize configurations with their respective sections
        self.environment = EnvironmentConfig(
            path=self._get_config_with_env("environment", "path"),
            monitoring=self._get_config_with_env("environment", "monitoring"),
        )

        self.model = ModelSettingsConfig(
            prediction=self._get_config_with_env("model", "prediction"),
            storage=self._get_config_with_env("model", "storage"),
        )

        self.training = TrainingSettingsConfig(
            model=self._get_config_with_env("training", "model"),
            training=self._get_config_with_env("training", "training"),
            metrics=self._get_config_with_env("training", "metrics"),
        )

    def _get_config_with_env(self, config_file: str, section: str) -> dict[str, Any]:
        """Get configuration with environment variable overrides."""
        cfg = dict(self._config.get(config_file, {}).get(section, {}))

        # Override with environment variables
        env_prefix = "ML_"
        for key in cfg.keys():
            env_key = f"{env_prefix}{key.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                # Convert value to appropriate type
                if isinstance(cfg[key], bool):
                    cfg[key] = value.lower() in ("true", "1", "yes")
                elif isinstance(cfg[key], int):
                    cfg[key] = int(value)
                elif isinstance(cfg[key], float):
                    cfg[key] = float(value)
                else:
                    cfg[key] = value

        return cfg

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get configuration value using dot notation."""
        try:
            config_file, section, *parts = key.split(".")
            value = self._config[config_file][section]
            for part in parts:
                value = value[part]
            return str(value)
        except (KeyError, TypeError):
            return default


@lru_cache()
def config() -> Config:
    """Get the global configuration instance."""
    return Config()
