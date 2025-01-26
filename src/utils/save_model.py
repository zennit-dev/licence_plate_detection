"""Model saving and versioning utilities."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Protocol

import keras

from src.settings import config, logger


class SupportsWrite(Protocol):
    def write(self, __s: str) -> Any: ...


class SaveModel:
    """
    Handles model saving and versioning.
    """

    def save(self, model: keras.Model, metrics: Dict[str, Dict[str, Any]]) -> Path:
        """Save model and version info with metrics in both Keras and TensorFlow formats.

        Args:
            model: The trained Keras model to save
            metrics: Dictionary of model metrics (e.g. training history)

        Returns:
            Path: Path where model was saved

        Raises:
            ValueError: If required metrics are missing
        """
        # Create version-specific directory
        base_dir = Path(config.model.storage.storage_dir)
        version = f"{config.model.storage.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version_dir = base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save in Keras format (with augmentation)
            keras_path = version_dir / "model.keras"
            model.save(keras_path)
            logger.info(f"Keras model saved to: {keras_path}")

            # Save metadata
            version_info = {
                "version": version,
                "path": str(version_dir),
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics,
                "keras_model_path": str(keras_path),
                "production_ready": self._is_production_ready(
                    metrics.get("evaluation", {}).get("metrics", {})
                ),
            }

            metadata_file = version_dir / "metadata.json"
            f: SupportsWrite
            with open(metadata_file, "w") as f:
                json.dump(version_info, f, indent=2)
            logger.info(f"Model metadata saved to: {metadata_file}")

            return version_dir
        except Exception as e:
            raise ValueError(f"Failed to save model: {str(e)}")

    @staticmethod
    def get_latest_version() -> Dict[str, Any]:
        """Get latest version info.

        Returns:
            Dictionary containing version information or empty dict if no version exists
        """
        base_dir = Path(config.model.storage.storage_dir)
        if not base_dir.exists():
            return {}

        # Find the latest version directory
        version_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()], reverse=True)
        if not version_dirs:
            return {}

        latest_metadata = version_dirs[0] / "metadata.json"
        if not latest_metadata.exists():
            return {}

        try:
            with open(latest_metadata) as f:
                return dict(json.load(f))
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _is_production_ready(metrics: Dict[str, float]) -> bool:
        """Check if model meets production requirements.

        Args:
            metrics: Dictionary of model metrics

        Returns:
            True if model meets all requirements
        """
        # Check accuracy threshold
        accuracy = metrics.get("accuracy", 0)
        min_accuracy = config.training.metrics.min_accuracy
        if accuracy < min_accuracy:
            return False

        # Check inference time if specified
        max_inference_time = config.training.prediction.timeout_ms
        if "inference_time" in metrics and metrics["inference_time"] > max_inference_time:
            return False

        # Verify all required metrics are above minimum accuracy
        required_metrics = (
            config.model.storage.required_metrics
            if hasattr(config.model.storage, "required_metrics")
            else []
        )
        for metric in required_metrics:
            metric_value = metrics.get(metric, 0)
            if metric_value < min_accuracy:
                return False

        return True
