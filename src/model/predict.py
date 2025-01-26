"""Module for managing model operations including data and prediction."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import keras
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from src.settings import config
from src.utils import PreprocessorFactory

# Type aliases for cleaner type hints
NDArray = npt.NDArray[np.float32]
PredictionResults = Dict[str, Dict[str, Union[int, float, str, list[float]]]]


class PredictModel:
    """Handles model operations including data, saving, loading and prediction."""

    def __init__(self) -> None:
        """Initialize model manager."""
        self.keras_model: Optional[keras.Model] = None
        self._class_mapping: Optional[Dict[str, int]] = None
        self._idx_to_class: Optional[Dict[int, str]] = None

    def __load_class_mapping(self) -> None:
        """Load class mapping from the processed data directory."""
        latest_dir = self.__load_latest_version()

        if self._class_mapping is None:
            mapping_file = latest_dir / "metadata.json"
            if not mapping_file.exists():
                raise FileNotFoundError(f"Class mapping file not found: {mapping_file}")

            with open(mapping_file) as f:
                self._class_mapping = json.load(f)
                # Create reverse mapping (index to class name)
                self._idx_to_class = {v: k for k, v in self._class_mapping.items()}
                self._idx_to_class = {v: k for k, v in self._class_mapping.items()}

    def __evaluate(
        self,
        start_time: float,
        input_data: Union[NDArray, tf.Tensor],
        extra_kwargs: Dict[str, Any],
        results: PredictionResults,
    ) -> PredictionResults:
        """Evaluate the input data using the Keras model.

        Args:
            start_time: Start time for timeout checking
            input_data: Preprocessed input data
            extra_kwargs: Additional arguments for model evaluation
            results: Dictionary to store results

        Returns:
            Updated results dictionary

        Raises:
            TimeoutError: If evaluation time exceeds timeout
            ValueError: If model is not loaded
        """
        timeout = config.training.prediction.timeout_ms / 1000
        if time.time() - start_time > timeout:
            raise TimeoutError("Prediction timeout exceeded")

        if self.keras_model is None:
            raise ValueError("Keras model not loaded")

        # Load class mapping if not already loaded
        if self._idx_to_class is None:
            self.__load_class_mapping()

        # After loading, we can assert the mapping exists
        assert self._idx_to_class is not None, "Class mapping not loaded"

        # Apply any model-specific preprocessing from extra_kwargs
        model_input = input_data
        if extra_kwargs.get("add_batch_dim", False):
            # Ensure input has shape (height, width, channels) before adding batch dimension
            if len(model_input.shape) == 2:
                if isinstance(model_input, np.ndarray):
                    model_input = np.expand_dims(model_input, axis=-1)
                else:
                    model_input = tf.expand_dims(model_input, axis=-1)

            # Add batch dimension if not present
            if len(model_input.shape) == 3:
                if isinstance(model_input, np.ndarray):
                    model_input = np.expand_dims(model_input, axis=0)
                else:
                    model_input = tf.expand_dims(model_input, axis=0)

        # Keras prediction
        prediction = self.keras_model.predict(model_input)
        digit = int(tf.argmax(prediction[0]).numpy())
        results["prediction"] = {
            "class": self._idx_to_class[digit],
            "confidence": float(prediction[0][digit]),
        }

        if config.training.prediction.return_probabilities:
            results["prediction"]["probabilities"] = prediction[0].tolist()

        return results

    @staticmethod
    def __post_processing(results: PredictionResults) -> PredictionResults:
        """Post-process prediction results.

        Args:
            results: Raw prediction results

        Returns:
            Post-processed results
        """
        # No ensemble needed since we only have one model
        return results

    def __load_latest_model(self) -> keras.Model:
        """Load the latest version of both models.

        Returns:
            Tuple of (keras_model, tensorflow_model)

        Raises:
            FileNotFoundError: If no models are found
        """
        latest_dir = self.__load_latest_version()
        return keras.models.load_model(latest_dir / "model.keras")

    @staticmethod
    def __load_latest_version() -> Path:
        model_dir = Path(config.model.storage.storage_dir)
        version_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        if not version_dirs:
            raise FileNotFoundError(f"No model versions found in {model_dir}")

        # Sort by timestamp to get latest timestamp for each version
        version_dirs = sorted(
            version_dirs, key=lambda d: d.name.split("_")[1] if "_" in d.name else "", reverse=True
        )

        version_groups: Dict[str, Path] = {}
        for dir_path in version_dirs:
            version = dir_path.name.split("_")[0] if "_" in dir_path.name else dir_path.name
            if version not in version_groups:
                version_groups[version] = dir_path

        current_version = config.model.storage.version
        if current_version in version_groups:
            return version_groups[current_version]
        else:
            latest_version = sorted(version_groups.keys(), reverse=True)[0]
            return version_groups[latest_version]

    def predict(
        self,
        input_data: Union[str, Path, NDArray, tf.Tensor],
        preprocessor_type: str,
        preprocess_kwargs: Optional[Dict[str, Any]] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> PredictionResults:
        """Make predictions using the Keras model.

        Args:
            input_data: Raw input data (file path, text, or preprocessed array)
            preprocessor_type: Type of preprocessor to use (if input needs preprocessing)
            preprocess_kwargs: Optional preprocessing parameters
            extra_kwargs: Additional arguments for model evaluation

        Returns:
            Dictionary containing predictions from the Keras model

        Raises:
            TimeoutError: If prediction time exceeds timeout
            ValueError: If model is not loaded or input is invalid
        """

        preprocess_kwargs = preprocess_kwargs or {}
        extra_kwargs = extra_kwargs or {}
        processed_input: Union[str, Path, NDArray, tf.Tensor]

        if isinstance(input_data, (str, Path)):
            preprocessor = PreprocessorFactory.get_preprocessor(preprocessor_type)
            processed_input = preprocessor.preprocess(input_data, **preprocess_kwargs)
        elif isinstance(input_data, (np.ndarray, tf.Tensor)):
            processed_input = input_data
        else:
            raise ValueError("Invalid input type")

        if self.keras_model is None:
            self.keras_model = self.__load_latest_model()

        # Start timing for timeout check
        start_time = time.time()

        results: PredictionResults = {}
        try:
            results = self.__evaluate(start_time, processed_input, extra_kwargs or {}, results)
        except TimeoutError as e:
            raise TimeoutError(
                f"Prediction timeout exceeded: {config.training.prediction.timeout_ms}ms"
            ) from e

        if config.training.prediction.enable_postprocessing:
            results = self.__post_processing(results)

        return results
