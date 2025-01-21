"""Module for managing model operations including data and prediction."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import keras
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from src.settings import config, logger
from src.utils import PreprocessorFactory

# Type aliases for cleaner type hints
NDArray = npt.NDArray[np.float32]
PredictionResults = Dict[str, Dict[str, Union[int, float, str, list[float]]]]


class PredictModel:
    """Handles model operations including data, saving, loading and prediction."""

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        """Initialize model manager.

        Args:
            model_dir: Optional directory containing saved models. If None, uses config default.
        """
        self.model_dir = (
            Path(config.environment.path.model_dir) if model_dir is None else Path(model_dir)
        )
        self.keras_model: Optional[tf.keras.Model] = None  # noqa
        self.tf_model: Optional[tf.keras.Model] = None  # noqa
        self._prediction_cache: Dict[str, PredictionResults] = {}
        self._class_mapping: Optional[Dict[str, int]] = None
        self._idx_to_class: Optional[Dict[int, str]] = None

    def _load_class_mapping(self) -> None:
        """Load class mapping from the processed data directory."""
        if self._class_mapping is None:
            mapping_file = self.model_dir / "class_mapping.json"
            if not mapping_file.exists():
                raise FileNotFoundError(f"Class mapping file not found: {mapping_file}")

            with open(mapping_file) as f:
                self._class_mapping = json.load(f)
                # Create reverse mapping (index to class name)
                self._idx_to_class = {v: k for k, v in self._class_mapping.items()}
                self._idx_to_class = {v: k for k, v in self._class_mapping.items()}

    def _evaluate(
        self,
        start_time: float,
        input_data: Union[NDArray, tf.Tensor],
        extra_kwargs: Dict[str, Any],
        results: PredictionResults,
    ) -> PredictionResults:
        """Evaluate the input data using both models.

        Args:
            start_time: Start time for timeout checking
            input_data: Preprocessed input data
            extra_kwargs: Additional arguments for model evaluation
            results: Dictionary to store results

        Returns:
            Updated results dictionary

        Raises:
            TimeoutError: If evaluation time exceeds timeout
            ValueError: If models are not loaded
        """
        timeout = config.model.prediction.timeout_ms / 1000
        if time.time() - start_time > timeout:
            raise TimeoutError("Prediction timeout exceeded")

        if self.keras_model is None:
            raise ValueError("Keras model not loaded")

        # Load class mapping if not already loaded
        if self._idx_to_class is None:
            self._load_class_mapping()

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
        keras_prediction = self.keras_model.predict(model_input)
        keras_digit = int(tf.argmax(keras_prediction[0]).numpy())
        results["keras"] = {
            "class": self._idx_to_class[keras_digit],
            "confidence": float(keras_prediction[0][keras_digit]),
        }

        if config.model.prediction.return_probabilities:
            results["keras"]["probabilities"] = keras_prediction[0].tolist()

        # TensorFlow prediction
        if time.time() - start_time > timeout:
            raise TimeoutError("Prediction timeout exceeded")

        if self.tf_model is None:
            raise ValueError("TensorFlow model not loaded")

        infer = self.tf_model.signatures["serving_default"]
        input_name = list(infer.structured_input_signature[1].keys())[0]
        tf_prediction = infer(**{input_name: model_input})
        tf_probs = tf_prediction["output_0"].numpy()[0]
        tf_digit = int(tf.argmax(tf_probs).numpy())

        results["tensorflow"] = {
            "class": self._idx_to_class[tf_digit],
            "confidence": float(tf_probs[tf_digit]),
        }

        if config.model.prediction.return_probabilities:
            results["tensorflow"]["probabilities"] = tf_probs.tolist()

        return results

    @staticmethod
    def _post_processing(results: PredictionResults) -> PredictionResults:
        """Post-process prediction results.

        Args:
            results: Raw prediction results

        Returns:
            Post-processed results
        """
        if results["keras"]["class"] == results["tensorflow"]["class"]:
            keras_conf = results["keras"].get("confidence", 0.0)
            tf_conf = results["tensorflow"].get("confidence", 0.0)
            if not isinstance(keras_conf, (int, float)) or not isinstance(tf_conf, (int, float)):
                raise ValueError("Confidence values must be numeric")

            results["ensemble"] = {
                "class": results["keras"]["class"],
                "confidence": (float(keras_conf) + float(tf_conf)) / 2,
            }

        return results

    def _load_latest_model(self) -> Tuple[tf.keras.Model, tf.keras.Model]:  # noqa
        """Load the latest version of both models.

        Returns:
            Tuple of (keras_model, tensorflow_model)

        Raises:
            FileNotFoundError: If no models are found
        """
        model_dir = Path(config.environment.path.model_dir)

        version_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        if not version_dirs:
            raise FileNotFoundError(f"No model versions found in {model_dir}")

        # Sort by timestamp to get latest timestamp for each version
        version_dirs = sorted(
            version_dirs, key=lambda d: d.name.split("_")[1] if "_" in d.name else "", reverse=True
        )

        # Then group by version and get the latest timestamp for each version
        version_groups: Dict[str, Path] = {}
        for dir_path in version_dirs:
            version = dir_path.name.split("_")[0] if "_" in dir_path.name else dir_path.name
            if version not in version_groups:
                version_groups[version] = dir_path

        # Try to find the current version first
        current_version = config.model.storage.version
        if current_version in version_groups:
            latest_dir = version_groups[current_version]
        else:
            # If no matching version found, use the latest version available
            latest_version = sorted(version_groups.keys(), reverse=True)[0]
            latest_dir = version_groups[latest_version]
            logger.warning(
                f"No models found for version {current_version}. "
                f"Using latest available version: {latest_dir.name}"
            )

        keras_path = latest_dir / "model.keras"
        keras_model = keras.models.load_model(keras_path)

        tf_path = latest_dir / "tensorflow_model"
        tf_model = tf.saved_model.load(str(tf_path))

        return keras_model, tf_model

    @staticmethod
    def _cache_predictions(input_data: Union[NDArray, tf.Tensor]) -> str:
        """Cache predictions to avoid redundant computation.

        Args:
            input_data: Input data used for prediction

        Returns:
            Cache key for the input data
        """
        if isinstance(input_data, tf.Tensor):
            tensor_bytes = tf.io.serialize_tensor(input_data).numpy()
            cache_key = str(hash(tensor_bytes.tobytes()))
        else:
            cache_key = str(hash(input_data.tobytes()))

        return cache_key

    def predict(
        self,
        input_data: Union[str, Path, NDArray, tf.Tensor],
        preprocessor_type: Optional[str] = None,
        preprocess_kwargs: Optional[Dict[str, Any]] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> PredictionResults:
        """Make predictions using both Keras and TensorFlow models.

        Args:
            input_data: Raw input data (file path, text, or preprocessed array)
            preprocessor_type: Type of preprocessor to use (if input needs preprocessing)
            preprocess_kwargs: Optional preprocessing parameters
            extra_kwargs: Additional arguments for model evaluation

        Returns:
            Dictionary containing predictions from both models

        Raises:
            TimeoutError: If prediction time exceeds timeout
            ValueError: If models are not loaded or input is invalid
        """
        processed_input: Union[NDArray, tf.Tensor]

        # Handle preprocessing if needed
        if preprocessor_type and isinstance(input_data, (str, Path)):
            preprocessor = PreprocessorFactory.get_preprocessor(preprocessor_type)
            processed_input = preprocessor.preprocess(input_data, **(preprocess_kwargs or {}))
        elif isinstance(input_data, (np.ndarray, tf.Tensor)):
            processed_input = input_data
        else:
            raise ValueError(
                "Input must be either a file path with preprocessor_type specified, "
                "or a preprocessed numpy array/tensor"
            )

        # Generate cache key for preprocessed data
        cache_key = (
            self._cache_predictions(processed_input)
            if config.model.prediction.cache_predictions
            else None
        )
        if cache_key and cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]

        if self.keras_model is None or self.tf_model is None:
            self.keras_model, self.tf_model = self._load_latest_model()

        # Start timing for timeout check
        start_time = time.time()

        results: PredictionResults = {}
        try:
            results = self._evaluate(start_time, processed_input, extra_kwargs or {}, results)
        except TimeoutError as e:
            raise TimeoutError(
                f"Prediction timeout exceeded: {config.model.prediction.timeout_ms}ms"
            ) from e

        if config.model.prediction.enable_postprocessing:
            results = self._post_processing(results)

        if cache_key:
            self._prediction_cache[cache_key] = results

        return results
