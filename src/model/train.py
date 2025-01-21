from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import keras
import numpy as np
import numpy.typing as npt
from keras import Input
from keras.src.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

from src.settings import config, logger
from src.utils.preprocessor_factory import PreprocessorFactory


class TrainModel:
    """Class to handle data of a Keras model."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int,
        preprocessor_type: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        data_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the data configuration.

        Args:
            input_shape: Shape of input data (excluding batch dimension)
            num_classes: Number of output classes
            preprocessor_type: Type of preprocessor to use ('image', 'text', or 'tabular')
            model_config: Configuration for model architecture (layers, activations, etc.)
            data_config: Configuration for data loading (normalization, preprocessing, etc.)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_config = model_config or {}
        self.data_config = data_config or {
            "normalization_factor": 1.0,
            "data_format": "npz",
            "feature_key": "features",
            "label_key": "labels",
        }

        # Initialize preprocessor if specified
        self.preprocessor = None
        if preprocessor_type:
            self.preprocessor = PreprocessorFactory.get_preprocessor(preprocessor_type)

    @staticmethod
    def __load_data(data_path: Path) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Load preprocessed data from npz files."""
        logger.info(f"Loading data from {data_path}")

        data = np.load(data_path / "data.npz")
        features, labels = data["features"], data["labels"]

        # Ensure features are float32 and in range [0, 255]
        features = features.astype(np.float32)
        if features.max() <= 1.0:
            features *= 255.0

        # Ensure labels are properly formatted integers
        labels = labels.astype(np.int32)

        return features, labels

    def __create_model(self) -> keras.Model:
        """Create and compile the model based on configuration."""
        model = keras.Sequential(
            [
                # Input and preprocessing
                Input(shape=self.input_shape),
                keras.layers.Rescaling(1.0 / 255),
                # Data augmentation
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.1),
                # First convolutional block
                Conv2D(64, (3, 3), activation="relu", padding="same"),
                Conv2D(64, (3, 3), activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.2),
                # Second convolutional block
                Conv2D(128, (3, 3), activation="relu", padding="same"),
                Conv2D(128, (3, 3), activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.2),
                # Third convolutional block
                Conv2D(256, (3, 3), activation="relu", padding="same"),
                Conv2D(256, (3, 3), activation="relu"),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.2),
                # Dense layers
                Flatten(),
                Dense(512, activation="relu"),
                Dropout(0.3),
                Dense(self.num_classes, activation="softmax"),
            ]
        )

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def train(
        self,
        train_data: Path,
        evaluate_data: Path,
        save_path: Optional[Path] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> Path:
        """Train a model on the provided data, evaluate it, and save results."""
        try:
            # Load and validate data
            if not train_data.exists():
                raise ValueError(f"Invalid data directory: {train_data}")

            x_train, y_train = self.__load_data(train_data)
            x_val, y_val = self.__load_data(evaluate_data)

            # Print data statistics
            logger.info(f"Training data shape: {x_train.shape}")
            logger.info(f"Training labels shape: {y_train.shape}")
            logger.info(f"Number of classes: {len(np.unique(y_train))}")

            # Create and train model
            model = self.__create_model()
            logger.info("Starting model training...")

            model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=epochs or 50,
                batch_size=batch_size or 32,
                verbose=1,
            )

            # Save the model
            logger.info("Saving model...")
            save_dir = Path(save_path) if save_path else Path(config.environment.path.model_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            model_path = save_dir / "model.keras"
            model.save(model_path)

            return model_path

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}")
