from pathlib import Path
from typing import Tuple

import numpy as np
import numpy.typing as npt
from keras.src.callbacks import EarlyStopping

from src.model.static import model as model_setup
from src.settings import config, logger
from src.utils import PreprocessorFactory, SaveModel


class TrainModel:
    """Class to handle data of a Keras model."""

    def __init__(self, preprocessor_type: str, num_classes: int) -> None:
        """
        Initialize the data configuration.

        Args:
            preprocessor_type: Type of preprocessor to use ('image', 'text', or 'tabular')
            num_classes: Number of classes in the dataset
        """
        self.num_classes = num_classes
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

    def train(
        self,
        train_data: Path,
        evaluate_data: Path,
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
            model = model_setup.create_model(num_classes=self.num_classes)
            logger.info("Starting model training...")

            early_stopping = EarlyStopping(
                monitor=config.training.early_stopping.monitor,
                patience=config.training.early_stopping.patience,
                restore_best_weights=config.training.early_stopping.restore_best_weights,
                verbose=config.training.early_stopping.verbose,
            )

            # Train the model with early stopping
            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                epochs=config.training.training.epochs,
                batch_size=config.training.training.batch_size,
                callbacks=[early_stopping] if config.training.early_stopping.enabled else [],
                verbose=config.training.training.verbose,
            )
            metrics = {
                "training": {
                    "history": history.history,
                    "parameters": {
                        "epochs": config.training.training.epochs,
                        "batch_size": config.training.training.batch_size,
                        "train_samples": len(x_train),
                        "val_samples": len(x_val),
                    },
                },
            }

            # Save the model
            logger.info("Saving model...")
            return SaveModel().save(model=model, metrics=metrics)

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
