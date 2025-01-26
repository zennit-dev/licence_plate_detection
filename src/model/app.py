"""Main application class for the ML project."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.model.predict import PredictModel
from src.model.train import TrainModel
from src.settings import config, logger


class MLApp:
    """Main application class handling data and prediction workflows."""

    def __init__(self) -> None:
        """Initialize the ML application."""
        self.model_manager = PredictModel()

    @staticmethod
    def train(train_data: Path, validation_data: Path) -> Path:
        """Run the training pipeline.

        Args:
            train_data: Path to the training data directory
            validation_data: Path to the validation data directory

        Returns:
            Path to saved model directory
        """
        try:
            # Load class mapping to determine number of classes
            processed_dir = Path(config.training.data.data_dir) / "processed"
            with open(processed_dir / "class_mapping.json") as f:
                class_mapping = json.load(f)
            num_classes = len(class_mapping)

            # Train and evaluate the model
            trainer = TrainModel(preprocessor_type="image", num_classes=num_classes)
            save_path = trainer.train(train_data, validation_data)

            logger.info("Training completed successfully")
            logger.info(f"Model saved to: {save_path}")

            return save_path

        except ImportError as e:
            logger.error(f"Failed to import required modules: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def predict(
        self,
        input_data: Union[str, Path],
        preprocessor_type: str,
        preprocess_kwargs: Optional[Dict[str, Any]] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Union[int, float, str, list[float]]]]:
        """Make predictions using the trained model.

        Args:
            input_data: Path to the input file to predict on
            preprocessor_type: Type of preprocessor to use
            preprocess_kwargs: Additional preprocessing arguments
            extra_kwargs: Additional prediction arguments

        Returns:
            Dictionary containing predictions and optionally probabilities
        """
        try:
            return self.model_manager.predict(
                input_data=input_data,
                preprocessor_type=preprocessor_type,
                preprocess_kwargs=preprocess_kwargs,
                extra_kwargs=extra_kwargs,
            )
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
