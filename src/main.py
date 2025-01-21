"""Main entry point for the ML project."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Union

from src.data import DataPreparation
from src.model import app
from src.settings import config, logger
from src.utils import PreprocessorType
from tools.lint import run_lint


def get_latest_saved_version() -> str:
    """Get the version of the latest saved model (only the base version part before timestamp)."""
    model_dir = Path(config.environment.path.model_dir)
    if not model_dir.exists():
        return ""

    version_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir()], reverse=True)
    if not version_dirs:
        return ""

    # Get only the base version part before the timestamp
    latest_version = version_dirs[0].name
    return latest_version.split("_")[0] if "_" in latest_version else latest_version


def print_predictions(
    predictions: Dict[str, Dict[str, Union[int, float, str, list[float]]]]
) -> None:
    min_accuracy = config.training.metrics.min_accuracy
    for model_type, result in predictions.items():
        confidence = result.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            logger.warning(f"Unexpected confidence type for {model_type}")
            continue
        if float(confidence) < min_accuracy:
            logger.warning(
                f"{model_type.capitalize()} model prediction confidence "
                f"({confidence:.4f}) is below minimum threshold ({min_accuracy})"
            )


def train(current_version: str, latest_saved_version: str) -> None:
    """Train a new model.

    Args:
        current_version: Current model version
        latest_saved_version: Latest saved model version
    """
    reason = "versions differ" if current_version != latest_saved_version else "force flag is set"
    logger.info(
        f"Training new model because {reason} "
        f"(current: {current_version}, latest: {latest_saved_version})"
    )

    # Prepare the dataset if needed
    raw_data_dir = Path(config.environment.path.data_dir) / "raw"
    processed_dir = Path(config.environment.path.data_dir) / "processed"
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    train_data = train_dir / "data.npz"
    val_data = val_dir / "data.npz"

    # Check if data preparation can be skipped
    data_is_valid = (
        train_data.exists()
        and val_data.exists()
        and (processed_dir / "class_mapping.json").exists()
    )

    if data_is_valid:
        # Calculate total size of training data
        train_size = train_data.stat().st_size
        logger.info("Found existing processed data, skipping data preparation")
        logger.info(f"Train data size: {train_size / 1e6:.2f} MB")
    else:
        logger.info("Preparing dataset...")
        data_prep = DataPreparation(data_dir=raw_data_dir, target_size=(224, 224), channels=3)
        data_prep.prepare(output_path=processed_dir, split_ratio=0.2)

        # Verify data was created
        if not train_data.exists() or not val_data.exists():
            raise RuntimeError("Data preparation failed: Training or validation data not created")

    # Train the model
    app.train(train_dir, val_dir)


def main() -> None:
    """Main entry point that demonstrates data and prediction."""
    parser = argparse.ArgumentParser(description="ML project command line interface")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force data even if versions match",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to the image file to predict",
        required=True,
    )
    args = parser.parse_args()

    if not run_lint():
        raise ValueError("Linting failed. Please fix the issues before continuing.")

    try:
        # Validate image path
        input_data = Path(args.image)
        if not input_data.exists():
            raise FileNotFoundError(f"Image file not found: {args.image}")
        if not input_data.is_file():
            raise ValueError(f"Image path is not a file: {args.image}")

        current_version = config.model.storage.version
        latest_saved_version = get_latest_saved_version()
        if current_version != latest_saved_version or args.force:
            train(current_version, latest_saved_version)
        else:
            logger.info(f"Using existing model version {current_version}")

        predictions = app.predict(
            input_data=input_data,
            preprocessor_type=str(PreprocessorType.IMAGE.value),
            preprocess_kwargs={"target_size": (224, 224), "channels": 3},
            extra_kwargs={"add_batch_dim": True},
        )
        print_predictions(predictions)

        logger.info(f"Predictions:\n {json.dumps(predictions, indent=4)}")
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
