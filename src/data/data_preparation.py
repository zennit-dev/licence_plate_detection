import json
import shutil
from pathlib import Path
from typing import Tuple

from src.data.processors import BatchCombiner, BatchProcessor, DatasetScanner, DataSplitter
from src.settings import logger
from src.utils.preprocessor_factory import PreprocessorFactory, PreprocessorType
from src.utils.save_model import SupportsWrite


class DataPreparation:
    """Orchestrates data preparation workflow."""

    def __init__(
        self,
        data_dir: Path,
        target_size: Tuple[int, int] = (224, 224),
        channels: int = 3,
        batch_size: int = 32,
    ) -> None:
        """Initialize data preparation.

        Args:
            data_dir: Directory containing raw data
            target_size: Target size for images (height, width)
            channels: Number of channels (3 for RGB)
            batch_size: Batch size for processing
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.channels = channels
        self.batch_size = batch_size
        self.preprocessor = PreprocessorFactory.get_preprocessor(PreprocessorType.IMAGE.value)

    def prepare(self, output_path: Path, split_ratio: float = 0.2) -> None:
        """Prepare dataset from directory structure.

        Args:
            output_path: Path to save processed data
            split_ratio: Validation split ratio
        """
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        scanner = DatasetScanner(self.data_dir)
        splitter = DataSplitter()
        processor = BatchProcessor(
            self.preprocessor,
            self.target_size,
            self.channels,
            self.batch_size,
        )
        combiner = BatchCombiner()

        # Set up directory structure
        train_dir = output_path / "train"
        val_dir = output_path / "val"
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)

        train_temp = train_dir / "temp"
        val_temp = val_dir / "temp"

        # Clean up any existing temp directories
        for temp_dir in [train_temp, val_temp]:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean up {temp_dir}: {e}")
            temp_dir.mkdir(exist_ok=True)

        # Scan dataset
        image_paths, labels, class_to_idx = scanner.scan_dataset()
        total_images = len(image_paths)

        # Split dataset
        train_indices, train_size, val_size = splitter.split_indices(total_images, split_ratio)

        # Process batches
        processed_count, failed_count, train_batch_count, val_batch_count = (
            processor.process_batches(image_paths, labels, train_indices, (train_temp, val_temp))
        )

        # Combine batches
        logger.info("Combining training batches...")
        combiner.combine_batches(train_temp, train_dir / "data.npz", train_batch_count)
        logger.info("Combining validation batches...")
        combiner.combine_batches(val_temp, val_dir / "data.npz", val_batch_count)

        # Clean up temp directories
        logger.info("Cleaning up temporary files...")
        try:
            shutil.rmtree(train_temp)
            shutil.rmtree(val_temp)
            logger.info("Temporary files cleaned up successfully")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directories: {e}")

        # Save class mapping
        f: SupportsWrite
        with open(output_path / "class_mapping.json", "w") as f:
            json.dump(class_to_idx, f)

        logger.info("Dataset preparation completed:")
        logger.info(f"- {len(class_to_idx)} classes")
        logger.info(f"- {processed_count} images processed ({failed_count} failed)")
        logger.info(f"- Training: {train_size} images in {train_batch_count} batches")
        logger.info(f"- Validation: {val_size} images in {val_batch_count} batches")
