from pathlib import Path
from typing import Any, List, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from src.settings import logger


class BatchProcessor:
    """Handles processing and saving data batches."""

    def __init__(
        self,
        preprocessor: Any,
        target_size: Tuple[int, int],
        channels: int,
        batch_size: int = 32,
        save_threshold: int = 100,
    ) -> None:
        self.preprocessor = preprocessor
        self.target_size = target_size
        self.channels = channels
        self.batch_size = batch_size
        self.save_threshold = save_threshold

    @staticmethod
    def save_batch(
        file_path: Path,
        features: List[NDArray[np.float32]],
        labels: List[int],
    ) -> None:
        """Save a batch of processed data to a file."""
        features_array = np.stack(features)
        # Ensure features have shape (batch_size, height, width, channels)
        if len(features_array.shape) == 3:
            features_array = np.expand_dims(features_array, axis=-1)
        labels_array = np.array(labels, dtype=np.int32)
        np.savez(file_path, features=features_array, labels=labels_array)

    def _handle_batch_save(
        self,
        features: List[NDArray[np.float32]],
        labels: List[int],
        temp_dir: Path,
        batch_count: int,
    ) -> Tuple[List[NDArray[np.float32]], List[int], int]:
        """Handle saving a batch when threshold is reached.

        Args:
            features: List of processed features
            labels: List of corresponding labels
            temp_dir: Directory to save batch file
            batch_count: Current batch count

        Returns:
            Tuple of (new_features, new_labels, new_batch_count)
        """
        if len(features) >= self.save_threshold:
            self.save_batch(
                temp_dir / f"batch_{batch_count}.npz",
                features,
                labels,
            )
            return [], [], batch_count + 1
        return features, labels, batch_count

    def process_batches(
        self,
        image_paths: List[Path],
        labels: List[int],
        train_indices: Set[int],
        temp_dirs: Tuple[Path, Path],
    ) -> Tuple[int, int, int, int]:
        """Process data in batches and save to temporary files.

        Returns:
            Tuple of (processed_count, failed_count, train_batch_count, val_batch_count)
        """
        train_temp, val_temp = temp_dirs
        total_images = len(image_paths)

        train_features: List[NDArray[np.float32]] = []
        train_labels: List[int] = []
        val_features: List[NDArray[np.float32]] = []
        val_labels: List[int] = []

        train_batch_count = 0
        val_batch_count = 0
        processed_count = 0
        failed_count = 0

        logger.info(
            f"Processing images (batch size: {self.batch_size}, "
            f"save threshold: {self.save_threshold})..."
        )
        with tqdm(total=total_images) as pbar:
            for idx in range(0, total_images, self.batch_size):
                batch_indices = range(idx, min(idx + self.batch_size, total_images))

                for i in batch_indices:
                    try:
                        img_array = self.preprocessor.preprocess(
                            image_paths[i],
                            target_size=self.target_size,
                            channels=self.channels,
                            normalize=True,
                            normalize_range=(0, 1),
                        )
                        if i in train_indices:
                            train_features.append(img_array)
                            train_labels.append(labels[i])
                            train_features, train_labels, train_batch_count = (
                                self._handle_batch_save(
                                    train_features, train_labels, train_temp, train_batch_count
                                )
                            )
                        else:
                            val_features.append(img_array)
                            val_labels.append(labels[i])
                            val_features, val_labels, val_batch_count = self._handle_batch_save(
                                val_features, val_labels, val_temp, val_batch_count
                            )
                        processed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to process {image_paths[i]}: {str(e)}")
                        failed_count += 1
                    pbar.update(1)

        # Save remaining batches
        if train_features:
            self.save_batch(
                train_temp / f"batch_{train_batch_count}.npz",
                train_features,
                train_labels,
            )
            train_batch_count += 1
        if val_features:
            self.save_batch(
                val_temp / f"batch_{val_batch_count}.npz",
                val_features,
                val_labels,
            )
            val_batch_count += 1

        return processed_count, failed_count, train_batch_count, val_batch_count
