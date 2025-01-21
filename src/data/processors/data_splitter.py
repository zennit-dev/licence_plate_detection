from typing import Set, Tuple

import numpy as np

from src.settings import logger


class DataSplitter:
    """Handles splitting data into train and validation sets."""

    @staticmethod
    def split_indices(total_size: int, split_ratio: float) -> Tuple[Set[int], int, int]:
        """Create train/validation split indices.

        Args:
            total_size: Total number of samples
            split_ratio: Validation split ratio

        Returns:
            Tuple of (train_indices, train_size, val_size)
        """
        indices = np.random.permutation(total_size)
        split_idx = int(total_size * (1 - split_ratio))
        train_indices = set(indices[:split_idx])
        train_size = len(train_indices)
        val_size = total_size - train_size

        logger.info(f"Split dataset: {train_size} training images, {val_size} validation images")
        return train_indices, train_size, val_size
