import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.settings import logger


class BatchCombiner:
    """Handles combining batch files into final datasets."""

    @staticmethod
    def _clear_memory() -> None:
        """Force clear memory."""
        import gc

        gc.collect()

    @staticmethod
    def combine_batches(temp_dir: Path, output_file: Path, batch_count: int) -> None:
        """Combine multiple batch files into a single file incrementally.

        Args:
            temp_dir: Directory containing batch files
            output_file: Path to save combined output
            batch_count: Total number of batches to combine
        """
        if batch_count == 0:
            return

        first_batch = temp_dir / "batch_0.npz"
        if not first_batch.exists():
            return

        logger.info("Starting batch combination process...")
        # Copy first batch as base
        shutil.copy(first_batch, output_file)
        BatchCombiner._clear_memory()

        with tqdm(total=batch_count - 1, desc="Combining batches") as pbar:
            for i in range(1, batch_count):
                try:
                    batch_file = temp_dir / f"batch_{i}.npz"
                    if not batch_file.exists():
                        continue

                    # Load current combined state
                    current = np.load(output_file)
                    current_features = current["features"]
                    current_labels = current["labels"]
                    current_size = len(current_labels)

                    # Load new batch
                    batch = np.load(batch_file)
                    batch_features = batch["features"]
                    batch_labels = batch["labels"]
                    batch_size = len(batch_labels)

                    # Combine and save
                    features = np.concatenate([current_features, batch_features])
                    labels = np.concatenate([current_labels, batch_labels])
                    np.savez(output_file, features=features, labels=labels)

                    # Clean up memory
                    del current, current_features, current_labels
                    del batch, batch_features, batch_labels
                    del features, labels
                    BatchCombiner._clear_memory()

                    logger.debug(
                        f"Added batch {i} with {batch_size} "
                        f"samples (total: {current_size + batch_size})"
                    )
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Error processing batch {i}: {str(e)}")
                    continue
