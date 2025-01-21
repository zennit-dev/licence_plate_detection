from pathlib import Path
from typing import Any, Optional, Tuple, Union, cast

import numpy as np

from src.utils.preprocessors.base import BaseProcessor, NDArray, NDArrayAny


class TabularProcessor(BaseProcessor):
    """Preprocessor for tabular data."""

    def preprocess(
        self,
        input_data: Union[str, Path, NDArrayAny],
        normalize: bool = True,
        normalize_range: Tuple[float, float] = (0.0, 1.0),
        handle_missing: bool = True,
        categorical_columns: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> NDArray:
        """Preprocess tabular data.

        Args:
            input_data: Path to CSV/Excel file or numpy array
            normalize: Whether to normalize numerical columns
            normalize_range: Target range for normalization
            handle_missing: Whether to handle missing values
            categorical_columns: List of categorical column names
            **kwargs: Additional preprocessing parameters

        Returns:
            Preprocessed data as numpy array

        Raises:
            ValueError: If data preprocessing fails
        """
        try:
            # Load data if it's a file path
            if isinstance(input_data, (str, Path)):
                data = np.genfromtxt(input_data, delimiter=",", dtype=np.float32)
            else:
                data = input_data.astype(np.float32)

            if handle_missing:
                # Replace missing values with column means
                column_means = np.nanmean(data, axis=0)
                nan_mask = np.isnan(data)
                data[nan_mask] = np.take(column_means, np.where(nan_mask)[1])

            if normalize:
                min_val, max_val = normalize_range
                data_min = np.min(data, axis=0)
                data_max = np.max(data, axis=0)
                data = (data - data_min) / (data_max - data_min)
                data = data * (max_val - min_val) + min_val

            return cast(NDArray, data.astype(np.float32))
        except Exception as e:
            raise ValueError(f"Failed to preprocess tabular data: {str(e)}")
