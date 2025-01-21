from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[np.float32]
NDArrayAny = npt.NDArray[Any]


class BaseProcessor(ABC):
    """Base class for all preprocessors."""

    @abstractmethod
    def preprocess(self, input_data: Any, **kwargs: Any) -> NDArray:
        """Preprocess the input data.

        Args:
            input_data: Raw input data
            **kwargs: Additional preprocessing parameters

        Returns:
            Preprocessed data as numpy array

        Raises:
            ValueError: If preprocessing fails
        """
        pass
