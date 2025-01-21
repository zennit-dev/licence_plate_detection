from typing import Any, Optional, cast

import numpy as np

from src.utils.preprocessors.base import BaseProcessor, NDArray


class TextProcessor(BaseProcessor):
    """Preprocessor for text data."""

    def preprocess(
        self,
        input_data: str,
        max_length: Optional[int] = None,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        **kwargs: Any,
    ) -> NDArray:
        """Preprocess text data.

        Args:
            input_data: Input text
            max_length: Maximum sequence length (pad/truncate to this length)
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
            remove_numbers: Whether to remove numbers
            **kwargs: Additional preprocessing parameters

        Returns:
            Preprocessed text as numpy array

        Raises:
            ValueError: If text preprocessing fails
        """
        try:
            # Basic text preprocessing
            if lowercase:
                input_data = input_data.lower()

            if remove_punctuation:
                input_data = "".join(c for c in input_data if c.isalnum() or c.isspace())

            if remove_numbers:
                input_data = "".join(c for c in input_data if not c.isdigit())

            # Convert to character-level or token-level representation
            tokens = input_data.split()
            if max_length:
                tokens = tokens[:max_length]

            # Convert to numerical representation
            char_to_int = {char: i for i, char in enumerate(set("".join(tokens)))}
            numerical = [[char_to_int[char] for char in token] for token in tokens]

            # Pad sequences if needed
            if max_length:
                padded = [seq + [0] * (max_length - len(seq)) for seq in numerical]
                return cast(NDArray, np.array(padded, dtype=np.float32))

            return cast(NDArray, np.array(numerical, dtype=np.float32))
        except Exception as e:
            raise ValueError(f"Failed to preprocess text: {str(e)}")
