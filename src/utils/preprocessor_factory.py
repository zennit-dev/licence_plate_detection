"""Factory for creating and managing preprocessors."""

from enum import Enum
from typing import Type

from src.utils.preprocessors import ImageProcessor, TabularProcessor, TextProcessor
from src.utils.preprocessors.base import BaseProcessor


class PreprocessorType(Enum):
    """Supported preprocessor types."""

    IMAGE = "image"
    TEXT = "text"
    TABULAR = "tabular"


class PreprocessorFactory:
    """Factory for creating and managing preprocessors."""

    _preprocessors: dict[PreprocessorType, Type[BaseProcessor]] = {
        PreprocessorType.IMAGE: ImageProcessor,
        PreprocessorType.TEXT: TextProcessor,
        PreprocessorType.TABULAR: TabularProcessor,
    }

    @classmethod
    def get_preprocessor(cls, preprocessor_type: PreprocessorType | str) -> BaseProcessor:
        """Get a preprocessor instance.

        Args:
            preprocessor_type: Type of preprocessor to create

        Returns:
            Preprocessor instance

        Raises:
            ValueError: If preprocessor type is not supported
        """
        if isinstance(preprocessor_type, str):
            try:
                preprocessor_type = PreprocessorType(preprocessor_type.lower())
            except ValueError as e:
                raise ValueError(
                    f"Unsupported preprocessor type: {preprocessor_type}. "
                    f"Supported types: {[t.value for t in PreprocessorType]}"
                ) from e

        preprocessor_class = cls._preprocessors.get(preprocessor_type)
        if not preprocessor_class:
            raise ValueError(
                f"No preprocessor found for type: {preprocessor_type}. "
                f"Supported types: {[t.value for t in PreprocessorType]}"
            )

        return preprocessor_class()
