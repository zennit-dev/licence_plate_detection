from pathlib import Path
from typing import Any, Optional, Tuple, Union, cast

import numpy as np
import tensorflow as tf
from PIL import Image
from PIL.Image import Image as PILImage

from src.utils.preprocessors.base import BaseProcessor, NDArray


class ImageProcessor(BaseProcessor):
    """Preprocessor for image data."""

    def preprocess(
        self,
        input_data: Union[str, Path],
        target_size: Optional[Tuple[int, int]] = None,
        channels: int = 1,
        normalize: bool = True,
        normalize_range: Tuple[float, float] = (0.0, 1.0),
        preserve_aspect_ratio: bool = True,
        binarize: bool = False,
        binarize_threshold: float = 127.5,
        invert_colors: bool = False,
        **kwargs: Any,
    ) -> NDArray:
        """Preprocess image data.

        Args:
            input_data: Path to image file
            target_size: Optional tuple of (height, width) to resize the image to
            channels: Number of color channels (1 for grayscale, 3 for RGB)
            normalize: Whether to normalize pixel values
            normalize_range: Tuple of (min, max) values for normalization
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing
            binarize: Whether to binarize the image
            binarize_threshold: Threshold value for binarization (0-255)
            invert_colors: Whether to invert the image colors
            **kwargs: Additional preprocessing parameters

        Returns:
            Preprocessed image as numpy array

        Raises:
            ValueError: If the image cannot be loaded or preprocessed
        """
        try:
            # Handle ICC profile issues first
            img: PILImage = Image.open(str(input_data))
            if img.mode != "RGB" and channels == 3:
                img = img.convert("RGB")

            # Remove ICC profile if present to avoid warnings
            if "icc_profile" in img.info:
                img_array = np.array(img)
                img = Image.fromarray(img_array)

            # Convert to numpy array first
            img_array = np.array(img, dtype=np.float32)

            # Convert to tensor for TF operations
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

            if invert_colors:
                img_tensor = tf.cast(255.0, tf.float32) - img_tensor

            if binarize:
                threshold = tf.cast(binarize_threshold, tf.float32)
                zeros = tf.zeros_like(img_tensor)
                ones = tf.ones_like(img_tensor) * tf.cast(255.0, tf.float32)
                img_tensor = tf.where(img_tensor > threshold, ones, zeros)

            if target_size is not None:
                if preserve_aspect_ratio:
                    shape = tf.shape(img_tensor)
                    h, w = shape[0], shape[1]
                    max_dim = tf.maximum(h, w)
                    img_tensor = tf.image.resize_with_pad(img_tensor, max_dim, max_dim)
                img_tensor = tf.image.resize(img_tensor, target_size, method="bilinear")

                # Ensure the tensor has the correct shape (height, width, channels)
                if len(img_tensor.shape) == 2:
                    img_tensor = tf.expand_dims(img_tensor, axis=-1)
                elif len(img_tensor.shape) == 3 and img_tensor.shape[-1] != channels:
                    if channels == 1:
                        img_tensor = tf.reduce_mean(img_tensor, axis=-1, keepdims=True)
                    elif channels == 3 and img_tensor.shape[-1] == 1:
                        img_tensor = tf.image.grayscale_to_rgb(img_tensor)

            if normalize:
                min_val, max_val = normalize_range
                min_tensor = tf.reduce_min(img_tensor)
                max_tensor = tf.reduce_max(img_tensor)
                img_tensor = (img_tensor - min_tensor) / (max_tensor - min_tensor)
                img_tensor = img_tensor * tf.cast(max_val - min_val, tf.float32) + tf.cast(
                    min_val, tf.float32
                )

            result_array = img_tensor.numpy()
            if channels == 1:
                result_array = np.squeeze(result_array)

            # Ensure result has shape (height, width, channels)
            if len(result_array.shape) == 2:
                result_array = np.expand_dims(result_array, axis=-1)
            elif len(result_array.shape) == 3 and result_array.shape[-1] != channels:
                if channels == 1:
                    result_array = np.mean(result_array, axis=-1, keepdims=True)
                elif channels == 3 and result_array.shape[-1] == 1:
                    result_array = np.repeat(result_array, 3, axis=-1)

            # Explicitly cast to ensure correct type
            return cast(NDArray, result_array.astype(np.float32))

        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {str(e)}")
