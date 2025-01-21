import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

from src.settings import logger


class DatasetScanner:
    """Handles scanning dataset directory and creating class mappings."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.images_dir = data_dir / "images"
        self.annotations_dir = data_dir / "annotations"

    def parse_annotation(self, xml_path: Path) -> str:
        """Parse XML annotation file to get the class label.

        Args:
            xml_path: Path to XML annotation file

        Returns:
            Class label from the annotation
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            # Assuming the class label is stored in a specific tag - adjust as needed
            # This is an example - modify according to your XML structure
            class_elem = root.find(".//object/name")
            if class_elem is not None and class_elem.text:
                return class_elem.text
            raise ValueError(f"No class label found in {xml_path}")
        except Exception as e:
            raise ValueError(f"Failed to parse annotation {xml_path}: {str(e)}")

    def scan_dataset(self) -> Tuple[List[Path], List[int], Dict[str, int]]:
        """Scan dataset directory and collect image paths and labels.

        Returns:
            Tuple of (image_paths, labels, class_mapping)
        """
        if not self.images_dir.exists() or not self.annotations_dir.exists():
            raise ValueError(f"Images or annotations directory not found in {self.data_dir}")

        image_paths: List[Path] = []
        labels: List[int] = []
        class_labels: set[str] = set()

        # First pass: collect all unique class labels
        logger.info("Scanning annotations to collect class labels...")
        for xml_path in self.annotations_dir.glob("*.xml"):
            try:
                class_label = self.parse_annotation(xml_path)
                class_labels.add(class_label)
            except ValueError as e:
                logger.warning(str(e))
                continue

        # Create class mapping
        class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(class_labels))}

        # Second pass: collect image paths and labels
        logger.info("Collecting image paths and labels...")
        for xml_path in self.annotations_dir.glob("*.xml"):
            try:
                # Get corresponding image path
                image_name = xml_path.stem + ".png"  # Adjust extension if needed
                image_path = self.images_dir / image_name

                if not image_path.exists():
                    logger.warning(f"Image file not found for annotation: {image_name}")
                    continue

                class_label = self.parse_annotation(xml_path)
                class_idx = class_to_idx[class_label]

                image_paths.append(image_path)
                labels.append(class_idx)

            except ValueError as e:
                logger.warning(str(e))
                continue

        if not image_paths:
            raise ValueError(f"No valid image-annotation pairs found in {self.data_dir}")

        logger.info(f"Found {len(image_paths)} valid image-annotation pairs")
        logger.info(f"Found {len(class_to_idx)} unique classes")

        return image_paths, labels, class_to_idx
