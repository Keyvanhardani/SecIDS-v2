"""
SecIDS-v2 Data Pipeline

CAN-Bus specific feature engineering and dataset loading.
"""

from .features import CANFeatureExtractor
from .dataset import CANDataset, MultiTaskCANDataset, create_dataloaders
from .preprocessing import CANPreprocessor
from .augmentation import CANAugmentation

__all__ = [
    "CANFeatureExtractor",
    "CANDataset",
    "MultiTaskCANDataset",
    "create_dataloaders",
    "CANPreprocessor",
    "CANAugmentation",
]
