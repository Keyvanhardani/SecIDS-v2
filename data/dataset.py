"""
PyTorch Datasets for CAN-Bus IDS

Supports:
  - Single-task (binary classification)
  - Multi-task (multiple attack types)
  - Streaming mode (real-time inference)
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from .features import CANFeatureExtractor
from .preprocessing import CANPreprocessor


class CANDataset(Dataset):
    """
    CAN-Bus Dataset for single-task classification.

    Loads CAN frames and creates sliding windows.
    """

    def __init__(
        self,
        data_path: str,
        window_size: int = 128,
        stride: int = 64,
        feature_extractor: Optional[CANFeatureExtractor] = None,
        preprocessor: Optional[CANPreprocessor] = None,
        label_column: str = 'label',
        cache: bool = True
    ):
        """
        Args:
            data_path: Path to Parquet/CSV file
            window_size: Number of frames per sample
            stride: Step size for sliding window
            feature_extractor: Feature extraction pipeline
            preprocessor: Data normalization/scaling
            label_column: Name of label column
            cache: Cache processed features in memory
        """
        self.window_size = window_size
        self.stride = stride
        self.label_column = label_column
        self.cache = cache

        # Load data
        data_path = Path(data_path)
        if data_path.suffix == '.parquet':
            self.df = pd.read_parquet(data_path)
        elif data_path.suffix == '.csv':
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file type: {data_path.suffix}")

        # Feature extraction
        self.feature_extractor = feature_extractor or CANFeatureExtractor()
        self.preprocessor = preprocessor

        # Extract features
        print(f"Extracting features from {len(self.df)} frames...")
        self.features = self.feature_extractor.extract_all_features(self.df)

        # Preprocessing (normalization)
        if self.preprocessor:
            self.features = self.preprocessor.fit_transform(self.features)

        # Convert to numpy for speed
        self.feature_matrix = self.features.values.astype(np.float32)

        # Labels
        if label_column in self.df.columns:
            self.labels = self.df[label_column].values
        else:
            self.labels = np.zeros(len(self.df), dtype=np.int64)

        # Create windows
        self.windows = self._create_windows()

        print(f"Created {len(self.windows)} windows from {len(self.df)} frames")

    def _create_windows(self) -> List[Tuple[int, int]]:
        """Create sliding windows (start, end indices)"""
        windows = []
        for start in range(0, len(self.df) - self.window_size + 1, self.stride):
            end = start + self.window_size
            windows.append((start, end))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single window.

        Returns:
            {
                'features': [window_size, num_features],
                'label': scalar (0 or 1),
                'can_ids': [window_size] (for debugging)
            }
        """
        start, end = self.windows[idx]

        # Features
        features = self.feature_matrix[start:end]  # [window_size, num_features]

        # Label (majority vote in window)
        window_labels = self.labels[start:end]
        label = int(window_labels.sum() > len(window_labels) / 2)  # 1 if >50% attack

        # CAN IDs (for debugging)
        can_ids = self.df['can_id'].iloc[start:end].values

        return {
            'features': torch.from_numpy(features),
            'label': torch.tensor(label, dtype=torch.long),
            'can_ids': torch.from_numpy(can_ids.astype(np.int32))
        }


class MultiTaskCANDataset(Dataset):
    """
    Multi-Task CAN Dataset.

    Supports multiple label columns for different attack types.
    """

    def __init__(
        self,
        data_path: str,
        window_size: int = 128,
        stride: int = 64,
        task_columns: Dict[str, str] = None,
        feature_extractor: Optional[CANFeatureExtractor] = None,
        preprocessor: Optional[CANPreprocessor] = None,
        cache: bool = True
    ):
        """
        Args:
            task_columns: Dict[task_name -> column_name]
                         e.g., {"dos": "label_dos", "fuzzy": "label_fuzzy"}
        """
        self.window_size = window_size
        self.stride = stride
        self.task_columns = task_columns or {}
        self.cache = cache

        # Load data
        data_path = Path(data_path)
        if data_path.suffix == '.parquet':
            self.df = pd.read_parquet(data_path)
        elif data_path.suffix == '.csv':
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file type: {data_path.suffix}")

        # Feature extraction
        self.feature_extractor = feature_extractor or CANFeatureExtractor()
        self.preprocessor = preprocessor

        print(f"Extracting features from {len(self.df)} frames...")
        self.features = self.feature_extractor.extract_all_features(self.df)

        if self.preprocessor:
            self.features = self.preprocessor.fit_transform(self.features)

        self.feature_matrix = self.features.values.astype(np.float32)

        # Load labels for each task
        self.task_labels = {}
        for task_name, col_name in self.task_columns.items():
            if col_name in self.df.columns:
                self.task_labels[task_name] = self.df[col_name].values
            else:
                print(f"Warning: Column {col_name} not found for task {task_name}")
                self.task_labels[task_name] = np.zeros(len(self.df), dtype=np.int64)

        # Create windows
        self.windows = self._create_windows()
        print(f"Created {len(self.windows)} windows for {len(self.task_columns)} tasks")

    def _create_windows(self) -> List[Tuple[int, int]]:
        windows = []
        for start in range(0, len(self.df) - self.window_size + 1, self.stride):
            end = start + self.window_size
            windows.append((start, end))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                'features': [window_size, num_features],
                'labels': {
                    'dos': scalar,
                    'fuzzy': scalar,
                    ...
                }
            }
        """
        start, end = self.windows[idx]

        features = self.feature_matrix[start:end]

        # Labels for each task (majority vote)
        labels = {}
        for task_name, task_labels in self.task_labels.items():
            window_labels = task_labels[start:end]
            label = int(window_labels.sum() > len(window_labels) / 2)
            labels[task_name] = torch.tensor(label, dtype=torch.long)

        return {
            'features': torch.from_numpy(features),
            'labels': labels
        }


# Utility functions
def create_dataloaders(
    train_path: str,
    val_path: Optional[str] = None,
    batch_size: int = 32,
    window_size: int = 128,
    num_workers: int = 4,
    multitask: bool = False,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train/val dataloaders.

    Args:
        train_path: Path to training data
        val_path: Path to validation data (optional)
        batch_size: Batch size
        window_size: Window size
        num_workers: DataLoader workers
        multitask: Use MultiTaskCANDataset
        **kwargs: Additional dataset arguments

    Returns:
        train_loader, val_loader (or None)
    """
    DatasetClass = MultiTaskCANDataset if multitask else CANDataset

    # Filter kwargs based on dataset type
    if not multitask:
        # Remove multitask-specific kwargs for single-task dataset
        kwargs = {k: v for k, v in kwargs.items() if k != 'task_columns'}

    # Training set
    train_dataset = DatasetClass(
        train_path,
        window_size=window_size,
        stride=window_size // 2,  # 50% overlap
        **kwargs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Validation set
    val_loader = None
    if val_path:
        # Filter kwargs for validation set too
        val_kwargs = {k: v for k, v in kwargs.items() if k != 'task_columns'} if not multitask else kwargs

        val_dataset = DatasetClass(
            val_path,
            window_size=window_size,
            stride=window_size,  # No overlap for validation
            **val_kwargs
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader


# Test
if __name__ == "__main__":
    # Create synthetic data
    import tempfile

    np.random.seed(42)
    data = {
        'timestamp': np.cumsum(np.random.exponential(0.01, 10000)),
        'can_id': np.random.choice([0x100, 0x200, 0x300], 10000),
        'dlc': 8,
        'data': [''.join(f'{b:02x}' for b in np.random.randint(0, 256, 8)) for _ in range(10000)],
        'label': np.random.choice([0, 1], 10000, p=[0.9, 0.1])
    }

    df = pd.DataFrame(data)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        df.to_parquet(f.name)
        temp_path = f.name

    # Test dataset
    dataset = CANDataset(temp_path, window_size=64, stride=32)
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Features shape: {sample['features'].shape}")
    print(f"Label: {sample['label']}")

    # Test dataloader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"\nBatch features: {batch['features'].shape}")
    print(f"Batch labels: {batch['label'].shape}")
