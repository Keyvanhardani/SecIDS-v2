"""
SecIDS-v2 Training Pipeline

PyTorch Lightning-based training with:
  - Single-task and multi-task learning
  - WandB logging
  - Automatic checkpointing
  - Learning rate scheduling
"""

from .trainer import IDSTrainer, MultiTaskIDSTrainer
from .train import train_model

__all__ = [
    "IDSTrainer",
    "MultiTaskIDSTrainer",
    "train_model",
]
