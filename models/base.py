"""
Base model interface for all IDS models
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class BaseConfig:
    """Base configuration for all models"""
    input_dim: int = 32  # Feature dimension
    hidden_dim: int = 256
    num_classes: int = 2  # Binary by default
    dropout: float = 0.1


class BaseIDS(nn.Module, ABC):
    """
    Abstract base class for all IDS models.
    Ensures consistent interface for training, export, and inference.
    """

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [batch, sequence_length, features]

        Returns:
            logits: Output logits [batch, num_classes]
        """
        pass

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate features for analysis/visualization

        Args:
            x: Input tensor

        Returns:
            features: Feature tensor [batch, hidden_dim]
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata"""
        return {
            "name": self.__class__.__name__,
            "config": self.config.__dict__,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "num_trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def compute_receptive_field(self) -> int:
        """Compute theoretical receptive field (override in subclasses)"""
        return 1
