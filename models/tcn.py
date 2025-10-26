"""
Temporal Convolutional Network (TCN) for IDS

Fast and efficient architecture for sequential intrusion detection.
Better than LSTM for CAN-Bus: lower latency, parallelizable, stable gradients.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from .base import BaseIDS, BaseConfig


@dataclass
class TCNConfig(BaseConfig):
    """TCN-specific configuration"""
    num_channels: list = None  # [256, 256, 512, 512] - channel progression
    kernel_size: int = 3
    dropout: float = 0.1
    num_layers: int = 4
    activation: str = "relu"

    def __post_init__(self):
        if self.num_channels is None:
            # Default: 4 layers with increasing channels
            self.num_channels = [256] * self.num_layers


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution with dilation.
    Ensures no future information leakage (critical for real-time IDS).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Weight normalization for stability
        self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply convolution
        out = self.conv(x)

        # Remove future timesteps (causal)
        if self.padding > 0:
            out = out[:, :, :-self.padding]

        out = self.relu(out)
        out = self.dropout(out)
        return out


class ResidualBlock(nn.Module):
    """
    TCN Residual Block with skip connections.
    Enables deep networks (8-16 layers) without vanishing gradients.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation, dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation, dropout)

        # Residual connection with 1x1 conv if dimensions don't match
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.residual is None else self.residual(x)

        out = self.conv1(x)
        out = self.conv2(out)

        return self.relu(out + residual)


class TemporalCNN(BaseIDS):
    """
    Temporal Convolutional Network for Intrusion Detection.

    Architecture:
        Input -> [TCN Blocks with increasing dilation] -> Global Pool -> Classifier

    Advantages over LSTM:
        - 3-5x faster inference
        - Parallelizable (GPU-friendly)
        - Stable training (no vanishing gradients)
        - Large receptive field with dilated convolutions

    Typical receptive field: 2^num_layers * kernel_size (e.g., 2^8 * 3 = 768 frames)
    """

    def __init__(self, config: TCNConfig):
        super().__init__(config)
        self.config = config

        # Input projection
        self.input_proj = nn.Conv1d(config.input_dim, config.num_channels[0], kernel_size=1)

        # TCN blocks with exponentially increasing dilation
        layers = []
        num_levels = len(config.num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = config.num_channels[i-1] if i > 0 else config.num_channels[0]
            out_ch = config.num_channels[i]

            layers.append(ResidualBlock(
                in_ch, out_ch,
                config.kernel_size,
                dilation,
                config.dropout
            ))

        self.network = nn.Sequential(*layers)

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.num_channels[-1], config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features]
            return_features: If True, return (logits, features)

        Returns:
            logits: [batch, num_classes]
        """
        # Transpose for Conv1d: [batch, features, seq_len]
        x = x.transpose(1, 2)

        # Input projection
        x = self.input_proj(x)

        # TCN blocks
        x = self.network(x)

        # Global pooling
        features = self.pool(x).squeeze(-1)  # [batch, channels]

        # Classification
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representation"""
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.network(x)
        features = self.pool(x).squeeze(-1)
        return features

    def compute_receptive_field(self) -> int:
        """Compute theoretical receptive field"""
        receptive_field = 1
        for i in range(len(self.config.num_channels)):
            dilation = 2 ** i
            receptive_field += (self.config.kernel_size - 1) * dilation
        return receptive_field


# Quick test
if __name__ == "__main__":
    config = TCNConfig(
        input_dim=32,
        num_channels=[128, 128, 256, 256],
        kernel_size=3,
        num_classes=5,  # Multi-task
        dropout=0.1
    )

    model = TemporalCNN(config)
    print(f"Model: {model.get_model_info()}")
    print(f"Receptive field: {model.compute_receptive_field()} frames")

    # Test forward pass
    batch_size, seq_len, features = 4, 128, 32
    x = torch.randn(batch_size, seq_len, features)
    logits = model(x)
    print(f"Input: {x.shape} -> Output: {logits.shape}")
