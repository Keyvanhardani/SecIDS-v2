"""
Mamba-based Intrusion Detection System

State-Space Model (SSM) for long-range dependencies in CAN-Bus streams.
Better than LSTM/Transformer for:
  - Long sequences (1000+ frames)
  - Low latency inference
  - Stable training
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from .base import BaseIDS, BaseConfig

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("WARNING: mamba_ssm not installed. Install with: pip install mamba-ssm")


@dataclass
class MambaConfig(BaseConfig):
    """Mamba-specific configuration"""
    d_model: int = 256  # Model dimension
    d_state: int = 16   # SSM state dimension
    d_conv: int = 4     # Convolution kernel size
    expand: int = 2     # Expansion factor
    num_layers: int = 4
    dropout: float = 0.1


class MambaBlock(nn.Module):
    """
    Single Mamba block with residual connection.
    """

    def __init__(self, config: MambaConfig):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm required. Install: pip install mamba-ssm")

        self.mamba = Mamba(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual block
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class MambaIDS(BaseIDS):
    """
    Mamba-based Intrusion Detection System.

    Architecture:
        Input -> Embedding -> [Mamba Blocks] -> Pool -> Classifier

    Advantages:
        - Linear O(n) complexity (vs O(n²) for Transformer)
        - Long-range dependencies (1000+ frames)
        - Fast inference (comparable to TCN)
        - State-space formulation (principled approach)

    References:
        - Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)
        - https://github.com/state-spaces/mamba
    """

    def __init__(self, config: MambaConfig):
        super().__init__(config)
        self.config = config

        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm required for MambaIDS. "
                "Install: pip install mamba-ssm causal-conv1d"
            )

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.d_model)

        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(config) for _ in range(config.num_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(config.d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
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
        # Input projection
        x = self.input_proj(x)  # [batch, seq_len, d_model]

        # Mamba blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        # Global average pooling
        features = x.mean(dim=1)  # [batch, d_model]

        # Classification
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representation"""
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        features = x.mean(dim=1)
        return features


class MambaFallback(BaseIDS):
    """
    Fallback implementation when mamba_ssm is not available.
    Uses a simple LSTM instead (for development/testing).
    """

    def __init__(self, config: MambaConfig):
        super().__init__(config)
        self.config = config

        print("WARNING: Using LSTM fallback (mamba_ssm not available)")

        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.lstm = nn.LSTM(
            config.d_model,
            config.d_model,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.norm(x)
        features = x[:, -1, :]  # Last timestep
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.norm(x)
        return x[:, -1, :]


# Factory function
def create_mamba_ids(config: MambaConfig) -> BaseIDS:
    """Create MambaIDS or fallback to LSTM if mamba_ssm unavailable"""
    if MAMBA_AVAILABLE:
        return MambaIDS(config)
    else:
        return MambaFallback(config)


# Quick test
if __name__ == "__main__":
    config = MambaConfig(
        input_dim=32,
        d_model=256,
        d_state=16,
        num_layers=4,
        num_classes=5
    )

    model = create_mamba_ids(config)
    print(f"Model: {model.get_model_info()}")

    # Test forward pass
    batch_size, seq_len, features = 4, 256, 32
    x = torch.randn(batch_size, seq_len, features)
    logits = model(x)
    print(f"Input: {x.shape} -> Output: {logits.shape}")
