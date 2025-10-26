"""
Multi-Task Learning for IDS

Single backbone model with multiple task-specific heads:
  - DoS detection
  - Fuzzy attack detection
  - Spoofing detection
  - Replay attack detection
  - Normal/Benign classification

Advantages:
  - Shared representations (better generalization)
  - Single model deployment (less memory)
  - Cross-task knowledge transfer
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .base import BaseIDS, BaseConfig
from .tcn import TemporalCNN, TCNConfig
from .mamba import MambaIDS, MambaConfig, MAMBA_AVAILABLE


@dataclass
class MultiTaskConfig(BaseConfig):
    """Multi-task learning configuration"""
    backbone: str = "tcn"  # "tcn" or "mamba"
    task_names: List[str] = field(default_factory=lambda: ["dos", "fuzzy", "spoofing", "replay"])
    shared_hidden_dim: int = 256
    task_hidden_dim: int = 128

    # Backbone-specific configs
    tcn_config: Optional[TCNConfig] = None
    mamba_config: Optional[MambaConfig] = None


class TaskHead(nn.Module):
    """
    Task-specific classification head.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskIDS(BaseIDS):
    """
    Multi-Task Intrusion Detection System.

    Architecture:
        Input -> Backbone (TCN/Mamba) -> [Task Heads]

    Single forward pass produces predictions for all attack types.
    """

    def __init__(self, config: MultiTaskConfig):
        super().__init__(config)
        self.config = config

        # Build backbone
        if config.backbone == "tcn":
            if config.tcn_config is None:
                # Default TCN config
                config.tcn_config = TCNConfig(
                    input_dim=config.input_dim,
                    num_channels=[256, 256, 512, 512],
                    hidden_dim=config.shared_hidden_dim,
                    num_classes=2  # Dummy, not used
                )
            # Remove classification head (we'll add task-specific heads)
            self.backbone = TemporalCNN(config.tcn_config)
            self.feature_dim = config.tcn_config.num_channels[-1]

        elif config.backbone == "mamba":
            if not MAMBA_AVAILABLE:
                raise ImportError("mamba_ssm required for mamba backbone")

            if config.mamba_config is None:
                config.mamba_config = MambaConfig(
                    input_dim=config.input_dim,
                    d_model=256,
                    num_layers=4,
                    hidden_dim=config.shared_hidden_dim,
                    num_classes=2  # Dummy
                )
            self.backbone = MambaIDS(config.mamba_config)
            self.feature_dim = config.mamba_config.d_model

        else:
            raise ValueError(f"Unknown backbone: {config.backbone}")

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name in config.task_names:
            self.task_heads[task_name] = TaskHead(
                self.feature_dim,
                config.task_hidden_dim,
                num_classes=2  # Binary for each task
            )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, features]
            return_features: If True, return features in dict

        Returns:
            outputs: Dict[task_name -> logits]
        """
        # Extract features from backbone
        features = self.backbone.extract_features(x)  # [batch, feature_dim]

        # Task-specific predictions
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(features)

        if return_features:
            outputs["features"] = features

        return outputs

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shared features"""
        return self.backbone.extract_features(x)

    def get_task_names(self) -> List[str]:
        """Get list of task names"""
        return list(self.task_heads.keys())


class MultiTaskLoss(nn.Module):
    """
    Multi-Task Loss with uncertainty-based weighting.

    Reference: "Multi-Task Learning Using Uncertainty to Weigh Losses" (Kendall et al.)
    Learns task-specific weights automatically.
    """

    def __init__(self, task_names: List[str], init_log_var: float = 0.0):
        super().__init__()
        self.task_names = task_names

        # Learnable log-variance for each task (uncertainty weighting)
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.tensor(init_log_var))
            for task in task_names
        })

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Dict[task_name -> logits]
            targets: Dict[task_name -> labels]

        Returns:
            loss_dict: Dict with 'total', 'task_losses', 'weights'
        """
        total_loss = 0
        task_losses = {}
        weights = {}

        for task in self.task_names:
            if task not in targets or targets[task] is None:
                continue  # Skip tasks without labels

            # Cross-entropy loss
            loss = nn.functional.cross_entropy(predictions[task], targets[task])
            task_losses[task] = loss

            # Uncertainty weighting: loss_weighted = loss * exp(-log_var) + log_var
            log_var = self.log_vars[task]
            weighted_loss = torch.exp(-log_var) * loss + log_var
            total_loss += weighted_loss

            # Track weight for logging (exp(-log_var))
            weights[task] = torch.exp(-log_var).item()

        return {
            "total": total_loss,
            "task_losses": task_losses,
            "weights": weights
        }


# Quick test
if __name__ == "__main__":
    config = MultiTaskConfig(
        input_dim=32,
        backbone="tcn",
        task_names=["dos", "fuzzy", "spoofing", "replay"],
        shared_hidden_dim=256,
        task_hidden_dim=128
    )

    model = MultiTaskIDS(config)
    print(f"Model: {model.get_model_info()}")
    print(f"Tasks: {model.get_task_names()}")

    # Test forward pass
    batch_size, seq_len, features = 4, 128, 32
    x = torch.randn(batch_size, seq_len, features)
    outputs = model(x)

    print(f"\nInput: {x.shape}")
    for task, logits in outputs.items():
        print(f"  {task}: {logits.shape}")

    # Test loss
    loss_fn = MultiTaskLoss(config.task_names)
    targets = {task: torch.randint(0, 2, (batch_size,)) for task in config.task_names}
    loss_dict = loss_fn(outputs, targets)
    print(f"\nTotal loss: {loss_dict['total'].item():.4f}")
    print("Task weights:", loss_dict['weights'])
