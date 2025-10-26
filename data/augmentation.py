"""
CAN-Bus Data Augmentation

Training-time augmentation for improved robustness:
  - Byte-level bit flips (fuzzy simulation)
  - ID spoofing patterns
  - Temporal jitter
  - Payload masking
"""

import numpy as np
import torch
from typing import Dict, Tuple


class CANAugmentation:
    """
    Augmentation for CAN-Bus data during training.

    Critical: Preserve timing statistics (don't break domain)
    """

    def __init__(
        self,
        bit_flip_prob: float = 0.01,
        id_spoof_prob: float = 0.05,
        temporal_jitter_std: float = 0.001,
        payload_mask_prob: float = 0.1,
        apply_prob: float = 0.5
    ):
        """
        Args:
            bit_flip_prob: Probability of flipping each bit
            id_spoof_prob: Probability of spoofing CAN ID
            temporal_jitter_std: Std dev for timing noise (seconds)
            payload_mask_prob: Probability of masking payload bytes
            apply_prob: Probability of applying augmentation
        """
        self.bit_flip_prob = bit_flip_prob
        self.id_spoof_prob = id_spoof_prob
        self.temporal_jitter_std = temporal_jitter_std
        self.payload_mask_prob = payload_mask_prob
        self.apply_prob = apply_prob

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply augmentation to a sample.

        Args:
            sample: {
                'features': [seq_len, num_features],
                'label': scalar
            }

        Returns:
            Augmented sample (same structure)
        """
        if np.random.rand() > self.apply_prob:
            return sample  # No augmentation

        features = sample['features'].clone()
        seq_len, num_features = features.shape

        # Apply different augmentations
        features = self._byte_bit_flip(features, num_features)
        features = self._temporal_jitter(features, num_features)
        features = self._payload_masking(features, num_features)

        sample['features'] = features
        return sample

    def _byte_bit_flip(self, features: torch.Tensor, num_features: int) -> torch.Tensor:
        """
        Flip random bits in payload bytes.

        Simulates:
          - Fuzzy attacks (random corruption)
          - Transmission errors
        """
        # Assume features 2-9 are payload bytes (byte_0 to byte_7)
        byte_start_idx = 2  # After can_id, dlc
        byte_end_idx = min(10, num_features)

        if byte_end_idx <= byte_start_idx:
            return features

        # Random bit flips
        for i in range(features.shape[0]):
            for j in range(byte_start_idx, byte_end_idx):
                if np.random.rand() < self.bit_flip_prob:
                    # Flip a random bit
                    byte_val = int(features[i, j].item())
                    bit_pos = np.random.randint(0, 8)
                    byte_val ^= (1 << bit_pos)
                    features[i, j] = float(byte_val)

        return features

    def _temporal_jitter(self, features: torch.Tensor, num_features: int) -> torch.Tensor:
        """
        Add noise to timing features.

        Simulates:
          - Variable bus load
          - Measurement noise
        """
        # Find timing features (delta_t, delta_t_rolling_*, etc.)
        # Assume they're after basic features
        timing_start_idx = 10
        timing_end_idx = min(20, num_features)

        if timing_end_idx <= timing_start_idx:
            return features

        # Add Gaussian noise
        noise = torch.randn(features.shape[0], timing_end_idx - timing_start_idx)
        noise *= self.temporal_jitter_std

        features[:, timing_start_idx:timing_end_idx] += noise

        # Ensure non-negative (delta_t can't be negative)
        features[:, timing_start_idx:timing_end_idx] = torch.clamp(
            features[:, timing_start_idx:timing_end_idx],
            min=0
        )

        return features

    def _payload_masking(self, features: torch.Tensor, num_features: int) -> torch.Tensor:
        """
        Randomly mask (zero out) payload bytes.

        Forces model to not rely on single bytes.
        """
        byte_start_idx = 2
        byte_end_idx = min(10, num_features)

        if byte_end_idx <= byte_start_idx:
            return features

        # Random masking
        for i in range(features.shape[0]):
            for j in range(byte_start_idx, byte_end_idx):
                if np.random.rand() < self.payload_mask_prob:
                    features[i, j] = 0

        return features


class MixUp:
    """
    MixUp augmentation for time series.

    Mixes two samples linearly to create synthetic data.
    Reference: "mixup: Beyond Empirical Risk Minimization" (Zhang et al.)
    """

    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter (lower = less mixing)
        """
        self.alpha = alpha

    def __call__(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Apply MixUp to a batch.

        Args:
            batch: {
                'features': [batch, seq_len, features],
                'label': [batch]
            }

        Returns:
            mixed_batch, lambda (mixing ratio)
        """
        batch_size = batch['features'].shape[0]

        # Sample mixing ratio
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Random permutation
        index = torch.randperm(batch_size)

        # Mix features
        mixed_features = lam * batch['features'] + (1 - lam) * batch['features'][index]

        # Mixed labels (for soft labels)
        if 'label' in batch:
            mixed_labels = lam * batch['label'] + (1 - lam) * batch['label'][index]
        else:
            mixed_labels = None

        mixed_batch = {
            'features': mixed_features,
            'label': mixed_labels
        }

        return mixed_batch, lam


# Test
if __name__ == "__main__":
    # Synthetic sample
    sample = {
        'features': torch.randn(128, 32),  # [seq_len, features]
        'label': torch.tensor(1)
    }

    print(f"Original features range: [{sample['features'].min():.3f}, {sample['features'].max():.3f}]")

    # Apply augmentation
    aug = CANAugmentation(
        bit_flip_prob=0.05,
        temporal_jitter_std=0.01,
        apply_prob=1.0  # Always apply for testing
    )

    augmented = aug(sample)
    print(f"Augmented features range: [{augmented['features'].min():.3f}, {augmented['features'].max():.3f}]")

    # Test MixUp
    batch = {
        'features': torch.randn(4, 128, 32),
        'label': torch.tensor([0, 1, 0, 1])
    }

    mixup = MixUp(alpha=0.2)
    mixed_batch, lam = mixup(batch)

    print(f"\nMixUp lambda: {lam:.3f}")
    print(f"Original labels: {batch['label']}")
    print(f"Mixed labels: {mixed_batch['label']}")
