"""
CAN-Bus Feature Engineering

Extracts temporal and statistical features from CAN frames:
  - Inter-arrival time (Δt)
  - Payload entropy
  - Hamming distance
  - Per-ID rolling statistics
  - Burst detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import defaultdict
from scipy.stats import entropy


class CANFeatureExtractor:
    """
    Extract advanced features from CAN-Bus frames.

    CAN Frame format:
        - Timestamp (float)
        - CAN ID (0-2047 for standard, 0-536870911 for extended)
        - DLC (Data Length Code, 0-8)
        - Payload (8 bytes)
        - Label (optional, for training)
    """

    def __init__(self, window_size: int = 10, normalize: bool = True):
        """
        Args:
            window_size: Number of frames for rolling statistics
            normalize: Whether to normalize features
        """
        self.window_size = window_size
        self.normalize = normalize

        # Per-ID state (for streaming inference)
        self.id_history: Dict[int, List[bytes]] = defaultdict(list)
        self.id_timestamps: Dict[int, List[float]] = defaultdict(list)

    def extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic features from CAN frames.

        Expected columns: ['timestamp', 'can_id', 'dlc', 'data']
        """
        features = pd.DataFrame(index=df.index)

        # 1. CAN ID (categorical -> one-hot or embedding)
        features['can_id'] = df['can_id']

        # 2. DLC (data length)
        features['dlc'] = df['dlc']

        # 3. Payload bytes (8 bytes)
        if 'data' in df.columns:
            # Assume data is hex string or bytes
            payload_bytes = df['data'].apply(self._parse_payload)
            for i in range(8):
                features[f'byte_{i}'] = payload_bytes.apply(lambda x: x[i] if len(x) > i else 0)

        return features

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract timing-based features.

        Critical for attack detection:
        - DoS: High frequency (low Δt)
        - Replay: Fixed Δt patterns
        - Fuzzy: Random Δt
        """
        features = pd.DataFrame(index=df.index)

        # 1. Inter-arrival time (Δt)
        features['delta_t'] = df['timestamp'].diff().fillna(0)

        # 2. Global statistics
        features['delta_t_rolling_mean'] = features['delta_t'].rolling(
            window=self.window_size, min_periods=1
        ).mean()
        features['delta_t_rolling_std'] = features['delta_t'].rolling(
            window=self.window_size, min_periods=1
        ).std().fillna(0)

        # 3. Per-ID Δt (more specific)
        per_id_delta = df.groupby('can_id')['timestamp'].diff().fillna(0)
        features['delta_t_per_id'] = per_id_delta

        # 4. Burst detection (many frames in short time)
        # Count frames in last 100ms
        features['burst_count'] = 0
        for idx in df.index[1:]:
            current_time = df.loc[idx, 'timestamp']
            recent_mask = (df['timestamp'] >= current_time - 0.1) & (df['timestamp'] < current_time)
            features.loc[idx, 'burst_count'] = recent_mask.sum()

        # 5. Bus load (frames per second)
        features['bus_load'] = features['burst_count'] * 10  # Approximate

        return features

    def extract_payload_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract statistical features from payload data.

        Detects:
        - Fuzzy attacks: High entropy (random bytes)
        - Normal traffic: Low entropy (structured data)
        """
        features = pd.DataFrame(index=df.index)

        if 'data' not in df.columns:
            return features

        # Parse payload
        payload_bytes = df['data'].apply(self._parse_payload)

        # 1. Entropy (randomness)
        features['payload_entropy'] = payload_bytes.apply(self._compute_entropy)

        # 2. Hamming distance to previous frame (same ID)
        features['hamming_distance'] = 0.0
        for can_id in df['can_id'].unique():
            id_mask = df['can_id'] == can_id
            id_payloads = payload_bytes[id_mask]

            if len(id_payloads) > 1:
                hamming_dists = [0] + [
                    self._hamming_distance(id_payloads.iloc[i-1], id_payloads.iloc[i])
                    for i in range(1, len(id_payloads))
                ]
                features.loc[id_mask, 'hamming_distance'] = hamming_dists

        # 3. Byte diversity (unique bytes in payload)
        features['byte_diversity'] = payload_bytes.apply(lambda x: len(set(x)) / max(len(x), 1))

        # 4. Zero-byte ratio (padding detection)
        features['zero_ratio'] = payload_bytes.apply(lambda x: x.count(0) / max(len(x), 1))

        return features

    def extract_id_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Per-ID statistical features.

        Detects ID-spoofing and replay attacks.
        """
        features = pd.DataFrame(index=df.index)

        # 1. Frame count per ID (rolling)
        features['id_frame_count'] = df.groupby('can_id').cumcount()

        # 2. Time since last frame from this ID
        features['time_since_last_id'] = df.groupby('can_id')['timestamp'].diff().fillna(0)

        # 3. Mean Δt for this ID (deviation = anomaly)
        id_delta_mean = df.groupby('can_id')['timestamp'].diff().rolling(
            window=self.window_size, min_periods=1
        ).mean()
        features['id_delta_mean'] = id_delta_mean.reset_index(level=0, drop=True)

        # 4. ID frequency rank (rare IDs = suspicious)
        id_counts = df['can_id'].value_counts()
        features['id_frequency'] = df['can_id'].map(id_counts)
        features['id_frequency_rank'] = df['can_id'].map(id_counts.rank(ascending=False))

        return features

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features at once.

        Returns:
            Feature matrix with columns:
            - Basic: can_id, dlc, byte_0..byte_7
            - Temporal: delta_t, delta_t_rolling_*, burst_count, bus_load
            - Payload: entropy, hamming_distance, byte_diversity
            - ID Stats: id_frame_count, time_since_last_id, id_delta_mean
        """
        all_features = []

        all_features.append(self.extract_basic_features(df))
        all_features.append(self.extract_temporal_features(df))
        all_features.append(self.extract_payload_features(df))
        all_features.append(self.extract_id_statistics(df))

        # Combine
        features = pd.concat(all_features, axis=1)

        # Handle NaN/Inf
        features = features.replace([np.inf, -np.inf], 0).fillna(0)

        return features

    # Helper functions
    def _parse_payload(self, data) -> List[int]:
        """Parse payload to list of bytes"""
        if isinstance(data, str):
            # Hex string "0102030405060708"
            try:
                return [int(data[i:i+2], 16) for i in range(0, len(data), 2)]
            except:
                return [0] * 8
        elif isinstance(data, bytes):
            return list(data)
        elif isinstance(data, list):
            return data
        else:
            return [0] * 8

    def _compute_entropy(self, payload: List[int]) -> float:
        """Shannon entropy of payload bytes"""
        if not payload or len(payload) == 0:
            return 0.0

        # Count byte frequencies
        counts = np.bincount(payload, minlength=256)
        probs = counts / counts.sum()
        probs = probs[probs > 0]  # Remove zeros

        return entropy(probs, base=2)

    def _hamming_distance(self, payload1: List[int], payload2: List[int]) -> int:
        """Hamming distance between two payloads (bit-level)"""
        if len(payload1) != len(payload2):
            # Pad shorter one
            max_len = max(len(payload1), len(payload2))
            payload1 = payload1 + [0] * (max_len - len(payload1))
            payload2 = payload2 + [0] * (max_len - len(payload2))

        # XOR and count bits
        distance = 0
        for b1, b2 in zip(payload1, payload2):
            xor = b1 ^ b2
            distance += bin(xor).count('1')

        return distance


# Example usage
if __name__ == "__main__":
    # Synthetic CAN data
    np.random.seed(42)

    data = {
        'timestamp': np.cumsum(np.random.exponential(0.01, 1000)),  # ~100 fps
        'can_id': np.random.choice([0x100, 0x200, 0x300, 0x400], 1000),
        'dlc': np.random.choice([8], 1000),
        'data': [''.join(f'{b:02x}' for b in np.random.randint(0, 256, 8)) for _ in range(1000)]
    }

    df = pd.DataFrame(data)

    # Extract features
    extractor = CANFeatureExtractor(window_size=10)
    features = extractor.extract_all_features(df)

    print(f"Input: {df.shape}")
    print(f"Features: {features.shape}")
    print(f"Columns: {features.columns.tolist()}")
    print(f"\nSample:\n{features.head()}")
