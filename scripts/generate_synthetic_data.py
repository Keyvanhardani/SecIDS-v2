"""
Generate Synthetic CAN-Bus Data for Training

Creates realistic CAN traffic with injected attacks:
- DoS (high frequency)
- Fuzzy (random payloads)
- Spoofing (ID manipulation)
- Replay (repeated patterns)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm


class CANDataGenerator:
    """Generate synthetic CAN-Bus traffic with attacks"""

    def __init__(self, seed=42):
        np.random.seed(seed)

        # Common CAN IDs in automotive networks
        self.normal_can_ids = [
            0x100,  # Engine RPM
            0x200,  # Vehicle Speed
            0x300,  # Steering Angle
            0x400,  # Brake Pressure
            0x500,  # Transmission
            0x600,  # Climate Control
        ]

    def generate_benign_traffic(self, num_frames: int, base_time: float = 0.0):
        """Generate normal CAN traffic"""

        frames = []
        current_time = base_time

        for _ in range(num_frames):
            # Random CAN ID
            can_id = np.random.choice(self.normal_can_ids)

            # Realistic inter-arrival time (exponential distribution)
            # Mean: 10ms (100 fps)
            delta_t = np.random.exponential(0.01)
            current_time += delta_t

            # DLC (always 8 for simplicity)
            dlc = 8

            # Payload: semi-realistic data
            # Engine RPM: bytes 0-1
            # Speed: bytes 2-3
            # Other: bytes 4-7
            if can_id == 0x100:  # Engine RPM
                rpm = np.random.randint(800, 6000)
                data = [
                    (rpm >> 8) & 0xFF,  # High byte
                    rpm & 0xFF,         # Low byte
                    *np.random.randint(0, 256, 6)
                ]
            elif can_id == 0x200:  # Speed
                speed = np.random.randint(0, 200)
                data = [
                    0,
                    speed,
                    *np.random.randint(0, 256, 6)
                ]
            else:
                data = np.random.randint(0, 256, 8).tolist()

            data_hex = ''.join(f'{b:02x}' for b in data)

            frames.append({
                'timestamp': current_time,
                'can_id': can_id,
                'dlc': dlc,
                'data': data_hex,
                'label': 0,  # Benign
                'attack_type': 'benign'
            })

        return frames, current_time

    def generate_dos_attack(self, num_frames: int, base_time: float = 0.0):
        """Generate DoS attack (high frequency flooding)"""

        frames = []
        current_time = base_time

        # DoS: very short inter-arrival time
        dos_id = 0x7FF  # Malicious ID

        for _ in range(num_frames):
            # Very short delta_t (flooding)
            delta_t = np.random.uniform(0.0001, 0.001)  # 0.1-1ms
            current_time += delta_t

            dlc = 8
            data = np.random.randint(0, 256, 8).tolist()
            data_hex = ''.join(f'{b:02x}' for b in data)

            frames.append({
                'timestamp': current_time,
                'can_id': dos_id,
                'dlc': dlc,
                'data': data_hex,
                'label': 1,  # Attack
                'attack_type': 'dos'
            })

        return frames, current_time

    def generate_fuzzy_attack(self, num_frames: int, base_time: float = 0.0):
        """Generate Fuzzy attack (random payloads)"""

        frames = []
        current_time = base_time

        for _ in range(num_frames):
            # Random CAN ID (spoofing)
            can_id = np.random.randint(0, 0x7FF)

            delta_t = np.random.exponential(0.01)
            current_time += delta_t

            dlc = 8
            # Completely random payload (high entropy)
            data = np.random.randint(0, 256, 8).tolist()
            data_hex = ''.join(f'{b:02x}' for b in data)

            frames.append({
                'timestamp': current_time,
                'can_id': can_id,
                'dlc': dlc,
                'data': data_hex,
                'label': 1,
                'attack_type': 'fuzzy'
            })

        return frames, current_time

    def generate_spoofing_attack(self, num_frames: int, base_time: float = 0.0):
        """Generate Spoofing attack (fake ID)"""

        frames = []
        current_time = base_time

        # Spoof a legitimate ID
        spoofed_id = 0x100  # Pretend to be Engine RPM

        for _ in range(num_frames):
            delta_t = np.random.exponential(0.01)
            current_time += delta_t

            dlc = 8
            # Malicious payload
            data = np.random.randint(0, 256, 8).tolist()
            data_hex = ''.join(f'{b:02x}' for b in data)

            frames.append({
                'timestamp': current_time,
                'can_id': spoofed_id,
                'dlc': dlc,
                'data': data_hex,
                'label': 1,
                'attack_type': 'spoofing'
            })

        return frames, current_time

    def generate_replay_attack(self, num_frames: int, base_time: float = 0.0):
        """Generate Replay attack (repeated patterns)"""

        frames = []
        current_time = base_time

        # Create a pattern to replay
        pattern_id = 0x200
        pattern_data = ''.join(f'{b:02x}' for b in np.random.randint(0, 256, 8))

        for _ in range(num_frames):
            delta_t = np.random.exponential(0.01)
            current_time += delta_t

            dlc = 8

            frames.append({
                'timestamp': current_time,
                'can_id': pattern_id,
                'dlc': dlc,
                'data': pattern_data,  # Same data repeated!
                'label': 1,
                'attack_type': 'replay'
            })

        return frames, current_time

    def generate_mixed_dataset(
        self,
        total_frames: int,
        attack_ratio: float = 0.15,
        attack_distribution: dict = None
    ):
        """
        Generate mixed benign + attack traffic

        Args:
            total_frames: Total number of frames
            attack_ratio: Ratio of attack frames (0.15 = 15%)
            attack_distribution: Dict of attack types and their ratios
                                e.g., {'dos': 0.4, 'fuzzy': 0.3, 'spoofing': 0.2, 'replay': 0.1}
        """

        if attack_distribution is None:
            attack_distribution = {
                'dos': 0.4,
                'fuzzy': 0.3,
                'spoofing': 0.2,
                'replay': 0.1
            }

        num_benign = int(total_frames * (1 - attack_ratio))
        num_attack = total_frames - num_benign

        all_frames = []
        current_time = 0.0

        # Generate benign traffic
        print(f"Generating {num_benign:,} benign frames...")
        benign_frames, current_time = self.generate_benign_traffic(num_benign, current_time)
        all_frames.extend(benign_frames)

        # Generate attacks
        for attack_type, ratio in attack_distribution.items():
            num_frames = int(num_attack * ratio)
            print(f"Generating {num_frames:,} {attack_type} frames...")

            if attack_type == 'dos':
                frames, current_time = self.generate_dos_attack(num_frames, current_time)
            elif attack_type == 'fuzzy':
                frames, current_time = self.generate_fuzzy_attack(num_frames, current_time)
            elif attack_type == 'spoofing':
                frames, current_time = self.generate_spoofing_attack(num_frames, current_time)
            elif attack_type == 'replay':
                frames, current_time = self.generate_replay_attack(num_frames, current_time)

            all_frames.extend(frames)

        # Convert to DataFrame
        df = pd.DataFrame(all_frames)

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic CAN data")

    parser.add_argument('--train-size', type=int, default=100000, help="Training set size")
    parser.add_argument('--val-size', type=int, default=20000, help="Validation set size")
    parser.add_argument('--test-size', type=int, default=20000, help="Test set size")
    parser.add_argument('--attack-ratio', type=float, default=0.15, help="Attack ratio (0.15 = 15%)")
    parser.add_argument('--output-dir', type=str, default='data', help="Output directory")
    parser.add_argument('--format', type=str, default='parquet', choices=['parquet', 'csv'])

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = CANDataGenerator(seed=42)

    # Generate datasets
    print("\n" + "="*60)
    print("Generating Synthetic CAN-Bus Data")
    print("="*60 + "\n")

    datasets = {
        'train': args.train_size,
        'val': args.val_size,
        'test': args.test_size
    }

    for split, size in datasets.items():
        print(f"\n{'='*60}")
        print(f"{split.upper()} SET ({size:,} frames)")
        print(f"{'='*60}")

        df = generator.generate_mixed_dataset(
            total_frames=size,
            attack_ratio=args.attack_ratio
        )

        # Statistics
        print(f"\nStatistics:")
        print(f"  Total frames: {len(df):,}")
        print(f"  Benign: {(df['label'] == 0).sum():,} ({(df['label'] == 0).sum() / len(df) * 100:.1f}%)")
        print(f"  Attack: {(df['label'] == 1).sum():,} ({(df['label'] == 1).sum() / len(df) * 100:.1f}%)")
        print(f"\n  Attack breakdown:")
        for attack_type in ['dos', 'fuzzy', 'spoofing', 'replay']:
            count = (df['attack_type'] == attack_type).sum()
            print(f"    {attack_type:10s}: {count:,} ({count / len(df) * 100:.1f}%)")

        # Save
        if args.format == 'parquet':
            output_path = output_dir / f'{split}.parquet'
            df.to_parquet(output_path, index=False)
        else:
            output_path = output_dir / f'{split}.csv'
            df.to_csv(output_path, index=False)

        print(f"\n✓ Saved to {output_path}")

    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE!")
    print("="*60)
    print(f"\nDatasets saved to: {output_dir}/")
    print(f"  - train.{args.format}")
    print(f"  - val.{args.format}")
    print(f"  - test.{args.format}")
    print(f"\nNext steps:")
    print(f"  1. Train model: python -m training.train --data {output_dir}/train.{args.format}")
    print(f"  2. Evaluate: python -m eval.evaluate --test-data {output_dir}/test.{args.format}")


if __name__ == "__main__":
    main()
