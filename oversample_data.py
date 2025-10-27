"""
Oversample Attack Class for Balanced Training
==============================================

Creates a new training dataset with more Attack samples.
"""

import pandas as pd
from sklearn.utils import resample
import argparse


def oversample_attacks(
    input_path: str,
    output_path: str,
    attack_ratio: float = 0.20,
    seed: int = 42
):
    """
    Oversample attack class to achieve desired ratio.

    Args:
        input_path: Path to original parquet file
        output_path: Path to save oversampled data
        attack_ratio: Target ratio for attack class (0.0-1.0)
        seed: Random seed
    """

    print("="*70)
    print("Attack Class Oversampling")
    print("="*70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Target Attack Ratio: {attack_ratio*100:.1f}%")
    print("="*70)

    # Load data
    print(f"\n📦 Loading data...")
    df = pd.read_parquet(input_path)

    # Split by class
    normal = df[df['label'] == 0].copy()
    attack = df[df['label'] == 1].copy()

    print(f"\n📊 Original Distribution:")
    print(f"  Normal: {len(normal):,} ({len(normal)/len(df)*100:.2f}%)")
    print(f"  Attack: {len(attack):,} ({len(attack)/len(df)*100:.2f}%)")

    # Calculate target attack count
    # If we want X% attacks, and we have N normals:
    # N / (1 - X) * X = target_attacks
    target_attack_count = int(len(normal) * attack_ratio / (1 - attack_ratio))

    print(f"\n🎯 Target Attack Count: {target_attack_count:,}")

    # Oversample attack class
    if target_attack_count > len(attack):
        print(f"  → Oversampling by {target_attack_count/len(attack):.1f}x")
        attack_oversampled = resample(
            attack,
            replace=True,  # Sample with replacement
            n_samples=target_attack_count,
            random_state=seed
        )
    else:
        print(f"  → Undersampling (already have enough attacks)")
        attack_oversampled = attack.sample(n=target_attack_count, random_state=seed)

    # Combine
    df_balanced = pd.concat([normal, attack_oversampled], ignore_index=True)

    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\n📊 New Distribution:")
    print(f"  Normal: {len(normal):,} ({len(normal)/len(df_balanced)*100:.2f}%)")
    print(f"  Attack: {len(attack_oversampled):,} ({len(attack_oversampled)/len(df_balanced)*100:.2f}%)")
    print(f"  Total: {len(df_balanced):,}")

    # Verify ratio
    actual_ratio = len(attack_oversampled) / len(df_balanced)
    print(f"\n✅ Actual Attack Ratio: {actual_ratio*100:.2f}% (target: {attack_ratio*100:.1f}%)")

    # Save
    df_balanced.to_parquet(output_path, index=False)
    print(f"\n✅ Saved to: {output_path}")

    return df_balanced


def main():
    parser = argparse.ArgumentParser(description="Oversample Attack Class")

    parser.add_argument("--input", type=str, required=True, help="Input parquet file")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file")
    parser.add_argument("--attack-ratio", type=float, default=0.20,
                        help="Target attack ratio (0.0-1.0, default 0.20 = 20%%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.attack_ratio <= 0 or args.attack_ratio >= 1:
        raise ValueError("attack-ratio must be between 0 and 1")

    oversample_attacks(
        input_path=args.input,
        output_path=args.output,
        attack_ratio=args.attack_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
