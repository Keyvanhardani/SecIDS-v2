"""
Balanced Training Script for SecIDS-v2
========================================

Fixes class imbalance with weighted loss function.
"""

import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pathlib import Path
import pandas as pd
import numpy as np

from models.tcn import TemporalCNN, TCNConfig
from training.trainer import IDSTrainer
from data.dataset import create_dataloaders


def calculate_class_weights(data_path: str, balance_factor: float = 10.0):
    """
    Calculate class weights for imbalanced dataset.

    Args:
        data_path: Path to training data
        balance_factor: How much to weight minority class (default 10x)

    Returns:
        torch.Tensor: Class weights [weight_normal, weight_attack]
    """
    print(f"\n📊 Calculating class weights from {data_path}...")

    df = pd.read_parquet(data_path)

    # Count classes
    class_counts = df['label'].value_counts().sort_index()
    total = len(df)

    print(f"\nClass Distribution:")
    print(f"  Normal (0): {class_counts[0]:,} ({class_counts[0]/total*100:.2f}%)")
    print(f"  Attack (1): {class_counts[1]:,} ({class_counts[1]/total*100:.2f}%)")

    # Calculate weights - simple approach
    # Normal class gets weight 1.0, Attack class gets balance_factor
    weights = torch.FloatTensor([
        1.0,
        balance_factor
    ])

    print(f"\n⚖️ Class Weights (normalized):")
    print(f"  Normal: {weights[0]:.2f}x")
    print(f"  Attack: {weights[1]:.2f}x")
    print(f"\nThis means Attack samples are {weights[1]:.1f}x more important during training!")

    return weights


def train_balanced_model(
    data_path: str,
    val_path: str,
    output_dir: str,
    batch_size: int = 32,
    max_epochs: int = 100,
    learning_rate: float = 1e-3,
    window_size: int = 128,
    num_workers: int = 0,
    balance_factor: float = 10.0,
    gpu: int = 0
):
    """
    Train IDS model with balanced class weights.
    """

    print("="*70)
    print("SecIDS-v2 Balanced Training")
    print("="*70)
    print(f"Data: {data_path}")
    print(f"Val: {val_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {max_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Balance factor: {balance_factor}x")
    print("="*70)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate class weights
    class_weights = calculate_class_weights(data_path, balance_factor)

    # Create dataloaders
    print(f"\n📦 Loading data...")
    train_loader, val_loader = create_dataloaders(
        train_path=data_path,
        val_path=val_path,
        batch_size=batch_size,
        window_size=window_size,
        num_workers=num_workers,
        multitask=False
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Get input dimension from first batch
    first_batch = next(iter(train_loader))
    input_dim = first_batch['features'].shape[-1]
    print(f"Input dimension: {input_dim}")

    # Build TCN model
    print(f"\n🧠 Building Temporal CNN model...")
    config = TCNConfig(
        input_dim=input_dim,
        num_channels=[256, 256, 512, 512],
        kernel_size=3,
        num_classes=2,
        dropout=0.1
    )
    model = TemporalCNN(config)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create Lightning Module with class weights
    lightning_model = IDSTrainer(
        model=model,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        class_weights=class_weights  # ✅ Use balanced weights!
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best-epoch{epoch:02d}-f1{val/f1:.4f}",
            monitor="val/f1",
            mode="max",
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor="val/f1",  # Monitor F1 instead of loss!
            patience=15,
            mode="max",
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices=[gpu] if torch.cuda.is_available() else "auto",
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Train
    print("\n🚀 Starting balanced training...")
    print("="*70)
    trainer.fit(lightning_model, train_loader, val_loader)

    # Save final model
    final_path = output_dir / "final_model.ckpt"
    trainer.save_checkpoint(final_path)

    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"Model saved to: {final_path}")
    print(f"\nBest metrics:")
    print(f"  Best F1: {trainer.callback_metrics.get('val/f1', 0):.4f}")
    print(f"  Best Acc: {trainer.callback_metrics.get('val/acc', 0):.4f}")

    # Training summary
    print(f"\n📊 Training Summary:")
    print(f"  Epochs completed: {trainer.current_epoch + 1}")
    print(f"  Class weights used: Normal={class_weights[0]:.2f}, Attack={class_weights[1]:.2f}")

    return lightning_model, trainer


def main():
    parser = argparse.ArgumentParser(description="Balanced Training for SecIDS-v2")

    parser.add_argument("--data", type=str, required=True, help="Training data path")
    parser.add_argument("--val", type=str, required=True, help="Validation data path")
    parser.add_argument("--output", type=str, default="outputs/tcn_balanced", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--balance-factor", type=float, default=10.0,
                        help="How much to weight minority class (default 10x)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")

    args = parser.parse_args()

    train_balanced_model(
        data_path=args.data,
        val_path=args.val,
        output_dir=args.output,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        window_size=args.window_size,
        num_workers=args.workers,
        balance_factor=args.balance_factor,
        gpu=args.gpu
    )


if __name__ == "__main__":
    main()
