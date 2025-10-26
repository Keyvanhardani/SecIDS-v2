"""
Training script for SecIDS-v2 models

Usage:
    python -m training.train --config configs/tcn_fuzzy.yaml
"""

import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pathlib import Path
import sys
sys.path.append('..')

from models import TemporalCNN, TCNConfig, MambaIDS, MambaConfig, MultiTaskIDS, MultiTaskConfig
from data import CANDataset, MultiTaskCANDataset, create_dataloaders
from training.trainer import IDSTrainer, MultiTaskIDSTrainer


def train_model(
    model_type: str = "tcn",
    data_path: str = "data/train.parquet",
    val_path: str = "data/val.parquet",
    output_dir: str = "outputs",
    batch_size: int = 32,
    max_epochs: int = 100,
    learning_rate: float = 1e-3,
    window_size: int = 128,
    num_workers: int = 4,
    use_wandb: bool = False,
    multitask: bool = False,
    task_names: list = None,
    **kwargs
):
    """
    Train an IDS model.

    Args:
        model_type: "tcn" or "mamba"
        data_path: Path to training data (Parquet)
        val_path: Path to validation data
        output_dir: Where to save checkpoints
        batch_size: Batch size
        max_epochs: Maximum training epochs
        learning_rate: Initial learning rate
        window_size: Sequence length
        num_workers: DataLoader workers
        use_wandb: Enable WandB logging
        multitask: Use multi-task learning
        task_names: List of task names for multi-task
    """

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders first to get actual feature count
    print(f"Loading data from {data_path}...")

    train_loader, val_loader = create_dataloaders(
        train_path=data_path,
        val_path=val_path,
        batch_size=batch_size,
        window_size=window_size,
        num_workers=num_workers,
        multitask=multitask,
        task_columns={task: f"label_{task}" for task in task_names} if multitask else None
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader) if val_loader else 0}")

    # Get actual input dimension from first batch
    first_batch = next(iter(train_loader))
    actual_input_dim = first_batch['features'].shape[-1]
    print(f"Detected input dimension: {actual_input_dim}")

    # Build model
    print(f"Building {model_type} model...")

    if multitask:
        if task_names is None:
            task_names = ["dos", "fuzzy", "spoofing", "replay"]

        config = MultiTaskConfig(
            input_dim=actual_input_dim,
            backbone=model_type,
            task_names=task_names
        )
        model = MultiTaskIDS(config)

    else:
        if model_type == "tcn":
            config = TCNConfig(
                input_dim=actual_input_dim,
                num_channels=[256, 256, 512, 512],
                kernel_size=3,
                num_classes=2,
                dropout=0.1
            )
            model = TemporalCNN(config)

        elif model_type == "mamba":
            config = MambaConfig(
                input_dim=actual_input_dim,
                d_model=256,
                d_state=16,
                num_layers=4,
                num_classes=2,
                dropout=0.1
            )
            model = MambaIDS(config)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    print(f"Model info: {model.get_model_info()}")

    # Lightning Module
    if multitask:
        lightning_model = MultiTaskIDSTrainer(
            model=model,
            task_names=task_names,
            learning_rate=learning_rate,
            max_epochs=max_epochs
        )
    else:
        lightning_model = IDSTrainer(
            model=model,
            learning_rate=learning_rate,
            max_epochs=max_epochs
        )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="best-{epoch:02d}-{val/f1:.4f}",
            monitor="val/f1" if not multitask else "val/f1_dos",
            mode="max",
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor="val/loss" if not multitask else "val/loss_total",
            patience=10,
            mode="min"
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]

    # Logger
    if use_wandb:
        logger = WandbLogger(
            project="SecIDS-v2",
            name=f"{model_type}_{multitask and 'multitask' or 'single'}",
            save_dir=output_dir
        )
    else:
        logger = TensorBoardLogger(
            save_dir=output_dir,
            name="tensorboard_logs"
        )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices=1,
        precision="16-mixed",  # Mixed precision for speed
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        deterministic=False  # Set True for reproducibility (slower)
    )

    # Train
    print("\nStarting training...")
    trainer.fit(lightning_model, train_loader, val_loader)

    # Save final model
    final_path = output_dir / "final_model.ckpt"
    trainer.save_checkpoint(final_path)
    print(f"\nTraining complete! Model saved to {final_path}")

    return lightning_model, trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SecIDS-v2 model")

    parser.add_argument("--model", type=str, default="tcn", choices=["tcn", "mamba"])
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val", type=str, default=None, help="Path to validation data")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wandb", action="store_true", help="Use WandB logging")
    parser.add_argument("--multitask", action="store_true", help="Multi-task learning")
    parser.add_argument("--tasks", nargs="+", default=["dos", "fuzzy", "spoofing", "replay"])

    args = parser.parse_args()

    train_model(
        model_type=args.model,
        data_path=args.data,
        val_path=args.val,
        output_dir=args.output,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        window_size=args.window_size,
        num_workers=args.workers,
        use_wandb=args.wandb,
        multitask=args.multitask,
        task_names=args.tasks
    )
