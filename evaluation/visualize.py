"""
Visualization tools for SecIDS-v2 evaluation

Generates comprehensive evaluation reports with:
- Confusion matrices
- ROC and PR curves
- Training history plots
- Feature importance
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    classification_report
)
import sys
sys.path.append('..')

from data import CANDataset, create_dataloaders
from training.trainer import IDSTrainer
from models import TemporalCNN, TCNConfig


def generate_evaluation_report(
    checkpoint_path: str,
    test_data_path: str,
    output_dir: str = "evaluation/visualizations",
    model_type: str = "tcn"
):
    """
    Generate comprehensive evaluation report with visualizations.

    Args:
        checkpoint_path: Path to trained model checkpoint
        test_data_path: Path to test data (Parquet)
        output_dir: Where to save visualizations
        model_type: "tcn" or "mamba"
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Get hyperparameters from checkpoint
    input_dim = 25  # Our feature dimension

    # Create model
    if model_type == "tcn":
        from models import TCNConfig
        config = TCNConfig(
            input_dim=input_dim,
            num_channels=[256, 256, 512, 512],
            kernel_size=3,
            num_classes=2,
            dropout=0.1
        )
        model = TemporalCNN(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    # Create test dataloader
    print(f"Loading test data from {test_data_path}...")
    test_dataset = CANDataset(
        test_data_path,
        window_size=64,
        stride=64
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    # Run inference
    print("Running inference on test set...")
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch['features']
            labels = batch['label']

            logits = model(features)
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of attack class
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Generate visualizations
    print("Generating visualizations...")

    # 1. Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds, output_dir / "confusion_matrix.png")

    # 2. ROC Curve
    plot_roc_curve(all_labels, all_probs, output_dir / "roc_curve.png")

    # 3. Precision-Recall Curve
    plot_pr_curve(all_labels, all_probs, output_dir / "pr_curve.png")

    # 4. Classification Report
    report = classification_report(all_labels, all_preds, target_names=['Normal', 'Attack'])
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)

    print(f"\nClassification Report:\n{report}")

    # 5. Model Architecture Summary
    save_model_summary(model, output_dir / "model_summary.txt")

    print(f"\nEvaluation complete! Results saved to {output_dir}")

    return {
        'accuracy': (all_preds == all_labels).mean(),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
        'report': report
    }


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Normal', 'Attack'],
        yticklabels=['Normal', 'Attack'],
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - SecIDS-v2 TCN Model', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_roc_curve(y_true, y_probs, save_path):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - SecIDS-v2 TCN Model', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to {save_path}")


def plot_pr_curve(y_true, y_probs, save_path):
    """Plot Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - SecIDS-v2 TCN Model', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curve to {save_path}")


def save_model_summary(model, save_path):
    """Save model architecture summary"""
    info = model.get_model_info()

    summary = f"""
SecIDS-v2 Model Summary
=======================

Architecture: {info['name']}
Parameters: {info['num_parameters']:,}
Trainable Parameters: {info['num_trainable_params']:,}

Configuration:
{'-' * 50}
"""
    for key, value in info['config'].items():
        summary += f"{key:20s}: {value}\n"

    with open(save_path, 'w') as f:
        f.write(summary)

    print(f"Saved model summary to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation report for SecIDS-v2")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to test data")
    parser.add_argument("--output", type=str, default="evaluation/visualizations", help="Output directory")
    parser.add_argument("--model-type", type=str, default="tcn", choices=["tcn", "mamba"])

    args = parser.parse_args()

    generate_evaluation_report(
        checkpoint_path=args.checkpoint,
        test_data_path=args.data,
        output_dir=args.output,
        model_type=args.model_type
    )
