"""
Simple Model Evaluation Script - TCN Only
==========================================
No dependencies on Mamba or other unused modules.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)

from models.tcn import TemporalCNN, TCNConfig
from training.trainer import IDSTrainer
from data.dataset import CANDataset
from torch.utils.data import DataLoader


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained TCN model"""
    print(f"Loading model from {checkpoint_path}...")

    try:
        # Try loading Lightning module
        model = IDSTrainer.load_from_checkpoint(
            checkpoint_path,
            map_location=device
        )
        print("✅ Loaded as IDSTrainer")
    except Exception as e:
        print(f"Error loading as IDSTrainer: {e}")
        print(f"Loading checkpoint manually...")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Get input dim from checkpoint
        if 'hyper_parameters' in checkpoint:
            input_dim = checkpoint['hyper_parameters'].get('input_dim', 25)
        else:
            input_dim = 25

        # Create TCN model with config
        config = TCNConfig(
            input_dim=input_dim,
            num_channels=[256, 256, 512, 512],
            kernel_size=3,
            num_classes=2,
            dropout=0.1
        )
        model_net = TemporalCNN(config)

        # Load weights
        state_dict = checkpoint.get('state_dict', checkpoint)
        # Remove 'model.' prefix if exists
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('model.', '') if k.startswith('model.') else k
            new_state_dict[new_k] = v

        model_net.load_state_dict(new_state_dict, strict=False)

        # Wrap in IDSTrainer
        model = IDSTrainer(model_net, learning_rate=1e-3)
        print("✅ Loaded checkpoint manually")

    model.eval()
    model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    return model


def run_inference(model, dataloader, device='cuda'):
    """Run inference on test set"""
    model.eval()

    all_preds = []
    all_probs = []
    all_targets = []

    print(f"Running inference on {len(dataloader)} batches...")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 10 == 0:
                print(f"  Batch {i}/{len(dataloader)}")

            features = batch['features'].to(device)
            labels = batch['label'].to(device)

            # Forward
            logits = model(features)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    predictions = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)

    print(f"✅ Generated {len(predictions)} predictions")
    return predictions, probabilities, targets


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all metrics"""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0))
    }

    # ROC/PR curves
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        metrics['auroc'] = float(auc(fpr, tpr))

        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob[:, 1])
        metrics['auprc'] = float(auc(recall_curve, precision_curve))
    else:
        metrics['auroc'] = 0.0
        metrics['auprc'] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics['classification_report'] = report

    return metrics


def plot_confusion_matrix(cm, output_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))

    # Handle different matrix sizes
    if cm.shape == (2, 2):
        labels = ['Normal', 'Attack']
    elif cm.shape == (1, 1):
        labels = ['Single Class']
    else:
        labels = [f'Class {i}' for i in range(cm.shape[0])]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)

    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved confusion matrix: {output_path}")


def plot_roc_curve(y_true, y_prob, output_path):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved ROC curve: {output_path}")


def plot_pr_curve(y_true, y_prob, output_path):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR (AUC = {pr_auc:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved PR curve: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate SecIDS TCN Model')
    parser.add_argument('--model', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to test data')
    parser.add_argument('--output', type=str, default='results', help='Output dir')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Create output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("SecIDS Model Evaluation")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print("="*70)

    # Load model
    model = load_model(args.model, args.device)

    # Load test data
    print(f"\nLoading test data...")
    test_dataset = CANDataset(
        args.data,
        window_size=128,
        stride=128,
        cache=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(args.device == 'cuda')
    )

    print(f"Test set: {len(test_dataset)} samples")

    # Run inference
    print("\n" + "="*70)
    print("Running Inference...")
    print("="*70)
    predictions, probabilities, targets = run_inference(model, test_loader, args.device)

    # Compute metrics
    print("\n" + "="*70)
    print("Computing Metrics...")
    print("="*70)
    metrics = compute_metrics(targets, predictions, probabilities)

    # Print results
    print(f"\n📊 Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  AUPRC:     {metrics['auprc']:.4f}")

    # Save metrics
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Saved metrics: {metrics_file}")

    # Save classification report
    report_file = output_dir / 'classification_report.txt'
    with open(report_file, 'w') as f:
        f.write("SecIDS-v2 Evaluation Report\n")
        f.write("="*70 + "\n\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1']:.4f}\n")
        f.write(f"AUROC:     {metrics['auroc']:.4f}\n")
        f.write(f"AUPRC:     {metrics['auprc']:.4f}\n\n")

        # Class distribution
        f.write("Class Distribution:\n")
        f.write(f"  True Normal:  {(targets == 0).sum()}\n")
        f.write(f"  True Attack:  {(targets == 1).sum()}\n")
        f.write(f"  Pred Normal:  {(predictions == 0).sum()}\n")
        f.write(f"  Pred Attack:  {(predictions == 1).sum()}\n\n")

        f.write("="*70 + "\n")

        # Only generate classification report if we have both classes
        unique_true = np.unique(targets)
        unique_pred = np.unique(predictions)

        if len(unique_true) > 1 or len(unique_pred) > 1:
            # Use labels parameter to handle missing classes
            f.write(classification_report(
                targets, predictions,
                labels=[0, 1],
                target_names=['Normal', 'Attack'],
                zero_division=0
            ))
        else:
            f.write(f"⚠️ Warning: Only single class found in predictions!\n")
            f.write(f"True labels contain: {unique_true}\n")
            f.write(f"Predictions contain: {unique_pred}\n")
            f.write(f"\nThis suggests the test set or model output is imbalanced.\n")

    print(f"✅ Saved report: {report_file}")

    # Generate plots
    print("\n" + "="*70)
    print("Generating Visualizations...")
    print("="*70)

    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, output_dir / 'confusion_matrix.png')

    if metrics['auroc'] > 0:
        plot_roc_curve(targets, probabilities, output_dir / 'roc_curve.png')
        plot_pr_curve(targets, probabilities, output_dir / 'pr_curve.png')

    print("\n" + "="*70)
    print("✅ Evaluation Complete!")
    print("="*70)
    print(f"Results saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
