"""
Model Evaluation Script
========================

Evaluates trained SecIDS model on test set and generates:
- Confusion matrix
- ROC/PR curves
- Per-class metrics
- Latency profiling
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

from models.tcn import TemporalCNN
from models.mamba_ids import MambaIDS
from training.lightning_module import IDSLightningModule
from data.dataset import CANDataset
from torch.utils.data import DataLoader


def load_model(checkpoint_path: str, model_type: str = 'tcn') -> IDSLightningModule:
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")

    try:
        # Try loading with Lightning module
        model = IDSLightningModule.load_from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"Failed to load as LightningModule: {e}")
        print("Trying to load model directly...")

        # Load checkpoint manually
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Get model config
        if 'hyper_parameters' in checkpoint:
            config = checkpoint['hyper_parameters']
            input_dim = config.get('input_dim', 25)
        else:
            input_dim = 25

        # Create model
        if model_type == 'tcn':
            model_net = TemporalCNN(input_dim=input_dim)
        elif model_type == 'mamba':
            model_net = MambaIDS(input_dim=input_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if exists
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model_net.load_state_dict(state_dict)
        else:
            model_net.load_state_dict(checkpoint)

        # Wrap in Lightning module
        model = IDSLightningModule(model_net, lr=1e-3)

    model.eval()
    print(f"Model loaded successfully!")
    return model


def predict(model: IDSLightningModule, dataloader: DataLoader, device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on dataset

    Returns:
        predictions: [N] predicted class labels
        probabilities: [N, num_classes] class probabilities
        targets: [N] ground truth labels
    """
    model = model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    all_targets = []

    print(f"Running inference on {len(dataloader)} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}")

            # Get data
            features = batch['features'].to(device)  # [B, 128, 25]
            labels = batch['label'].to(device)  # [B]

            # Forward pass
            logits = model(features)  # [B, 2]
            probs = torch.softmax(logits, dim=-1)  # [B, 2]
            preds = torch.argmax(logits, dim=-1)  # [B]

            # Collect results
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    # Concatenate
    predictions = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)

    print(f"Inference complete! Generated {len(predictions)} predictions")
    return predictions, probabilities, targets


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Compute all evaluation metrics"""

    metrics = {}

    # Basic metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, average='binary', zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, average='binary', zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, average='binary', zero_division=0))

    # ROC AUC
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        metrics['auroc'] = float(auc(fpr, tpr))

        # PR AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob[:, 1])
        metrics['auprc'] = float(auc(recall_curve, precision_curve))
    else:
        metrics['auroc'] = 0.0
        metrics['auprc'] = 0.0

    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics['per_class'] = report

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # Class distribution
    metrics['class_distribution'] = {
        'true': {
            'normal': int((y_true == 0).sum()),
            'attack': int((y_true == 1).sum())
        },
        'predicted': {
            'normal': int((y_pred == 0).sum()),
            'attack': int((y_pred == 1).sum())
        }
    }

    return metrics


def plot_confusion_matrix(cm: np.ndarray, output_path: Path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])

    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to {output_path}")


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curve to {output_path}")


def profile_latency(model: IDSLightningModule, dataloader: DataLoader, device: str = 'cuda', num_runs: int = 100) -> Dict:
    """Profile inference latency"""
    model = model.to(device)
    model.eval()

    print(f"\nProfiling latency over {num_runs} runs...")

    # Warmup
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
            features = batch['features'].to(device)
            _ = model(features)

    # Profile
    latencies = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_runs:
                break

            features = batch['features'].to(device)

            # Time inference
            if device == 'cuda':
                torch.cuda.synchronize()

            start = time.time()
            _ = model(features)

            if device == 'cuda':
                torch.cuda.synchronize()

            end = time.time()

            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)

    latencies = np.array(latencies)

    latency_stats = {
        'mean_ms': float(latencies.mean()),
        'std_ms': float(latencies.std()),
        'min_ms': float(latencies.min()),
        'max_ms': float(latencies.max()),
        'median_ms': float(np.median(latencies)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'throughput_fps': float(1000 / latencies.mean())
    }

    print(f"  Mean latency: {latency_stats['mean_ms']:.2f} ms")
    print(f"  Std latency: {latency_stats['std_ms']:.2f} ms")
    print(f"  Throughput: {latency_stats['throughput_fps']:.1f} FPS")

    return latency_stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate SecIDS model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to test data (parquet)')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--model-type', type=str, default='tcn', choices=['tcn', 'mamba'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--workers', type=int, default=0, help='DataLoader workers')
    parser.add_argument('--profile', action='store_true', help='Profile latency')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SecIDS Model Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Load model
    model = load_model(args.model, args.model_type)

    # Load test data
    print(f"\nLoading test data from {args.data}...")
    test_dataset = CANDataset(
        args.data,
        window_size=128,
        stride=128,  # No overlap for testing
        cache=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True if args.device == 'cuda' else False
    )

    print(f"Test set: {len(test_dataset)} samples, {len(test_loader)} batches")

    # Run inference
    print("\n" + "=" * 60)
    print("Running Inference...")
    print("=" * 60)
    predictions, probabilities, targets = predict(model, test_loader, args.device)

    # Compute metrics
    print("\n" + "=" * 60)
    print("Computing Metrics...")
    print("=" * 60)
    metrics = compute_metrics(targets, predictions, probabilities)

    # Print results
    print(f"\nResults:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  AUPRC:     {metrics['auprc']:.4f}")

    print(f"\nClass Distribution:")
    print(f"  True Normal:  {metrics['class_distribution']['true']['normal']}")
    print(f"  True Attack:  {metrics['class_distribution']['true']['attack']}")
    print(f"  Pred Normal:  {metrics['class_distribution']['predicted']['normal']}")
    print(f"  Pred Attack:  {metrics['class_distribution']['predicted']['attack']}")

    # Save metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, output_dir / 'confusion_matrix.png')

    # Plot ROC curve
    if metrics['auroc'] > 0:
        plot_roc_curve(targets, probabilities, output_dir / 'roc_curve.png')
        plot_precision_recall_curve(targets, probabilities, output_dir / 'pr_curve.png')

    # Profile latency
    if args.profile:
        print("\n" + "=" * 60)
        print("Profiling Latency...")
        print("=" * 60)
        latency_stats = profile_latency(model, test_loader, args.device)

        # Save latency stats
        latency_path = output_dir / 'latency.json'
        with open(latency_path, 'w') as f:
            json.dump(latency_stats, f, indent=2)
        print(f"\nSaved latency stats to {latency_path}")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
