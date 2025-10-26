"""
Model Evaluation CLI

Compare multiple models on standard metrics:
  - Accuracy, Precision, Recall, F1
  - ROC-AUC, PR-AUC
  - Latency (ms/frame)
  - Memory footprint
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import time
from rich.console import Console
from rich.table import Table
import sys
sys.path.append('..')

from training.trainer import IDSTrainer, MultiTaskIDSTrainer
from data import CANDataset, create_dataloaders
from eval.metrics import compute_metrics


console = Console()


def evaluate_model(
    checkpoint_path: str,
    test_data_path: str,
    output_dir: Optional[str] = None,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    """
    Evaluate a single model on test data.

    Args:
        checkpoint_path: Path to model checkpoint
        test_data_path: Path to test data
        output_dir: Where to save results (optional)
        batch_size: Batch size
        device: Device for inference

    Returns:
        Dict of metrics
    """

    console.print(f"\n[bold cyan]Evaluating: {checkpoint_path}[/bold cyan]")

    # Load model
    try:
        lightning_model = IDSTrainer.load_from_checkpoint(checkpoint_path)
        multitask = False
    except:
        lightning_model = MultiTaskIDSTrainer.load_from_checkpoint(checkpoint_path)
        multitask = True

    model = lightning_model.model.to(device)
    model.eval()

    # Load test data
    _, test_loader = create_dataloaders(
        train_path=test_data_path,  # Dummy, not used
        val_path=test_data_path,
        batch_size=batch_size,
        multitask=multitask
    )

    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []
    latencies = []

    console.print(f"Running inference on {len(test_loader)} batches...")

    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            labels = batch['label'] if not multitask else batch['labels']

            # Measure latency
            start = time.perf_counter()
            outputs = model(features)
            end = time.perf_counter()
            latencies.append((end - start) * 1000 / features.size(0))  # ms per sample

            # Get predictions
            if multitask:
                # Just evaluate first task for now
                task = list(outputs.keys())[0]
                logits = outputs[task]
                labels = labels[task]
            else:
                logits = outputs

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs[:, 1].cpu().numpy())  # Prob of attack class

    # Concatenate
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)

    # Add latency
    metrics['latency_mean_ms'] = np.mean(latencies)
    metrics['latency_std_ms'] = np.std(latencies)
    metrics['latency_p50_ms'] = np.percentile(latencies, 50)
    metrics['latency_p95_ms'] = np.percentile(latencies, 95)

    # Model size
    checkpoint_size_mb = Path(checkpoint_path).stat().st_size / (1024 * 1024)
    metrics['model_size_mb'] = checkpoint_size_mb

    # Print results
    _print_metrics_table(metrics)

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)

        # Save predictions
        results_df = pd.DataFrame({
            'label': all_labels,
            'prediction': all_preds,
            'probability': all_probs
        })
        results_df.to_csv(output_dir / "predictions.csv", index=False)

        console.print(f"\n✓ Results saved to {output_dir}")

    return metrics


def benchmark_models(
    checkpoints: Dict[str, str],
    test_data_path: str,
    output_path: str = "benchmark_results.csv"
):
    """
    Benchmark multiple models and compare.

    Args:
        checkpoints: Dict[model_name -> checkpoint_path]
        test_data_path: Path to test data
        output_path: Where to save comparison table
    """

    console.print("\n[bold cyan]Benchmarking Models[/bold cyan]\n")

    results = []

    for model_name, checkpoint_path in checkpoints.items():
        console.print(f"\n{'='*60}")
        console.print(f"Model: {model_name}")
        console.print(f"{'='*60}")

        try:
            metrics = evaluate_model(checkpoint_path, test_data_path)
            metrics['model_name'] = model_name
            results.append(metrics)
        except Exception as e:
            console.print(f"[red]Error evaluating {model_name}: {e}[/red]")

    # Create comparison table
    df = pd.DataFrame(results)

    # Reorder columns
    cols = ['model_name', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
            'latency_p50_ms', 'latency_p95_ms', 'model_size_mb']
    df = df[[c for c in cols if c in df.columns]]

    # Save
    df.to_csv(output_path, index=False)

    # Print comparison
    _print_comparison_table(df)

    console.print(f"\n✓ Benchmark results saved to {output_path}")

    return df


def _print_metrics_table(metrics: Dict[str, float]):
    """Print metrics in a formatted table"""

    table = Table(title="Evaluation Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    # Classification metrics
    table.add_row("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
    table.add_row("Precision", f"{metrics.get('precision', 0):.4f}")
    table.add_row("Recall", f"{metrics.get('recall', 0):.4f}")
    table.add_row("F1-Score", f"{metrics.get('f1', 0):.4f}")
    table.add_row("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
    table.add_row("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}")

    table.add_row("", "")  # Separator

    # Latency
    table.add_row("Latency (mean)", f"{metrics.get('latency_mean_ms', 0):.2f} ms")
    table.add_row("Latency (p50)", f"{metrics.get('latency_p50_ms', 0):.2f} ms")
    table.add_row("Latency (p95)", f"{metrics.get('latency_p95_ms', 0):.2f} ms")

    table.add_row("", "")

    # Model size
    table.add_row("Model Size", f"{metrics.get('model_size_mb', 0):.2f} MB")

    console.print(table)


def _print_comparison_table(df: pd.DataFrame):
    """Print model comparison table"""

    table = Table(title="Model Comparison", show_header=True)

    # Add columns
    for col in df.columns:
        table.add_column(col, justify="right" if col != "model_name" else "left")

    # Add rows
    for _, row in df.iterrows():
        table.add_row(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])

    console.print("\n")
    console.print(table)


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SecIDS-v2 models")

    parser.add_argument("--checkpoint", type=str, help="Single model checkpoint")
    parser.add_argument("--checkpoints", nargs="+", help="Multiple checkpoints for comparison")
    parser.add_argument("--names", nargs="+", help="Model names (for --checkpoints)")
    parser.add_argument("--test-data", type=str, required=True, help="Test data path")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    if args.checkpoint:
        # Single model evaluation
        evaluate_model(
            checkpoint_path=args.checkpoint,
            test_data_path=args.test_data,
            output_dir=args.output,
            batch_size=args.batch_size
        )

    elif args.checkpoints:
        # Multi-model benchmark
        if not args.names or len(args.names) != len(args.checkpoints):
            raise ValueError("--names must match --checkpoints in length")

        checkpoints_dict = dict(zip(args.names, args.checkpoints))

        benchmark_models(
            checkpoints=checkpoints_dict,
            test_data_path=args.test_data,
            output_path=Path(args.output) / "benchmark.csv"
        )

    else:
        parser.print_help()
