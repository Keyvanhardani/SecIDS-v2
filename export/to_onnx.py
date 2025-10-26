"""
Export PyTorch models to ONNX format

ONNX enables:
  - Cross-platform inference (CPU, GPU, mobile)
  - Optimization with ONNX Runtime
  - Base for TensorRT conversion
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys
sys.path.append('..')

from models import BaseIDS
from training.trainer import IDSTrainer, MultiTaskIDSTrainer


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    input_shape: Tuple[int, int, int] = (1, 128, 32),
    opset_version: int = 14,
    dynamic_axes: bool = True,
    verify: bool = True
) -> str:
    """
    Export trained model to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch Lightning checkpoint
        output_path: Where to save ONNX model
        input_shape: (batch, seq_len, features) for dummy input
        opset_version: ONNX opset version (14 for broad compatibility)
        dynamic_axes: Allow variable batch/sequence length
        verify: Verify output matches PyTorch

    Returns:
        Path to exported ONNX model
    """

    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load model
    try:
        # Try single-task first
        lightning_model = IDSTrainer.load_from_checkpoint(checkpoint_path)
        model = lightning_model.model
        multitask = False
    except:
        # Try multi-task
        lightning_model = MultiTaskIDSTrainer.load_from_checkpoint(checkpoint_path)
        model = lightning_model.model
        multitask = True

    model.eval()

    print(f"Model loaded: {model.__class__.__name__}")
    print(f"Multi-task: {multitask}")

    # Prepare dummy input
    batch_size, seq_len, num_features = input_shape
    dummy_input = torch.randn(batch_size, seq_len, num_features)

    # Dynamic axes (variable batch/seq length)
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch', 1: 'sequence'},
        }

        if multitask:
            # Multi-task: multiple outputs
            for i, task in enumerate(model.get_task_names()):
                dynamic_axes_dict[f'output_{task}'] = {0: 'batch'}
        else:
            dynamic_axes_dict['output'] = {0: 'batch'}
    else:
        dynamic_axes_dict = None

    # Export
    print(f"\nExporting to ONNX...")
    print(f"  Input shape: {input_shape}")
    print(f"  Dynamic axes: {dynamic_axes}")
    print(f"  Opset version: {opset_version}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # For multi-task, we need to handle dict outputs
    if multitask:
        # Wrap model to return tuple instead of dict
        class MultiTaskWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.task_names = model.get_task_names()

            def forward(self, x):
                outputs = self.model(x)
                # Convert dict to tuple (ONNX doesn't support dict outputs)
                return tuple(outputs[task] for task in self.task_names)

        wrapped_model = MultiTaskWrapper(model)
        output_names = [f'output_{task}' for task in model.get_task_names()]
    else:
        wrapped_model = model
        output_names = ['output']

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(output_path),
        input_names=['input'],
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True
    )

    print(f"✓ Exported to {output_path}")

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")

    # Compare outputs (sanity check)
    if verify:
        print("\nComparing PyTorch vs ONNX outputs...")
        _verify_onnx_output(model, str(output_path), dummy_input, multitask)

    # Print model info
    print(f"\nModel info:")
    print(f"  IR version: {onnx_model.ir_version}")
    print(f"  Opset: {onnx_model.opset_import[0].version}")
    print(f"  Inputs: {[i.name for i in onnx_model.graph.input]}")
    print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")

    # File size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")

    return str(output_path)


def _verify_onnx_output(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    dummy_input: torch.Tensor,
    multitask: bool
):
    """Verify ONNX model produces same output as PyTorch"""

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input)

    # ONNX inference
    ort_session = ort.InferenceSession(onnx_path)
    onnx_input = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    onnx_output = ort_session.run(None, onnx_input)

    # Compare
    if multitask:
        # Dict output from PyTorch, tuple from ONNX
        task_names = list(pytorch_output.keys())
        for i, task in enumerate(task_names):
            pt_out = pytorch_output[task].numpy()
            onnx_out = onnx_output[i]
            diff = np.abs(pt_out - onnx_out).max()
            print(f"  {task}: max diff = {diff:.6f}")
            assert diff < 1e-4, f"Large difference for task {task}: {diff}"
    else:
        pt_out = pytorch_output.numpy()
        onnx_out = onnx_output[0]
        diff = np.abs(pt_out - onnx_out).max()
        print(f"  Max difference: {diff:.6f}")
        assert diff < 1e-4, f"Large difference: {diff}"

    print("✓ Outputs match!")


def benchmark_onnx(
    onnx_path: str,
    input_shape: Tuple[int, int, int] = (1, 128, 32),
    num_runs: int = 100
):
    """
    Benchmark ONNX model inference latency.

    Args:
        onnx_path: Path to ONNX model
        input_shape: Input shape
        num_runs: Number of iterations for averaging
    """
    import time

    print(f"Benchmarking {onnx_path}...")

    # Create session
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    # Dummy input
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(10):
        ort_session.run(None, {input_name: dummy_input})

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        ort_session.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    # Statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"\nLatency (over {num_runs} runs):")
    print(f"  Mean: {mean_time:.2f} ms")
    print(f"  Std:  {std_time:.2f} ms")
    print(f"  Min:  {min_time:.2f} ms")
    print(f"  Max:  {max_time:.2f} ms")
    print(f"  Throughput: {1000 / mean_time:.1f} samples/sec")


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="PyTorch checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX path")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-features", type=int, default=32)
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")

    args = parser.parse_args()

    input_shape = (args.batch_size, args.seq_len, args.num_features)

    onnx_path = export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        input_shape=input_shape
    )

    if args.benchmark:
        benchmark_onnx(onnx_path, input_shape)
