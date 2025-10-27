#!/usr/bin/env python3
"""
Automated testing script for SecIDS-v2
Usage: python test_model.py
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.tcn import TemporalCNN, TCNConfig

def test_model_loading(checkpoint_path):
    """Test 1: Load model"""
    print("Test 1: Model Loading...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = TCNConfig(
            input_dim=25,
            num_channels=[256, 256, 512, 512],
            kernel_size=3,
            num_classes=2
        )
        model = TemporalCNN(config)

        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove 'model.' prefix if present
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.eval()

        num_params = sum(p.numel() for p in model.parameters())
        print(f"✅ PASSED: Model loaded successfully")
        print(f"   Parameters: {num_params:,}")
        return model
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return None

def test_inference(model):
    """Test 2: Run inference"""
    print("\nTest 2: Inference...")
    try:
        dummy_input = torch.randn(1, 128, 25)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (1, 2), f"Expected shape (1, 2), got {output.shape}"

        # Check if output is valid
        probs = torch.softmax(output, dim=1)
        assert torch.all(probs >= 0) and torch.all(probs <= 1), "Invalid probabilities"
        assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0])), "Probabilities don't sum to 1"

        print(f"✅ PASSED: Inference successful")
        print(f"   Output shape: {output.shape}")
        print(f"   Probabilities: Normal={probs[0][0]:.4f}, Attack={probs[0][1]:.4f}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_latency(model, num_runs=100):
    """Test 3: Latency benchmark"""
    print(f"\nTest 3: Latency Benchmark ({num_runs} runs)...")
    try:
        import time
        dummy_input = torch.randn(1, 128, 25)

        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)  # Convert to ms

        mean_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        std_latency = np.std(latencies)
        fps = 1000 / mean_latency

        print(f"✅ PASSED: Latency benchmark complete")
        print(f"   Mean: {mean_latency:.2f}ms")
        print(f"   Min: {min_latency:.2f}ms")
        print(f"   Max: {max_latency:.2f}ms")
        print(f"   Std: {std_latency:.2f}ms")
        print(f"   FPS: {fps:.1f}")

        # Warning if too slow
        if mean_latency > 10:
            print(f"   ⚠️ WARNING: Latency > 10ms may not be suitable for real-time applications")

        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_batch_inference(model):
    """Test 4: Batch inference"""
    print("\nTest 4: Batch Inference...")
    try:
        batch_sizes = [1, 8, 32, 64]
        for batch_size in batch_sizes:
            dummy_input = torch.randn(batch_size, 128, 25)
            with torch.no_grad():
                output = model(dummy_input)
            assert output.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {output.shape}"

        print(f"✅ PASSED: Batch inference successful")
        print(f"   Tested batch sizes: {batch_sizes}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_data_loading(data_path):
    """Test 5: Data loading"""
    print(f"\nTest 5: Data Loading...")
    try:
        df = pd.read_parquet(data_path)
        print(f"✅ PASSED: Data loaded successfully")
        print(f"   Total samples: {len(df):,}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")

        if 'label' in df.columns:
            class_dist = df['label'].value_counts()
            print(f"   Class distribution:")
            for label, count in class_dist.items():
                percentage = (count / len(df)) * 100
                print(f"     Class {label}: {count:,} ({percentage:.1f}%)")

            # Check for severe imbalance
            if class_dist.min() / class_dist.max() < 0.1:
                print(f"   ⚠️ WARNING: Severe class imbalance detected (ratio < 10%)")

        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    print("=" * 70)
    print("SecIDS-v2 Automated Testing Suite")
    print("=" * 70)
    print()

    # Configuration
    checkpoint_path = "outputs/tcn_production/final_model.ckpt"
    data_path = "data/test.parquet"

    # Check if files exist
    if not Path(checkpoint_path).exists():
        print(f"❌ ERROR: Checkpoint not found at {checkpoint_path}")
        print("   Please train a model first or update the checkpoint path.")
        return 1

    # Run tests
    results = {}

    # Test 1: Model Loading
    model = test_model_loading(checkpoint_path)
    results['model_loading'] = model is not None

    if model:
        # Test 2: Inference
        results['inference'] = test_inference(model)

        # Test 3: Latency
        results['latency'] = test_latency(model, num_runs=100)

        # Test 4: Batch Inference
        results['batch_inference'] = test_batch_inference(model)
    else:
        results['inference'] = False
        results['latency'] = False
        results['batch_inference'] = False

    # Test 5: Data Loading
    if Path(data_path).exists():
        results['data_loading'] = test_data_loading(data_path)
    else:
        print(f"\n⚠️ Skipping data test: {data_path} not found")
        results['data_loading'] = None

    # Summary
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    for test_name, result in results.items():
        status = "✅ PASSED" if result is True else ("❌ FAILED" if result is False else "⚠️ SKIPPED")
        print(f"{test_name:20s}: {status}")

    print()
    print(f"Total: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 70)

    # Return exit code
    if failed > 0:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1
    elif passed == 0:
        print("\n⚠️ No tests passed. Please check your setup.")
        return 1
    else:
        print("\n✅ All tests passed! Model is ready for deployment.")
        return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
