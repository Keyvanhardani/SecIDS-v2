# SecIDS-v2 Testing Guide

Complete guide for testing your trained SecIDS-v2 model.

---

## Quick Test

### 1. Basic Model Loading Test

```python
import torch
from models.tcn import TemporalCNN, TCNConfig

# Load checkpoint
checkpoint_path = "outputs/tcn_production/final_model.ckpt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Create model config
config = TCNConfig(
    input_dim=25,
    hidden_dim=256,
    num_classes=2,
    dropout=0.1,
    num_channels=[256, 256, 512, 512],
    kernel_size=3
)

# Initialize model
model = TemporalCNN(config)

# Load weights
state_dict = checkpoint.get('state_dict', checkpoint)
# Remove 'model.' prefix if present
state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

model.eval()
print("✅ Model loaded successfully!")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## Full Evaluation Pipeline

### 2. Evaluate on Test Set

```bash
# Run evaluation script
python evaluate_model.py \
    --model outputs/tcn_production/final_model.ckpt \
    --data data/test.parquet \
    --output results/evaluation \
    --batch-size 64 \
    --workers 0
```

**Expected Output:**
```
Loading model from outputs/tcn_production/final_model.ckpt
Loading test data from data/test.parquet
Found 30000 test samples

Running evaluation...
100%|████████████████| 469/469 [00:15<00:00, 30.12it/s]

Classification Report:
              precision    recall  f1-score   support

      Normal       0.98      0.99      0.99     21000
      Attack       0.97      0.96      0.97      9000

    accuracy                           0.98     30000
   macro avg       0.98      0.98      0.98     30000
weighted avg       0.98      0.98      0.98     30000

Confusion Matrix saved to: results/evaluation/confusion_matrix.png
ROC Curve saved to: results/evaluation/roc_curve.png
PR Curve saved to: results/evaluation/pr_curve.png
```

---

## Inference Testing

### 3. Single Frame Prediction

```python
import torch
import pandas as pd
import numpy as np
from models.tcn import TemporalCNN, TCNConfig

# Load model
checkpoint = torch.load("outputs/tcn_production/final_model.ckpt")
config = TCNConfig(input_dim=25, num_channels=[256, 256, 512, 512], kernel_size=3, num_classes=2)
model = TemporalCNN(config)
state_dict = {k.replace('model.', ''): v for k, v in checkpoint.get('state_dict', checkpoint).items()}
model.load_state_dict(state_dict)
model.eval()

# Create dummy input (batch=1, seq_len=128, features=25)
dummy_input = torch.randn(1, 128, 25)

# Inference
with torch.no_grad():
    logits = model(dummy_input)
    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(logits, dim=1)

print(f"Prediction: {'Attack' if prediction == 1 else 'Normal'}")
print(f"Confidence: {probabilities[0][prediction].item():.2%}")
print(f"Normal probability: {probabilities[0][0].item():.2%}")
print(f"Attack probability: {probabilities[0][1].item():.2%}")
```

---

## Data Loading Test

### 4. Test Data Pipeline

```python
import pandas as pd
from data.dataset import CANDataset
from torch.utils.data import DataLoader

# Load test data
test_df = pd.read_parquet("data/test.parquet")
print(f"Test data shape: {test_df.shape}")
print(f"Columns: {test_df.columns.tolist()}")
print(f"Class distribution:\n{test_df['label'].value_counts()}")

# Create dataset
dataset = CANDataset(test_df, window_size=128)
print(f"\nDataset size: {len(dataset)} windows")

# Create dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Get one batch
batch = next(iter(dataloader))
X, y = batch
print(f"\nBatch shapes:")
print(f"  Input (X): {X.shape}")  # Should be [32, 128, 25]
print(f"  Labels (y): {y.shape}")  # Should be [32]
print(f"  Label values: {y.unique()}")
```

---

## Performance Testing

### 5. Latency Benchmark

```python
import torch
import time
import numpy as np
from models.tcn import TemporalCNN, TCNConfig

# Load model
checkpoint = torch.load("outputs/tcn_production/final_model.ckpt")
config = TCNConfig(input_dim=25, num_channels=[256, 256, 512, 512], kernel_size=3, num_classes=2)
model = TemporalCNN(config)
state_dict = {k.replace('model.', ''): v for k, v in checkpoint.get('state_dict', checkpoint).items()}
model.load_state_dict(state_dict)
model.eval()

# Warm up
dummy_input = torch.randn(1, 128, 25)
with torch.no_grad():
    for _ in range(10):
        _ = model(dummy_input)

# Benchmark
latencies = []
num_runs = 100

with torch.no_grad():
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(dummy_input)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

print(f"Latency Statistics (ms):")
print(f"  Mean: {np.mean(latencies):.2f}")
print(f"  Median: {np.median(latencies):.2f}")
print(f"  Min: {np.min(latencies):.2f}")
print(f"  Max: {np.max(latencies):.2f}")
print(f"  Std: {np.std(latencies):.2f}")
print(f"  FPS: {1000 / np.mean(latencies):.1f}")
```

---

## ONNX Export Test

### 6. Export and Test ONNX Model

```bash
# Export to ONNX
python export_onnx.py \
    --checkpoint outputs/tcn_production/final_model.ckpt \
    --output models/secids_v2.onnx \
    --input-dim 25 \
    --window-size 128

# Test ONNX model
python -c "
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession('models/secids_v2.onnx')

# Print model info
print('ONNX Model Inputs:')
for input in session.get_inputs():
    print(f'  Name: {input.name}, Shape: {input.shape}, Type: {input.type}')

print('\nONNX Model Outputs:')
for output in session.get_outputs():
    print(f'  Name: {output.name}, Shape: {output.shape}, Type: {output.type}')

# Test inference
input_data = np.random.randn(1, 128, 25).astype(np.float32)
outputs = session.run(None, {'input': input_data})
print(f'\nInference output shape: {outputs[0].shape}')
print(f'Predictions: {outputs[0]}')
print('✅ ONNX export successful!')
"
```

---

## Common Issues & Solutions

### Issue 1: "RuntimeError: Error(s) in loading state_dict"

**Solution**: The checkpoint might contain wrapped keys (e.g., `model.layer1.weight`). Strip the prefix:

```python
state_dict = checkpoint.get('state_dict', checkpoint)
state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
```

### Issue 2: "KeyError: 'state_dict'"

**Solution**: The checkpoint structure might be different. Try:

```python
# Option 1: Direct load
model.load_state_dict(checkpoint)

# Option 2: Check available keys
print(checkpoint.keys())
```

### Issue 3: "Shape mismatch"

**Solution**: Verify config matches training settings:

```python
# Check saved config
if 'hyper_parameters' in checkpoint:
    print("Training config:", checkpoint['hyper_parameters'])

# Ensure matching config
config = TCNConfig(
    input_dim=25,          # Must match training
    num_channels=[256, 256, 512, 512],  # Must match training
    kernel_size=3,
    num_classes=2
)
```

### Issue 4: "Class imbalance - model predicts only Normal"

**Solution**: This is expected with the unbalanced training. Retrain with class weights:

```bash
python train_balanced.py \
    --data data/train.parquet \
    --val data/val.parquet \
    --output outputs/tcn_balanced \
    --balance-factor 15.0 \
    --epochs 100
```

---

## Automated Test Script

Create `test_model.py`:

```python
#!/usr/bin/env python3
"""
Automated testing script for SecIDS-v2
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
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
        state_dict = {k.replace('model.', ''): v for k, v in checkpoint.get('state_dict', checkpoint).items()}
        model.load_state_dict(state_dict)
        model.eval()
        print("✅ PASSED: Model loaded successfully")
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
        print(f"✅ PASSED: Inference successful, output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_latency(model, num_runs=100):
    """Test 3: Latency benchmark"""
    print(f"\nTest 3: Latency Benchmark ({num_runs} runs)...")
    try:
        dummy_input = torch.randn(1, 128, 25)

        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

                if start:
                    start.record()
                import time
                t0 = time.perf_counter()
                _ = model(dummy_input)
                t1 = time.perf_counter()
                if end:
                    end.record()
                    torch.cuda.synchronize()
                    latencies.append(start.elapsed_time(end))
                else:
                    latencies.append((t1 - t0) * 1000)

        mean_latency = np.mean(latencies)
        print(f"✅ PASSED: Mean latency: {mean_latency:.2f}ms")
        print(f"  Min: {np.min(latencies):.2f}ms, Max: {np.max(latencies):.2f}ms")
        print(f"  FPS: {1000 / mean_latency:.1f}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_data_loading(data_path):
    """Test 4: Data loading"""
    print(f"\nTest 4: Data Loading from {data_path}...")
    try:
        df = pd.read_parquet(data_path)
        print(f"✅ PASSED: Loaded {len(df)} samples")
        print(f"  Shape: {df.shape}")
        if 'label' in df.columns:
            print(f"  Class distribution: {df['label'].value_counts().to_dict()}")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SecIDS-v2 Automated Testing")
    print("=" * 60)

    checkpoint_path = "outputs/tcn_production/final_model.ckpt"
    data_path = "data/test.parquet"

    # Run tests
    model = test_model_loading(checkpoint_path)
    if model:
        test_inference(model)
        test_latency(model)

    if Path(data_path).exists():
        test_data_loading(data_path)
    else:
        print(f"\n⚠️ Skipping data test: {data_path} not found")

    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)
```

Run it:

```bash
python test_model.py
```

---

## Production Readiness Checklist

Before deploying, ensure:

- [ ] Model loads successfully without errors
- [ ] Inference produces expected output shape (batch_size, 2)
- [ ] Latency is acceptable for your use case (< 10ms for real-time)
- [ ] Evaluation metrics meet minimum thresholds:
  - [ ] Accuracy > 95%
  - [ ] F1-Score > 0.90
  - [ ] Recall > 0.85 (important for attack detection!)
- [ ] ONNX export successful (if deploying on edge devices)
- [ ] Class imbalance addressed (if F1 for attacks = 0)
- [ ] Model tested on held-out validation set (not seen during training)

---

## Next Steps

1. **If tests pass**: Proceed to deployment (see `USAGE.md`)
2. **If class imbalance detected**: Retrain with `train_balanced.py`
3. **If latency too high**: Consider quantization (INT8) or model pruning
4. **If accuracy too low**: Try Mamba model or increase training epochs

---

**Version**: 2.1.0
**Last Updated**: October 2024
