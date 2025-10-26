# SecIDS-v2 Usage Guide

Complete guide for training, evaluating, and deploying models.

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Export & Deployment](#export--deployment)
6. [UI & Monitoring](#ui--monitoring)
7. [Docker Deployment](#docker-deployment)

---

## Installation

### Requirements
- Python 3.10+
- PyTorch 2.1+
- CUDA 11.8+ (for GPU training)
- NVIDIA Jetson Nano/Orin (for edge deployment)

### Setup

```bash
# Clone repository
git clone https://github.com/Keyvanhardani/SecIDS-v2.git
cd SecIDS-v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install Mamba (for Mamba model)
pip install mamba-ssm causal-conv1d
```

---

## Data Preparation

### CAN-Bus Data Format

Your data should be in **Parquet** or **CSV** format with columns:

```
timestamp,can_id,dlc,data,label
1609459200.123,0x100,8,0102030405060708,0
1609459200.133,0x200,8,AA BB CC DD EE FF 00 11,1
...
```

**Columns:**
- `timestamp`: Float (Unix timestamp in seconds)
- `can_id`: Integer (0-2047 for standard, 0-536870911 for extended)
- `dlc`: Integer (0-8, data length code)
- `data`: String (hex bytes, e.g., "0102030405060708")
- `label`: Integer (0 = Benign, 1 = Attack)

### Multi-Task Data

For multi-task learning, add task-specific label columns:

```
timestamp,can_id,dlc,data,label_dos,label_fuzzy,label_spoofing,label_replay
1609459200.123,0x100,8,0102...,0,0,0,0
1609459200.133,0x200,8,AABB...,1,0,0,0  # DoS attack
...
```

### Example: Create Synthetic Data

```python
import pandas as pd
import numpy as np

# Generate 10000 synthetic CAN frames
data = {
    'timestamp': np.cumsum(np.random.exponential(0.01, 10000)),
    'can_id': np.random.choice([0x100, 0x200, 0x300], 10000),
    'dlc': 8,
    'data': [''.join(f'{b:02x}' for b in np.random.randint(0, 256, 8))
             for _ in range(10000)],
    'label': np.random.choice([0, 1], 10000, p=[0.9, 0.1])
}

df = pd.DataFrame(data)
df.to_parquet('data/train.parquet')
```

---

## Training

### Single-Task Training (TCN)

```bash
python -m training.train \
    --model tcn \
    --data data/train.parquet \
    --val data/val.parquet \
    --output outputs/tcn_dos \
    --batch-size 32 \
    --epochs 100 \
    --lr 1e-3 \
    --window-size 128
```

### Multi-Task Training (Mamba)

```bash
python -m training.train \
    --model mamba \
    --data data/train_multitask.parquet \
    --val data/val_multitask.parquet \
    --output outputs/mamba_multitask \
    --multitask \
    --tasks dos fuzzy spoofing replay \
    --batch-size 32 \
    --epochs 100
```

### With WandB Logging

```bash
python -m training.train \
    --model tcn \
    --data data/train.parquet \
    --val data/val.parquet \
    --output outputs/tcn_wandb \
    --wandb
```

### Training Output

```
outputs/tcn_dos/
├── checkpoints/
│   ├── best-epoch=42-val_f1=0.9820.ckpt
│   ├── last.ckpt
├── tensorboard_logs/
└── final_model.ckpt
```

---

## Evaluation

### Single Model Evaluation

```bash
python -m eval.evaluate \
    --checkpoint outputs/tcn_dos/final_model.ckpt \
    --test-data data/test.parquet \
    --output eval_results \
    --batch-size 32
```

**Output:**
```
┌─────────────────┬──────────┐
│ Metric          │ Value    │
├─────────────────┼──────────┤
│ Accuracy        │ 0.9820   │
│ Precision       │ 0.9830   │
│ Recall          │ 0.9810   │
│ F1-Score        │ 0.9820   │
│ ROC-AUC         │ 0.9950   │
│ Latency (p50)   │ 4.20 ms  │
│ Model Size      │ 0.80 MB  │
└─────────────────┴──────────┘
```

### Multi-Model Benchmark

```bash
python -m eval.evaluate \
    --checkpoints \
        outputs/lstm_v1/final.ckpt \
        outputs/tcn_v2/final.ckpt \
        outputs/mamba_v2/final.ckpt \
    --names LSTM-v1 TCN-v2 Mamba-v2 \
    --test-data data/test.parquet \
    --output benchmark_results.csv
```

---

## Export & Deployment

### Export to ONNX

```bash
python -m export.to_onnx \
    --checkpoint outputs/tcn_dos/final_model.ckpt \
    --output models/tcn_dos.onnx \
    --batch-size 1 \
    --seq-len 128 \
    --num-features 32 \
    --benchmark
```

**Output:**
```
✓ Exported to models/tcn_dos.onnx
✓ ONNX model is valid
✓ Outputs match!

Latency (over 100 runs):
  Mean: 4.23 ms
  Throughput: 236.4 samples/sec
```

### Export to TensorRT (Jetson)

```bash
# On NVIDIA Jetson device
python -m export.to_tensorrt \
    --onnx models/tcn_dos.onnx \
    --output models/tcn_dos_fp16.trt \
    --precision fp16 \
    --batch-size 1 \
    --benchmark
```

**TensorRT Results (Jetson Nano):**
```
Latency:
  Mean: 3.85 ms  (vs 4.23 ms ONNX)
  Throughput: 259.7 samples/sec
```

### INT8 Quantization

```bash
python -m export.to_tensorrt \
    --onnx models/tcn_dos.onnx \
    --output models/tcn_dos_int8.trt \
    --precision int8 \
    --calibration-data data/calibration.parquet
```

---

## UI & Monitoring

### Streamlit Dashboard

```bash
streamlit run serving/dashboard.py
```

Open browser: http://localhost:8501

**Features:**
- Real-time CAN-Bus monitoring
- Model performance comparison
- Attack pattern analysis
- Scientific report generation

### FastAPI Inference Server

```bash
python -m serving.app \
    --model-path models/tcn_dos.onnx \
    --device cuda \
    --port 8000
```

**API Endpoints:**

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model_info

# Predict (example)
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
      "frames": [
        {"timestamp": 1609459200.123, "can_id": 256, "dlc": 8, "data": "0102030405060708"},
        {"timestamp": 1609459200.133, "can_id": 512, "dlc": 8, "data": "AABBCCDDEEFF0011"}
      ]
    }'
```

**Response:**
```json
{
  "prediction": "DoS",
  "confidence": 0.987,
  "latency_ms": 4.12,
  "details": {
    "Benign": 0.013,
    "Attack": 0.987
  }
}
```

---

## Docker Deployment

### Build & Run

```bash
# Build image
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Services:**
- FastAPI: http://localhost:8000
- Dashboard: http://localhost:8501
- Jupyter: http://localhost:8888

### Production Deployment

```bash
# API only
docker run -d \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    --name secids-api \
    secids-v2:latest
```

---

## Quick Start Examples

### 1. Train TCN on DoS Detection

```bash
# Prepare data
python scripts/prepare_data.py --input raw_can_logs.csv --output data/train.parquet

# Train
python -m training.train --model tcn --data data/train.parquet --output outputs/tcn_dos

# Evaluate
python -m eval.evaluate --checkpoint outputs/tcn_dos/final_model.ckpt --test-data data/test.parquet

# Export
python -m export.to_onnx --checkpoint outputs/tcn_dos/final_model.ckpt --output models/tcn_dos.onnx

# Deploy
python -m serving.app --model-path models/tcn_dos.onnx --device cuda
```

### 2. Multi-Task Learning (All Attacks)

```bash
python -m training.train \
    --model mamba \
    --data data/multitask_train.parquet \
    --multitask \
    --tasks dos fuzzy spoofing replay \
    --output outputs/mamba_all
```

### 3. Benchmark v1 vs v2

```bash
python -m eval.evaluate \
    --checkpoints \
        old_models/lstm_dos.ckpt \
        outputs/tcn_dos/final_model.ckpt \
    --names LSTM-v1 TCN-v2 \
    --test-data data/test.parquet \
    --output benchmark_v1_vs_v2.csv
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python -m training.train --batch-size 16  # Instead of 32

# Or use gradient accumulation
python -m training.train --batch-size 8 --accumulate-grad-batches 4
```

### Mamba Not Available
```bash
# Install Mamba dependencies
pip install mamba-ssm causal-conv1d

# Or use TCN fallback (automatic)
```

### TensorRT Not Available
```bash
# Install on Jetson
sudo apt-get install tensorrt

# Or export to ONNX only (works everywhere)
```

---

## Next Steps

1. **Data**: Prepare your CAN-Bus dataset
2. **Train**: Start with TCN (fastest to train)
3. **Evaluate**: Compare against baseline
4. **Export**: Convert to ONNX for deployment
5. **Deploy**: Use Docker for production

Use the Streamlit dashboard to generate scientific reports and visualizations.

---

**Questions?** Open an issue on GitHub or contact keyvan.hardani@example.com
