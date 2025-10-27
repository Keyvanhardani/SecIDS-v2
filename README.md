# SecIDS-v2: Next-Generation Automotive Intrusion Detection

[![GitHub](https://img.shields.io/badge/GitHub-SecIDS--v2-blue?logo=github)](https://github.com/Keyvanhardani/SecIDS-v2)
[![License](https://img.shields.io/badge/License-CC--BY--NC--4.0-green.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Live-brightgreen)](https://secids.keyvan.ai/)

**State-of-the-art CAN-Bus intrusion detection using Temporal Convolutional Networks (TCN) and State-Space Models (Mamba)**

---

## 🎯 Overview

SecIDS-v2 is a production-ready intrusion detection system specifically designed for automotive CAN-Bus networks. Built with modern deep learning architectures, it provides **real-time threat detection** with **3-5x faster inference** than traditional LSTM-based approaches.

### Key Features

- ⚡ **Real-Time Performance**: 4.2ms inference latency on NVIDIA Jetson Nano
- 🎯 **High Accuracy**: 98.2% detection accuracy on real-world CAN traffic
- 🚗 **Multi-Attack Detection**: DoS, Fuzzy, Spoofing, and Replay attacks
- 🔧 **Edge-Optimized**: Deployment on resource-constrained automotive hardware
- 📊 **Production-Ready**: Complete pipeline from training to deployment

### Performance Highlights

| Metric | SecIDS-v2 (TCN) | Baseline (LSTM) | Improvement |
|--------|-----------------|-----------------|-------------|
| **Accuracy** | 98.2% | 97.2% | +1.0% |
| **F1-Score** | 97.5% | 96.4% | +1.1% |
| **Latency (Jetson Nano)** | **4.2ms** | 18.5ms | **4.4× faster** |
| **Model Size** | 15.2 MB | 8.4 MB | +81% (more params) |
| **Parameters** | 3.8M | 2.8M | +36% |

---

## 🏗️ Architecture

SecIDS-v2 offers two model architectures:

### 1. **Temporal CNN (TCN)** - Production Model
- **Architecture**: 4-layer dilated CNN with residual connections
- **Parameters**: 3.8M
- **Receptive Field**: 2047 frames
- **Best for**: Real-time deployment, resource-constrained devices

### 2. **Mamba (State-Space Model)** - Experimental
- **Architecture**: Selective state-space model with attention
- **Parameters**: 4.2M
- **Accuracy**: 98.7% (+0.5% over TCN)
- **Best for**: Research, scenarios requiring highest accuracy

### Feature Engineering

SecIDS-v2 extracts **25 CAN-specific features**:

**Temporal Features:**
- Inter-arrival time (Δt)
- Δt rolling statistics (mean, std)
- Burst count (frames in 100ms window)
- Per-ID Δt

**Payload Features:**
- Shannon entropy
- Hamming distance to previous frame
- Byte diversity
- Zero-byte ratio

**ID Statistics:**
- Frame count per ID
- ID frequency deviation
- Time since last ID occurrence

---

## 📊 Benchmarks

### Per-Attack Type Performance (TCN-v2)

| Attack Type | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| **DoS** | 98.5% | 98.0% | 98.2% |
| **Fuzzy** | 97.8% | 97.2% | 97.5% |
| **Spoofing** | 97.1% | 96.5% | 96.8% |
| **Replay** | 97.4% | 96.8% | 97.1% |

### Latency Across Platforms

| Platform | TCN-v2 | Mamba-v2 | LSTM-v1 |
|----------|--------|----------|---------|
| **Jetson Nano (INT8)** | **4.2ms** | 6.1ms | 18.5ms |
| **Jetson Xavier NX (FP16)** | **2.8ms** | 4.3ms | 12.1ms |
| **Desktop GPU (RTX 4060)** | **0.9ms** | 1.4ms | 3.2ms |
| **Raspberry Pi 4** | 12.1ms | 18.7ms | 45.2ms |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Keyvanhardani/SecIDS-v2.git
cd SecIDS-v2
pip install -r requirements.txt
```

### Training a Model

```bash
# Train TCN model (100 epochs, production settings)
python -m training.train \
    --model tcn \
    --data data/train.parquet \
    --val data/val.parquet \
    --output outputs/tcn_production \
    --batch-size 64 \
    --epochs 100 \
    --lr 1e-3 \
    --window-size 128

# Train with balanced class weights (recommended)
python train_balanced.py \
    --data data/train.parquet \
    --val data/val.parquet \
    --output outputs/tcn_balanced \
    --balance-factor 15.0
```

### Evaluation

```bash
# Evaluate model on test set
python evaluate_model.py \
    --model outputs/tcn_production/final_model.ckpt \
    --data data/test.parquet \
    --output results/evaluation \
    --batch-size 64
```

### Inference

```bash
# Quick inference test
python -m eval.predict \
    --model outputs/tcn_production/final_model.ckpt \
    --input data/test_stream.csv
```

### Export for Deployment

```bash
# Export to ONNX
python export_onnx.py \
    --checkpoint outputs/tcn_production/final_model.ckpt \
    --output models/secids_v2.onnx \
    --input-dim 25 \
    --window-size 128

# Convert to TensorRT (for NVIDIA Jetson)
trtexec --onnx=models/secids_v2.onnx \
        --saveEngine=models/secids_v2_int8.trt \
        --int8 \
        --workspace=4096
```

---

## 🧪 Testing the Model

### Quick Test Script

```python
import torch
import pandas as pd
from models import TemporalCNN, TCNConfig

# Load model
checkpoint = torch.load("outputs/tcn_production/final_model.ckpt")

config = TCNConfig(
    input_dim=25,
    num_channels=[256, 256, 512, 512],
    kernel_size=3,
    num_classes=2
)

model = TemporalCNN(config)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Load test data
test_data = pd.read_parquet("data/test.parquet")

# Prepare input: [batch, sequence_length, features]
# Shape: [1, 128, 25]
input_window = ...  # Your CAN frame window

# Inference
with torch.no_grad():
    logits = model(input_window)
    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(logits, dim=1)

print(f"Prediction: {'Attack' if prediction == 1 else 'Normal'}")
print(f"Confidence: {probabilities[0][prediction]:.2%}")
```

See **TESTING_GUIDE.md** for comprehensive testing instructions.

---

## 📦 Project Structure

```
SecIDS-v2/
├── data/                       # Dataset preprocessing and loaders
│   ├── dataset.py             # PyTorch Dataset implementation
│   ├── preprocessing.py       # Feature extraction
│   └── augmentation.py        # Data augmentation
├── models/                     # Model architectures
│   ├── tcn.py                 # Temporal CNN
│   ├── mamba.py               # State-Space Model
│   └── multi_task.py          # Multi-task learning
├── training/                   # Training pipeline
│   ├── train.py               # Main training script
│   └── trainer.py             # PyTorch Lightning trainer
├── evaluation/                 # Evaluation scripts
│   ├── visualize.py           # Generate evaluation plots
│   └── benchmark.py           # Benchmark across models
├── export/                     # Model export
│   ├── to_onnx.py            # ONNX export
│   └── to_tensorrt.py        # TensorRT export
├── serving/                    # Deployment (excluded from git)
│   ├── api.py                # FastAPI server
│   └── dashboard_v2.py       # Streamlit dashboard
├── scripts/                    # Utility scripts
│   ├── evaluate.py           # Evaluation helper
│   └── generate_synthetic_data.py
├── check_data.py              # Data validation
├── evaluate_model.py          # Model evaluation CLI
├── export_onnx.py             # ONNX export CLI
├── train_balanced.py          # Balanced training
└── oversample_data.py         # Data balancing
```

---

## 🎓 Dataset

### Car Hacking Challenge 2021

- **Source**: IEEE Dataport / OCSLab HK Security
- **Total Frames**: 200,000 CAN messages
- **Split**: 70% Train / 15% Val / 15% Test
- **Attack Types**: DoS, Fuzzy, Spoofing, Replay
- **Class Distribution**: 70% Normal, 30% Attack

**Note**: The model trained on this dataset may require fine-tuning for specific vehicle models or CAN configurations.

---

## 🔗 Links

- 🌐 **Live Dashboard**: [secids.keyvan.ai](https://secids.keyvan.ai/)
- 📦 **GitHub Repository**: [Keyvanhardani/SecIDS-v2](https://github.com/Keyvanhardani/SecIDS-v2)
- 📚 **Documentation**: See `USAGE.md` and `QUICKSTART.md`

---

## 🛠️ Deployment

### Docker

```bash
# Build image
docker build -t secids-v2 .

# Run inference server
docker run -p 8000:8000 secids-v2
```

### NVIDIA Jetson

```bash
# Install dependencies
sudo apt-get install python3-pip
pip3 install -r requirements.txt

# Export to TensorRT INT8
python export_onnx.py --checkpoint model.ckpt --output model.onnx
trtexec --onnx=model.onnx --int8 --saveEngine=model_int8.trt

# Run inference
python -m serving.api --model model_int8.trt --device cuda
```

---

## ⚠️ Known Limitations

1. **Class Imbalance**: Current model may exhibit bias towards "Normal" class due to 70/30 distribution. Use `train_balanced.py` with class weights for better recall.
2. **Domain-Specific**: Trained on Car Hacking Challenge dataset. May require fine-tuning for specific vehicle models.
3. **Fixed Window**: Operates on fixed-size windows of 128 frames. Variable-length support coming in v2.2.

---

## 📄 License

**Creative Commons Attribution Non-Commercial 4.0 (CC-BY-NC-4.0)**

This work is licensed for:
- ✅ Academic research
- ✅ Educational purposes
- ✅ Non-commercial security research
- ❌ Commercial deployment (contact author for licensing)

---

## 📖 Citation

```bibtex
@software{hardani2024secids_v2,
  author = {Hardani, Keyvan},
  title = {SecIDS-v2: Next-Generation Automotive Intrusion Detection with Temporal CNNs},
  year = {2024},
  version = {2.1.0},
  url = {https://github.com/Keyvanhardani/SecIDS-v2},
  license = {CC-BY-NC-4.0}
}
```

---

## 👤 Author

**Keyvan Hardani**
Security Researcher & ML Engineer

- 🐙 GitHub: [@Keyvanhardani](https://github.com/Keyvanhardani)
- 📧 Email: hardani@hotmail.de

---

## 🙏 Acknowledgments

- **Dataset**: Car Hacking Challenge 2021 (IEEE Dataport, OCSLab HK Security)
- **Framework**: PyTorch Lightning
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Deployment**: ONNX Runtime, TensorRT, FastAPI, Streamlit

---

**Version**: 2.1.0
**Last Updated**: October 2024
**Status**: Production Ready
