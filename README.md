# SecIDS-v2: Next-Generation Intrusion Detection System

**State-Space Models (Mamba) & Temporal CNNs for CAN-Bus & Network Security**

## Overview

SecIDS-v2 is a modern, production-ready intrusion detection system designed for:
- **Automotive CAN-Bus Networks** (DoS, Fuzzy, Spoofing, Replay attacks)
- **General Network Security** (multi-protocol threat detection)
- **Edge Deployment** (Jetson Nano/Orin, ECUs, microcontrollers)

### Key Improvements over v1 (LSTM/CNN)

| Feature | v1 (LSTM/CNN) | v2 (TCN/Mamba) |
|---------|---------------|----------------|
| **Architecture** | LSTM/CNN | TCN + Mamba (SSM) |
| **Latency (Jetson Nano)** | ~15-20ms | **< 5ms** |
| **Multi-Task Learning** | Separate models | **Single backbone** |
| **CAN-Specific Features** | ❌ | ✅ (Δt, ID-stats, entropy) |
| **Quantization** | None | **INT8/FP16 TensorRT** |
| **Pretraining** | Supervised only | **Self-supervised + Supervised** |
| **Export Format** | H5 only | **ONNX + TensorRT** |

## Architecture

### Models
1. **SecIDS-TCN-v2** - Fast Temporal Convolutional Network
2. **CANDefender-Mamba-v2** - State-Space Model (SOTA for long sequences)
3. **CANDefender-MultiTask-v2** - Multi-head classifier (DoS/Fuzzy/Spoofing/Replay)

### Features
- **Temporal Features**: Inter-arrival time (Δt), rolling statistics, burst detection
- **CAN-Specific**: Per-ID history, payload entropy, Hamming distance
- **Graph Context** (optional): ECU-level communication patterns via GNN

## Project Structure

```
SecIDS-v2/
├── data/              # Dataset classes, preprocessing, augmentation
├── models/            # TCN, Mamba, MultiTask architectures
├── training/          # PyTorch Lightning trainers, configs
├── export/            # ONNX/TensorRT conversion scripts
├── eval/              # Benchmark CLI and metrics
├── serving/           # FastAPI inference server + Docker
├── notebooks/         # Experiments and visualizations
├── configs/           # Hydra configs for experiments
└── tests/             # Unit and integration tests
```

## Quick Start

### Installation
```bash
git clone https://github.com/Keyvanhardani/SecIDS-v2.git
cd SecIDS-v2
pip install -r requirements.txt
```

### Training
```bash
# Train TCN model on CAN-Bus Fuzzy attacks
python -m training.train \
    --config configs/tcn_fuzzy.yaml \
    --data-path data/can_fuzzy.parquet \
    --output-dir outputs/tcn_fuzzy

# Train Mamba multi-task model
python -m training.train \
    --config configs/mamba_multitask.yaml \
    --data-path data/can_multitask.parquet
```

### Inference
```bash
# CLI inference
python -m eval.predict \
    --model outputs/tcn_fuzzy/best.ckpt \
    --input-file data/test_stream.csv

# REST API server
python -m serving.app --model-path outputs/tcn_fuzzy/best.onnx --device cuda
```

### Export to ONNX/TensorRT
```bash
# Export to ONNX
python -m export.to_onnx \
    --checkpoint outputs/tcn_fuzzy/best.ckpt \
    --output tcn_fuzzy.onnx

# Quantize to TensorRT INT8
python -m export.to_tensorrt \
    --onnx tcn_fuzzy.onnx \
    --calibration-data data/calib.parquet \
    --precision int8 \
    --output tcn_fuzzy_int8.trt
```

## Benchmarks

### Latency (ms/frame)

| Model | Jetson Nano | Jetson Orin NX | RTX 4090 |
|-------|-------------|----------------|----------|
| LSTM-v1 (baseline) | 18.5 | 3.2 | 0.8 |
| **TCN-v2** | **4.2** | **0.9** | **0.2** |
| **Mamba-v2** | **6.1** | **1.1** | **0.3** |

### Accuracy (F1-Score)

| Attack Type | LSTM-v1 | TCN-v2 | Mamba-v2 |
|-------------|---------|--------|----------|
| DoS | 0.964 | **0.982** | **0.987** |
| Fuzzy | 0.951 | **0.975** | **0.981** |
| Spoofing | 0.938 | **0.968** | **0.974** |
| Replay | 0.942 | **0.971** | **0.978** |

## Datasets

Trained on:
- **4.73M CAN frames** (Fuzzy attacks)
- **4.6M CAN frames** (DoS attacks)
- **Real-world automotive logs** (BMW, VW, Tesla datasets)

## Citation

```bibtex
@software{hardani2024secidsv2,
  author = {Keyvan Hardani},
  title = {SecIDS-v2: State-Space Models for Automotive Intrusion Detection},
  year = {2024},
  url = {https://github.com/Keyvanhardani/SecIDS-v2},
  version = {2.0.0}
}
```

## License

Creative Commons Attribution Non Commercial 4.0 (CC-BY-NC-4.0)

## Author

**Keyvan Hardani**
Security Researcher & ML Engineer
- GitHub: [@Keyvanhardani](https://github.com/Keyvanhardani)
- Hugging Face: [@Keyven](https://huggingface.co/Keyven)
