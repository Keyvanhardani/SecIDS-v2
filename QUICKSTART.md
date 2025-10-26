# SecIDS-v2 - Quick Start Guide

Schnellanleitung, um sofort loszulegen (5 Minuten!)

## 1. Setup (einmalig)

```bash
# In das Projektverzeichnis wechseln
cd /home/Security-Models/SecIDS-v2

# Dependencies installieren
pip3 install -r requirements.txt

# Oder mit Virtual Environment (empfohlen):
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## 2. Synthetische Daten generieren

```bash
# Generiere 50k Training, 10k Validation, 10k Test
python3 scripts/generate_synthetic_data.py \
    --train-size 50000 \
    --val-size 10000 \
    --test-size 10000 \
    --attack-ratio 0.15 \
    --output-dir data \
    --format parquet
```

**Output:**
```
data/
├── train.parquet  (50,000 frames, 15% attacks)
├── val.parquet    (10,000 frames)
└── test.parquet   (10,000 frames)
```

## 3. Training starten

### Option A: Schnelles Training (TCN, CPU)

```bash
python3 -m training.train \
    --model tcn \
    --data data/train.parquet \
    --val data/val.parquet \
    --output outputs/tcn_demo \
    --batch-size 16 \
    --epochs 10 \
    --lr 1e-3
```

**Dauer:** ~5-10 Minuten auf CPU

### Option B: Vollständiges Training (TCN, GPU)

```bash
python3 -m training.train \
    --model tcn \
    --data data/train.parquet \
    --val data/val.parquet \
    --output outputs/tcn_full \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-3 \
    --wandb  # Optional: WandB logging
```

**Dauer:** ~30-60 Minuten auf GPU

### Option C: Mamba Model (SOTA)

```bash
python3 -m training.train \
    --model mamba \
    --data data/train.parquet \
    --val data/val.parquet \
    --output outputs/mamba_full \
    --batch-size 32 \
    --epochs 50
```

**Hinweis:** Mamba benötigt `mamba-ssm`:
```bash
pip install mamba-ssm causal-conv1d
```

### Option D: Multi-Task Learning

```bash
python3 -m training.train \
    --model tcn \
    --data data/train.parquet \
    --val data/val.parquet \
    --output outputs/multitask \
    --multitask \
    --tasks dos fuzzy spoofing replay \
    --batch-size 32 \
    --epochs 50
```

## 4. Training Output

Nach dem Training findest du:

```
outputs/tcn_demo/
├── checkpoints/
│   ├── best-epoch=XX-val_f1=0.98XX.ckpt
│   └── last.ckpt
├── tensorboard_logs/
│   └── version_0/
└── final_model.ckpt
```

**Live-Monitoring während Training:**
```bash
# In separatem Terminal
tensorboard --logdir outputs/tcn_demo/tensorboard_logs
# Öffne: http://localhost:6006
```

## 5. Evaluation

```bash
python3 -m eval.evaluate \
    --checkpoint outputs/tcn_demo/final_model.ckpt \
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
└─────────────────┴──────────┘
```

## 6. Export zu ONNX

```bash
python3 -m export.to_onnx \
    --checkpoint outputs/tcn_demo/final_model.ckpt \
    --output models/tcn_demo.onnx \
    --batch-size 1 \
    --seq-len 128 \
    --num-features 32 \
    --benchmark
```

**Output:**
```
✓ Exported to models/tcn_demo.onnx
✓ ONNX model is valid
✓ Outputs match!

Latency (over 100 runs):
  Mean: 4.23 ms
  Throughput: 236.4 samples/sec
```

## 7. UI Dashboard starten

```bash
# Streamlit Dashboard
streamlit run serving/dashboard.py

# Öffne im Browser: http://localhost:8501
```

## 8. FastAPI Server starten

```bash
python3 -m serving.app \
    --model-path models/tcn_demo.onnx \
    --device cpu \
    --port 8000

# API docs: http://localhost:8000/docs
```

**Test API:**
```bash
curl http://localhost:8000/health
```

## 9. Docker Deployment

```bash
# Build & Start
docker-compose up -d

# Services:
# - API: http://localhost:8000
# - Dashboard: http://localhost:8501
# - Jupyter: http://localhost:8888

# Logs ansehen
docker-compose logs -f

# Stoppen
docker-compose down
```

## Troubleshooting

### Out of Memory (CPU)
```bash
# Reduziere Batch Size
python3 -m training.train --batch-size 8 ...
```

### CUDA Out of Memory (GPU)
```bash
# Reduziere Batch Size oder Window Size
python3 -m training.train --batch-size 16 --window-size 64 ...
```

### Mamba nicht verfügbar
```bash
# Installiere Mamba-Dependencies
pip install mamba-ssm causal-conv1d

# Oder nutze TCN (funktioniert überall)
python3 -m training.train --model tcn ...
```

### Zu langsames Training
```bash
# Nutze GPU (falls verfügbar)
# PyTorch nutzt automatisch CUDA wenn verfügbar

# Oder: Reduziere Daten
python3 scripts/generate_synthetic_data.py --train-size 10000 ...
```

## Kompletter Workflow (Copy-Paste)

```bash
# 1. Setup
cd /home/Security-Models/SecIDS-v2
pip3 install -r requirements.txt

# 2. Daten generieren
python3 scripts/generate_synthetic_data.py \
    --train-size 50000 \
    --val-size 10000 \
    --test-size 10000

# 3. Training (schnell für Demo)
python3 -m training.train \
    --model tcn \
    --data data/train.parquet \
    --val data/val.parquet \
    --output outputs/tcn_demo \
    --batch-size 16 \
    --epochs 10

# 4. Evaluation
python3 -m eval.evaluate \
    --checkpoint outputs/tcn_demo/final_model.ckpt \
    --test-data data/test.parquet

# 5. Export
python3 -m export.to_onnx \
    --checkpoint outputs/tcn_demo/final_model.ckpt \
    --output models/tcn_demo.onnx

# 6. Dashboard
streamlit run serving/dashboard.py
```

## Nächste Schritte

- **Eigene Daten**: Ersetze synthetische Daten mit echten CAN-Logs
- **Hyperparameter Tuning**: Experimentiere mit Learning Rate, Batch Size, etc.
- **Model Comparison**: Trainiere mehrere Modelle und vergleiche
- **Jetson Deployment**: Exportiere zu TensorRT und teste auf Jetson Nano
- **Documentation**: Nutze Dashboard für Präsentationen und Reports

---

**Questions?** Check `USAGE.md` für Details oder öffne ein Issue auf GitHub.
