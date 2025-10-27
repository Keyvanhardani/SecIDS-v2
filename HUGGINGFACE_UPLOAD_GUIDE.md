# HuggingFace Model Upload Guide

## Schritt 1: Account und Repository erstellen

1. Gehe zu https://huggingface.co/ und erstelle einen Account (falls noch nicht vorhanden)
2. Klicke auf dein Profil → **"New Model"**
3. Repository Name: `SecIDS-v2`
4. License: **CC-BY-NC-4.0**
5. Repository Type: **Public**

## Schritt 2: Model Card vorbereiten

Die Model Card ist bereits fertig: `MODEL_CARD.md`

Diese Datei wird zu `README.md` im HuggingFace Repository.

## Schritt 3: Dateien vorbereiten

### Benötigte Dateien:

```
SecIDS-v2/
├── README.md                          # = MODEL_CARD.md (umbenannt)
├── config.json                        # Model config
├── pytorch_model.bin oder .ckpt       # Trained model
├── visuals/                           # Bilder für README
│   ├── architecture.png
│   ├── performance_comparison.png
│   ├── feature_importance.png
│   └── github_banner.png
└── requirements.txt                   # Dependencies
```

## Schritt 4: config.json erstellen

Erstelle eine `config.json` mit Model-Details:

```json
{
  "model_type": "temporal_cnn",
  "architecture": "SecIDS-v2",
  "input_dim": 25,
  "hidden_dim": 256,
  "num_classes": 2,
  "num_channels": [256, 256, 512, 512],
  "kernel_size": 3,
  "dropout": 0.1,
  "activation": "relu",
  "num_parameters": 3815170,
  "framework": "pytorch",
  "pytorch_version": "2.0+",
  "license": "cc-by-nc-4.0"
}
```

## Schritt 5: Upload via Web Interface

### Option A: Web Upload (Einfach)

1. Gehe zu deinem Model Repository auf HuggingFace
2. Klicke auf **"Files and versions"** → **"Add file"** → **"Upload files"**
3. Uploade folgende Dateien:
   - `README.md` (kopiere MODEL_CARD.md)
   - `config.json`
   - Model Checkpoint (z.B. `secids_v2_tcn.ckpt`)
   - Bilder aus `visuals/` Ordner
4. Commit Message: "Initial upload: SecIDS-v2 v2.1.0"

### Option B: Git LFS (Für große Dateien)

```bash
# Install Git LFS
git lfs install

# Clone dein HuggingFace Repository
git clone https://huggingface.co/Keyvanhardani/SecIDS-v2

cd SecIDS-v2

# Track große Dateien (Model Checkpoints)
git lfs track "*.ckpt"
git lfs track "*.bin"
git lfs track "*.onnx"

# Kopiere Dateien
cp /home/Security-Models/SecIDS-v2/MODEL_CARD.md README.md
cp /home/Security-Models/SecIDS-v2/visuals/* .
cp outputs/tcn_production/final_model.ckpt secids_v2_tcn.ckpt

# Commit und push
git add .
git commit -m "Initial upload: SecIDS-v2 v2.1.0"
git push
```

## Schritt 6: README mit Bildern

Die Bilder müssen im Repository sein und dann im README referenziert werden:

```markdown
# SecIDS-v2

![Architecture](architecture.png)

![Performance](performance_comparison.png)

![Features](feature_importance.png)
```

HuggingFace rendert die Bilder automatisch, wenn sie im selben Repository sind.

## Schritt 7: Model Tags hinzufügen

Im HuggingFace UI → **"Model card"** → **"Edit"** → Tags hinzufügen:

```yaml
tags:
  - automotive
  - intrusion-detection
  - can-bus
  - security
  - temporal-cnn
  - pytorch-lightning
```

## Schritt 8: Spaces Widget (Optional)

HuggingFace bietet ein Widget um das Model direkt zu testen. Das erfordert:

1. Ein inference API script
2. Oder ein Gradio/Streamlit App

Für später - erst mal nur Model Upload!

## Quick Checklist

- [ ] HuggingFace Account erstellt
- [ ] Repository "SecIDS-v2" erstellt
- [ ] MODEL_CARD.md → README.md kopiert
- [ ] config.json erstellt
- [ ] Model Checkpoint hochgeladen
- [ ] Visuals hochgeladen
- [ ] Tags hinzugefügt
- [ ] License auf CC-BY-NC-4.0 gesetzt

## Wichtige URLs nach Upload

- **Model URL:** https://huggingface.co/Keyvanhardani/SecIDS-v2
- **Dashboard URL:** https://secids.keyvan.ai (dein Dashboard)
- **GitHub URL:** https://github.com/Keyvanhardani/SecIDS-v2

## Nächste Schritte

1. LinkedIn Post mit HuggingFace Link
2. GitHub README aktualisieren mit HuggingFace Badge
3. Model in Paper/Portfolio referenzieren
