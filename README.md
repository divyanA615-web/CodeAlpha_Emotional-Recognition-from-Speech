# 🎙️ Speech Emotion Recognition from Audio

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Librosa-0.10-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  A production-grade deep learning system that detects human emotions (happy, angry, sad, fearful, etc.)
  directly from speech audio using a <strong>CNN-BiLSTM-Attention</strong> hybrid architecture.
</p>

---

## 📌 Project Overview

This project implements an end-to-end **Speech Emotion Recognition (SER)** pipeline that:

- Extracts **MFCC + delta + delta-delta** features from raw `.wav` audio
- Classifies speech into **8 emotions** using a deep learning model
- Combines **CNN** (spectral pattern extraction) → **BiLSTM** (temporal dynamics) → **Self-Attention** (emotional peak focus)
- Trained and evaluated on the **RAVDESS** and **TESS** benchmark datasets

> Built as part of the **CodeAlpha AI/ML Internship** — Task: Emotion Recognition from Speech

---

## 🧠 Architecture

```
Raw Audio (.wav)
      │
      ▼
Feature Extraction
  ├─ MFCC (40 coefficients)
  ├─ Delta (velocity)
  └─ Delta-Delta (acceleration)
       → Output shape: (120, 130)
      │
      ▼
CNN Feature Extractor  [4 blocks: 32 → 64 → 128 → 256 channels]
      │  ← learns local spectro-temporal patterns
      ▼
Bidirectional LSTM     [2 layers, hidden=256, both directions]
      │  ← captures long-range temporal emotion dynamics
      ▼
Self-Attention         [128-dim projection]
      │  ← focuses on emotionally salient speech frames
      ▼
Global Average Pooling
      │
      ▼
FC Classifier          [512 → 256 → 8 emotions]
      │
      ▼
Predicted Emotion + Confidence Score
```

---

## 🎭 Recognized Emotions

| Label | Emotion | Datasets |
|-------|---------|----------|
| 0 | 😐 Neutral | RAVDESS + TESS |
| 1 | 😌 Calm | RAVDESS |
| 2 | 😊 Happy | RAVDESS + TESS |
| 3 | 😢 Sad | RAVDESS + TESS |
| 4 | 😠 Angry | RAVDESS + TESS |
| 5 | 😨 Fearful | RAVDESS + TESS |
| 6 | 🤢 Disgust | RAVDESS + TESS |
| 7 | 😲 Surprised | RAVDESS + TESS |

---

## 📦 Datasets

### RAVDESS
- **Full name:** Ryerson Audio-Visual Database of Emotional Speech and Song
- **Actors:** 24 professional actors (12 male, 12 female)
- **Samples:** ~1,440 audio clips
- **Emotions:** 8 (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
- **Download:** [Zenodo — RAVDESS](https://zenodo.org/record/1188976)

### TESS
- **Full name:** Toronto Emotional Speech Set
- **Actors:** 2 actresses (young and old)
- **Samples:** ~2,800 audio clips
- **Emotions:** 7 (angry, disgust, fear, happy, neutral, sad, pleasant surprise)
- **Download:** [UofT TESS](https://tspace.library.utoronto.ca/handle/1807/24487)

---

## 🗂️ Project Structure

```
CodeAlpha_Emotional-Recognition-from-Speech/
│
├── config.py                  # All hyperparameters & paths (edit here)
├── train.py                   # Main training entry point
├── predict.py                 # Predict emotion from a single .wav file
├── evaluate.py                # Evaluate saved checkpoint on test set
├── requirements.txt           # All Python dependencies
│
├── data/
│   ├── loader.py              # RAVDESS + TESS dataset parser
│   ├── augmentation.py        # Noise, time-stretch, pitch-shift
│   ├── dataset.py             # PyTorch Dataset + DataLoader factory
│   └── raw/
│       ├── RAVDESS/           # ← place Actor_01/ ... Actor_24/ here
│       └── TESS/              # ← place OAF_angry/ YAF_fear/ ... here
│
├── features/
│   └── extractor.py           # MFCC + delta + delta-delta extraction
│
├── models/
│   ├── attention.py           # Scaled dot-product self-attention
│   └── ser_model.py           # CNN-BiLSTM-Attention classifier
│
├── training/
│   └── trainer.py             # Training loop, early stopping, AMP, checkpointing
│
├── evaluation/
│   └── metrics.py             # Accuracy, F1, confusion matrix, plots
│
├── inference/
│   └── predictor.py           # Production inference API
│
├── utils/
│   └── logger.py              # Structured logging
│
├── checkpoints/               # Saved model weights (auto-created)
├── logs/                      # Training logs (auto-created)
└── results/                   # Confusion matrix & training plots (auto-created)
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/divyanA615-web/CodeAlpha_Emotional-Recognition-from-Speech.git
cd CodeAlpha_Emotional-Recognition-from-Speech
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download and place datasets

**RAVDESS:**
```
Download → https://zenodo.org/record/1188976
Extract  → data/raw/RAVDESS/Actor_01/ ... Actor_24/
```

**TESS:**
```
Download → https://tspace.library.utoronto.ca/handle/1807/24487
Extract  → data/raw/TESS/OAF_angry/ YAF_fear/ YAF_ps/ ...
```

---

## 🚀 Usage

### Train the model
```bash
# Default settings (from config.py)
python train.py

# Custom hyperparameters
python train.py --epochs 80 --batch_size 64 --lr 5e-4

# Resume from last checkpoint
python train.py --resume checkpoints/last_ser_model.pt
```

### Predict emotion from audio
```bash
python predict.py --audio path/to/your_speech.wav
```

**Example output:**
```
┌─────────────────────────────────────────┐
│  File:                  your_speech.wav │
├─────────────────────────────────────────┤
│  Detected emotion : HAPPY               │
│  Confidence       :  92.37%             │
├─────────────────────────────────────────┤
│  All probabilities:                     │
│  😊 happy      ████████████████████   92.4%  │
│  😐 neutral    ██░░░░░░░░░░░░░░░░░░    4.1%  │
│  😢 sad        █░░░░░░░░░░░░░░░░░░░    2.1%  │
└─────────────────────────────────────────┘
```

### Evaluate a saved checkpoint
```bash
python evaluate.py --checkpoint checkpoints/best_ser_model.pt
```

---

## 🔬 Key Features

| Feature | Details |
|---------|---------|
| **Feature Extraction** | MFCC (40) + Δ + ΔΔ → (120, 130) tensor |
| **Augmentation** | White noise, time-stretch (0.9×/1.1×), pitch-shift (±2 semitones) |
| **Model** | CNN-BiLSTM-Attention hybrid |
| **Loss** | CrossEntropy + label smoothing (0.1) + class weighting |
| **Optimizer** | AdamW with decoupled weight decay |
| **Scheduler** | ReduceLROnPlateau (patience=7, factor=0.5) |
| **Training** | Mixed-precision (FP16), gradient clipping, early stopping |
| **Inference** | Single-file & batch prediction API |

---

## 📊 Results

| Dataset | Test Accuracy | Weighted F1 |
|---------|--------------|-------------|
| RAVDESS only | ~80% | ~0.80 |
| TESS only | ~93% | ~0.93 |
| **RAVDESS + TESS (combined)** | **~85–88%** | **~0.86** |

---

## 🛠️ Configuration

All settings are in `config.py`. Key knobs:

```python
SAMPLE_RATE   = 22050    # audio sample rate
DURATION      = 3        # fixed clip length (seconds)
N_MFCC        = 40       # MFCC coefficients
LSTM_HIDDEN   = 256      # BiLSTM hidden size
BATCH_SIZE    = 32       # training batch size
EPOCHS        = 100      # max epochs (early stopping applies)
LEARNING_RATE = 1e-3     # initial learning rate
DROPOUT       = 0.4      # classifier dropout
AUGMENT_PROB  = 0.5      # augmentation probability per sample
```

---

## 📈 Training Pipeline

```
Load Data (RAVDESS + TESS)
      │
      ▼
Stratified Train/Val/Test Split  (70% / 15% / 15%)
      │
      ▼
DataLoader (augmentation ON for train, OFF for val/test)
      │
      ▼
Train Loop
  ├─ Mixed-precision forward pass (FP16 on GPU)
  ├─ CrossEntropy loss + label smoothing
  ├─ AdamW backward + gradient clipping
  ├─ ReduceLROnPlateau scheduler
  └─ Best-model checkpointing + early stopping
      │
      ▼
Evaluation
  ├─ Classification report (per-class precision, recall, F1)
  ├─ Confusion matrix (raw + normalized)
  └─ Training history plots
```

---

## 🧩 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Deep Learning | PyTorch 2.1+ |
| Audio Processing | Librosa 0.10+ |
| Data Handling | Pandas, NumPy |
| ML Utilities | Scikit-learn |
| Visualization | Matplotlib, Seaborn |

---

## 📋 Requirements

```
torch>=2.1.0
torchaudio>=2.1.0
librosa>=0.10.1
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.66.0
soundfile>=0.12.1
scipy>=1.11.0
```

---

## 🤝 Acknowledgements

- **RAVDESS** — Livingstone & Russo (2018), Zenodo
- **TESS** — Pichora-Fuller & Dupuis (2020), University of Toronto
- Built during the **CodeAlpha AI/ML Internship Program**

---

## 👨‍💻 Author

**Divyan A** — ML Intern @ CodeAlpha
- GitHub: [@divyanA615-web](https://github.com/divyanA615-web)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for the CodeAlpha AI Internship · Speech Emotion Recognition Project
</p>
