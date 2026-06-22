"""
config.py — Central configuration for Speech Emotion Recognition.
All hyperparameters, paths, and constants live here.
Edit this file to tune the model — never hardcode values elsewhere.
"""
from pathlib import Path
import torch


class Config:
    # ─── Project Paths ────────────────────────────────────────────────────────
    ROOT_DIR       = Path(__file__).parent
    DATA_DIR       = ROOT_DIR / "data" / "raw"
    RAVDESS_DIR    = DATA_DIR / "RAVDESS"          # place RAVDESS actor folders here
    TESS_DIR       = DATA_DIR / "TESS"             # place TESS emotion folders here
    PROCESSED_DIR  = ROOT_DIR / "data" / "processed"
    CHECKPOINTS    = ROOT_DIR / "checkpoints"
    LOGS_DIR       = ROOT_DIR / "logs"
    RESULTS_DIR    = ROOT_DIR / "results"

    # ─── Audio ────────────────────────────────────────────────────────────────
    SAMPLE_RATE    = 22050     # Hz — standard for librosa
    DURATION       = 3         # seconds — clips padded/trimmed to this length
    HOP_LENGTH     = 512
    N_FFT          = 2048
    N_MFCC         = 40        # number of MFCC coefficients
    N_MELS         = 128
    N_CHROMA       = 12
    FMIN           = 20.0
    FMAX           = 8000.0

    # ─── Features ─────────────────────────────────────────────────────────────
    USE_DELTA      = True      # first-order delta (velocity)
    USE_DELTA2     = True      # second-order delta (acceleration)
    # Combined feature dim: 40 * (1 + 1 + 1) = 120
    FEATURE_DIM    = N_MFCC * (1 + int(USE_DELTA) + int(USE_DELTA2))
    # Time frames: int((22050 * 3) / 512) + 1 = 130
    TIME_FRAMES    = int((SAMPLE_RATE * DURATION) / HOP_LENGTH) + 1

    # ─── Emotion Labels ───────────────────────────────────────────────────────
    EMOTIONS = {
        "neutral":   0,
        "calm":      1,
        "happy":     2,
        "sad":       3,
        "angry":     4,
        "fearful":   5,
        "disgust":   6,
        "surprised": 7,
    }
    IDX_TO_EMOTION = {v: k for k, v in EMOTIONS.items()}
    NUM_CLASSES    = len(EMOTIONS)   # 8

    # RAVDESS: filename 3rd field encodes emotion (e.g. 03-01-05-01-01-01-12.wav)
    RAVDESS_MAP = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    # TESS: folder suffix encodes emotion (e.g. YAF_fear/, OAF_happy/)
    TESS_MAP = {
        "angry":   "angry",
        "disgust": "disgust",
        "fear":    "fearful",
        "happy":   "happy",
        "neutral": "neutral",
        "sad":     "sad",
        "ps":      "surprised",   # ps = pleasant surprise
    }

    # ─── Augmentation ─────────────────────────────────────────────────────────
    AUGMENT_PROB        = 0.5
    NOISE_FACTOR        = 0.005
    PITCH_SHIFT_STEPS   = 2
    TIME_STRETCH_RATES  = [0.9, 1.1]

    # ─── Model Architecture ───────────────────────────────────────────────────
    CNN_CHANNELS   = [32, 64, 128, 256]  # output channels per conv block
    KERNEL_SIZE    = (3, 3)
    POOL_SIZE      = (2, 2)
    LSTM_HIDDEN    = 256
    LSTM_LAYERS    = 2
    LSTM_DROPOUT   = 0.3
    ATTENTION_DIM  = 128
    FC_HIDDEN      = 256
    DROPOUT        = 0.4

    # ─── Training ─────────────────────────────────────────────────────────────
    BATCH_SIZE              = 32
    EPOCHS                  = 100
    LEARNING_RATE           = 1e-3
    WEIGHT_DECAY            = 1e-4
    GRAD_CLIP               = 1.0
    LABEL_SMOOTHING         = 0.1
    EARLY_STOPPING_PATIENCE = 15
    SCHEDULER_PATIENCE      = 7
    SCHEDULER_FACTOR        = 0.5
    MIN_LR                  = 1e-6
    MIXED_PRECISION         = True      # FP16 on GPU, ignored on CPU

    # ─── Data Split ───────────────────────────────────────────────────────────
    TEST_SIZE      = 0.15
    VAL_SIZE       = 0.15
    RANDOM_STATE   = 42

    # ─── Hardware ─────────────────────────────────────────────────────────────
    DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS    = 4         # DataLoader workers; set 0 on Windows if errors occur

    # ─── Checkpoint Paths ─────────────────────────────────────────────────────
    BEST_MODEL_PATH = CHECKPOINTS / "best_ser_model.pt"
    LAST_MODEL_PATH = CHECKPOINTS / "last_ser_model.pt"