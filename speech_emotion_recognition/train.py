"""
train.py — Main training entry point for the SER system.

Usage:
    # Basic run with defaults from config.py
    python train.py

    # Custom hyperparameters
    python train.py --epochs 50 --batch_size 64 --lr 5e-4

    # Resume from last checkpoint
    python train.py --resume checkpoints/last_ser_model.pt

    # Skip TESS, use only RAVDESS
    python train.py --no_tess
"""
import argparse
import logging
import sys

import torch

from config import Config
from utils.logger import setup_logger
from data.loader import build_dataframe, split_data
from data.dataset import build_dataloaders
from models.ser_model import SERModel
from training.trainer import Trainer
from evaluation.metrics import evaluate, plot_confusion_matrix, plot_training_history

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the CNN-BiLSTM-Attention Speech Emotion Recognizer"
    )
    p.add_argument("--epochs",      type=int,   default=Config.EPOCHS,
                   help=f"Max epochs (default {Config.EPOCHS})")
    p.add_argument("--batch_size",  type=int,   default=Config.BATCH_SIZE,
                   help=f"Batch size (default {Config.BATCH_SIZE})")
    p.add_argument("--lr",          type=float, default=Config.LEARNING_RATE,
                   help=f"Learning rate (default {Config.LEARNING_RATE})")
    p.add_argument("--workers",     type=int,   default=Config.NUM_WORKERS,
                   help=f"DataLoader workers (default {Config.NUM_WORKERS})")
    p.add_argument("--resume",      type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--no_ravdess",  action="store_true",
                   help="Exclude RAVDESS dataset")
    p.add_argument("--no_tess",     action="store_true",
                   help="Exclude TESS dataset")
    return p.parse_args()


def main() -> None:
    setup_logger()
    args = parse_args()

    logger.info("╔═══════════════════════════════════════════════════╗")
    logger.info("║  Speech Emotion Recognition — Training            ║")
    logger.info("╚═══════════════════════════════════════════════════╝")
    logger.info("Device  : %s", Config.DEVICE)
    logger.info("AMP     : %s", Config.MIXED_PRECISION and Config.DEVICE.type == "cuda")
    logger.info("Epochs  : %d", args.epochs)
    logger.info("Batch   : %d", args.batch_size)
    logger.info("LR      : %s", args.lr)

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    logger.info("\n[1/5] Loading datasets …")
    df = build_dataframe(
        ravdess_dir=None if args.no_ravdess else Config.RAVDESS_DIR,
        tess_dir   =None if args.no_tess    else Config.TESS_DIR,
    )
    train_df, val_df, test_df = split_data(df)

    # ── 2. Build DataLoaders ──────────────────────────────────────────────────
    logger.info("\n[2/5] Building DataLoaders …")
    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        train_df, val_df, test_df,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # ── 3. Initialize Model ───────────────────────────────────────────────────
    logger.info("\n[3/5] Initializing model …")
    model = SERModel()
    model.summary()

    if args.resume:
        ckpt = torch.load(args.resume, map_location=Config.DEVICE)
        model.load_state_dict(ckpt["model_state"])
        logger.info("Resumed from %s", args.resume)

    # Patch learning rate from CLI if provided
    if args.lr != Config.LEARNING_RATE:
        Config.LEARNING_RATE = args.lr

    # ── 4. Train ──────────────────────────────────────────────────────────────
    logger.info("\n[4/5] Training …")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
    )
    history = trainer.train(epochs=args.epochs)

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    logger.info("\n[5/5] Evaluating on test set …")

    # Load best (not last) checkpoint for final evaluation
    best_ckpt = torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
    model.load_state_dict(best_ckpt["model_state"])

    results = evaluate(model, test_loader)

    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        results,
        save_path=str(Config.RESULTS_DIR / "confusion_matrix.png"),
    )
    plot_training_history(
        history,
        save_path=str(Config.RESULTS_DIR / "training_history.png"),
    )

    logger.info("\n╔══════════════════════════════════════╗")
    logger.info("║  FINAL RESULTS                       ║")
    logger.info("║  Test Accuracy  : %.4f              ║", results["accuracy"])
    logger.info("║  Weighted F1    : %.4f              ║", results["f1_score"])
    logger.info("║  Best model     : %s", Config.BEST_MODEL_PATH)
    logger.info("╚══════════════════════════════════════╝")


if __name__ == "__main__":
    main()