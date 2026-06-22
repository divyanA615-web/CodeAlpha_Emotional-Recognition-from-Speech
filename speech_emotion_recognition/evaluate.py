"""
evaluate.py — Evaluate any saved checkpoint on the test set.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_ser_model.pt
    python evaluate.py --checkpoint checkpoints/best_ser_model.pt --workers 2
"""
import argparse
import logging

import torch
from torch.utils.data import DataLoader

from utils.logger import setup_logger
from config import Config
from data.loader import build_dataframe, split_data
from data.dataset import SpeechEmotionDataset
from models.ser_model import SERModel
from evaluation.metrics import evaluate, plot_confusion_matrix

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved SER checkpoint")
    p.add_argument("--checkpoint", default=str(Config.BEST_MODEL_PATH))
    p.add_argument("--workers",    type=int, default=Config.NUM_WORKERS)
    p.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    return p.parse_args()


def main() -> None:
    setup_logger(log_to_file=False)
    args = parse_args()

    logger.info("Evaluating checkpoint: %s", args.checkpoint)

    # ── Data ──────────────────────────────────────────────────────────────────
    df = build_dataframe()
    _, _, test_df = split_data(df)

    test_ds     = SpeechEmotionDataset(test_df, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(Config.DEVICE.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    ckpt    = torch.load(args.checkpoint, map_location=Config.DEVICE)
    hparams = ckpt.get("hparams", {})
    model   = SERModel(**{k: v for k, v in hparams.items()}) if hparams else SERModel()
    model.load_state_dict(ckpt["model_state"])

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = evaluate(model, test_loader)

    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        results,
        save_path=str(Config.RESULTS_DIR / "confusion_matrix_eval.png"),
    )

    logger.info("Accuracy : %.4f", results["accuracy"])
    logger.info("F1 Score : %.4f", results["f1_score"])
    logger.info("Plot saved → %s/confusion_matrix_eval.png", Config.RESULTS_DIR)


if __name__ == "__main__":
    main()