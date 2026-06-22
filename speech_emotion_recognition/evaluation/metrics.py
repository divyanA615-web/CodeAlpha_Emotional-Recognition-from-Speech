"""
evaluation/metrics.py — Evaluate trained SER model and generate plots.

Outputs:
  • Per-class precision, recall, F1 (classification_report)
  • Overall accuracy and weighted F1
  • Confusion matrix (raw counts + normalized)
  • Training loss/accuracy curves
"""
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import seaborn as sns

from models.ser_model import SERModel
from config import Config

logger = logging.getLogger(__name__)


# ─── Core Evaluation ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:       SERModel,
    test_loader: DataLoader,
    device:      torch.device = Config.DEVICE,
) -> Dict:
    """
    Run inference on the test set and collect metrics.

    Args:
        model:       Trained SERModel (any device).
        test_loader: Test DataLoader.
        device:      Device to run inference on.

    Returns:
        Dict with keys: predictions, labels, accuracy, f1_score,
                        report (dict), emotion_names.
    """
    model.eval()
    model = model.to(device)

    use_amp    = Config.MIXED_PRECISION and device.type == "cuda"
    all_preds  = []
    all_labels = []

    for features, labels in test_loader:
        features = features.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits, _ = model(features)

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    emotion_names = [Config.IDX_TO_EMOTION[i] for i in range(Config.NUM_CLASSES)]

    accuracy = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="weighted")
    report   = classification_report(
        all_labels, all_preds,
        target_names=emotion_names,
        output_dict=True,
    )

    logger.info("Test accuracy  : %.4f", accuracy)
    logger.info("Weighted F1    : %.4f", f1)
    logger.info(
        "\n%s",
        classification_report(all_labels, all_preds, target_names=emotion_names),
    )

    return {
        "predictions":   all_preds,
        "labels":        all_labels,
        "accuracy":      accuracy,
        "f1_score":      f1,
        "report":        report,
        "emotion_names": emotion_names,
    }


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    results:   Dict,
    save_path: str = None,
) -> None:
    """
    Draw side-by-side confusion matrices (raw counts + row-normalized).

    Args:
        results:   Output dict from evaluate().
        save_path: If given, save the figure to this path (PNG/PDF).
    """
    cm      = confusion_matrix(results["labels"], results["predictions"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    names   = results["emotion_names"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Confusion matrix — counts", "Confusion matrix — row-normalized"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=names, yticklabels=names,
            ax=ax, linewidths=0.5,
        )
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix → %s", save_path)

    plt.close(fig)


def plot_training_history(
    history:   Dict,
    save_path: str = None,
) -> None:
    """
    Plot training and validation loss + accuracy curves.

    Args:
        history:   Dict with keys train_loss, val_loss, train_acc, val_acc.
        save_path: Optional save path.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train", color="#2563EB")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="#DC2626", linestyle="--")
    axes[0].set_title("Loss over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train", color="#2563EB")
    axes[1].plot(epochs, history["val_acc"],   label="Val",   color="#DC2626", linestyle="--")
    axes[1].set_title("Accuracy over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Training curves → %s", save_path)

    plt.close(fig)