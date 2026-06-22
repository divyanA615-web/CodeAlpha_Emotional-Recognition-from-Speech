"""
training/trainer.py — Production training engine for the SER model.

Features:
  • Mixed-precision training (FP16 on CUDA via torch.cuda.amp)
  • Gradient clipping to prevent exploding gradients in the LSTM
  • ReduceLROnPlateau scheduler for adaptive learning rate decay
  • Early stopping with best-model checkpointing
  • Label smoothing in cross-entropy loss for better calibration
  • Inverse-frequency class weighting for imbalanced datasets
"""
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from models.ser_model import SERModel
from config import Config

logger = logging.getLogger(__name__)


# ─── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when a metric stops improving.

    Args:
        patience:  Epochs to wait after last improvement before stopping.
        min_delta: Minimum improvement to count as 'improvement'.
        mode:      'max' (accuracy) or 'min' (loss).
    """

    def __init__(
        self,
        patience:  int   = Config.EARLY_STOPPING_PATIENCE,
        min_delta: float = 1e-4,
        mode:      str   = "max",
    ) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.counter    = 0
        self.best_score: Optional[float] = None

    def __call__(self, score: float) -> bool:
        """Returns True when training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(
                    "Early stopping after %d epochs without improvement.", self.counter
                )
                return True
        return False


# ─── Trainer ─────────────────────────────────────────────────────────────────

class Trainer:
    """
    Encapsulates the full training loop.

    Args:
        model:         SERModel instance (not yet moved to device).
        train_loader:  Training DataLoader.
        val_loader:    Validation DataLoader.
        class_weights: Inverse-frequency weights for cross-entropy loss.
        device:        Target computation device.
    """

    def __init__(
        self,
        model:         SERModel,
        train_loader:  DataLoader,
        val_loader:    DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        device:        torch.device = Config.DEVICE,
    ) -> None:
        self.device       = device
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader

        # Loss with class weighting + label smoothing
        weights = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=Config.LABEL_SMOOTHING,
        )

        # AdamW — decoupled weight decay, better than vanilla Adam for transformers
        self.optimizer = AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler — halve LR when val_acc plateaus
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.MIN_LR,
        )

        self.early_stopping = EarlyStopping(mode="max")

        # Mixed precision — enabled on CUDA only
        self.use_amp = Config.MIXED_PRECISION and device.type == "cuda"
        self.scaler  = GradScaler(enabled=self.use_amp)

        self.history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }

        Config.CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    # ─── Single Epoch Helpers ─────────────────────────────────────────────────

    def _train_epoch(self) -> Tuple[float, float]:
        """One full pass through the training DataLoader."""
        self.model.train()
        total_loss = total_correct = total_n = 0

        for features, labels in self.train_loader:
            features = features.to(self.device, non_blocking=True)
            labels   = labels.to(self.device,   non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                logits, _ = self.model(features)
                loss      = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            # Unscale before clipping so clip works in float32 units
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            bs = features.size(0)
            total_loss    += loss.item() * bs
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n       += bs

        return total_loss / total_n, total_correct / total_n

    @torch.no_grad()
    def _validate_epoch(self) -> Tuple[float, float]:
        """One full pass through the validation DataLoader."""
        self.model.eval()
        total_loss = total_correct = total_n = 0

        for features, labels in self.val_loader:
            features = features.to(self.device, non_blocking=True)
            labels   = labels.to(self.device,   non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits, _ = self.model(features)
                loss      = self.criterion(logits, labels)

            bs = features.size(0)
            total_loss    += loss.item() * bs
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n       += bs

        return total_loss / total_n, total_correct / total_n

    # ─── Checkpointing ───────────────────────────────────────────────────────

    def _save(self, path: Path, epoch: int, val_acc: float) -> None:
        """Save a full training checkpoint."""
        torch.save(
            {
                "epoch":       epoch,
                "val_acc":     val_acc,
                "model_state": self.model.state_dict(),
                "optim_state": self.optimizer.state_dict(),
                "hparams":     self.model.hparams,
                "history":     self.history,
            },
            path,
        )
        logger.debug("Checkpoint saved → %s", path)

    # ─── Main Training Loop ───────────────────────────────────────────────────

    def train(self, epochs: int = Config.EPOCHS) -> Dict[str, list]:
        """
        Run the full training loop.

        Args:
            epochs: Maximum training epochs.

        Returns:
            History dict with keys [train_loss, train_acc, val_loss, val_acc].
        """
        best_val_acc = 0.0
        logger.info("═" * 60)
        logger.info("  Training on %s  |  AMP: %s  |  Params: %s",
                    self.device, self.use_amp, f"{self.model.count_parameters():,}")
        logger.info("═" * 60)

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()

            train_loss, train_acc = self._train_epoch()
            val_loss,   val_acc   = self._validate_epoch()

            elapsed = time.perf_counter() - t0
            lr      = self.optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch [%03d/%d] | "
                "Train loss %.4f acc %.4f | "
                "Val loss %.4f acc %.4f | "
                "LR %.2e | %.1fs",
                epoch, epochs,
                train_loss, train_acc,
                val_loss, val_acc,
                lr, elapsed,
            )

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save(Config.BEST_MODEL_PATH, epoch, val_acc)
                logger.info("  ↳ New best val_acc = %.4f — checkpoint saved.", best_val_acc)

            # Always save last checkpoint (allows resuming)
            self._save(Config.LAST_MODEL_PATH, epoch, val_acc)

            # Scheduler and early-stopping use val_acc (higher is better)
            self.scheduler.step(val_acc)
            if self.early_stopping(val_acc):
                break

        logger.info("Training complete. Best val_acc = %.4f", best_val_acc)
        return self.history