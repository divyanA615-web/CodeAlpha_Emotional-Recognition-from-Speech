"""
inference/predictor.py — Production inference API for Speech Emotion Recognition.

Usage:
    predictor = EmotionPredictor("checkpoints/best_ser_model.pt")

    # Single file
    result = predictor.predict("audio.wav")
    print(result["emotion"])       # "happy"
    print(result["confidence"])    # 0.93

    # Batch
    results = predictor.predict_batch(["a.wav", "b.wav", "c.wav"])
"""
import logging
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F

from models.ser_model import SERModel
from features.extractor import load_audio, extract_mfcc, normalize_feature, pad_or_trim
from config import Config

logger = logging.getLogger(__name__)


class EmotionPredictor:
    """
    Wraps a trained SERModel for single-file and batch inference.

    Args:
        checkpoint_path: Path to a .pt checkpoint saved by Trainer.
        device:          Inference device (defaults to Config.DEVICE).
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: torch.device = Config.DEVICE,
    ) -> None:
        self.device = device
        self.model  = self._load(checkpoint_path)
        logger.info("EmotionPredictor ready on %s", device)

    # ─── Model Loading ────────────────────────────────────────────────────────

    def _load(self, path: Union[str, Path]) -> SERModel:
        """Restore SERModel from checkpoint, using saved hparams."""
        ckpt   = torch.load(path, map_location=self.device)
        hparams = ckpt.get("hparams", {})

        model = SERModel(
            num_classes   = hparams.get("num_classes",   Config.NUM_CLASSES),
            feature_dim   = hparams.get("feature_dim",   Config.FEATURE_DIM),
            time_frames   = hparams.get("time_frames",   Config.TIME_FRAMES),
            cnn_channels  = hparams.get("cnn_channels",  Config.CNN_CHANNELS),
            lstm_hidden   = hparams.get("lstm_hidden",   Config.LSTM_HIDDEN),
            lstm_layers   = hparams.get("lstm_layers",   Config.LSTM_LAYERS),
            attention_dim = hparams.get("attention_dim", Config.ATTENTION_DIM),
            fc_hidden     = hparams.get("fc_hidden",     Config.FC_HIDDEN),
            dropout       = hparams.get("dropout",       Config.DROPOUT),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        model.to(self.device)

        val_acc = ckpt.get("val_acc", 0.0)
        logger.info("Loaded checkpoint %s  (val_acc = %.4f)", path, val_acc)
        return model

    # ─── Feature Prep ────────────────────────────────────────────────────────

    def _prepare(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """Load WAV → MFCC → normalize → pad/trim → (1, 1, F, T) tensor."""
        signal  = load_audio(str(audio_path))
        feature = extract_mfcc(signal)
        feature = normalize_feature(feature)
        feature = pad_or_trim(feature)
        # (F, T) → (1, 1, F, T) — batch + channel dims
        return torch.from_numpy(feature).unsqueeze(0).unsqueeze(0).to(self.device)

    # ─── Inference ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, audio_path: Union[str, Path]) -> Dict:
        """
        Predict emotion from a single WAV file.

        Args:
            audio_path: Path to a .wav audio file.

        Returns:
            Dict with keys:
                emotion       (str)  — top predicted emotion label
                confidence    (float) — probability of top emotion
                probabilities (dict)  — {emotion: probability} for all classes
        """
        tensor      = self._prepare(audio_path)
        logits, _   = self.model(tensor)
        probs       = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx    = int(np.argmax(probs))
        emotion     = Config.IDX_TO_EMOTION[pred_idx]
        confidence  = float(probs[pred_idx])

        probabilities = {
            Config.IDX_TO_EMOTION[i]: float(probs[i])
            for i in range(len(probs))
        }

        return {
            "emotion":       emotion,
            "confidence":    confidence,
            "probabilities": probabilities,
        }

    @torch.no_grad()
    def predict_batch(self, audio_paths: List[Union[str, Path]]) -> List[Dict]:
        """
        Predict emotions for a list of WAV files.

        Args:
            audio_paths: List of paths to .wav files.

        Returns:
            List of result dicts (same structure as predict()).
        """
        return [self.predict(p) for p in audio_paths]