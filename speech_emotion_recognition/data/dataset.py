"""
data/dataset.py — PyTorch Dataset and DataLoader factory for SER.

SpeechEmotionDataset loads audio on-the-fly, applies augmentation during
training, and returns (feature_tensor, label_tensor) pairs ready for the model.
"""
import logging
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from features.extractor import load_audio, extract_mfcc, normalize_feature, pad_or_trim
from data.augmentation import apply_augmentation
from config import Config

logger = logging.getLogger(__name__)


class SpeechEmotionDataset(Dataset):
    """
    On-the-fly audio dataset for Speech Emotion Recognition.

    Each __getitem__ call:
        1. Loads WAV file from disk.
        2. (Optional) Applies random augmentation.
        3. Extracts MFCC + delta + delta-delta.
        4. Normalizes per-coefficient.
        5. Pads/trims to fixed length.
        6. Returns (tensor of shape (1, 120, 130), label tensor).

    Args:
        dataframe:  DataFrame with columns [path, label].
        augment:    Enable data augmentation (training only).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        augment:   bool = False,
    ) -> None:
        self.df      = dataframe.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row      = self.df.iloc[idx]
        filepath = row["path"]
        label    = int(row["label"])

        # 1. Load audio
        signal = load_audio(filepath)

        # 2. Augment (training only)
        if self.augment:
            signal = apply_augmentation(signal)

        # 3–5. Feature pipeline
        feature = extract_mfcc(signal)
        feature = normalize_feature(feature)
        feature = pad_or_trim(feature)            # (120, 130)

        # 6. Add channel dim for CNN: (1, 120, 130)
        feat_tensor  = torch.from_numpy(feature).unsqueeze(0)   # float32
        label_tensor = torch.tensor(label, dtype=torch.long)

        return feat_tensor, label_tensor

    @property
    def class_weights(self) -> torch.Tensor:
        """
        Inverse-frequency class weights to handle class imbalance.
        Pass to nn.CrossEntropyLoss(weight=...) during training.

        Returns:
            Float tensor of shape (NUM_CLASSES,).
        """
        counts  = self.df["label"].value_counts().sort_index()
        total   = len(self.df)
        weights = total / (len(counts) * counts.values.astype(float))
        return torch.FloatTensor(weights)


# ─── DataLoader Factory ───────────────────────────────────────────────────────

def build_dataloaders(
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    test_df:    pd.DataFrame,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build train, val, and test DataLoaders.

    Args:
        train_df:    Training split DataFrame.
        val_df:      Validation split DataFrame.
        test_df:     Test split DataFrame.
        batch_size:  Samples per batch.
        num_workers: Worker processes for data loading.

    Returns:
        (train_loader, val_loader, test_loader, class_weights)
    """
    train_ds = SpeechEmotionDataset(train_df, augment=True)
    val_ds   = SpeechEmotionDataset(val_df,   augment=False)
    test_ds  = SpeechEmotionDataset(test_df,  augment=False)

    class_weights = train_ds.class_weights

    _common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(Config.DEVICE.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=False, **_common)
    val_loader   = DataLoader(val_ds,   shuffle=False, **_common)
    test_loader  = DataLoader(test_ds,  shuffle=False, **_common)

    logger.info(
        "DataLoaders ready — Train: %d | Val: %d | Test: %d | Workers: %d",
        len(train_df), len(val_df), len(test_df), num_workers,
    )
    return train_loader, val_loader, test_loader, class_weights