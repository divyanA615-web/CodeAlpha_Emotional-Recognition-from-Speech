"""
data/loader.py — Load RAVDESS and TESS datasets into a unified DataFrame.

RAVDESS naming convention (3rd field = emotion code):
    03-01-05-01-01-01-12.wav  →  emotion code "05" → "angry"

TESS folder naming convention (suffix = emotion):
    OAF_fear/  →  "fear" → "fearful"
    YAF_ps/    →  "ps"   → "surprised"
"""
import glob
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config

logger = logging.getLogger(__name__)


# ─── Dataset Loaders ─────────────────────────────────────────────────────────

def load_ravdess(directory: Path) -> List[Dict]:
    """
    Parse every WAV file under the RAVDESS directory tree.

    Args:
        directory: Root RAVDESS folder (contains Actor_01/ … Actor_24/).

    Returns:
        List of dicts: {path, emotion, label, dataset}.
    """
    if not directory.exists():
        logger.warning("RAVDESS directory not found: %s", directory)
        return []

    records: List[Dict] = []
    files = glob.glob(str(directory / "**" / "*.wav"), recursive=True)

    for filepath in files:
        parts = Path(filepath).stem.split("-")
        if len(parts) < 3:
            logger.debug("Skipping unexpected filename: %s", filepath)
            continue

        emotion_code = parts[2]
        emotion      = Config.RAVDESS_MAP.get(emotion_code)
        if emotion is None:
            logger.debug("Unknown RAVDESS emotion code %s in %s", emotion_code, filepath)
            continue

        records.append(
            {
                "path":    filepath,
                "emotion": emotion,
                "label":   Config.EMOTIONS[emotion],
                "dataset": "RAVDESS",
            }
        )

    logger.info("RAVDESS → %d samples loaded from %s", len(records), directory)
    return records


def load_tess(directory: Path) -> List[Dict]:
    """
    Parse every WAV file under the TESS directory tree.

    TESS folder structure:
        TESS/
            OAF_angry/   *.wav
            OAF_disgust/ *.wav
            YAF_fear/    *.wav  (maps to "fearful")
            YAF_ps/      *.wav  (maps to "surprised")
            ...

    Args:
        directory: Root TESS folder.

    Returns:
        List of dicts: {path, emotion, label, dataset}.
    """
    if not directory.exists():
        logger.warning("TESS directory not found: %s", directory)
        return []

    records: List[Dict] = []

    for folder in directory.iterdir():
        if not folder.is_dir():
            continue

        # Emotion keyword is the last part of the folder name after '_'
        emotion_key = folder.name.lower().split("_")[-1]
        emotion     = Config.TESS_MAP.get(emotion_key)
        if emotion is None:
            logger.debug("Unknown TESS emotion key '%s' in folder %s", emotion_key, folder.name)
            continue

        for wav_file in folder.glob("*.wav"):
            records.append(
                {
                    "path":    str(wav_file),
                    "emotion": emotion,
                    "label":   Config.EMOTIONS[emotion],
                    "dataset": "TESS",
                }
            )

    logger.info("TESS → %d samples loaded from %s", len(records), directory)
    return records


# ─── DataFrame Builder ────────────────────────────────────────────────────────

def build_dataframe(
    ravdess_dir: Optional[Path] = Config.RAVDESS_DIR,
    tess_dir:    Optional[Path] = Config.TESS_DIR,
) -> pd.DataFrame:
    """
    Merge RAVDESS and TESS into one DataFrame and print class statistics.

    Args:
        ravdess_dir: Path to RAVDESS root (None to skip).
        tess_dir:    Path to TESS root (None to skip).

    Returns:
        DataFrame with columns [path, emotion, label, dataset].

    Raises:
        ValueError: If no samples were loaded from either dataset.
    """
    records: List[Dict] = []

    if ravdess_dir:
        records.extend(load_ravdess(Path(ravdess_dir)))
    if tess_dir:
        records.extend(load_tess(Path(tess_dir)))

    if not records:
        raise ValueError(
            "No audio files found!\n"
            f"  RAVDESS expected at: {ravdess_dir}\n"
            f"  TESS    expected at: {tess_dir}\n"
            "Download datasets and place them in data/raw/ as shown above."
        )

    df = pd.DataFrame(records)
    logger.info("Total samples : %d", len(df))
    logger.info("Per-class distribution:\n%s", df["emotion"].value_counts().to_string())
    return df


# ─── Train / Val / Test Split ─────────────────────────────────────────────────

def split_data(
    df:           pd.DataFrame,
    test_size:    float = Config.TEST_SIZE,
    val_size:     float = Config.VAL_SIZE,
    random_state: int   = Config.RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / validation / test split.

    Args:
        df:           Full dataset DataFrame.
        test_size:    Fraction reserved for the test set (e.g. 0.15).
        val_size:     Fraction reserved for validation (e.g. 0.15).
        random_state: Reproducibility seed.

    Returns:
        (train_df, val_df, test_df) — each reset-indexed.
    """
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )

    # Adjust val fraction relative to the remaining train+val pool
    adjusted_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val,
        stratify=train_val["label"],
        random_state=random_state,
    )

    logger.info(
        "Split → Train: %d | Val: %d | Test: %d",
        len(train), len(val), len(test),
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )