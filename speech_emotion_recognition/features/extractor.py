"""
features/extractor.py — Audio feature extraction for Speech Emotion Recognition.

Pipeline per sample:
  load_audio()  →  extract_mfcc()  →  normalize_feature()  →  pad_or_trim()
                         ↓
              MFCC (40) + Δ (40) + ΔΔ (40)  →  shape (120, 130)
"""
import logging
from typing import Optional

import numpy as np
import librosa

from config import Config

logger = logging.getLogger(__name__)


# ─── Audio Loading ────────────────────────────────────────────────────────────

def load_audio(
    filepath: str,
    sample_rate: int   = Config.SAMPLE_RATE,
    duration:   float  = Config.DURATION,
    offset:     float  = 0.0,
) -> np.ndarray:
    """
    Load a WAV file, pad or trim to a fixed duration, and normalize amplitude.

    Args:
        filepath:    Path to the audio file.
        sample_rate: Target sample rate (resamples if needed).
        duration:    Target length in seconds.
        offset:      Start offset in seconds.

    Returns:
        Float32 array of shape (sample_rate * duration,).
    """
    target_len = int(sample_rate * duration)

    try:
        signal, _ = librosa.load(
            filepath,
            sr=sample_rate,
            duration=duration,
            offset=offset,
            mono=True,
        )
    except Exception as exc:
        logger.warning("Failed to load %s — %s. Returning silence.", filepath, exc)
        return np.zeros(target_len, dtype=np.float32)

    # Pad short clips with zeros
    if len(signal) < target_len:
        signal = np.pad(signal, (0, target_len - len(signal)), mode="constant")

    # Hard trim long clips
    signal = signal[:target_len]

    # Peak-normalize to [-1, 1]
    peak = np.max(np.abs(signal))
    if peak > 0.0:
        signal = signal / peak

    return signal.astype(np.float32)


# ─── Feature Extraction ───────────────────────────────────────────────────────

def extract_mfcc(
    signal:      np.ndarray,
    sample_rate: int  = Config.SAMPLE_RATE,
    n_mfcc:      int  = Config.N_MFCC,
    n_fft:       int  = Config.N_FFT,
    hop_length:  int  = Config.HOP_LENGTH,
    use_delta:   bool = Config.USE_DELTA,
    use_delta2:  bool = Config.USE_DELTA2,
) -> np.ndarray:
    """
    Extract MFCC features with optional delta (velocity) and delta-delta
    (acceleration) coefficients.

    Args:
        signal:      Normalized audio signal.
        sample_rate: Sample rate.
        n_mfcc:      Number of MFCC coefficients.
        n_fft:       FFT window size.
        hop_length:  Hop length between frames.
        use_delta:   Append first-order delta.
        use_delta2:  Append second-order delta.

    Returns:
        Feature array of shape (n_mfcc * [1|2|3], time_frames).
        Default: (120, ~130) for 3 s at 22 050 Hz / hop 512.
    """
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    parts = [mfcc]
    if use_delta:
        parts.append(librosa.feature.delta(mfcc, order=1))
    if use_delta2:
        parts.append(librosa.feature.delta(mfcc, order=2))

    return np.vstack(parts).astype(np.float32)


# ─── Post-Processing ─────────────────────────────────────────────────────────

def normalize_feature(feature: np.ndarray) -> np.ndarray:
    """
    Per-coefficient z-score normalization.

    Args:
        feature: Shape (n_features, time_frames).

    Returns:
        Normalized array of the same shape.
    """
    mean = feature.mean(axis=1, keepdims=True)
    std  = feature.std(axis=1, keepdims=True)
    std  = np.where(std < 1e-8, 1e-8, std)   # guard against zero-std coefficients
    return (feature - mean) / std


def pad_or_trim(
    feature:       np.ndarray,
    target_frames: int = Config.TIME_FRAMES,
) -> np.ndarray:
    """
    Pad (repeat-pad) or trim feature along the time axis to a fixed length.

    Args:
        feature:       Shape (n_features, n_frames).
        target_frames: Target number of time frames.

    Returns:
        Array of shape (n_features, target_frames).
    """
    n_feat, n_frames = feature.shape

    if n_frames < target_frames:
        pad = target_frames - n_frames
        feature = np.pad(feature, ((0, 0), (0, pad)), mode="constant")
    else:
        feature = feature[:, :target_frames]

    return feature.astype(np.float32)


# ─── Master Extractor ────────────────────────────────────────────────────────

def extract_features(filepath: str) -> np.ndarray:
    """
    End-to-end feature extraction: load → MFCC → normalize → pad/trim.

    Args:
        filepath: Path to a WAV audio file.

    Returns:
        Feature tensor ready for the model, shape (FEATURE_DIM, TIME_FRAMES).
        Default: (120, 130).
    """
    signal  = load_audio(filepath)
    feature = extract_mfcc(signal)
    feature = normalize_feature(feature)
    feature = pad_or_trim(feature)
    return feature