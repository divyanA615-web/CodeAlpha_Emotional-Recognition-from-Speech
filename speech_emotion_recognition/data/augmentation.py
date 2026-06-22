"""
data/augmentation.py — On-the-fly audio augmentation for training robustness.

Applied randomly during training only (never during validation or inference).
Each call to apply_augmentation() randomly picks ONE technique with probability
Config.AUGMENT_PROB, so the model sees clean and augmented samples equally.
"""
import random
import numpy as np
import librosa

from config import Config


def add_white_noise(
    signal:       np.ndarray,
    noise_factor: float = Config.NOISE_FACTOR,
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise scaled by noise_factor × signal amplitude.

    Rationale: simulates microphone hiss and recording environment noise,
    which is common in real-world speech capture.
    """
    noise = np.random.normal(0.0, noise_factor, size=signal.shape).astype(np.float32)
    return signal + noise


def time_stretch(
    signal: np.ndarray,
    rates:  list = Config.TIME_STRETCH_RATES,
) -> np.ndarray:
    """
    Speed up or slow down without changing pitch.

    Rationale: different speakers have different speaking rates; this teaches
    the model to be invariant to speed while keeping emotional content.

    Args:
        signal: Audio signal at Config.SAMPLE_RATE.
        rates:  List of rate multipliers to sample from (e.g. [0.9, 1.1]).
    """
    rate      = random.choice(rates)
    stretched = librosa.effects.time_stretch(y=signal, rate=rate)

    # Restore original length after stretching
    target = len(signal)
    if len(stretched) < target:
        stretched = np.pad(stretched, (0, target - len(stretched)), mode="constant")
    return stretched[:target].astype(np.float32)


def pitch_shift(
    signal:      np.ndarray,
    sample_rate: int = Config.SAMPLE_RATE,
    n_steps:     int = Config.PITCH_SHIFT_STEPS,
) -> np.ndarray:
    """
    Shift pitch up or down by n_steps semitones.

    Rationale: different speakers (age, gender) have different fundamental
    frequencies; pitch shift makes the model speaker-independent.

    Args:
        signal:      Audio signal.
        sample_rate: Sample rate.
        n_steps:     Maximum semitone shift (randomly ±n_steps).
    """
    steps    = random.choice([-n_steps, n_steps])
    shifted  = librosa.effects.pitch_shift(y=signal, sr=sample_rate, n_steps=steps)
    return shifted.astype(np.float32)


def apply_augmentation(
    signal: np.ndarray,
    prob:   float = Config.AUGMENT_PROB,
) -> np.ndarray:
    """
    Randomly apply one augmentation technique with probability `prob`.

    Args:
        signal: Raw audio signal (normalized, fixed-length).
        prob:   Probability of applying any augmentation (0.0–1.0).

    Returns:
        Augmented or original signal (same shape).
    """
    if random.random() > prob:
        return signal

    fn = random.choice([add_white_noise, time_stretch, pitch_shift])
    try:
        return fn(signal)
    except Exception:
        # Fallback to clean signal if augmentation fails (rare edge case)
        return signal