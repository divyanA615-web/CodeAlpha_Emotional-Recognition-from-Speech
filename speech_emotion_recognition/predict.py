"""
predict.py — Predict emotion from a speech audio file.

Usage:
    python predict.py --audio path/to/speech.wav
    python predict.py --audio speech.wav --checkpoint checkpoints/best_ser_model.pt
    python predict.py --audio speech.wav --top 3
"""
import argparse
import logging

from utils.logger import setup_logger
from inference.predictor import EmotionPredictor
from config import Config

logger = logging.getLogger(__name__)

EMOTION_EMOJI = {
    "neutral":   "😐",
    "calm":      "😌",
    "happy":     "😊",
    "sad":       "😢",
    "angry":     "😠",
    "fearful":   "😨",
    "disgust":   "🤢",
    "surprised": "😲",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict speech emotion")
    p.add_argument("--audio",      required=True,  help="Path to .wav audio file")
    p.add_argument("--checkpoint", default=str(Config.BEST_MODEL_PATH),
                   help="Model checkpoint path")
    p.add_argument("--top",        type=int, default=None,
                   help="Show top-N emotions (default: all)")
    return p.parse_args()


def main() -> None:
    setup_logger(log_to_file=False)
    args   = parse_args()

    predictor = EmotionPredictor(args.checkpoint)
    result    = predictor.predict(args.audio)

    emotion    = result["emotion"]
    confidence = result["confidence"]
    probs      = result["probabilities"]

    # Sort by probability descending
    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
    if args.top:
        sorted_probs = sorted_probs[: args.top]

    # Pretty print
    bar_width = 28
    print()
    print("┌─────────────────────────────────────────┐")
    print(f"│  File: {args.audio[-30:]:>30}   │")
    print("├─────────────────────────────────────────┤")
    print(f"│  Detected emotion : {emotion.upper():<10}          │")
    print(f"│  Confidence       : {confidence*100:>6.2f}%               │")
    print("├─────────────────────────────────────────┤")
    print("│  All probabilities:                     │")
    for emo, prob in sorted_probs:
        filled = int(prob * bar_width)
        bar    = "█" * filled + "░" * (bar_width - filled)
        emoji  = EMOTION_EMOJI.get(emo, "  ")
        print(f"│  {emoji} {emo:<10} {bar} {prob*100:>5.1f}%  │")
    print("└─────────────────────────────────────────┘")
    print()


if __name__ == "__main__":
    main()