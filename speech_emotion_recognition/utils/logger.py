"""
utils/logger.py — Logging configuration for SER system.
Call setup_logger() once at the start of any script.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

from config import Config


def setup_logger(name: str = "ser", log_to_file: bool = True) -> logging.Logger:
    """
    Configure root logger with console and optional rotating file output.

    Args:
        name:        Logger namespace label.
        log_to_file: When True, writes timestamped log to Config.LOGS_DIR.

    Returns:
        Configured root logger.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called more than once
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stdout so tqdm plays nicely)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_to_file:
        Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(Config.LOGS_DIR / f"ser_{timestamp}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info("Log file → %s", Config.LOGS_DIR / f"ser_{timestamp}.log")

    return logger