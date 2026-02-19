"""
Logging Utilities for NAS-YOLO
================================
Standard logging setup with file and console handlers,
plus training-specific meters for loss/metric tracking.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "NAS-YOLO",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up a logger with console and optional file output.

    Args:
        name: Logger name.
        log_file: Optional path to log file.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class AverageMeter:
    """Computes and stores the average and current value.

    Useful for tracking loss and metric values during training.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"


class ProgressMeter:
    """Display training progress with multiple meters."""

    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.num_batches = num_batches
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> str:
        entries = [self.prefix + f" [{batch}/{self.num_batches}]"]
        entries += [str(meter) for meter in self.meters]
        return "  ".join(entries)
