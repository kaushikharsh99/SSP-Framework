"""
Utility functions including loggers, checkpointing helpers, seeding, and hardware configuration.
"""

import logging
import random
import os
import torch
import numpy as np


def setup_logger(name: str = "ssp-framework", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns the standard project-wide logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure determinism if needed (may impact performance)
    os.environ["PYTHONHASHSEED"] = str(seed)
