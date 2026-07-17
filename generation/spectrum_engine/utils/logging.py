"""Structured logging setup for the Spectrum Generation Engine."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from ..core.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """Configure structured logging for the engine.
    
    Args:
        config: Logging configuration.
        
    Returns:
        The root 'spectrum-engine' logger.
    """
    logger = logging.getLogger("spectrum-engine")
    logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))
    logger.handlers.clear()

    # Format
    if config.structured:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    else:
        fmt = "%(levelname)s: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional)
    if config.file:
        log_path = Path(config.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
