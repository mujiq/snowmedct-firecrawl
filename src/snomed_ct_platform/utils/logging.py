"""
Logging Configuration for SNOMED-CT Platform

This module sets up structured logging with appropriate formatting and handlers.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    rotation: str = "10 MB",
    retention: str = "1 week"
) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        rotation: Log file rotation setting
        retention: Log file retention setting
    """
    # Remove default handler
    logger.remove()
    
    # Console handler with colored output
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True,
    )
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(exist_ok=True)
        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | "
                   "{level: <8} | "
                   "{name}:{function}:{line} - "
                   "{message}",
            rotation=rotation,
            retention=retention,
            compression="zip",
        )
    
    logger.info(f"Logging initialized with level: {log_level}")


def get_logger(name: str):
    """Get a logger instance for the specified module."""
    return logger.bind(name=name) 