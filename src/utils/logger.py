"""
Centralised logging configuration.

Import get_logger() in any module instead of calling logging.getLogger()
directly, so all loggers share the same format.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("training started")
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a consistent format.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level. Default: INFO.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

