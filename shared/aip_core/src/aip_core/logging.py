"""Consistent structured-ish logging across all modules."""

from __future__ import annotations

import logging
import sys

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    from aip_core.config import settings

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(handler)
    logger.setLevel(level or settings.log_level)
    return logger
