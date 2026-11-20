"""Logging: rich console output plus an optional plain-text file (a run's stdout.log)."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler

_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_ROOT = "qopd"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return ``qopd`` or a child logger (``qopd.<name>``)."""
    if name is None or name == _ROOT:
        return logging.getLogger(_ROOT)
    return logging.getLogger(name if name.startswith(f"{_ROOT}.") else f"{_ROOT}.{name}")


def setup_logging(log_file: str | Path | None = None, level: int | str = "INFO") -> logging.Logger:
    """Configure the ``qopd`` logger once; calling again only adds a missing file handler."""
    logger = get_logger()
    logger.setLevel(level)
    logger.propagate = False
    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        logger.addHandler(RichHandler(rich_tracebacks=True, show_path=False))
    if log_file is not None:
        log_file = Path(log_file).resolve()
        have = {
            Path(h.baseFilename).resolve()
            for h in logger.handlers
            if isinstance(h, logging.FileHandler)
        }
        if log_file not in have:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter(_FMT))
            logger.addHandler(fh)
    return logger
