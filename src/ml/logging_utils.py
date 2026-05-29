"""
Logging helpers for the ML layer.

Provides a single, idempotent ``configure_logging`` entry point plus a thin
``get_logger`` wrapper, so every module logs consistently without each one
re-configuring the root logger.
"""
from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once (idempotent).

    Safe to call multiple times: if handlers are already attached, the level is
    updated but no duplicate handlers are added.
    """
    root = logging.getLogger()
    resolved = getattr(logging, level.upper(), logging.INFO)
    if root.handlers:
        root.setLevel(resolved)
        return
    logging.basicConfig(level=resolved, format=_LOG_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
