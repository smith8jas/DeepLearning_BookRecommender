"""
Shared utilities for the Book Recommender System.
"""
from __future__ import annotations

import random
import re

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python and NumPy for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def clean_query(text: str) -> str:
    """Lowercase, remove punctuation, and collapse whitespace for a user query."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate(text: str, max_chars: int = 300) -> str:
    """Truncate text to max_chars, appending ellipsis if cut."""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"
