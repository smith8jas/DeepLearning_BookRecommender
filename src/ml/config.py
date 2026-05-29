"""
Centralised, environment-driven configuration for the recommender ML layer.

All tunable parameters live here and are read from environment variables with
sensible defaults, so the same code runs unchanged in local, CI and cloud
environments. Nothing in the ML layer should hard-code paths or magic numbers;
inject them through :class:`Settings` instead.

Environment variables
----------------------
BOOKWISE_DATA_PATH            Path to the processed books CSV.
BOOKWISE_MODEL_DIR            Directory where the fitted model is persisted.
BOOKWISE_TFIDF_MAX_FEATURES   Max vocabulary size for TF-IDF (default 20000).
BOOKWISE_TFIDF_NGRAM_MAX      Upper bound of the n-gram range (default 2).
BOOKWISE_TFIDF_MIN_DF         Minimum document frequency (default 2).
BOOKWISE_TEXT_WEIGHT          Default content-vs-metadata weight (default 0.7).
BOOKWISE_POP_WEIGHT           Popularity weight inside the metadata score (0.6).
BOOKWISE_RATING_WEIGHT        Rating weight inside the metadata score (0.4).
BOOKWISE_LOG_LEVEL            Logging level (default INFO).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Project root: src/ml/config.py -> ml -> src -> <root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Typed environment readers (single source of truth for parsing/validation)
# ---------------------------------------------------------------------------
def _env_str(name: str, default: str) -> str:
    """Return the env var as a string, falling back to ``default``."""
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    """Return the env var parsed as int, logging and falling back on error."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int for %s=%r; using default %s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    """Return the env var parsed as float, logging and falling back on error."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using default %s", name, raw, default)
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable configuration snapshot for the ML layer."""

    data_path: Path
    model_dir: Path

    # TF-IDF hyper-parameters
    tfidf_max_features: int
    tfidf_ngram_max: int
    tfidf_min_df: int

    # Hybrid scoring weights
    text_weight: float
    popularity_weight: float
    rating_weight: float

    log_level: str

    @property
    def tfidf_ngram_range(self) -> tuple[int, int]:
        """N-gram range tuple consumed by scikit-learn's ``TfidfVectorizer``."""
        return (1, self.tfidf_ngram_max)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build and cache the :class:`Settings` from the current environment.

    Cached so every component shares one consistent configuration instance.
    """
    return Settings(
        data_path=Path(
            _env_str(
                "BOOKWISE_DATA_PATH",
                str(PROJECT_ROOT / "data" / "processed" / "cleaned_books.csv"),
            )
        ),
        model_dir=Path(
            _env_str("BOOKWISE_MODEL_DIR", str(PROJECT_ROOT / "outputs" / "models"))
        ),
        tfidf_max_features=_env_int("BOOKWISE_TFIDF_MAX_FEATURES", 20_000),
        tfidf_ngram_max=_env_int("BOOKWISE_TFIDF_NGRAM_MAX", 2),
        tfidf_min_df=_env_int("BOOKWISE_TFIDF_MIN_DF", 2),
        text_weight=_env_float("BOOKWISE_TEXT_WEIGHT", 0.7),
        popularity_weight=_env_float("BOOKWISE_POP_WEIGHT", 0.6),
        rating_weight=_env_float("BOOKWISE_RATING_WEIGHT", 0.4),
        log_level=_env_str("BOOKWISE_LOG_LEVEL", "INFO"),
    )
