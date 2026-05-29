"""
Data loading and preprocessing for the book recommender.

This module reproduces the project's existing preprocessing pipeline exactly
(text columns -> empty string, numeric columns -> 0), but exposes it as small,
single-responsibility, reusable functions with type hints, logging and robust
error handling. The same ``load_dataset`` is used for the bundled CSV and for
any user-uploaded dataset, guaranteeing identical treatment in train,
validation and production.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .logging_utils import get_logger

logger = get_logger(__name__)

# --- Column groups (declared once; no inline magic strings) ----------------
TEXT_COLUMNS: tuple[str, ...] = (
    "combined_text",
    "description",
    "clean_title",
    "clean_authors",
    "clean_categories",
    "clean_description",
)
NUMERIC_COLUMNS: tuple[str, ...] = (
    "published_year",
    "average_rating",
    "popularity_score",
)
INTEGER_COLUMNS: tuple[str, ...] = ("published_year",)


def impute_text_columns(df: pd.DataFrame, columns: tuple[str, ...] = TEXT_COLUMNS) -> pd.DataFrame:
    """Fill missing text values with empty strings so TF-IDF can process them.

    Only columns present in ``df`` are touched; the frame is mutated in place
    and returned for convenient chaining.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


def impute_numeric_columns(
    df: pd.DataFrame,
    columns: tuple[str, ...] = NUMERIC_COLUMNS,
    integer_columns: tuple[str, ...] = INTEGER_COLUMNS,
) -> pd.DataFrame:
    """Coerce numeric columns and fill missing values with 0.

    Non-numeric tokens are coerced to NaN first (``errors="coerce"``), then
    filled with 0 to preserve the project's original behaviour. Integer columns
    are cast to ``int`` afterwards.
    """
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if col in integer_columns:
            df[col] = df[col].astype(int)
    return df


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and preprocess the books dataset from a CSV file.

    Parameters
    ----------
    path:
        Filesystem path to the processed books CSV.

    Returns
    -------
    pd.DataFrame
        Imputed dataframe ready for vectorisation.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file cannot be parsed as a non-empty CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading dataset from %s", path)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset is empty: {path}") from exc
    except Exception as exc:  # noqa: BLE001 - surface a clear, wrapped error
        raise ValueError(f"Failed to read dataset {path}: {exc}") from exc

    df = impute_text_columns(df)
    df = impute_numeric_columns(df)

    logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])
    return df
