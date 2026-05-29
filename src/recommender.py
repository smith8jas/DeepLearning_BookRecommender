"""
Backwards-compatible facade over the :mod:`ml` package.

The application (``app.py``) imports the same names it always has:
``load_data``, ``build_tfidf_matrix``, ``get_categories``,
``recommend_by_title``, ``recommend_by_query`` and ``browse_by_category``.

The real implementation now lives in the modular, professionally structured
``ml`` package (config / preprocessing / model / inference). This file keeps the
public signatures stable so nothing else in the project needs to change, while
delegating to that clean, testable, cloud-ready core.
"""
from __future__ import annotations

import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from ml import (
    browse_by_category,
    build_tfidf,
    get_categories,
    get_settings,
    load_dataset,
)
from ml import recommend_by_query as _recommend_by_query
from ml import recommend_by_title as _recommend_by_title
from ml.logging_utils import configure_logging

# Configure logging once, using the environment-driven level.
configure_logging(get_settings().log_level)

# Re-exported unchanged (identical signatures).
__all__ = [
    "load_data",
    "build_tfidf_matrix",
    "get_categories",
    "recommend_by_title",
    "recommend_by_query",
    "browse_by_category",
]


def load_data(path: str) -> pd.DataFrame:
    """Load and preprocess the books dataset (facade for ``ml.load_dataset``)."""
    return load_dataset(path)


def build_tfidf_matrix(df: pd.DataFrame) -> tuple[TfidfVectorizer, spmatrix]:
    """Build the TF-IDF model, returning ``(vectorizer, matrix)`` as before."""
    model = build_tfidf(df)
    return model.vectorizer, model.matrix


def recommend_by_title(
    query_title: str,
    df: pd.DataFrame,
    tfidf_matrix: spmatrix,
    vectorizer: TfidfVectorizer,
    n: int = 5,
    min_year: int = 0,
    max_year: int = 0,
    min_rating: float = 0.0,
    category: str = "All",
    text_weight: float = 0.7,
) -> pd.DataFrame:
    """Title-to-title content recommendations (facade)."""
    return _recommend_by_title(
        query_title=query_title,
        df=df,
        tfidf_matrix=tfidf_matrix,
        vectorizer=vectorizer,
        n=n,
        min_year=min_year,
        max_year=max_year,
        min_rating=min_rating,
        category=category,
        text_weight=text_weight,
    )


def recommend_by_query(
    query_text: str,
    df: pd.DataFrame,
    tfidf_matrix: spmatrix,
    vectorizer: TfidfVectorizer,
    n: int = 5,
    min_year: int = 0,
    max_year: int = 0,
    min_rating: float = 0.0,
    category: str = "All",
    text_weight: float = 0.7,
) -> pd.DataFrame:
    """Free-text query recommendations (facade)."""
    return _recommend_by_query(
        query_text=query_text,
        df=df,
        tfidf_matrix=tfidf_matrix,
        vectorizer=vectorizer,
        n=n,
        min_year=min_year,
        max_year=max_year,
        min_rating=min_rating,
        category=category,
        text_weight=text_weight,
    )
