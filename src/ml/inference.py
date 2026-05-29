"""
Inference: turn a fitted TF-IDF model into ranked book recommendations.

Responsibilities (single concern: scoring + ranking):
  * cosine similarity between vectors
  * hybrid score (content similarity + popularity + rating)
  * filtering (year / rating / category)
  * the three public entry points used by the UI

The functions are intentionally small and composable. Behaviour is identical to
the project's original ``recommender.py``; only the structure, typing, logging
and error handling have been improved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Settings, get_settings
from .logging_utils import get_logger

logger = get_logger(__name__)

# Output schema expected by the UI's HTML renderer.
_RESULT_COLUMNS = ["Title", "Authors", "Category", "Rating", "Year", "Score", "Description"]
_RENAME_MAP = {
    "title": "Title",
    "authors": "Authors",
    "categories": "Category",
    "average_rating": "Rating",
    "published_year": "Year",
    "description": "Description",
}


def get_categories(df: pd.DataFrame) -> list[str]:
    """Return the sorted unique category list, prefixed with ``"All"``."""
    if "categories" not in df.columns:
        return ["All"]
    cats = sorted(df["categories"].dropna().unique().tolist())
    return ["All"] + cats


def _hybrid_score(
    text_sim: np.ndarray,
    df: pd.DataFrame,
    text_weight: float,
    settings: Settings,
) -> np.ndarray:
    """Blend content similarity with popularity and rating into one score.

    ``final = text_weight * similarity + (1 - text_weight) * metadata`` where
    ``metadata = pop_w * normalised_log_popularity + rating_w * (rating / 5)``.
    Popularity is log-compressed so mega-popular titles don't dominate.
    """
    log_pop = np.log1p(df["popularity_score"].values)
    pop = log_pop / log_pop.max() if log_pop.max() > 0 else log_pop
    rating = df["average_rating"].values / 5.0
    meta = settings.popularity_weight * pop + settings.rating_weight * rating
    return text_weight * text_sim + (1 - text_weight) * meta


def _apply_filters(
    df: pd.DataFrame,
    min_year: int,
    max_year: int,
    min_rating: float,
    category: str,
) -> pd.Index:
    """Return the index of rows passing the year/rating/category filters."""
    mask = pd.Series(True, index=df.index)
    if min_year > 0:
        mask &= df["published_year"] >= min_year
    if max_year > 0:
        mask &= df["published_year"] <= max_year
    if min_rating > 0:
        mask &= df["average_rating"] >= min_rating
    if category and category != "All":
        mask &= df["categories"] == category
    return df.index[mask]


def _build_result(df: pd.DataFrame, indices: pd.Index, scores: np.ndarray) -> pd.DataFrame:
    """Assemble the UI-facing result frame for the selected rows."""
    rows = df.loc[indices].copy()
    rows["Score"] = scores[indices]
    return rows.rename(columns=_RENAME_MAP)[_RESULT_COLUMNS]


def _rank_top_n(scores: np.ndarray, valid: pd.Index, n: int) -> pd.Index:
    """Return the ``n`` highest-scoring indices among ``valid``."""
    order = np.argsort(scores[valid])[::-1][:n]
    return valid[order]


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
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Recommend books similar to a matched title (title -> title)."""
    settings = settings or get_settings()
    if df.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    needle = query_title.lower()
    matches = df[df["clean_title"].str.contains(needle, na=False)]
    if matches.empty:
        matches = df[df["combined_text"].str.contains(needle, na=False)]
    if matches.empty:
        logger.info("No title match for %r", query_title)
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    idx = matches.index[0]
    sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Pin the matched book to the top.
    sim[idx] = 1.0
    scores = _hybrid_score(sim, df, text_weight, settings)
    scores[idx] = 1.0

    valid = _apply_filters(df, min_year, max_year, min_rating, category)
    if len(valid) == 0:
        return pd.DataFrame(columns=_RESULT_COLUMNS)
    return _build_result(df, _rank_top_n(scores, valid, n), scores)


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
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Recommend books matching a free-text query (description -> books)."""
    settings = settings or get_settings()
    if df.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    # Transform with the fitted vocabulary (never re-fit at inference time).
    query_vec = vectorizer.transform([query_text.lower()])
    sim = cosine_similarity(query_vec, tfidf_matrix).flatten()

    scores = _hybrid_score(sim, df, text_weight, settings)
    valid = _apply_filters(df, min_year, max_year, min_rating, category)
    if len(valid) == 0:
        return pd.DataFrame(columns=_RESULT_COLUMNS)
    return _build_result(df, _rank_top_n(scores, valid, n), scores)


def browse_by_category(
    category: str,
    df: pd.DataFrame,
    n: int = 20,
    sort_by: str = "popularity",
    min_rating: float = 0.0,
    min_year: int = 0,
    max_year: int = 0,
) -> pd.DataFrame:
    """List top books in a category, sorted by popularity or rating."""
    if df.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    cat = category.strip() if category else ""
    if not cat or cat == "All":
        subset = df.copy()
    else:
        mask = df["categories"].fillna("").str.strip().str.lower() == cat.lower()
        subset = df[mask].copy()

    if min_rating > 0:
        subset = subset[subset["average_rating"] >= min_rating]
    if min_year > 0:
        subset = subset[subset["published_year"] >= min_year]
    if max_year > 0:
        subset = subset[subset["published_year"] <= max_year]

    if subset.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    sort_col = "average_rating" if sort_by == "rating" else "popularity_score"
    subset = subset.nlargest(n, sort_col).copy()

    log_pop = np.log1p(subset["popularity_score"].values)
    max_log = log_pop.max()
    subset["Score"] = log_pop / max_log if max_log > 0 else log_pop

    return subset.rename(columns=_RENAME_MAP)[_RESULT_COLUMNS]
