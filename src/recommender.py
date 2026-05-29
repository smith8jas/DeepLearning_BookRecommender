"""
Core recommendation logic: data loading, TF-IDF model, and recommendation functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["combined_text"]  = df["combined_text"].fillna("")
    df["description"]    = df["description"].fillna("")
    df["published_year"] = pd.to_numeric(df["published_year"], errors="coerce").fillna(0).astype(int)
    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce").fillna(0.0)
    df["popularity_score"] = pd.to_numeric(df["popularity_score"], errors="coerce").fillna(0.0)
    for col in ["clean_title", "clean_authors", "clean_categories", "clean_description"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


def build_tfidf_matrix(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    matrix = vectorizer.fit_transform(df["combined_text"])
    return vectorizer, matrix


def get_categories(df: pd.DataFrame) -> list[str]:
    cats = sorted(df["categories"].dropna().unique().tolist())
    return ["All"] + cats


def _hybrid_score(text_sim: np.ndarray, df: pd.DataFrame, text_weight: float) -> np.ndarray:
    # Log-normalize popularity so mega-popular books (Harry Potter) don't dominate
    log_pop = np.log1p(df["popularity_score"].values)
    pop = log_pop / log_pop.max() if log_pop.max() > 0 else log_pop
    rating = df["average_rating"].values / 5.0
    meta = 0.6 * pop + 0.4 * rating
    return text_weight * text_sim + (1 - text_weight) * meta


def _apply_filters(
    df: pd.DataFrame,
    min_year: int,
    max_year: int,
    min_rating: float,
    category: str,
) -> pd.Index:
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


def _build_result(df: pd.DataFrame, indices, scores: np.ndarray) -> pd.DataFrame:
    rows = df.loc[indices].copy()
    rows["Score"] = scores[indices]
    return rows.rename(columns={
        "title": "Title",
        "authors": "Authors",
        "categories": "Category",
        "average_rating": "Rating",
        "published_year": "Year",
        "description": "Description",
    })[["Title", "Authors", "Category", "Rating", "Year", "Score", "Description"]]


def recommend_by_title(
    query_title: str,
    df: pd.DataFrame,
    tfidf_matrix,
    vectorizer,
    n: int = 5,
    min_year: int = 0,
    max_year: int = 0,
    min_rating: float = 0.0,
    category: str = "All",
    text_weight: float = 0.7,
) -> pd.DataFrame:
    matches = df[df["clean_title"].str.contains(query_title.lower(), na=False)]
    if matches.empty:
        # fallback: search in description and combined text
        matches = df[df["combined_text"].str.contains(query_title.lower(), na=False)]
    if matches.empty:
        return pd.DataFrame()

    idx = matches.index[0]
    sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Give matched book a perfect score so it always appears first
    sim[idx] = 1.0
    scores = _hybrid_score(sim, df, text_weight)
    scores[idx] = 1.0

    valid = _apply_filters(df, min_year, max_year, min_rating, category)

    if len(valid) == 0:
        return pd.DataFrame()

    top_idx = valid[np.argsort(scores[valid])[::-1][:n]]
    return _build_result(df, top_idx, scores)


def browse_by_category(
    category: str,
    df: pd.DataFrame,
    n: int = 20,
    sort_by: str = "popularity",
    min_rating: float = 0.0,
    min_year: int = 0,
    max_year: int = 0,
) -> pd.DataFrame:
    cat = category.strip() if category else ""

    if not cat or cat == "All":
        subset = df.copy()
    else:
        # case-insensitive + strip to avoid mismatch
        mask = df["categories"].fillna("").str.strip().str.lower() == cat.lower()
        subset = df[mask].copy()

    if min_rating > 0:
        subset = subset[subset["average_rating"] >= min_rating]
    if min_year > 0:
        subset = subset[subset["published_year"] >= min_year]
    if max_year > 0:
        subset = subset[subset["published_year"] <= max_year]

    if subset.empty:
        return pd.DataFrame()

    sort_col = "average_rating" if sort_by == "rating" else "popularity_score"
    subset = subset.nlargest(n, sort_col).copy()

    log_pop = np.log1p(subset["popularity_score"].values)
    max_log = log_pop.max()
    subset["Score"] = log_pop / max_log if max_log > 0 else log_pop

    return subset.rename(columns={
        "title": "Title", "authors": "Authors", "categories": "Category",
        "average_rating": "Rating", "published_year": "Year", "description": "Description",
    })[["Title", "Authors", "Category", "Rating", "Year", "Score", "Description"]]


def recommend_by_query(
    query_text: str,
    df: pd.DataFrame,
    tfidf_matrix,
    vectorizer,
    n: int = 5,
    min_year: int = 0,
    max_year: int = 0,
    min_rating: float = 0.0,
    category: str = "All",
    text_weight: float = 0.7,
) -> pd.DataFrame:
    query_vec = vectorizer.transform([query_text.lower()])
    sim = cosine_similarity(query_vec, tfidf_matrix).flatten()

    scores = _hybrid_score(sim, df, text_weight)
    valid = _apply_filters(df, min_year, max_year, min_rating, category)

    if len(valid) == 0:
        return pd.DataFrame()

    top_idx = valid[np.argsort(scores[valid])[::-1][:n]]
    return _build_result(df, top_idx, scores)
