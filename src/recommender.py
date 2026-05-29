"""
Facade (bridge) between the UI and the ml/ package.

app.py always imports the same old names from here
(load_data, build_tfidf_matrix, get_categories, recommend_by_title,
recommend_by_query, browse_by_category). This file has no logic of its own:
it forwards those calls to the ml/ package (config, preprocessing, model,
inference), where the real code lives. That way ml/ can be reorganised
internally without touching the frontend.
"""
from __future__ import annotations

from ml import (
    browse_by_category,
    build_tfidf,
    get_categories,
    get_settings,
    load_dataset as load_data,
    recommend_by_query,
    recommend_by_title,
)
from ml.logging_utils import configure_logging

# Set up logging once, using the environment-driven level.
configure_logging(get_settings().log_level)

__all__ = [
    "load_data",
    "build_tfidf_matrix",
    "get_categories",
    "recommend_by_title",
    "recommend_by_query",
    "browse_by_category",
]


def build_tfidf_matrix(df):
    # The UI expects a (vectorizer, matrix) tuple; ml/ returns one object,
    # so we unpack it here to keep the original interface.
    model = build_tfidf(df)
    return model.vectorizer, model.matrix
