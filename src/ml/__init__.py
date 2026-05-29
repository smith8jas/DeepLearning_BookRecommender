"""
ML layer for the book recommender.

Public, stable surface used by the application and the ``recommender`` facade:
data loading, model building/persistence and inference. Internal structure may
evolve without breaking callers as long as these names stay stable.
"""
from __future__ import annotations

from .config import Settings, get_settings
from .inference import (
    browse_by_category,
    get_categories,
    recommend_by_query,
    recommend_by_title,
)
from .model import (
    TfidfModel,
    build_tfidf,
    get_or_build_model,
    load_model,
    model_exists,
    save_model,
)
from .preprocessing import load_dataset

__all__ = [
    "Settings",
    "get_settings",
    "load_dataset",
    "TfidfModel",
    "build_tfidf",
    "save_model",
    "load_model",
    "model_exists",
    "get_or_build_model",
    "get_categories",
    "recommend_by_title",
    "recommend_by_query",
    "browse_by_category",
]
