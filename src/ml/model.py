"""
TF-IDF model: building, persistence and loading.

The recommender is a classic content-based model: book text is vectorised with
TF-IDF and books are compared with cosine similarity. This module owns the
*model* concerns only (vectoriser + matrix), cleanly separated from data
preprocessing (``preprocessing.py``) and inference (``inference.py``).

Persistence uses ``joblib`` so a fitted model can be built once and reused
across processes or deployments (cloud-ready), instead of re-fitting on every
cold start.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Settings, get_settings
from .logging_utils import get_logger

logger = get_logger(__name__)

TEXT_FEATURE_COLUMN = "combined_text"
_MODEL_FILENAME = "tfidf_model.joblib"


@dataclass(frozen=True)
class TfidfModel:
    """A fitted TF-IDF vectoriser together with its document matrix."""

    vectorizer: TfidfVectorizer
    matrix: spmatrix


def build_tfidf(df: pd.DataFrame, settings: Settings | None = None) -> TfidfModel:
    """Fit a TF-IDF vectoriser on the corpus and return the model.

    Parameters
    ----------
    df:
        Preprocessed dataframe containing a ``combined_text`` column.
    settings:
        Optional settings override; defaults to :func:`get_settings`.

    Raises
    ------
    KeyError
        If the required text column is missing.
    """
    settings = settings or get_settings()
    if TEXT_FEATURE_COLUMN not in df.columns:
        raise KeyError(f"Missing required text column: {TEXT_FEATURE_COLUMN!r}")

    vectorizer = TfidfVectorizer(
        max_features=settings.tfidf_max_features,
        ngram_range=settings.tfidf_ngram_range,
        sublinear_tf=True,
        min_df=settings.tfidf_min_df,
    )
    matrix = vectorizer.fit_transform(df[TEXT_FEATURE_COLUMN])
    logger.info(
        "Built TF-IDF matrix: %d docs x %d features",
        matrix.shape[0],
        matrix.shape[1],
    )
    return TfidfModel(vectorizer=vectorizer, matrix=matrix)


def save_model(model: TfidfModel, model_dir: str | Path | None = None) -> Path:
    """Serialise the fitted model to ``model_dir`` and return the file path."""
    model_dir = Path(model_dir or get_settings().model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / _MODEL_FILENAME
    joblib.dump({"vectorizer": model.vectorizer, "matrix": model.matrix}, path)
    logger.info("Saved model to %s", path)
    return path


def model_exists(model_dir: str | Path | None = None) -> bool:
    """Return True if a serialised model is present in ``model_dir``."""
    model_dir = Path(model_dir or get_settings().model_dir)
    return (model_dir / _MODEL_FILENAME).exists()


def load_model(model_dir: str | Path | None = None) -> TfidfModel:
    """Load a previously serialised model from ``model_dir``.

    Raises
    ------
    FileNotFoundError
        If no serialised model exists.
    """
    model_dir = Path(model_dir or get_settings().model_dir)
    path = model_dir / _MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No serialised model at {path}")
    payload = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return TfidfModel(vectorizer=payload["vectorizer"], matrix=payload["matrix"])


def get_or_build_model(
    df: pd.DataFrame,
    model_dir: str | Path | None = None,
    settings: Settings | None = None,
) -> TfidfModel:
    """Load the model from disk if available, otherwise build and persist it.

    This is the cloud-ready entry point: it avoids re-fitting on every cold
    start while still working out-of-the-box when no cache exists.
    """
    if model_exists(model_dir):
        try:
            return load_model(model_dir)
        except Exception as exc:  # noqa: BLE001 - fall back to rebuilding
            logger.warning("Failed to load cached model (%s); rebuilding", exc)
    model = build_tfidf(df, settings=settings)
    try:
        save_model(model, model_dir)
    except Exception as exc:  # noqa: BLE001 - persistence is best-effort
        logger.warning("Could not persist model: %s", exc)
    return model
