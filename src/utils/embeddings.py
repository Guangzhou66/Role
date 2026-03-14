"""
Real text embedding computation.

Priority:
1. sentence-transformers (best quality)
2. TF-IDF + SVD (no GPU needed, always available)

Used by B6 (distribution shift) and the adapter training pipeline.
"""
from __future__ import annotations
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_SBERT_MODEL = None


def get_embeddings(texts: List[str], dim: int = 384,
                   model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Compute real text embeddings for a list of strings.
    Returns ndarray of shape (len(texts), dim).
    """
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)

    # Try sentence-transformers first
    try:
        return _sbert_embeddings(texts, model_name)
    except Exception as e:
        logger.info(f"sentence-transformers unavailable ({e}), using TF-IDF fallback")

    # Fallback: TF-IDF + TruncatedSVD
    return _tfidf_embeddings(texts, dim)


def _sbert_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    """Compute embeddings using sentence-transformers."""
    global _SBERT_MODEL
    from sentence_transformers import SentenceTransformer

    if _SBERT_MODEL is None or _SBERT_MODEL._model_name != model_name:
        logger.info(f"Loading sentence-transformers model: {model_name}")
        _SBERT_MODEL = SentenceTransformer(model_name)
        _SBERT_MODEL._model_name = model_name

    embeddings = _SBERT_MODEL.encode(
        texts, show_progress_bar=len(texts) > 100,
        batch_size=64, normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


def _tfidf_embeddings(texts: List[str], dim: int = 384) -> np.ndarray:
    """
    Compute TF-IDF vectors then reduce to `dim` dimensions via SVD.
    Works without GPU and without any model downloads.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    effective_dim = min(dim, len(texts) - 1, 5000)
    if effective_dim < 1:
        effective_dim = 1

    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    n_components = min(effective_dim, tfidf_matrix.shape[1] - 1)
    if n_components < 1:
        n_components = 1

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(tfidf_matrix)

    # Pad to requested dim if needed
    if reduced.shape[1] < dim:
        pad = np.zeros((reduced.shape[0], dim - reduced.shape[1]),
                        dtype=np.float32)
        reduced = np.hstack([reduced, pad])

    # L2 normalize
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    reduced = reduced / norms

    return reduced.astype(np.float32)
