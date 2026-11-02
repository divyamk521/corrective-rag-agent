
"""Local embedding wrapper for document and query vectors."""

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from crag.config import Settings


class Embeddings:
    """Generate normalized embeddings using a local SentenceTransformer."""

    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        model: Any = None,
    ):
        self.model_name = model_name
        self.query_prefix = query_prefix

        # Allow model injection for offline testing.
        self.model = (
            model
            if model is not None
            else SentenceTransformer(model_name)
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "Embeddings":
        """Create the embedding wrapper from application settings."""

        return cls(
            model_name=settings.embedding_model,
            query_prefix=settings.embedding_query_prefix,
        )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Encode text and return normalized vectors."""

        if not texts:
            return []

        if any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError(
                "All input texts must be non-empty strings."
            )

        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError(
                "Embedding model returned an invalid vector shape."
            )

        if not np.isfinite(vectors).all():
            raise ValueError(
                "Embedding model returned non-finite values."
            )

        # Normalize explicitly instead of relying only on the model.
        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        if np.any(norms == 0):
            raise ValueError(
                "Embedding model returned a zero-length vector."
            )

        normalized_vectors = vectors / norms

        return normalized_vectors.tolist()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for document chunks."""

        return self._encode(texts)

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a single user query."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "Query must be a non-empty string."
            )

        prefixed_query = f"{self.query_prefix}{query}"

        vectors = self._encode([prefixed_query])

        return vectors[0]
