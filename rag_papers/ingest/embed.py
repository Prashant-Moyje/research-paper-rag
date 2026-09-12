"""Local embedding via sentence-transformers.

Model: BAAI/bge-small-en-v1.5 (384-d, ~33M params).

Why this model:
  * Runs in seconds on CPU for a corpus this size, with no API key, no network
    at query time, and byte-identical results across runs - which is what makes
    the evaluation numbers reproducible.
  * Strong retrieval quality per parameter (competitive on MTEB retrieval with
    models several times its size).

Alternatives considered:
  * all-MiniLM-L6-v2 - smaller and faster, measurably weaker retrieval.
  * bge-base / bge-large - better, but 3-10x slower on CPU for a gain that a
    264-chunk corpus cannot really exercise.
  * OpenAI text-embedding-3-small - good, but adds a paid API dependency and
    breaks offline reproducibility. Ruled out by the local-only constraint.

Important detail: bge models are *asymmetric*. Queries must be prefixed with an
instruction string; passages must not. Getting this backwards silently degrades
retrieval, so the prefix lives in config and is applied only in `embed_query`.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

log = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Load and cache the encoder (loading costs seconds; do it once)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sentence-transformers is required for embeddings.\n"
            "Install it with:  pip install -r requirements.txt"
        ) from exc

    log.info("Loading embedding model %s ...", model_name)
    return SentenceTransformer(model_name)


class Embedder:
    """Thin wrapper giving symmetric/asymmetric encoding and a token counter."""

    def __init__(self, model_name: str, query_prefix: str = "", batch_size: int = 32) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._model = _load_model(model_name)

    @property
    def dim(self) -> int:
        # sentence-transformers 6.x renamed this; support both so the project
        # works across versions.
        getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        return int(getter())

    @property
    def max_tokens(self) -> int:
        return int(self._model.max_seq_length)

    def token_len(self, text: str) -> int:
        """Exact token count for this model.

        Used by the chunker so chunk sizes are enforced against the real
        tokenizer rather than a word-count estimate - the model truncates
        silently past `max_tokens`, so an estimate is not good enough.
        """
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))

    def embed_passages(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode documents. Returns L2-normalised float32 (N, dim)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,   # so dot product == cosine similarity
            show_progress_bar=show_progress,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a query, applying the asymmetric instruction prefix."""
        text = f"{self.query_prefix} {query}".strip() if self.query_prefix else query
        vec = self._model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vec, dtype=np.float32)[0]
