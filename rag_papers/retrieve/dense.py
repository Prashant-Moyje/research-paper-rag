"""Dense (vector) retrieval - the Stage 3 baseline.

Deliberately the simplest thing that can work: embed the query, cosine-rank
every chunk, return the top k. Stage 4 measures this before adding anything
(hybrid lexical search, reranking), so that added complexity has to earn its
place against a real number rather than an assumption.
"""

from __future__ import annotations

import logging

from ..config import Settings, settings as default_settings
from ..ingest.embed import Embedder
from ..ingest.store import get_store
from ..schema import RetrievedChunk

log = logging.getLogger(__name__)


class DenseRetriever:
    """Loads a prebuilt index and serves cosine-similarity queries."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or default_settings
        self.embedder = Embedder(
            self.cfg.embed_model, self.cfg.query_prefix, self.cfg.embed_batch
        )
        self.store = get_store(self.cfg.store, self.cfg.index_dir)
        self.store.load()
        log.info("DenseRetriever ready: %d chunks", len(self.store))

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Return the k most similar chunks, highest score first."""
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        k = k or self.cfg.top_k
        qvec = self.embedder.embed_query(query)
        hits = self.store.search(qvec, k)
        return [
            RetrievedChunk(chunk=c, score=s, source="dense", rank=i)
            for i, (c, s) in enumerate(hits, start=1)
        ]

    def __len__(self) -> int:
        return len(self.store)
