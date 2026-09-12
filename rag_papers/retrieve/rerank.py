"""Cross-encoder reranking.

Bi-encoders (the dense retriever) compress a chunk into one vector before ever
seeing the query, so fine-grained query-document interaction is lost. A
cross-encoder scores the pair jointly and is markedly more accurate - at the
cost of one forward pass per candidate, which is why it reranks a shortlist
rather than the whole corpus.

Motivating evidence from this corpus: for "Which generator model does RAG use
and how large is it?", the section that actually answers it ("2.3 Generator:
BART") sits at fused rank 13, behind chunks that merely discuss generation.
Lexical and dense signals both under-rank it; only reading the query and the
passage together separates them.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~22M params) - the standard
lightweight reranker, fast enough on CPU for a shortlist of 20-50.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from ..config import Settings, settings as default_settings
from ..schema import RetrievedChunk

log = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sentence-transformers is required for reranking.\n"
            "Install it with:  pip install -r requirements.txt"
        ) from exc
    log.info("Loading cross-encoder %s ...", model_name)
    return CrossEncoder(model_name)


class RerankingRetriever:
    """Wraps any retriever, reranking its candidate shortlist."""

    def __init__(
        self,
        base,
        cfg: Settings | None = None,
        model_name: str = DEFAULT_RERANK_MODEL,
        candidates: int | None = None,
    ) -> None:
        self.base = base
        self.cfg = cfg or default_settings
        self.model_name = model_name
        self.candidates = candidates or self.cfg.candidates
        self._model = _load_cross_encoder(model_name)

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or self.cfg.top_k
        shortlist = self.base.retrieve(query, k=max(self.candidates, k))
        if not shortlist:
            return []

        pairs = [(query, hit.chunk.text) for hit in shortlist]
        scores = self._model.predict(pairs, show_progress_bar=False)

        order = sorted(range(len(shortlist)), key=lambda i: -float(scores[i]))[:k]
        return [
            RetrievedChunk(
                chunk=shortlist[i].chunk,
                # Keep the retriever's cosine score so the calibrated abstention
                # threshold still applies; the cross-encoder score is a
                # different, unbounded scale and is not comparable to it.
                score=shortlist[i].score,
                source=f"{shortlist[i].source}+rerank",
                rank=rank,
            )
            for rank, i in enumerate(order, start=1)
        ]

    def __len__(self) -> int:
        return len(self.base)
