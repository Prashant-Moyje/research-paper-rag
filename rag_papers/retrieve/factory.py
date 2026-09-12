"""Retriever selection.

RAG_RETRIEVER picks the strategy. The default is `hybrid` because it was the
best measured configuration on the gold set (see docs/evaluation.md):

    retriever          Hit@5    P@1     MRR   Recall   latency
    dense              78.6%   57.1%   0.645   75.0%     21 ms
    bm25               92.9%   57.1%   0.729   87.5%      0.5 ms
    hybrid (RRF)       92.9%   78.6%   0.818   89.3%     22 ms
    hybrid + rerank    85.7%   57.1%   0.660   82.1%   2098 ms

`rerank` is kept selectable so the negative result stays reproducible, not
because it is recommended - the cross-encoder (trained on MS MARCO web text)
transfers poorly to dense academic prose and cost 100x the latency to lose
7 points of Hit@5.
"""

from __future__ import annotations

from ..config import Settings, settings as default_settings


def get_retriever(cfg: Settings | None = None, name: str | None = None):
    """Build the configured retriever. Shares one loaded index across backends."""
    cfg = cfg or default_settings
    name = (name or cfg.retriever).lower()

    from .dense import DenseRetriever

    dense = DenseRetriever(cfg)

    if name == "dense":
        return dense

    if name == "bm25":
        from .bm25 import BM25Retriever

        return BM25Retriever(cfg, chunks=dense.store.chunks)

    if name in {"hybrid", "rerank"}:
        from .bm25 import BM25Retriever
        from .hybrid import HybridRetriever

        hybrid = HybridRetriever(
            cfg, dense=dense, lexical=BM25Retriever(cfg, chunks=dense.store.chunks)
        )
        if name == "hybrid":
            return hybrid

        from .rerank import RerankingRetriever

        return RerankingRetriever(hybrid, cfg)

    raise ValueError(
        f"Unknown RAG_RETRIEVER={name!r}. Expected: dense | bm25 | hybrid | rerank"
    )
