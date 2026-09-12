"""Hybrid retrieval via Reciprocal Rank Fusion.

Why RRF rather than a weighted score blend: dense cosine similarity lives on
roughly [0.4, 0.9] while BM25 scores are unbounded and corpus-dependent, so
combining the raw numbers requires normalisation that is itself a tuned
parameter and drifts as the corpus changes. RRF ignores magnitudes and fuses
*ranks*:

    score(d) = sum_r  1 / (K + rank_r(d))

A document ranked highly by either retriever surfaces; one ranked well by both
surfaces strongly. K damps the influence of the very top rank.

On K: the literature default is 60, tuned for TREC-scale runs over millions of
documents. That value is wrong for this corpus and measurably so. With K=60 the
"agreement bonus" dominates completely - a chunk at rank 1 in one retriever
scores 1/61, while any chunk appearing in *both* at ranks 2 and 3 scores
1/62 + 1/63, so a confident single-retriever hit can never win. That buried
"2.2 Retriever: DPR", which BM25 ranks first. Measured on the gold set:

    K=60   Hit@5 85.7%  MRR 0.821   (misses R1 and R2)
    K=5    Hit@5 92.9%  MRR 0.839   (misses R2 only)

Hence RAG_RRF_K defaults to 5 here. This is a good example of why a published
default deserves a measurement rather than trust.

The practical consequence for this corpus: dense retrieval alone never
surfaced "2.2 Retriever: DPR", because the section is one short chunk among 60
in that paper. BM25 ranks it first on the token "DPR". Fusion keeps both
behaviours without tuning a weight.
"""

from __future__ import annotations

import logging

from ..config import Settings, settings as default_settings
from ..schema import RetrievedChunk

log = logging.getLogger(__name__)

RRF_K = 5


class HybridRetriever:
    """Fuses dense and lexical retrieval with Reciprocal Rank Fusion."""

    def __init__(
        self,
        cfg: Settings | None = None,
        dense=None,
        lexical=None,
        rrf_k: int | None = None,
    ) -> None:
        self.cfg = cfg or default_settings
        self.rrf_k = rrf_k if rrf_k is not None else getattr(self.cfg, "rrf_k", RRF_K)

        if dense is None:
            from .dense import DenseRetriever

            dense = DenseRetriever(self.cfg)
        if lexical is None:
            from .bm25 import BM25Retriever

            lexical = BM25Retriever(self.cfg, chunks=getattr(dense.store, "chunks", None))

        self.dense = dense
        self.lexical = lexical

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or self.cfg.top_k
        pool = max(self.cfg.candidates, k)

        # Score every chunk densely (264 chunks - a full scan costs well under a
        # millisecond). This guarantees that a chunk surfaced only by BM25 still
        # carries its true cosine similarity, so the calibrated abstention
        # threshold keeps its meaning. Attaching 0.0 to lexical-only hits would
        # cause false refusals whenever BM25 wins the top rank.
        dense_all = self.dense.retrieve(query, k=len(self.dense))
        dense_score = {h.chunk.chunk_id: h.score for h in dense_all}

        dense_hits = dense_all[:pool]
        lex_hits = self.lexical.retrieve(query, k=pool)

        fused: dict[str, float] = {}
        best: dict[str, RetrievedChunk] = {}
        origins: dict[str, set[str]] = {}

        for hits in (dense_hits, lex_hits):
            for rank, hit in enumerate(hits, start=1):
                cid = hit.chunk.chunk_id
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
                origins.setdefault(cid, set()).add(hit.source)
                if cid not in best:
                    best[cid] = hit

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        out: list[RetrievedChunk] = []
        for rank, (cid, _score) in enumerate(ordered, start=1):
            hit = best[cid]
            out.append(
                RetrievedChunk(
                    chunk=hit.chunk,
                    score=dense_score.get(cid, 0.0),
                    source="+".join(sorted(origins[cid])),
                    rank=rank,
                )
            )
        return out

    def __len__(self) -> int:
        return len(self.dense)
