"""Lexical (BM25) retrieval.

Dense embeddings capture meaning but blur rare surface forms. This corpus is
full of them - "DPR", "BART", "MIPS", "Adam", "LAMBADA" - and the Stage 3
baseline missed the RAG retriever/generator sections entirely even though their
own headings contain the answer tokens verbatim. BM25 scores exactly those rare
terms highly, which makes it complementary rather than redundant.

Two details that matter here:

* The indexed text includes the breadcrumb (paper + section title), so a query
  naming a section ("multi-head attention", "Retriever: DPR") matches the
  heading directly.
* Tokenisation splits on hyphens, so "few-shot" indexes as "few" + "shot" and
  still matches whichever form the query uses. This only works because
  normalisation already repaired the ligatures - without it "fine-tuning" would
  be spelled with U+FB01 and never match.
"""

from __future__ import annotations

import logging
import re

from ..config import Settings, settings as default_settings
from ..ingest.store import NumpyStore
from ..schema import Chunk, RetrievedChunk

log = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")

# Very common words carry no discriminative signal in a corpus that is entirely
# about ML papers; dropping them sharpens BM25's rare-term preference.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it its of on or that the
    this to was were what when where which who why with does do did can could
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics (so hyphens split), drop stopwords."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Retriever:
    """Okapi BM25 over the same chunks the dense index uses."""

    def __init__(self, cfg: Settings | None = None, chunks: list[Chunk] | None = None) -> None:
        self.cfg = cfg or default_settings

        if chunks is None:
            store = NumpyStore(self.cfg.index_dir)
            store.load()
            chunks = store.chunks
        self.chunks = chunks

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "rank-bm25 is required for lexical retrieval.\n"
                "Install it with:  pip install -r requirements.txt"
            ) from exc

        corpus = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(corpus)
        log.info("BM25Retriever ready: %d chunks", len(self.chunks))

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        k = k or self.cfg.top_k

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            RetrievedChunk(
                chunk=self.chunks[i], score=float(scores[i]), source="bm25", rank=rank
            )
            for rank, i in enumerate(order, start=1)
            if scores[i] > 0
        ]

    def __len__(self) -> int:
        return len(self.chunks)
