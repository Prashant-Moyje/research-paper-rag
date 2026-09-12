"""Tests for retrieval components.

These use stub retrievers so they run in milliseconds without loading the
embedding model or the index.
"""

from __future__ import annotations

import pytest

from rag_papers.retrieve.bm25 import tokenize
from rag_papers.retrieve.hybrid import HybridRetriever
from rag_papers.schema import Chunk, RetrievedChunk


def _chunk(cid: str, section: str = "S") -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id="d",
        paper_title="P",
        section=section,
        section_number="1",
        page_start=1,
        page_end=1,
        text=f"text {cid}",
        body=f"body {cid}",
        n_tokens=10,
    )


class _StubRetriever:
    """Returns a fixed ranking; `scores` lets dense expose cosine values."""

    def __init__(self, ids: list[str], source: str, scores: dict[str, float] | None = None):
        self.ids = ids
        self.source = source
        self.scores = scores or {}
        self._chunks = {cid: _chunk(cid) for cid in ids}

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or len(self.ids)
        return [
            RetrievedChunk(
                chunk=self._chunks[cid],
                score=self.scores.get(cid, 0.0),
                source=self.source,
                rank=i,
            )
            for i, cid in enumerate(self.ids[:k], start=1)
        ]

    def __len__(self) -> int:
        return len(self.ids)


class TestTokenize:
    def test_splits_on_hyphens(self):
        # So "few-shot" matches whichever form the query uses.
        assert tokenize("few-shot") == ["few", "shot"]

    def test_lowercases_and_drops_stopwords(self):
        assert tokenize("What IS the Adam optimizer") == ["adam", "optimizer"]

    def test_keeps_alphanumeric_terms(self):
        assert "400m" in tokenize("BART-large with 400M parameters")

    def test_empty_query(self):
        assert tokenize("???") == []


class TestHybridFusion:
    def test_reciprocal_rank_fusion_rewards_agreement(self):
        # "b" is ranked well by both retrievers and should win overall.
        dense = _StubRetriever(["a", "b", "c"], "dense", {"a": 0.9, "b": 0.8, "c": 0.7})
        lex = _StubRetriever(["b", "c", "a"], "bm25")
        hybrid = HybridRetriever(dense=dense, lexical=lex)
        out = hybrid.retrieve("q", k=3)
        assert out[0].chunk.chunk_id == "b"

    def test_lexical_only_hit_keeps_true_dense_score(self):
        # Regression test: a chunk surfaced only by BM25 previously received
        # score 0.0, which trips the calibrated abstention gate and causes a
        # false refusal. It must carry its real cosine similarity instead.
        dense = _StubRetriever(["a", "z"], "dense", {"a": 0.9, "z": 0.71})
        lex = _StubRetriever(["z"], "bm25")
        hybrid = HybridRetriever(dense=dense, lexical=lex)
        out = {h.chunk.chunk_id: h for h in hybrid.retrieve("q", k=2)}
        assert out["z"].score == pytest.approx(0.71)
        assert out["z"].score > 0.0

    def test_marks_fusion_source(self):
        dense = _StubRetriever(["a"], "dense", {"a": 0.9})
        lex = _StubRetriever(["a"], "bm25")
        hybrid = HybridRetriever(dense=dense, lexical=lex)
        assert hybrid.retrieve("q", k=1)[0].source == "bm25+dense"

    def test_respects_k(self):
        dense = _StubRetriever(list("abcdef"), "dense", {c: 0.9 for c in "abcdef"})
        lex = _StubRetriever(list("fedcba"), "bm25")
        hybrid = HybridRetriever(dense=dense, lexical=lex)
        assert len(hybrid.retrieve("q", k=3)) == 3

    def test_no_duplicate_chunks(self):
        dense = _StubRetriever(["a", "b"], "dense", {"a": 0.9, "b": 0.8})
        lex = _StubRetriever(["a", "b"], "bm25")
        hybrid = HybridRetriever(dense=dense, lexical=lex)
        out = hybrid.retrieve("q", k=5)
        assert len({h.chunk.chunk_id for h in out}) == len(out)


class TestRetrievalMetrics:
    def test_scores_hit_and_mrr(self):
        from rag_papers.eval.goldset import GoldQuestion
        from rag_papers.eval.retrieval_metrics import score_query

        gold = GoldQuestion("q1", "?", "d", gold_sections=["S2"])
        hits = [
            RetrievedChunk(chunk=_chunk("a", "S1"), score=0.9, rank=1),
            RetrievedChunk(chunk=_chunk("b", "S2"), score=0.8, rank=2),
        ]
        res = score_query(gold, hits)
        assert res.hit is True
        assert res.first_rank == 2
        assert res.reciprocal_rank == pytest.approx(0.5)
        assert res.precision_at_1 is False

    def test_miss_scores_zero(self):
        from rag_papers.eval.goldset import GoldQuestion
        from rag_papers.eval.retrieval_metrics import score_query

        gold = GoldQuestion("q1", "?", "d", gold_sections=["SX"])
        hits = [RetrievedChunk(chunk=_chunk("a", "S1"), score=0.9, rank=1)]
        res = score_query(gold, hits)
        assert res.hit is False
        assert res.reciprocal_rank == 0.0
        assert res.recall == 0.0

    def test_wrong_paper_is_not_relevant(self):
        from rag_papers.eval.goldset import GoldQuestion
        from rag_papers.eval.retrieval_metrics import is_relevant

        gold = GoldQuestion("q1", "?", "other-doc", gold_sections=["S1"])
        hit = RetrievedChunk(chunk=_chunk("a", "S1"), score=0.9, rank=1)
        assert is_relevant(hit, gold) is False


class TestGoldSet:
    def test_every_answerable_question_has_gold_sections(self):
        from rag_papers.eval.goldset import ANSWERABLE

        for gold in ANSWERABLE:
            assert gold.gold_sections, f"{gold.qid} has no gold sections"
            assert gold.doc_id, f"{gold.qid} has no doc_id"

    def test_unanswerable_questions_have_no_gold(self):
        from rag_papers.eval.goldset import UNANSWERABLE

        for gold in UNANSWERABLE:
            assert gold.doc_id is None
            assert not gold.gold_sections

    def test_five_sample_questions_present(self):
        from rag_papers.eval.goldset import SAMPLES

        assert len(SAMPLES) == 5
