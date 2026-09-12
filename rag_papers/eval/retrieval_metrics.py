"""Retrieval metrics scored against the gold section labels.

A retrieved chunk counts as relevant when it comes from the right paper *and*
one of the sections known to contain the answer. Metrics reported:

  Hit@k   - did any relevant chunk appear in the top k? (can the generator
            possibly succeed?)
  MRR     - 1/rank of the first relevant chunk. Sensitive to ordering, which
            matters because the LLM weights earlier context more heavily.
  Recall@k- fraction of the gold sections that were surfaced. Q1 needs several
            sections at once, so hit-rate alone would overstate success.
  P@1     - was the very first result relevant?
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schema import RetrievedChunk
from .goldset import GoldQuestion


def is_relevant(hit: RetrievedChunk, gold: GoldQuestion) -> bool:
    if gold.doc_id and hit.chunk.doc_id != gold.doc_id:
        return False
    return hit.chunk.section in gold.gold_sections


@dataclass
class QueryResult:
    qid: str
    hit: bool
    first_rank: int | None
    reciprocal_rank: float
    recall: float
    precision_at_1: bool
    top_score: float
    sections_found: list[str]
    latency_s: float = 0.0


def score_query(gold: GoldQuestion, hits: list[RetrievedChunk], latency_s: float = 0.0) -> QueryResult:
    first_rank: int | None = None
    for i, hit in enumerate(hits, start=1):
        if is_relevant(hit, gold):
            first_rank = i
            break

    found = {h.chunk.section for h in hits if is_relevant(h, gold)}
    recall = len(found) / len(gold.gold_sections) if gold.gold_sections else 0.0

    return QueryResult(
        qid=gold.qid,
        hit=first_rank is not None,
        first_rank=first_rank,
        reciprocal_rank=(1.0 / first_rank) if first_rank else 0.0,
        recall=recall,
        precision_at_1=bool(hits) and is_relevant(hits[0], gold),
        top_score=hits[0].score if hits else 0.0,
        sections_found=sorted(found),
        latency_s=latency_s,
    )


@dataclass
class RetrievalReport:
    name: str
    results: list[QueryResult]

    @property
    def hit_rate(self) -> float:
        return sum(r.hit for r in self.results) / max(len(self.results), 1)

    @property
    def mrr(self) -> float:
        return sum(r.reciprocal_rank for r in self.results) / max(len(self.results), 1)

    @property
    def recall(self) -> float:
        return sum(r.recall for r in self.results) / max(len(self.results), 1)

    @property
    def p_at_1(self) -> float:
        return sum(r.precision_at_1 for r in self.results) / max(len(self.results), 1)

    @property
    def mean_latency(self) -> float:
        return sum(r.latency_s for r in self.results) / max(len(self.results), 1)

    def summary_row(self) -> str:
        return (
            f"{self.name:<22} {self.hit_rate:>7.1%} {self.p_at_1:>7.1%} "
            f"{self.mrr:>7.3f} {self.recall:>8.1%} {self.mean_latency * 1000:>9.1f}ms"
        )

    @staticmethod
    def header() -> str:
        return (
            f"{'retriever':<22} {'Hit@k':>7} {'P@1':>7} {'MRR':>7} {'Recall':>8} {'latency':>11}\n"
            + "-" * 66
        )
