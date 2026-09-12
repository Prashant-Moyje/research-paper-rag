"""Tests for evaluation metrics and the inline-citation cleaner."""

from __future__ import annotations

import numpy as np
import pytest

from rag_papers.eval.answer_metrics import (
    SUPPORT_THRESHOLD,
    AnswerEvaluator,
    fact_coverage,
    split_sentences,
)
from rag_papers.eval.goldset import GoldQuestion
from rag_papers.ingest.clean import strip_inline_citations
from rag_papers.pipeline import Answer
from rag_papers.schema import Chunk, RetrievedChunk


class TestStripInlineCitations:
    def test_removes_numeric_reference(self):
        # The real bug: the model copied "[26]" out of the passage and emitted
        # it as a context-block citation when only 5 blocks existed.
        assert strip_inline_citations("DPR [26] follows a bi-encoder") == (
            "DPR follows a bi-encoder"
        )

    def test_removes_alphanumeric_reference(self):
        assert strip_inline_citations("conditioning [RWC+19], but no") == (
            "conditioning, but no"
        )

    def test_removes_multi_reference(self):
        assert strip_inline_citations("models [47, 51, 52] show") == "models show"

    def test_handles_pdf_spacing_artifact(self):
        # pypdf emits "[ 1]" with an internal space.
        assert strip_inline_citations("each of [ 1] the two") == "each of the two"

    def test_leaves_ordinary_text_alone(self):
        text = "The encoder has two sub-layers."
        assert strip_inline_citations(text) == text


class TestFactCoverage:
    def test_all_facts_present(self):
        assert fact_coverage("Adam with warmup 4000", ["Adam", "4000"])[0] == 1.0

    def test_missing_fact_reported(self):
        score, missing = fact_coverage("Adam optimizer", ["Adam", "4000"])
        assert score == 0.5
        assert missing == ["4000"]

    def test_alternatives_any_match(self):
        # "sin(" must satisfy a requirement written as "sine".
        score, _ = fact_coverage("PE = sin(pos/10000)", ["sine|sin("])
        assert score == 1.0

    def test_hyphen_and_case_insensitive(self):
        assert fact_coverage("a BI-ENCODER design", ["bi-encoder"])[0] == 1.0
        assert fact_coverage("a bi encoder design", ["bi-encoder"])[0] == 1.0

    def test_no_requirements_is_full_credit(self):
        assert fact_coverage("anything", [])[0] == 1.0


class TestSplitSentences:
    def test_ignores_short_fragments(self):
        out = split_sentences("Yes. This is a proper sentence with enough words.")
        assert len(out) == 1

    def test_splits_on_terminators(self):
        out = split_sentences(
            "First sentence has enough words here. Second sentence also has enough."
        )
        assert len(out) == 2


class _FakeEmbedder:
    """Deterministic stand-in: identical strings match, others do not."""

    def __init__(self, supported: set[str]):
        self.supported = supported

    def embed_passages(self, texts, show_progress=False):
        vecs = []
        for t in texts:
            v = np.zeros(2, dtype=np.float32)
            v[0 if t in self.supported else 1] = 1.0
            vecs.append(v)
        return np.vstack(vecs)


def _answer(text: str, *, answered=True, sections=("S1",)) -> Answer:
    hits = [
        RetrievedChunk(
            chunk=Chunk(f"c{i}", "d", "P", s, "1", 1, 1, "t", "Grounded context sentence here.", 10),
            score=0.8,
            rank=i,
        )
        for i, s in enumerate(sections, start=1)
    ]
    return Answer(question="q", text=text, answered=answered, hits=hits)


class TestAnswerEvaluator:
    def test_correct_abstention_scores_full_marks(self):
        gold = GoldQuestion("X1", "?", None, answerable=False)
        ev = AnswerEvaluator(_FakeEmbedder(set()))
        score = ev.score(gold, _answer("", answered=False), retrieval_hit=False)
        assert score.failure_mode is None
        assert score.correctness == 1.0

    def test_answering_an_unanswerable_question_is_a_hallucination(self):
        gold = GoldQuestion("X1", "?", None, answerable=False)
        ev = AnswerEvaluator(_FakeEmbedder(set()))
        score = ev.score(gold, _answer("Some confident claim about nothing."), False)
        assert score.failure_mode == "hallucination"

    def test_missing_gold_section_is_a_retrieval_failure(self):
        gold = GoldQuestion("Q", "?", "d", gold_sections=["SX"], must_include=["zzz"])
        ev = AnswerEvaluator(_FakeEmbedder(set()))
        score = ev.score(gold, _answer("An answer lacking the required fact."), False)
        assert score.failure_mode == "retrieval"

    def test_right_context_wrong_answer_is_a_generation_failure(self):
        gold = GoldQuestion("Q", "?", "d", gold_sections=["S1"], must_include=["zzz"])
        ev = AnswerEvaluator(_FakeEmbedder({"Grounded context sentence here."}))
        score = ev.score(gold, _answer("An answer lacking the required fact."), True)
        assert score.failure_mode == "generation"

    def test_groundedness_detects_unsupported_claim(self):
        gold = GoldQuestion("Q", "?", "d", gold_sections=["S1"])
        ev = AnswerEvaluator(_FakeEmbedder({"Grounded context sentence here."}))
        score = ev.score(gold, _answer("A totally invented claim with no basis."), True)
        assert score.groundedness == 0.0
        assert score.unsupported_sentences

    def test_support_threshold_is_a_documented_constant(self):
        assert 0.0 < SUPPORT_THRESHOLD < 1.0


class TestGoldSetIntegrity:
    def test_sample_questions_have_facts(self):
        from rag_papers.eval.goldset import SAMPLES

        for gold in SAMPLES:
            assert gold.must_include, f"{gold.qid} has no must_include facts"

    def test_qids_are_unique(self):
        from rag_papers.eval.goldset import GOLD

        qids = [g.qid for g in GOLD]
        assert len(qids) == len(set(qids))
