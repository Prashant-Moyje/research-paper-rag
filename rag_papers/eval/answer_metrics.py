"""Answer-quality metrics: correctness, groundedness, citation quality.

All metrics here are deterministic. The project runs fully locally, so an
LLM-as-judge would mean grading a 3B model's output with a 3B model - weak and
unreproducible. Embedding-based and rule-based measures give the same run the
same score every time, which is what makes the reported numbers meaningful.

Metric definitions
------------------
correctness   Fraction of the question's `must_include` facts that appear in
              the answer. Coarse but objective: the facts are short, verbatim
              strings taken from the papers ("Adam", "bi-encoder", "400M").
              Matching is lenient about hyphens and case, because the corpus
              itself is inconsistent about them.

groundedness  Fraction of the answer's sentences that are semantically
              supported by *some* sentence in the retrieved context, measured
              by cosine similarity between sentence embeddings. This is a
              proxy for NLI entailment, not entailment itself - it detects
              claims with no basis in the context, but cannot catch a claim
              that inverts the context's meaning while reusing its wording.
              That limitation is stated in the report rather than hidden.

citation      Three separate things, often conflated:
              * validity  - do cited block numbers exist? (fabrication check)
              * coverage  - what share of factual sentences carry a citation?
              * precision - does the *specific* block cited actually support
                            the sentence citing it?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..pipeline import Answer
from .goldset import GoldQuestion

_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]|$)")
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# A sentence counts as supported when its best match in the retrieved context
# reaches this cosine similarity. Calibrated by inspection on this corpus:
# verbatim-ish restatements land above 0.75, genuine paraphrases 0.6-0.75,
# and unrelated sentences below 0.45.
SUPPORT_THRESHOLD = 0.60


def split_sentences(text: str, min_words: int = 4) -> list[str]:
    """Split into substantive sentences, ignoring short fragments."""
    out = []
    for raw in _SENTENCE.findall(text):
        s = raw.strip()
        if len(s.split()) >= min_words:
            out.append(s)
    return out


def _normalize_fact(text: str) -> str:
    """Lowercase and neutralise hyphen/space differences for fact matching."""
    return re.sub(r"[\s\-]+", " ", text.lower())


def fact_coverage(answer_text: str, must_include: list[str]) -> tuple[float, list[str]]:
    """Fraction of required facts present, plus the ones that are missing.

    A fact may list alternatives separated by "|", satisfied if any one appears.
    This matters when a question has several equally correct answers: "Why
    self-attention?" is answered validly by computational complexity,
    parallelisation *or* path length, and requiring one specific term would
    score a correct answer as wrong.
    """
    if not must_include:
        return 1.0, []
    haystack = _normalize_fact(answer_text)
    missing = [
        f
        for f in must_include
        if not any(_normalize_fact(alt) in haystack for alt in f.split("|"))
    ]
    return (len(must_include) - len(missing)) / len(must_include), missing


@dataclass
class AnswerScore:
    qid: str
    answered: bool
    # correctness
    correctness: float = 0.0
    missing_facts: list[str] = field(default_factory=list)
    # groundedness
    groundedness: float = 0.0
    unsupported_sentences: list[str] = field(default_factory=list)
    # citations
    citations_valid: bool = True
    invalid_citations: list[int] = field(default_factory=list)
    citation_coverage: float = 0.0
    citation_precision: float = 0.0
    # retrieval context
    retrieval_hit: bool = False
    # attribution
    failure_mode: str | None = None  # None | retrieval | generation | abstention
    # cost
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    completion_tokens: int = 0


class AnswerEvaluator:
    """Scores answers using the same local embedding model as retrieval."""

    def __init__(self, embedder) -> None:
        self.embedder = embedder

    def _similarity_matrix(self, answers: list[str], contexts: list[str]) -> np.ndarray:
        """Cosine similarity between every answer and context sentence."""
        if not answers or not contexts:
            return np.zeros((len(answers), len(contexts)), dtype=np.float32)
        a = self.embedder.embed_passages(answers)
        c = self.embedder.embed_passages(contexts)
        return a @ c.T  # both are L2-normalised

    def score(self, gold: GoldQuestion, answer: Answer, retrieval_hit: bool) -> AnswerScore:
        score = AnswerScore(
            qid=gold.qid,
            answered=answer.answered,
            retrieval_hit=retrieval_hit,
            retrieval_s=answer.retrieval_s,
            generation_s=answer.generation_s,
            completion_tokens=answer.completion_tokens,
        )

        # ---- Abstention cases -------------------------------------------
        if not answer.answered:
            if gold.answerable:
                # Refused a question the corpus can answer.
                score.failure_mode = "retrieval" if not retrieval_hit else "generation"
            else:
                score.correctness = 1.0      # correctly declined
                score.groundedness = 1.0
                score.citation_coverage = 1.0
                score.citation_precision = 1.0
            return score

        if not gold.answerable:
            # Answered something the corpus cannot support - a hallucination.
            score.failure_mode = "hallucination"

        # ---- Correctness -------------------------------------------------
        score.correctness, score.missing_facts = fact_coverage(answer.text, gold.must_include)

        # ---- Groundedness -------------------------------------------------
        answer_sents = split_sentences(answer.text)
        context_sents: list[str] = []
        chunk_of_sent: list[int] = []      # which block each context sentence came from
        for idx, hit in enumerate(answer.hits, start=1):
            for sent in split_sentences(hit.chunk.body, min_words=4):
                context_sents.append(sent)
                chunk_of_sent.append(idx)

        if answer_sents and context_sents:
            sim = self._similarity_matrix(answer_sents, context_sents)
            best = sim.max(axis=1)
            supported = best >= SUPPORT_THRESHOLD
            score.groundedness = float(supported.mean())
            score.unsupported_sentences = [
                s for s, ok in zip(answer_sents, supported) if not ok
            ]

            # ---- Citation precision --------------------------------------
            # For each sentence carrying a citation, does the cited block
            # actually support it (rather than some other retrieved block)?
            checks: list[bool] = []
            for row, sent in enumerate(answer_sents):
                cited = [
                    int(p)
                    for m in _CITATION.finditer(sent)
                    for p in m.group(1).split(",")
                    if p.strip().isdigit()
                ]
                if not cited:
                    continue
                for block in cited:
                    cols = [i for i, c in enumerate(chunk_of_sent) if c == block]
                    if not cols:
                        checks.append(False)
                        continue
                    checks.append(bool(sim[row, cols].max() >= SUPPORT_THRESHOLD))
            score.citation_precision = float(np.mean(checks)) if checks else 0.0

        # ---- Citation validity / coverage ---------------------------------
        if answer.citations is not None:
            score.invalid_citations = answer.citations.invalid_indices
            score.citations_valid = not answer.citations.invalid_indices
            score.citation_coverage = answer.citations.citation_coverage

        # ---- Failure attribution -------------------------------------------
        if gold.answerable and score.failure_mode is None:
            if not retrieval_hit:
                score.failure_mode = "retrieval"
            elif score.correctness < 1.0 or score.groundedness < 0.8:
                score.failure_mode = "generation"
        return score


@dataclass
class EvaluationReport:
    scores: list[AnswerScore]

    def _mean(self, attr: str, subset=None) -> float:
        rows = subset if subset is not None else self.scores
        return float(np.mean([getattr(s, attr) for s in rows])) if rows else 0.0

    @property
    def answerable(self) -> list[AnswerScore]:
        return [s for s in self.scores if s.failure_mode != "hallucination" and s.answered or
                (s.failure_mode in {"retrieval", "generation"})]

    def summary(self, answerable_qids: set[str], unanswerable_qids: set[str]) -> dict:
        ans = [s for s in self.scores if s.qid in answerable_qids]
        una = [s for s in self.scores if s.qid in unanswerable_qids]
        answered = [s for s in ans if s.answered]

        return {
            "n_answerable": len(ans),
            "n_unanswerable": len(una),
            "answer_rate": float(np.mean([s.answered for s in ans])) if ans else 0.0,
            "correctness": self._mean("correctness", answered),
            "groundedness": self._mean("groundedness", answered),
            "citation_validity": (
                float(np.mean([s.citations_valid for s in answered])) if answered else 0.0
            ),
            "citation_coverage": self._mean("citation_coverage", answered),
            "citation_precision": self._mean("citation_precision", answered),
            "retrieval_hit_rate": self._mean("retrieval_hit", ans),
            "abstention_accuracy": (
                float(np.mean([not s.answered for s in una])) if una else 0.0
            ),
            "false_abstention": (
                float(np.mean([not s.answered for s in ans])) if ans else 0.0
            ),
            "retrieval_failures": sum(1 for s in ans if s.failure_mode == "retrieval"),
            "generation_failures": sum(1 for s in ans if s.failure_mode == "generation"),
            "hallucinations": sum(1 for s in una if s.failure_mode == "hallucination"),
            "mean_retrieval_s": self._mean("retrieval_s"),
            "mean_generation_s": self._mean("generation_s", answered),
            "mean_completion_tokens": self._mean("completion_tokens", answered),
        }
