"""The end-to-end RAG pipeline: question in, grounded cited answer out.

Abstention is enforced at two independent points, because either alone leaks:

  1. A *retrieval gate*. If the best chunk scores below RAG_MIN_SCORE, the
     corpus almost certainly does not cover the question, and we refuse without
     spending an LLM call. This catches the case where the model would happily
     confabulate from weakly-related context.
  2. A *generation instruction*. The prompt supplies an exact sentinel for the
     model to return when the supplied context is insufficient. This catches
     the case where chunks score well but still do not answer the question.

Keeping both, and reporting which one fired, is what makes failures
attributable during evaluation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .config import Settings, settings as default_settings
from .generate.citations import CitationReport, validate_citations
from .generate.prompt import INSUFFICIENT, SYSTEM_PROMPT, build_prompt
from .generate.provider import get_provider
from .retrieve.factory import get_retriever
from .schema import RetrievedChunk

log = logging.getLogger(__name__)


@dataclass
class Answer:
    question: str
    text: str
    answered: bool                      # False when the system abstained
    abstain_reason: str | None = None   # "low_retrieval_score" | "model_declined"
    hits: list[RetrievedChunk] = field(default_factory=list)
    citations: CitationReport | None = None
    retrieval_s: float = 0.0
    generation_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    @property
    def total_s(self) -> float:
        return self.retrieval_s + self.generation_s

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0


class RAGPipeline:
    """Wires retrieval, the confidence gate, generation and citation checking."""

    def __init__(self, cfg: Settings | None = None, retriever=None) -> None:
        self.cfg = cfg or default_settings
        self.retriever = retriever or get_retriever(self.cfg)
        self.provider = get_provider(self.cfg)

    def health(self) -> tuple[bool, str]:
        return self.provider.health()

    def answer(self, question: str, k: int | None = None) -> Answer:
        """Answer one question end to end."""
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")
        k = k or self.cfg.top_k

        t0 = time.perf_counter()
        hits = self.retriever.retrieve(question, k=k)
        retrieval_s = time.perf_counter() - t0

        # --- Gate 1: retrieval confidence -------------------------------
        if not hits or hits[0].score < self.cfg.min_score:
            best = hits[0].score if hits else 0.0
            log.info("Abstaining: top score %.3f < %.3f", best, self.cfg.min_score)
            return Answer(
                question=question,
                text=(
                    "I could not find sufficient evidence in the indexed papers to "
                    "answer this question."
                ),
                answered=False,
                abstain_reason="low_retrieval_score",
                hits=hits,
                retrieval_s=retrieval_s,
                model=self.cfg.ollama_model,
            )

        prompt = build_prompt(question, hits)
        try:
            response = self.provider.generate(prompt, system=SYSTEM_PROMPT)
        except RuntimeError as exc:
            raise RuntimeError(f"Generation failed: {exc}") from exc

        # --- Gate 2: model-declared insufficiency -----------------------
        if INSUFFICIENT.lower() in response.text.lower():
            return Answer(
                question=question,
                text=(
                    "The retrieved passages do not contain enough information to "
                    "answer this question."
                ),
                answered=False,
                abstain_reason="model_declined",
                hits=hits,
                retrieval_s=retrieval_s,
                generation_s=response.latency_s,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                model=response.model,
            )

        report = validate_citations(response.text, hits)
        if report.invalid_indices:
            log.warning(
                "Model cited non-existent blocks %s (only %d supplied)",
                report.invalid_indices,
                len(hits),
            )

        return Answer(
            question=question,
            text=report.answer,
            answered=True,
            hits=hits,
            citations=report,
            retrieval_s=retrieval_s,
            generation_s=response.latency_s,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            model=response.model,
        )
