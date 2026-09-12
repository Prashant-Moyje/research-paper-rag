"""Prompt construction for grounded, cited answering.

Design points:

* Context blocks are explicitly numbered [1]..[k] and carry their own
  provenance line, so the model has a concrete token to cite and the citation
  validator has something checkable.
* The abstention instruction is a single exact sentinel string rather than a
  vague "say you don't know", so refusal can be detected by exact match during
  evaluation instead of by fuzzy judgement.
* A worked example is included because small local models (llama3.2:3b) follow
  a demonstrated format far more reliably than a described one. During Stage 3
  testing the 3B model answered correctly but omitted citations entirely until
  the format was shown rather than explained.
"""

from __future__ import annotations

from ..schema import RetrievedChunk

INSUFFICIENT = "INSUFFICIENT EVIDENCE"

SYSTEM_PROMPT = (
    "You are a precise research assistant answering questions about AI research "
    "papers. You answer strictly from the provided context and never rely on "
    "prior knowledge. Every factual sentence you write must carry a bracketed "
    "citation naming the context block it came from."
)

_TEMPLATE = """\
Answer the QUESTION using ONLY the CONTEXT blocks below.

Rules:
1. Use only information present in the CONTEXT. Do not add outside knowledge.
2. Cite the block number in square brackets after each factual claim, like [1] or [2].
3. If the CONTEXT does not contain enough information, reply with exactly:
   {insufficient}
   Do not guess and do not pad the answer.
4. Be specific and concise. Prefer the paper's own terminology.

Example of the required style:
  Question: What optimizer was used?
  Answer: The model was trained with the Adam optimizer [2], using a warmup
  schedule over the first 4000 steps [2].

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def format_context(hits: list[RetrievedChunk], max_chars: int | None = None) -> str:
    """Render retrieved chunks as numbered, attributed blocks."""
    blocks: list[str] = []
    used = 0
    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        header = f"[{i}] {chunk.citation}"
        body = chunk.body.strip()
        block = f"{header}\n{body}"
        if max_chars is not None and used + len(block) > max_chars:
            remaining = max_chars - used - len(header) - 2
            if remaining < 200:
                break
            block = f"{header}\n{body[:remaining]}..."
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list[RetrievedChunk], max_context_chars: int = 8000) -> str:
    """Assemble the full user prompt."""
    return _TEMPLATE.format(
        insufficient=INSUFFICIENT,
        context=format_context(hits, max_context_chars),
        question=question.strip(),
    )
