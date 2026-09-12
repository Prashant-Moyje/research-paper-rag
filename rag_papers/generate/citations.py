"""Citation extraction and validation.

An LLM can emit a citation that looks perfect and refers to nothing - a [7]
when only five blocks were supplied, or a [3] attached to a claim that block 3
does not support. Treating the model's citations as trustworthy is the single
most common weakness in tutorial RAG systems, so here they are parsed and
checked mechanically:

* out-of-range markers are removed from the answer text and reported,
* the set of genuinely cited sources is resolved back to paper/section/page,
* answers that make claims with no citation at all are flagged.

This is what lets the UI distinguish "grounded and attributed" from "sounds
confident".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schema import RetrievedChunk

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE = re.compile(r"[^.!?]+(?:[.!?]|$)")


@dataclass
class CitationReport:
    answer: str                                   # answer after cleaning
    cited_indices: list[int] = field(default_factory=list)
    invalid_indices: list[int] = field(default_factory=list)
    sources: list[RetrievedChunk] = field(default_factory=list)
    uncited_sentences: int = 0
    total_sentences: int = 0

    @property
    def has_citations(self) -> bool:
        return bool(self.cited_indices)

    @property
    def citation_coverage(self) -> float:
        """Fraction of substantive sentences carrying at least one citation."""
        if not self.total_sentences:
            return 0.0
        return 1.0 - (self.uncited_sentences / self.total_sentences)


def _parse_markers(text: str) -> list[int]:
    found: list[int] = []
    for match in _CITATION.finditer(text):
        for part in match.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                found.append(int(part))
    return found


def validate_citations(answer: str, hits: list[RetrievedChunk]) -> CitationReport:
    """Parse, validate and resolve the citations in a generated answer."""
    n = len(hits)
    all_markers = _parse_markers(answer)

    valid = sorted({i for i in all_markers if 1 <= i <= n})
    invalid = sorted({i for i in all_markers if not (1 <= i <= n)})

    cleaned = answer
    if invalid:
        # Strip only the markers that resolve to nothing, leaving valid ones.
        def _scrub(match: re.Match[str]) -> str:
            keep = [
                p.strip()
                for p in match.group(1).split(",")
                if p.strip().isdigit() and 1 <= int(p.strip()) <= n
            ]
            return f"[{', '.join(keep)}]" if keep else ""

        cleaned = _CITATION.sub(_scrub, answer)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    # A sentence counts toward coverage if it is substantive (>= 4 words) OR it
    # carries a citation. Filtering on length alone was asymmetric: a short but
    # properly cited sentence ("Cited one [1].") dropped out of the denominator
    # while a long uncited one stayed, deflating the score.
    sentences = [
        s.strip()
        for s in _SENTENCE.findall(cleaned)
        if len(s.split()) >= 4 or _CITATION.search(s)
    ]
    uncited = sum(1 for s in sentences if not _CITATION.search(s))

    return CitationReport(
        answer=cleaned,
        cited_indices=valid,
        invalid_indices=invalid,
        sources=[hits[i - 1] for i in valid],
        uncited_sentences=uncited,
        total_sentences=len(sentences),
    )
