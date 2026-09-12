"""Shared data structures passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class Page:
    """One extracted PDF page."""

    number: int          # 1-indexed
    text: str


@dataclass
class Paper:
    """A parsed source document."""

    doc_id: str          # stable slug, e.g. "1706.03762v7"
    title: str
    filename: str
    n_pages: int
    pages: list[Page] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)


@dataclass
class Chunk:
    """An indexed unit of retrievable text.

    `text` is what gets embedded and shown as evidence. It is prefixed with a
    breadcrumb ("Paper > Section") because queries in this domain frequently
    echo section names ("positional encoding", "multi-head attention"), so
    including the heading measurably improves dense-retrieval match.
    """

    chunk_id: str
    doc_id: str
    paper_title: str
    section: str         # e.g. "3.5 Positional Encoding"
    section_number: str  # e.g. "3.5" ("" when unnumbered)
    page_start: int
    page_end: int
    text: str            # breadcrumb + body (embedded / shown)
    body: str            # body only (used for BM25 and exact quoting)
    n_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(**d)

    @property
    def citation(self) -> str:
        pages = (
            f"p.{self.page_start}"
            if self.page_start == self.page_end
            else f"pp.{self.page_start}-{self.page_end}"
        )
        sec = f" § {self.section}" if self.section else ""
        return f"{self.paper_title}{sec} ({pages})"


@dataclass
class RetrievedChunk:
    """A chunk plus its retrieval score and provenance."""

    chunk: Chunk
    score: float
    source: str = "dense"   # dense | bm25 | fused
    rank: int = 0
