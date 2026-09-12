"""Section-aware chunking.

Why section-aware rather than fixed-size sliding windows?

Every one of the project's benchmark questions is answered inside a single
*named* section of a paper (Transformer 3.1 / 3.2.2 / 3.5, RAG 2.1-2.3,
GPT-3 2 "Approach"). A fixed-size splitter cuts blindly and will happily sever
the two-sub-layer explanation in 3.1 across a boundary, leaving neither half
independently answerable. Splitting on headings first keeps each explanation
whole, and only then packs the remainder into windows.

Two further consequences of this design:

* Each chunk carries a breadcrumb ("Paper > Section") *inside the embedded
  text*. Queries here routinely echo section titles ("positional encoding",
  "multi-head attention"), so the heading is high-signal for dense retrieval.
* The section label is exact metadata for citations, so answers can be
  attributed to "Attention Is All You Need section 3.5 (p.6)" rather than an
  opaque chunk index.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence

from ..schema import Chunk, Page, Paper
from .clean import normalize_text, strip_references

log = logging.getLogger(__name__)

# Numbered headings: "3", "3.1", "3.2.1" followed by a Title Case phrase.
_SECTION = re.compile(
    r"^(?P<num>\d+(?:\.\d+){0,2})\s+(?P<title>[A-Z][A-Za-z0-9 ,'\-:/&()]{2,70})\s*$",
    re.M,
)
# Unnumbered front matter that still marks a real section.
_NAMED_SECTION = re.compile(r"^(?P<title>Abstract|Introduction|Conclusion)\s*$", re.M)

# A table-of-contents line is a heading whose title ends in a page number
# ("2 Approach 6"). The GPT-3 paper has a full TOC that would otherwise be
# mistaken for 8 real sections.
_TOC_TAIL = re.compile(r"\s+\d{1,3}$")


def approx_token_count(text: str) -> int:
    """Cheap token estimate used when no real tokenizer is supplied.

    Empirically ~1.45 tokens per whitespace word on this corpus (subword
    splitting of technical vocabulary). Only used for planning chunk sizes.
    """
    return int(len(text.split()) * 1.45)


class _Section:
    __slots__ = ("number", "title", "start", "end")

    def __init__(self, number: str, title: str, start: int, end: int) -> None:
        self.number, self.title, self.start, self.end = number, title, start, end

    @property
    def label(self) -> str:
        return f"{self.number} {self.title}".strip()


def _page_offsets(pages: Sequence[Page], sep: str = "\n") -> tuple[str, list[tuple[int, int]]]:
    """Concatenate page texts, returning the text plus (offset, page_no) marks."""
    parts: list[str] = []
    marks: list[tuple[int, int]] = []
    cursor = 0
    for page in pages:
        marks.append((cursor, page.number))
        parts.append(page.text)
        cursor += len(page.text) + len(sep)
    return sep.join(parts), marks


def _page_at(offset: int, marks: list[tuple[int, int]]) -> int:
    """Which page does this character offset fall on?"""
    page = marks[0][1] if marks else 1
    for start, number in marks:
        if offset >= start:
            page = number
        else:
            break
    return page


def _find_sections(text: str) -> list[_Section]:
    """Locate section headings, discarding table-of-contents entries."""
    found: list[tuple[int, str, str]] = []

    for m in _SECTION.finditer(text):
        title = m.group("title").strip()
        if _TOC_TAIL.search(title):      # "Approach 6" -> table of contents
            continue
        found.append((m.start(), m.group("num"), title))

    for m in _NAMED_SECTION.finditer(text):
        found.append((m.start(), "", m.group("title").strip()))

    found.sort(key=lambda t: t[0])

    # Drop duplicate headings that repeat (running headers), keeping the first.
    seen: set[str] = set()
    unique: list[tuple[int, str, str]] = []
    for start, num, title in found:
        key = f"{num} {title}".lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append((start, num, title))

    sections: list[_Section] = []
    for i, (start, num, title) in enumerate(unique):
        end = unique[i + 1][0] if i + 1 < len(unique) else len(text)
        sections.append(_Section(num, title, start, end))
    return sections


def _split_window(
    text: str,
    target_tokens: int,
    overlap: float,
    token_len: Callable[[str], int],
) -> list[tuple[int, int]]:
    """Pack text into overlapping windows, breaking on paragraph/sentence bounds.

    Returns (start, end) character offsets relative to `text`.
    """
    if token_len(text) <= target_tokens:
        return [(0, len(text))]

    # Split into atomic units (paragraphs, then long paragraphs into sentences).
    units: list[tuple[int, int]] = []
    for para in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text):
        p_start, p_end, p_text = para.start(), para.end(), para.group()
        if token_len(p_text) <= target_tokens:
            units.append((p_start, p_end))
            continue
        pos = p_start
        for sent in re.finditer(r".+?(?:[.!?](?=\s|$)|$)", p_text, re.S):
            s, e = p_start + sent.start(), p_start + sent.end()
            if e > pos:
                units.append((max(pos, s), e))
                pos = e
    if not units:
        units = [(0, len(text))]

    # Hard guard: a unit with no sentence or paragraph boundaries (dense result
    # tables in the GPT-3 paper are the real case) can still exceed the budget.
    # Left unsplit it would be silently truncated by the embedding model, so
    # fall back to splitting on whitespace.
    guarded: list[tuple[int, int]] = []
    for u_start, u_end in units:
        if token_len(text[u_start:u_end]) <= target_tokens:
            guarded.append((u_start, u_end))
            continue
        word_spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text[u_start:u_end])]
        if not word_spans:
            continue
        w = 0
        while w < len(word_spans):
            seg_start = word_spans[w][0]
            seg_end = word_spans[w][1]
            k = w
            while (
                k + 1 < len(word_spans)
                and token_len(text[u_start + seg_start : u_start + word_spans[k + 1][1]])
                <= target_tokens
            ):
                k += 1
                seg_end = word_spans[k][1]
            guarded.append((u_start + seg_start, u_start + seg_end))
            if k == w and k + 1 < len(word_spans):
                k += 1  # single word longer than budget; advance to avoid a stall
            w = k + 1
    units = guarded

    windows: list[tuple[int, int]] = []
    i = 0
    while i < len(units):
        start = units[i][0]
        end = units[i][1]
        j = i
        while j + 1 < len(units) and token_len(text[start : units[j + 1][1]]) <= target_tokens:
            j += 1
            end = units[j][1]
        windows.append((start, end))
        if j + 1 >= len(units):
            break
        # Step back by the overlap fraction, measured in units.
        span = max(1, j - i + 1)
        step = max(1, int(span * (1.0 - overlap)))
        i += step
    return windows


def chunk_paper(
    paper: Paper,
    *,
    target_tokens: int = 400,
    overlap: float = 0.15,
    min_tokens: int = 50,
    vocab=None,
    token_len: Callable[[str], int] | None = None,
) -> list[Chunk]:
    """Split one paper into section-aware, page-attributed chunks."""
    token_len = token_len or approx_token_count

    cleaned_pages = [
        Page(number=p.number, text=normalize_text(p.text, vocab)) for p in paper.pages
    ]
    text, marks = _page_offsets(cleaned_pages)

    kept = strip_references(text)
    if len(kept) < len(text):
        log.info("%s: dropped %d chars of references", paper.doc_id, len(text) - len(kept))
    text = kept

    sections = _find_sections(text)
    if not sections:
        log.warning("%s: no sections detected; falling back to whole-document windows",
                    paper.doc_id)
        sections = [_Section("", "", 0, len(text))]

    chunks: list[Chunk] = []
    for section in sections:
        body = text[section.start : section.end].strip()
        if not body:
            continue
        for w_start, w_end in _split_window(body, target_tokens, overlap, token_len):
            piece = body[w_start:w_end].strip()
            n_tok = token_len(piece)
            if n_tok < min_tokens:
                continue
            abs_start = section.start + w_start
            abs_end = section.start + w_end
            breadcrumb = f"{paper.title}"
            if section.label:
                breadcrumb += f" > {section.label}"
            chunks.append(
                Chunk(
                    chunk_id=f"{paper.doc_id}::{len(chunks):04d}",
                    doc_id=paper.doc_id,
                    paper_title=paper.title,
                    section=section.label,
                    section_number=section.number,
                    page_start=_page_at(abs_start, marks),
                    page_end=_page_at(max(abs_start, abs_end - 1), marks),
                    text=f"{breadcrumb}\n\n{piece}",
                    body=piece,
                    n_tokens=n_tok,
                )
            )

    log.info("%s: %d sections -> %d chunks", paper.doc_id, len(sections), len(chunks))
    return chunks


def chunk_papers(papers: Sequence[Paper], **kwargs) -> list[Chunk]:
    """Chunk a corpus, sharing one vocabulary across all papers."""
    from .clean import build_vocabulary

    vocab = kwargs.pop("vocab", None)
    if vocab is None:
        vocab = build_vocabulary([p.full_text for p in papers])

    out: list[Chunk] = []
    for paper in papers:
        out.extend(chunk_paper(paper, vocab=vocab, **kwargs))
    return out
