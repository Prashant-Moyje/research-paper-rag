"""Tests for chunking invariants and citation validation."""

from __future__ import annotations

from rag_papers.generate.citations import validate_citations
from rag_papers.ingest.chunk import _find_sections, _split_window, approx_token_count
from rag_papers.schema import Chunk, Page, Paper, RetrievedChunk


def _mk_hits(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id=f"d::{i}",
                doc_id="d",
                paper_title="Paper",
                section=f"{i} Section",
                section_number=str(i),
                page_start=i,
                page_end=i,
                text="t",
                body="b",
                n_tokens=10,
            ),
            score=0.9 - i * 0.01,
            rank=i,
        )
        for i in range(1, n + 1)
    ]


class TestSectionDetection:
    def test_finds_numbered_sections(self):
        text = "1 Introduction\nblah\n3.2.2 Multi-Head Attention\nmore\n"
        labels = [s.label for s in _find_sections(text)]
        assert "1 Introduction" in labels
        assert "3.2.2 Multi-Head Attention" in labels

    def test_discards_table_of_contents_entries(self):
        # A TOC line ends with a page number; the GPT-3 paper has a full TOC.
        text = "2 Approach 6\nreal body\n"
        assert [s.label for s in _find_sections(text)] == []

    def test_deduplicates_repeated_headings(self):
        text = "2 Approach\nbody\n2 Approach\nmore body\n"
        assert len(_find_sections(text)) == 1


class TestWindowing:
    def test_short_text_is_one_window(self):
        assert _split_window("short text", 400, 0.15, approx_token_count) == [(0, 10)]

    def test_oversized_unsplittable_unit_is_still_bounded(self):
        # A dense table has no sentence or paragraph boundaries. Without the
        # word-level guard this produced a 5297-token chunk that the embedding
        # model would silently truncate.
        table = " ".join(str(i) for i in range(4000))
        windows = _split_window(table, 100, 0.15, approx_token_count)
        assert len(windows) > 1
        for start, end in windows:
            assert approx_token_count(table[start:end]) <= 100 * 1.35

    def test_windows_cover_the_text(self):
        text = ". ".join(f"Sentence number {i} here" for i in range(200))
        windows = _split_window(text, 120, 0.15, approx_token_count)
        assert windows[0][0] == 0
        assert windows[-1][1] >= len(text) - 2


class TestCitationValidation:
    def test_extracts_valid_markers(self):
        report = validate_citations("Fact one [1]. Fact two [3].", _mk_hits(5))
        assert report.cited_indices == [1, 3]
        assert report.invalid_indices == []
        assert len(report.sources) == 2

    def test_strips_fabricated_citation(self):
        # The model cites [9] when only 3 blocks exist.
        report = validate_citations("A claim [9].", _mk_hits(3))
        assert report.invalid_indices == [9]
        assert "[9]" not in report.answer

    def test_keeps_valid_part_of_mixed_group(self):
        report = validate_citations("Claim [2, 8].", _mk_hits(3))
        assert report.cited_indices == [2]
        assert report.invalid_indices == [8]
        assert "[2]" in report.answer

    def test_detects_uncited_sentences(self):
        text = "This is a long uncited factual sentence about transformers. Cited one [1]."
        report = validate_citations(text, _mk_hits(3))
        assert report.uncited_sentences == 1
        assert 0.0 < report.citation_coverage < 1.0

    def test_no_citations_at_all(self):
        report = validate_citations("Plain answer with no markers whatsoever here.", _mk_hits(3))
        assert not report.has_citations
        assert report.citation_coverage == 0.0


class TestSchema:
    def test_citation_string_single_and_multi_page(self):
        one = Chunk("i", "d", "T", "3.5 PE", "3.5", 6, 6, "t", "b", 5)
        many = Chunk("i", "d", "T", "2 Approach", "2", 6, 7, "t", "b", 5)
        assert one.citation == "T § 3.5 PE (p.6)"
        assert many.citation == "T § 2 Approach (pp.6-7)"

    def test_chunk_roundtrip(self):
        c = Chunk("i", "d", "T", "S", "1", 1, 2, "t", "b", 5)
        assert Chunk.from_dict(c.to_dict()) == c

    def test_paper_full_text_joins_pages(self):
        p = Paper("d", "T", "f.pdf", 2, [Page(1, "one"), Page(2, "two")])
        assert p.full_text == "one\ntwo"
