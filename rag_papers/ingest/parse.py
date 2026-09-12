"""PDF -> Paper objects.

Uses pypdf: pure-python, no system dependencies, and verified to produce clean
single-column output on this corpus (section headings and inline math survive).
Alternatives considered: PyMuPDF (faster but AGPL), unstructured (heavyweight,
slow, and unnecessary for well-formed arXiv preprints).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pypdf import PdfReader

from ..schema import Page, Paper

log = logging.getLogger(__name__)

# Titles verified by reading page 1 of each PDF during design. The heuristic
# extractor below handles unknown files; this map guarantees clean titles for
# the known corpus (heuristics on PDF title lines are notoriously brittle).
KNOWN_TITLES: dict[str, str] = {
    "1706.03762": "Attention Is All You Need",
    "2005.11401": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "2005.14165": "Language Models are Few-Shot Learners (GPT-3)",
}

_BOILERPLATE = re.compile(
    r"provided proper attribution|arxiv:\d|preprint|under review", re.I
)


def _guess_title(first_page: str, fallback: str) -> str:
    """Pick the first substantial non-boilerplate line as the title."""
    for raw in first_page.splitlines():
        line = raw.strip()
        if len(line) < 8 or _BOILERPLATE.search(line):
            continue
        # Skip author/affiliation lines (emails, daggers, many commas)
        if "@" in line or line.count(",") > 2:
            continue
        return line
    return fallback


def parse_pdf(path: Path) -> Paper:
    """Extract per-page text from a single PDF.

    Raises:
        ValueError: if the PDF yields no extractable text (e.g. a pure scan,
            which would require OCR that this pipeline does not perform).
    """
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # corrupt/encrypted file
        raise ValueError(f"Could not open PDF {path.name}: {exc}") from exc

    pages: list[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            log.warning("%s page %d: extraction failed (%s); skipping", path.name, i, exc)
            text = ""
        pages.append(Page(number=i, text=text))

    total_chars = sum(len(p.text) for p in pages)
    if total_chars < 500:
        raise ValueError(
            f"{path.name}: extracted only {total_chars} characters. "
            "The PDF is likely a scanned image; OCR is required but not supported."
        )

    doc_id = path.stem
    base_id = doc_id.split("v")[0]
    title = KNOWN_TITLES.get(base_id) or _guess_title(pages[0].text, fallback=doc_id)

    log.info("Parsed %s: %d pages, %d chars, title=%r", path.name, len(pages), total_chars, title)
    return Paper(
        doc_id=doc_id, title=title, filename=path.name, n_pages=len(pages), pages=pages
    )


def parse_directory(papers_dir: Path) -> list[Paper]:
    """Parse every PDF in a directory, skipping (not failing on) bad files."""
    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs found in {papers_dir}. Add research papers there first."
        )

    papers: list[Paper] = []
    for pdf in pdfs:
        try:
            papers.append(parse_pdf(pdf))
        except ValueError as exc:
            log.error("Skipping %s: %s", pdf.name, exc)

    if not papers:
        raise RuntimeError(f"None of the {len(pdfs)} PDFs in {papers_dir} could be parsed.")
    return papers
