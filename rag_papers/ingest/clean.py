"""Normalisation of raw PDF text.

Each rule below fixes an artifact that was *measured* on this corpus, not one
assumed from general PDF folklore:

* Ligatures      - "fine-tuning" scores 0 literal hits in the GPT-3 text because
                   pypdf emits U+FB01 ("ﬁne-tuning"); 27 hits after NFKC. Left
                   unfixed this silently breaks BM25/keyword retrieval.
                   Counts: GPT-3 444, RAG 114, Transformer 0.
* Hyphen breaks  - a hyphen at a line break is ambiguous: it may be a soft
                   hyphen from justification ("knowl-edge" -> "knowledge") or a
                   real compound that happened to wrap ("position-wise"). On
                   this corpus the split is ~50/50 across 90 instances, so any
                   fixed rule is wrong about half the time. It matters: joining
                   "fine-tuning"/"few-shot" would break the GPT-3 queries.
                   Resolved against corpus vocabulary instead - see
                   `build_vocabulary` / `resolve_hyphenation`.
* Math spacing   - pypdf drops spaces around italic math: "wherepos", "andi",
                   "dimensiondmodel".
* Page furniture - running headers/footers and bare page numbers add noise.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Ligatures that NFKC handles, plus a few pypdf emits outside the NFKC table.
_EXTRA_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
}

# Quotes/dashes normalised so lexical matching is stable.
_PUNCT = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00ad": "",
}

_HYPHEN_BREAK = re.compile(r"(\w+)-\n(\w+)")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*")
_PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,3}\s*$", re.M)
_ARXIV_STAMP = re.compile(r"^\s*arXiv:\d{4}\.\d{4,5}v\d+\s+\[[^\]]+\]\s+.*$", re.M | re.I)
_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+$", re.M)

# "wherepos" -> "where pos": a lowercase run followed by a known math token.
_MATH_GLUE = re.compile(
    r"\b(where|and|with|let|if|when|dimension|size|of|is|to|by)"
    r"(pos|i|k|d model|dmodel|dk|dv|h|N|n ctx|nctx|K)\b"
)


def build_vocabulary(texts: list[str]) -> Counter[str]:
    """Count every token in the corpus, used to disambiguate line-break hyphens.

    Called on ligature-fixed but not yet hyphen-resolved text, so the evidence
    comes from the many *unbroken* occurrences of the same words elsewhere.
    """
    vocab: Counter[str] = Counter()
    for text in texts:
        for bad, good in _EXTRA_LIGATURES.items():
            text = text.replace(bad, good)
        text = unicodedata.normalize("NFKC", text)
        # Ignore tokens that are themselves split across a line break.
        text = _HYPHEN_BREAK.sub(" ", text)
        for tok in _TOKEN.findall(text):
            vocab[tok.lower()] += 1
    return vocab


def resolve_hyphenation(text: str, vocab: Counter[str] | None = None) -> str:
    """Resolve `A-
B` into either `AB` or `A-B` using corpus evidence.

    Decision order:
      1. If the joined form `AB` is attested elsewhere -> join (syllable split).
      2. If the hyphenated form `A-B` is attested elsewhere -> keep the hyphen.
      3. Otherwise keep the hyphen: technical compounds dominate this domain,
         and BM25 tokenisation splits on hyphens anyway, so a wrongly kept
         hyphen is far cheaper than a wrongly destroyed one.
    """

    def _sub(m: re.Match[str]) -> str:
        a, b = m.group(1), m.group(2)
        if vocab:
            joined = f"{a}{b}".lower()
            hyphenated = f"{a}-{b}".lower()
            n_joined, n_hyphen = vocab.get(joined, 0), vocab.get(hyphenated, 0)
            if n_joined > n_hyphen:
                return f"{a}{b}"
            if n_hyphen > 0:
                return f"{a}-{b}"
        return f"{a}-{b}"

    return _HYPHEN_BREAK.sub(_sub, text)


def normalize_text(text: str, vocab: Counter[str] | None = None) -> str:
    """Apply all cleaning rules. Safe to call on already-clean text.

    Pass `vocab` (from `build_vocabulary`) to enable evidence-based
    hyphenation; without it, hyphens at line breaks are preserved.
    """
    if not text:
        return ""

    for bad, good in _EXTRA_LIGATURES.items():
        text = text.replace(bad, good)

    # NFKC folds remaining compatibility forms (ligatures, fullwidth, some sub/superscripts).
    text = unicodedata.normalize("NFKC", text)

    for bad, good in _PUNCT.items():
        text = text.replace(bad, good)

    # Re-join words split across a line break before collapsing whitespace.
    text = resolve_hyphenation(text, vocab)

    text = strip_inline_citations(text)
    text = _ARXIV_STAMP.sub("", text)
    text = _PAGE_NUMBER_LINE.sub("", text)
    text = _MATH_GLUE.sub(r"\1 \2", text)

    text = _TRAILING_WS.sub("", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


_INLINE_CITATION = re.compile(
    r"\s*\[\s*"
    r"(?:\d+(?:\s*[,;-]\s*\d+)*"            # [26]  [1, 2]  [3-5]  [ 1]
    r"|[A-Z][A-Za-z&]*\+?\s*\d{2}[a-z]?"    # [RWC+19]  [Vas17]
    r"(?:\s*[,;]\s*[A-Z][A-Za-z&]*\+?\s*\d{2}[a-z]?)*)"
    r"\s*\]"
)


def strip_inline_citations(text: str) -> str:
    """Remove the papers' own bibliography markers from body text.

    Two reasons this is necessary rather than cosmetic:

    1. They collide with the prompt's context-block numbering. The generator
       was observed copying "DPR [26]" straight out of a passage, producing a
       citation to block 26 when only 5 blocks existed. The validator catches
       it, but the answer is left with a dangling reference.
    2. Once `strip_references` has removed the bibliography, these markers point
       at nothing, so they carry no information for question answering.

    Corpus counts: 238 numeric ("[26]") and 186 alphanumeric ("[RWC+19]").
    """
    return _INLINE_CITATION.sub("", text)


def strip_references(text: str) -> str:
    """Drop the bibliography.

    Reference lists are dense with paper titles that lexically resemble queries
    but carry no explanatory content, so they are a common source of false
    retrieval hits. Only strips when the heading appears in the last 40% of the
    document, to avoid cutting at an in-body mention of the word.
    """
    pattern = re.compile(r"^\s*(References|Bibliography)\s*$", re.M | re.I)
    matches = list(pattern.finditer(text))
    if not matches:
        return text
    cut = matches[-1].start()
    if cut < len(text) * 0.60:
        return text
    return text[:cut].rstrip()
