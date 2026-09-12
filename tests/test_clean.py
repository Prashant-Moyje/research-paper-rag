"""Tests for text normalisation.

Each case here corresponds to an artifact actually measured on the corpus, so
a regression means real retrieval damage, not a cosmetic change.
"""

from __future__ import annotations

from collections import Counter

from rag_papers.ingest.clean import (
    build_vocabulary,
    normalize_text,
    resolve_hyphenation,
    strip_references,
)


class TestLigatures:
    def test_fi_ligature_becomes_searchable(self):
        # The real bug: "fine-tuning" scored 0 hits in the GPT-3 text.
        raw = "We do not ﬁne-tune GPT-3 because of task-speciﬁc concerns."
        out = normalize_text(raw)
        assert "fine-tune" in out
        assert "specific" in out
        assert "ﬁ" not in out

    def test_all_ligature_forms(self):
        out = normalize_text("ﬀ ﬁ ﬂ ﬃ ﬄ")
        assert out == "ff fi fl ffi ffl"


class TestHyphenation:
    def test_joins_syllable_split_when_joined_form_is_attested(self):
        vocab = Counter({"knowledge": 12})
        assert resolve_hyphenation("knowl-\nedge", vocab) == "knowledge"

    def test_keeps_compound_when_hyphenated_form_is_attested(self):
        vocab = Counter({"position-wise": 4})
        assert resolve_hyphenation("position-\nwise", vocab) == "position-wise"

    def test_defaults_to_keeping_hyphen_when_unknown(self):
        # Safer for technical compounds; BM25 splits on hyphens anyway.
        assert resolve_hyphenation("foo-\nbar", Counter()) == "foo-bar"

    def test_preserves_query_critical_terms(self):
        # Joining these would break the GPT-3 benchmark questions.
        vocab = build_vocabulary(["few-shot learning and fine-tuning are common"])
        assert resolve_hyphenation("few-\nshot", vocab) == "few-shot"
        assert resolve_hyphenation("fine-\ntuning", vocab) == "fine-tuning"


class TestMathSpacing:
    def test_reinserts_dropped_spaces(self):
        out = normalize_text("wherepos is the position andi is the dimension")
        assert "where pos" in out
        assert "and i" in out


class TestStripReferences:
    def test_removes_trailing_bibliography(self):
        body = "Body text. " * 80
        out = strip_references(body + "\nReferences\n[1] Someone et al.")
        assert "Someone et al." not in out
        assert "Body text." in out

    def test_ignores_early_mention(self):
        # A "References" heading in the first half is not the bibliography.
        text = "References\n" + ("Real content follows. " * 100)
        assert strip_references(text) == text


class TestIdempotence:
    def test_normalizing_twice_changes_nothing(self):
        raw = "The ﬁne-tuning of knowl-\nedge — see [1]."
        once = normalize_text(raw)
        assert normalize_text(once) == once

    def test_empty_input(self):
        assert normalize_text("") == ""
