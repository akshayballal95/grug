"""Chunking: sentence boundaries, code passthrough, and structure preservation."""

from __future__ import annotations

import pytest

from grug.base import CompressionResult, CompressorBackend, count_tokens
from grug.chunking import (
    DEFAULT_CHUNK_TOKENS,
    INLINE_CODE_RE,
    URL_RE,
    chunk_document,
    compress_document,
    protect_spans,
    rejoin,
    restore_spans,
    split_sentences,
)


class _Upper(CompressorBackend):
    """Records what it was handed and shouts it back."""

    name = "upper-test"

    def __init__(self):
        self.seen: list[str] = []

    def compress(self, text, rate=0.5, **kwargs):
        self.seen.append(text)
        return CompressionResult.build(text, text.upper().strip(), self.name)


# -- sentence splitting -----------------------------------------------------


def test_splits_on_sentence_ends():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_keeps_trailing_fragment():
    assert split_sentences("Done. And more") == ["Done.", "And more"]


@pytest.mark.parametrize("abbrev", ["e.g.", "i.e.", "Dr.", "vs.", "etc.", "Inc."])
def test_abbreviations_do_not_end_sentences(abbrev):
    text = f"See {abbrev} the appendix for details."
    assert split_sentences(text) == [text]


def test_decimals_do_not_end_sentences():
    text = "The lag was 1.2 seconds on average."
    assert split_sentences(text) == [text]


def test_initials_do_not_end_sentences():
    text = "J. R. Tolkien wrote it."
    assert split_sentences(text) == [text]


def test_quoted_sentence_end_splits_after_the_quote():
    assert split_sentences('He said "no." Then he left.') == ['He said "no."', "Then he left."]


# -- chunk sizing -----------------------------------------------------------


def test_short_document_is_one_chunk():
    chunks = chunk_document("A short paragraph of prose.", max_tokens=DEFAULT_CHUNK_TOKENS)
    assert len(chunks) == 1
    assert chunks[0].compressible


def test_chunks_respect_the_token_ceiling():
    text = " ".join(f"Sentence number {i} carries a little content." for i in range(200))
    chunks = chunk_document(text, max_tokens=60)
    assert len(chunks) > 1
    for chunk in chunks:
        assert count_tokens(chunk.text) <= 60


def test_chunks_break_on_sentence_boundaries():
    text = " ".join(f"Sentence number {i} carries a little content." for i in range(60))
    chunks = chunk_document(text, max_tokens=60)
    for chunk in chunks[:-1]:
        assert chunk.text.rstrip().endswith(".")


def test_oversized_single_sentence_is_hard_split():
    """Only when one sentence alone exceeds the limit do we cut mid-sentence."""
    text = "word " * 400 + "end."
    chunks = chunk_document(text, max_tokens=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert count_tokens(chunk.text) <= 50


def test_zero_max_tokens_is_rejected():
    with pytest.raises(ValueError, match="max_tokens"):
        chunk_document("text", max_tokens=0)


def test_empty_document_yields_no_chunks():
    assert chunk_document("") == []


# -- code passthrough -------------------------------------------------------

FENCED = """Intro text here.

```python
def f(x):
    return x  # the  spacing   matters
```

Outro text here.
"""


def test_fenced_code_is_a_non_compressible_chunk():
    chunks = chunk_document(FENCED)
    code = [c for c in chunks if not c.compressible and c.text.strip()]
    assert len(code) == 1
    assert code[0].text.startswith("```python")
    assert "the  spacing   matters" in code[0].text


def test_fenced_code_survives_a_full_round_trip():
    backend = _Upper()
    result = compress_document(FENCED, backend)
    assert "def f(x):\n    return x  # the  spacing   matters" in result.text
    assert "INTRO TEXT HERE." in result.text
    assert result.metadata["code_blocks_preserved"] == 1
    assert not any("def f" in seen for seen in backend.seen)


def test_tilde_fences_are_also_preserved():
    text = "Before.\n\n~~~\nraw text\n~~~\n\nAfter."
    result = compress_document(text, _Upper())
    assert "~~~\nraw text\n~~~" in result.text


def test_unterminated_fence_runs_to_end_of_document():
    text = "Before.\n\n```\nnever closed\nstill code"
    chunks = chunk_document(text)
    assert chunks[-1].compressible is False
    assert "still code" in chunks[-1].text


def test_paragraph_structure_is_preserved():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    result = compress_document(text, _Upper())
    assert result.text == "FIRST PARAGRAPH.\n\nSECOND PARAGRAPH.\n\nTHIRD PARAGRAPH."


def test_blank_line_around_a_code_block_is_preserved():
    result = compress_document(FENCED, _Upper())
    assert "\n\n```python" in result.text
    assert "```\n\nOUTRO" in result.text


# -- inline spans -----------------------------------------------------------


def test_inline_code_is_shielded_from_the_backend():
    backend = _Upper()
    result = compress_document("Use the `--dry-run` flag now.", backend)
    assert "`--dry-run`" in result.text
    assert "--DRY-RUN" not in result.text


def test_urls_are_shielded_from_the_backend():
    result = compress_document("See https://example.com/Docs for more.", _Upper())
    assert "https://example.com/Docs" in result.text


def test_protection_can_be_switched_off():
    result = compress_document("Use the `--dry-run` flag.", _Upper(), preserve_inline_code=False)
    assert "`--DRY-RUN`" in result.text


def test_nested_protection_tags_do_not_collide():
    """Two protect passes over the same text must not consume each other."""
    text = "See https://x.test and `code` here."
    outer, outer_stash = protect_spans(text, URL_RE, tag="c")
    inner, inner_stash = protect_spans(outer, INLINE_CODE_RE, tag="r")
    assert restore_spans(restore_spans(inner, inner_stash, tag="r"), outer_stash, tag="c") == text


def test_restore_leaves_foreign_tags_alone():
    text, _ = protect_spans("go to https://x.test", URL_RE, tag="c")
    assert restore_spans(text, ["ignored"], tag="r") == text


# -- rejoin -----------------------------------------------------------------


def test_rejoin_rejects_a_length_mismatch():
    chunks = chunk_document("Some prose here.")
    with pytest.raises(ValueError, match="outputs for"):
        rejoin(chunks, [])


def test_batching_is_used_for_multiple_chunks():
    class _Batched(_Upper):
        name = "batched-test"

        def __init__(self):
            super().__init__()
            self.batch_calls = 0

        def compress_batch(self, texts, rate=0.5, **kwargs):
            self.batch_calls += 1
            return [self.compress(t, rate=rate, **kwargs) for t in texts]

    backend = _Batched()
    text = " ".join(f"Sentence number {i} has content." for i in range(80))
    result = compress_document(text, backend, max_tokens=50)
    assert backend.batch_calls == 1
    assert result.metadata["compressed_chunks"] > 1


def test_default_chunk_size_fits_the_bert_window():
    assert DEFAULT_CHUNK_TOKENS <= 450


# -- markdown structure -----------------------------------------------------

MARKDOWN = """# Title

Intro paragraph with some words in it.

## Section

- first bullet item
- second bullet item

> a quoted claim about things

| Flag | Effect |
| --- | --- |
| `--rate` | Fraction of tokens to keep. |
| `--json` | Emit JSON on stdout. |

Closing paragraph here.
"""


def test_tables_are_not_compressible():
    chunks = chunk_document(MARKDOWN)
    tables = [c for c in chunks if not c.compressible and c.text.lstrip().startswith("|")]
    assert len(tables) == 1
    assert "| `--rate` | Fraction of tokens to keep. |" in tables[0].text


def test_table_survives_a_round_trip_verbatim():
    result = compress_document(MARKDOWN, _Upper())
    for row in ("| Flag | Effect |", "| --- | --- |", "| `--json` | Emit JSON on stdout. |"):
        assert row in result.text


def test_heading_markers_survive():
    result = compress_document(MARKDOWN, _Upper())
    headings = [ln for ln in result.text.splitlines() if ln.startswith("#")]
    assert len(headings) == 2
    assert headings[0].startswith("# ")
    assert headings[1].startswith("## ")


def test_list_and_blockquote_markers_survive():
    result = compress_document(MARKDOWN, _Upper())
    assert len([ln for ln in result.text.splitlines() if ln.startswith("- ")]) == 2
    assert len([ln for ln in result.text.splitlines() if ln.startswith("> ")]) == 1


def test_blank_lines_are_preserved_exactly():
    """Paragraph breaks are chunk separators, so no backend can collapse them."""
    result = compress_document(MARKDOWN, _Upper())
    assert result.text.count("\n\n") == MARKDOWN.count("\n\n")


def test_paragraphs_are_never_packed_together():
    """A blank line is a hard chunk boundary."""
    text = "First para.\n\nSecond para.\n\nThird para."
    bodies = [
        c.text for c in chunk_document(text, max_tokens=DEFAULT_CHUNK_TOKENS) if c.compressible
    ]
    assert len(bodies) == 3
    assert all("\n\n" not in b for b in bodies)


def test_horizontal_rule_survives():
    result = compress_document("Above the line.\n\n---\n\nBelow the line.", _Upper())
    assert "---" in result.text


def test_markdown_preservation_can_be_switched_off():
    chunks = chunk_document(MARKDOWN, preserve_markdown=False)
    assert not any(not c.compressible and "|" in c.text for c in chunks)


def test_pipe_inside_a_fence_is_not_a_table():
    """The parser knows a fence's contents are not markdown."""
    text = "Before.\n\n```\n| not | a | table |\n| --- | --- | --- |\n```\n\nAfter."
    chunks = chunk_document(text)
    verbatim = [c.text for c in chunks if not c.compressible and c.text.strip()]
    assert len(verbatim) == 1
    assert verbatim[0].startswith("```")


def test_indented_code_block_is_verbatim():
    text = "Intro paragraph.\n\n    indented = code_block(1)\n    more = code(2)\n\nOutro."
    result = compress_document(text, _Upper())
    assert "    indented = code_block(1)" in result.text


def test_code_survives_even_if_the_parse_fails(monkeypatch):
    """A parser bug must cost us tables, not the whole document."""
    from grug import chunking

    def boom():
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(chunking, "_parser", boom)
    result = compress_document(FENCED, _Upper())
    assert "def f(x):\n    return x  # the  spacing   matters" in result.text


# -- protected spans --------------------------------------------------------


def test_wrapped_inline_code_span_is_protected():
    """CommonMark lets a code span wrap; the regex used to refuse a newline."""
    text = "The list is `not, no, never,\nnor, unless` and nothing else."
    result = compress_document(text, _Upper())
    assert "`not, no, never,\nnor, unless`" in result.text


def test_unmatched_backtick_does_not_swallow_the_document():
    text = "A stray ` backtick here.\n\nA whole separate paragraph follows."
    result = compress_document(text, _Upper())
    assert "SEPARATE PARAGRAPH" in result.text


@pytest.mark.parametrize("number", ["9.6", "1,250", "3-5", "12.5%", "1.2.3"])
def test_compound_numbers_are_protected(number):
    result = compress_document(f"The measured value was {number} in the trial.", _Upper())
    assert number in result.text


def test_number_protection_can_be_switched_off():
    chunks = chunk_document("Value was 9.6 here.", preserve_numbers=False)
    assert not any("9.6" in span for c in chunks for span in c.stash)
