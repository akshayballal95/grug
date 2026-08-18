"""Chunking: sentence boundaries, code passthrough, and structure preservation."""

from __future__ import annotations

import pytest

from grug.base import CompressionResult, CompressorBackend, count_tokens
from grug.chunking import (
    DEFAULT_CHUNK_TOKENS,
    IDENTIFIER_RE,
    INLINE_CODE_RE,
    URL_RE,
    chunk_document,
    compress_document,
    contains_placeholder,
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


@pytest.mark.parametrize(
    "identifier",
    [
        "node-07", "us-east-1", "v2.1.0-rc3", "RFC-7231", "utf-8", "sha256:9f2b",
        "log4j-2.17.1", "TRAINING.md", "notes.grug.md", "Node.js",
    ],
)
def test_identifiers_are_protected(identifier):
    """Internal punctuation is load-bearing: a backend that drops it renames the thing."""
    result = compress_document(f"The {identifier} target failed during the rollout.", _Upper())
    assert identifier in result.text


def test_identifier_protection_can_be_switched_off():
    chunks = chunk_document("Deploy to us-east-1 now.", preserve_identifiers=False)
    assert not any("us-east-1" in span for c in chunks for span in c.stash)


@pytest.mark.parametrize(
    "word", ["sign-off", "api-gateway", "well-known", "trade-off", "text/plain", "read:write"]
)
def test_plain_hyphenated_words_are_left_compressible(word):
    """A hyphen between two plain words is ordinary English, not an identifier.

    'sign-off' and 'api-gateway' are the same shape, so no rule can protect one
    and release the other. Protecting both would pin every hyphenated compound
    in the document; the rule asks for a digit or a non-hyphen separator instead.
    """
    chunks = chunk_document(f"Await {word} from the team.", preserve_identifiers=True)
    assert not any(word in span for c in chunks for span in c.stash)


def test_a_later_pattern_does_not_swallow_an_earlier_placeholder():
    """Nesting one placeholder inside another loses it: restore only unwraps once."""
    text, stash = protect_spans("see `-b`/`backend=` now", INLINE_CODE_RE, IDENTIFIER_RE)
    assert not any(contains_placeholder(span) for span in stash)
    assert restore_spans(text, stash) == "see `-b`/`backend=` now"


def test_no_placeholder_leaks_through_a_document_round_trip():
    text = "> 9.6 seconds\n\nUse `-b`/`backend=` on us-east-1.\n"
    result = compress_document(text, _Upper())
    assert not contains_placeholder(result.text), result.text


def test_per_chunk_lists_are_concatenated_not_overwritten():
    """'pinned_back' is how a user audits what a backend had to restore."""
    from grug.chunking import _merge_metadata

    merged = _merge_metadata(
        [
            {"pinned_back": ["not"], "origin_tokens": 10},
            {"pinned_back": ["never", "4,800"], "origin_tokens": 12},
            {"pinned_back": [], "origin_tokens": 8},
        ]
    )
    assert merged["pinned_back"] == ["not", "never", "4,800"]
    assert merged["origin_tokens"] == 30


def test_scalar_metadata_still_keeps_the_first_value():
    from grug.chunking import _merge_metadata

    merged = _merge_metadata([{"model": "a", "device": "cpu"}, {"model": "b", "device": "cuda"}])
    assert merged == {"model": "a", "device": "cpu"}


@pytest.mark.parametrize("phrase", ["and/or", "input/output", "he/she", "N/A", "e.g.", "U.S.A"])
def test_ordinary_english_punctuation_is_not_an_identifier(phrase):
    """A slash or dot between plain words is English; pinning it costs ratio."""
    chunks = chunk_document(f"Use {phrase} in the report.", preserve_identifiers=True)
    assert not any(phrase.strip(".") in span for c in chunks for span in c.stash)
