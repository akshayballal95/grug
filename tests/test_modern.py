"""The modern-encoder backend.

The pure functions run always. Anything needing weights is marked ``slow`` and
skipped when torch is absent, so the default suite downloads nothing.
"""

from __future__ import annotations

import importlib.util
import types

import pytest

from grug.backends.modern import (
    DEFAULT_FORCE_TOKENS,
    ModernBackend,
    _preserve_label_id,
    _windows,
    join_words,
    split_words,
)
from grug.verify import NEGATION_FORCE_TOKENS

HAS_TORCH = ModernBackend.is_available()
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="needs grug[modern]")

CACHED = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"


# -- word splitting ---------------------------------------------------------


def test_words_and_newlines_are_separate_units():
    assert split_words("a b\nc") == ["a", "b", "\n", "c"]


def test_blank_line_run_is_one_unit():
    assert split_words("a\n\n\nb") == ["a", "\n\n\n", "b"]


def test_round_trip_preserves_line_structure():
    for text in ("one two three", "line one\nline two", "para\n\nnext para", "# H\n- item"):
        assert join_words(split_words(text)) == text


def test_join_does_not_pad_around_newlines():
    assert join_words(["a", "\n", "b"]) == "a\nb"


def test_dropping_a_word_does_not_break_the_layout():
    words = split_words("keep drop keep\nsecond line")
    survivors = [w for w in words if w != "drop"]
    assert join_words(survivors) == "keep keep\nsecond line"


# -- windowing --------------------------------------------------------------


def test_short_input_is_one_window():
    assert _windows([1, 1, 1], 100) == [(0, 3)]


def test_windows_respect_the_budget():
    counts = [3] * 10
    spans = _windows(counts, 9)
    assert spans == [(0, 3), (3, 6), (6, 9), (9, 10)]
    for start, end in spans:
        assert sum(counts[start:end]) <= 9 or end - start == 1


def test_a_single_oversized_word_still_gets_a_window():
    assert _windows([50], 10) == [(0, 1)]


def test_windows_cover_every_word_exactly_once():
    counts = [2, 5, 1, 7, 3, 4]
    covered = [i for start, end in _windows(counts, 8) for i in range(start, end)]
    assert covered == list(range(len(counts)))


# -- label resolution -------------------------------------------------------


@pytest.mark.parametrize(
    ("id2label", "expected"),
    [
        ({0: "discard", 1: "preserve"}, 1),
        ({0: "preserve", 1: "discard"}, 0),
        ({0: "LABEL_0", 1: "LABEL_1"}, 1),
        ({}, 1),
    ],
)
def test_preserve_label_id(id2label, expected):
    config = types.SimpleNamespace(id2label=id2label, num_labels=2)
    assert _preserve_label_id(config) == expected


# -- defaults ---------------------------------------------------------------


def test_negations_are_forced_by_default():
    for word in NEGATION_FORCE_TOKENS:
        assert word in DEFAULT_FORCE_TOKENS
    assert "\n" in DEFAULT_FORCE_TOKENS


def test_import_does_not_pull_torch():
    import subprocess
    import sys

    code = (
        "import sys, grug.backends.modern; "
        "print([m for m in ('torch', 'transformers') if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


@pytest.mark.skipif(HAS_TORCH, reason="torch is installed")
def test_construction_without_torch_names_the_extra():
    from grug.base import MissingDependencyError

    with pytest.raises(MissingDependencyError, match=r"grug\[modern\]"):
        ModernBackend()


def test_is_registered():
    import grug

    assert "modern" in grug.list_backends()


# -- with weights -----------------------------------------------------------


@pytest.fixture(scope="module")
def backend():
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("transformers not installed")
    return ModernBackend(model_name=CACHED, device="cpu")


@pytest.mark.slow
@requires_torch
def test_compresses_and_tracks_the_rate(backend):
    doc = "The quarterly invoice does not include tax on the 1,250 units we shipped."
    loose = backend.compress(doc, rate=0.9)
    tight = backend.compress(doc, rate=0.3)
    assert tight.compressed_tokens < loose.compressed_tokens
    assert tight.backend == "modern"


@pytest.mark.slow
@requires_torch
def test_output_is_a_subsequence_of_the_input(backend):
    doc = "Alpha beta gamma delta epsilon zeta eta theta iota kappa."
    result = backend.compress(doc, rate=0.5)
    original = split_words(doc)
    position = 0
    for word in split_words(result.text):
        position = original.index(word, position) + 1  # raises if out of order


@pytest.mark.slow
@requires_torch
def test_numbers_and_negations_survive(backend):
    doc = "The migration is not automatic and no refund is issued for the 1,250 accounts."
    result = backend.compress(doc, rate=0.3)
    assert "1,250" in result.text
    assert "not" in result.text.split()


@pytest.mark.slow
@requires_torch
def test_newlines_survive(backend):
    result = backend.compress("first line here\nsecond line here", rate=0.5)
    assert "\n" in result.text


@pytest.mark.slow
@requires_torch
def test_unknown_kwargs_are_rejected(backend):
    with pytest.raises(TypeError, match="nonsense"):
        backend.compress("some text", rate=0.5, nonsense=True)


@pytest.mark.slow
@requires_torch
def test_blank_input_skips_the_model(backend):
    assert backend.compress("   ", rate=0.5).metadata == {"skipped": "blank"}
