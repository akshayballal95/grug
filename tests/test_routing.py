"""Picking a backend when the caller supplies a question.

A question only changes the choice when the caller left it open. Naming a
backend explicitly always wins -- silently swapping a model out from under
someone who asked for it by name would be worse than ignoring the question.
"""

from __future__ import annotations

import importlib.util

import pytest

import grug
from grug.backends.lingua2 import Lingua2Backend
from grug.backends.longlingua import LongLinguaBackend
from grug.base import CompressionResult, CompressorBackend
from grug.registry import default_backend_name, list_backends, register_backend, unregister_backend

HAS_LLMLINGUA = importlib.util.find_spec("llmlingua") is not None


class QuestionRecorder(CompressorBackend):
    """A question-aware backend that needs no weights, for routing tests."""

    name = "fakeqa"
    description = "test double"
    question_aware = True

    def __init__(self, **kwargs) -> None:
        self.seen: list[str | None] = []

    def compress(self, text: str, rate: float = 0.5, **kwargs) -> CompressionResult:
        self.seen.append(kwargs.pop("question", None))
        return CompressionResult.build(text, text, self.name)


@pytest.fixture
def fake_question_backend(monkeypatch):
    """Register the test double and take both ML backends out of play.

    Routing decides which *name* gets picked; leaving lingua2 installed would
    have these tests loading real weights to prove a point about strings.
    """
    monkeypatch.setattr(Lingua2Backend, "is_available", classmethod(lambda cls: False))
    monkeypatch.setattr(LongLinguaBackend, "is_available", classmethod(lambda cls: False))
    register_backend(QuestionRecorder)
    try:
        yield QuestionRecorder
    finally:
        unregister_backend(QuestionRecorder.name)


# -- resolution -------------------------------------------------------------


def test_longlingua_is_registered():
    assert "longlingua" in list_backends()


def test_plain_default_is_unchanged_by_the_new_backend():
    """Adding a backend must not move the default for callers with no question."""
    assert default_backend_name() in {"lingua2", "rules"}


@pytest.mark.skipif(not HAS_LLMLINGUA, reason="llmlingua not installed")
def test_a_question_selects_the_question_aware_backend():
    assert default_backend_name(question=True) == "longlingua"


def test_resolution_prefers_any_question_aware_backend(fake_question_backend):
    """Generic over the flag, so a third-party question-aware backend counts too."""
    assert default_backend_name(question=True) == "fakeqa"


def test_resolution_falls_back_when_nothing_is_question_aware(monkeypatch):
    monkeypatch.setattr(LongLinguaBackend, "is_available", classmethod(lambda cls: False))
    assert default_backend_name(question=True) == default_backend_name()


# -- the question reaching the backend --------------------------------------


def test_question_is_forwarded_to_the_backend(fake_question_backend):
    instance = QuestionRecorder()
    grug.compress("some text to compress", backend=instance, question="what broke?")
    assert instance.seen == ["what broke?"]


def test_no_question_forwards_nothing(fake_question_backend):
    instance = QuestionRecorder()
    grug.compress("some text to compress", backend=instance)
    assert instance.seen == [None]


def test_compressor_auto_switches_when_a_question_arrives(fake_question_backend):
    comp = grug.Compressor()
    comp.compress("some text to compress", question="what broke?")
    assert comp.backend_name == "fakeqa"


def test_compressor_without_a_question_keeps_the_plain_default(fake_question_backend):
    comp = grug.Compressor()
    comp.compress("some text to compress")
    assert comp.backend_name == default_backend_name()


# -- explicit choices win ---------------------------------------------------


def test_an_explicitly_named_backend_is_not_swapped_out(fake_question_backend):
    comp = grug.Compressor("rules")
    comp.compress("some text to compress", question="what broke?")
    assert comp.backend_name == "rules"


def test_ignoring_a_question_is_reported_as_a_warning():
    result = grug.Compressor("rules").compress(
        "the migration is not automatic", question="what broke?"
    )
    assert any("question" in w and "rules" in w for w in result.warnings)


def test_a_question_aware_backend_produces_no_such_warning(fake_question_backend):
    result = grug.compress("some text", backend=QuestionRecorder(), question="what broke?")
    assert not any("not question-aware" in w for w in result.warnings)


def test_no_warning_when_no_question_was_asked():
    result = grug.Compressor("rules").compress("the migration is not automatic")
    assert not any("question" in w for w in result.warnings)


def test_module_level_compress_auto_switches_too(fake_question_backend):
    """grug.compress() resolves its own backend, so it needs the same rule."""
    result = grug.compress("some text to compress", question="what broke?")
    assert result.backend == "fakeqa"


def test_module_level_compress_without_a_question_is_unchanged(fake_question_backend):
    result = grug.compress("some text to compress")
    assert result.backend == default_backend_name()


# -- the CLI's own resolution -----------------------------------------------


def test_cli_builder_picks_a_question_aware_backend(fake_question_backend):
    from grug.cli import _build_compressor

    built = _build_compressor(None, None, True, 450, question="what broke?")
    assert built.backend_name == "fakeqa"


def test_cli_builder_without_a_question_is_unchanged(fake_question_backend):
    from grug.cli import _build_compressor

    built = _build_compressor(None, None, True, 450, question=None)
    assert built.backend_name == default_backend_name()


def test_cli_builder_respects_an_explicit_backend(fake_question_backend):
    from grug.cli import _build_compressor

    built = _build_compressor("rules", None, True, 450, question="what broke?")
    assert built.backend_name == "rules"
