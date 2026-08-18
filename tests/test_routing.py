"""Picking a backend when the caller supplies a question.

A question only changes the choice when the caller left it open. Naming a
backend explicitly always wins -- silently swapping a model out from under
someone who asked for it by name would be worse than ignoring the question.
"""

from __future__ import annotations

import importlib.util

import pytest

import grug
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
def fake_question_backend():
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


def test_resolution_prefers_any_question_aware_backend(monkeypatch, fake_question_backend):
    """Generic over the flag, so a third-party question-aware backend counts too."""
    monkeypatch.setattr(LongLinguaBackend, "is_available", classmethod(lambda cls: False))
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


def test_compressor_auto_switches_when_a_question_arrives(monkeypatch, fake_question_backend):
    monkeypatch.setattr(LongLinguaBackend, "is_available", classmethod(lambda cls: False))
    comp = grug.Compressor()
    comp.compress("some text to compress", question="what broke?")
    assert comp.backend_name == "fakeqa"


def test_compressor_without_a_question_keeps_the_plain_default(monkeypatch, fake_question_backend):
    monkeypatch.setattr(LongLinguaBackend, "is_available", classmethod(lambda cls: False))
    comp = grug.Compressor()
    comp.compress("some text to compress")
    assert comp.backend_name == default_backend_name()


# -- explicit choices win ---------------------------------------------------


def test_an_explicitly_named_backend_is_not_swapped_out(monkeypatch, fake_question_backend):
    monkeypatch.setattr(LongLinguaBackend, "is_available", classmethod(lambda cls: False))
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
