"""The cascade runs the rules engine, then the classifier on what is left."""

from __future__ import annotations

from typing import ClassVar

import pytest

import grug
from grug.backends.cascade import CascadeBackend
from grug.base import CompressionResult, CompressorBackend
from grug.registry import register_backend


class _KeepFirst(CompressorBackend):
    """Stands in for the classifier: keeps a leading share of the words."""

    name = "keep-first-test"
    calls: ClassVar[list[tuple[str, float]]] = []

    def compress(self, text: str, rate: float = 0.5, **kwargs) -> CompressionResult:
        type(self).calls.append((text, rate))
        words = text.split()
        keep = max(1, round(rate * len(words)))
        return CompressionResult.build(text, " ".join(words[:keep]), self.name, metadata={})


@pytest.fixture
def cascade() -> CascadeBackend:
    backend = CascadeBackend.__new__(CascadeBackend)
    backend._rules = grug.create_backend("rules")
    backend._classifier = _KeepFirst()
    _KeepFirst.calls = []
    return backend


def test_it_is_registered_and_needs_a_model():
    rows = {row["name"]: row for row in grug.backend_info()}
    assert "cascade" in rows
    assert rows["cascade"]["requires_configuration"] is True


def test_the_target_is_a_share_of_the_original_not_of_the_leftovers(cascade):
    """The second stage sees a shorter document, so the rate has to be rescaled.

    Passing `rate` straight through would compress twice and undershoot badly.
    """
    text = " ".join(f"the sentence number {i} is here and it continues" for i in range(30))
    result = cascade.compress(text, rate=0.3)

    assert _KeepFirst.calls, "the classifier stage never ran"
    _, inner = _KeepFirst.calls[-1]
    survived = result.metadata["rules_ratio"]
    assert inner == pytest.approx(0.3 / survived, rel=1e-3)
    assert inner > 0.3, "the rescaled rate must be looser than the overall target"


def test_the_classifier_is_skipped_when_rules_already_went_far_enough(cascade):
    text = " ".join(f"the sentence number {i} is here and it continues" for i in range(30))
    result = cascade.compress(text, rate=0.99)
    assert result.metadata["classifier_ran"] is False
    assert _KeepFirst.calls == []


def test_blank_text_is_passed_through(cascade):
    result = cascade.compress("   ", rate=0.4)
    assert result.text == "   "
    assert result.metadata["skipped"] == "blank"


def test_an_invalid_rate_is_rejected(cascade):
    with pytest.raises(ValueError):
        cascade.compress("some words here", rate=1.5)


def test_output_is_shorter_than_the_rules_stage_alone(cascade):
    text = " ".join(f"the sentence number {i} is here and it continues" for i in range(30))
    rules_only = grug.compress(text, rate=0.33, backend="rules").text
    cascaded = cascade.compress(text, rate=0.25).text
    assert len(cascaded.split()) < len(rules_only.split())


register_backend(_KeepFirst)
