"""Token counting and the CompressionResult/CompressorBackend contract."""

from __future__ import annotations

import pytest

from grug import base
from grug.base import (
    CompressionResult,
    CompressorBackend,
    MissingDependencyError,
    count_tokens,
    tokenizer_name,
)

# -- token counting ---------------------------------------------------------


def test_counts_are_positive_and_monotonic():
    short = count_tokens("hello world")
    long = count_tokens("hello world " * 10)
    assert 0 < short < long


def test_empty_text_counts_zero():
    assert count_tokens("") == 0


def test_special_token_text_does_not_explode():
    """Documents may legitimately contain '<|endoftext|>' as literal text."""
    assert count_tokens("the string <|endoftext|> appears here") > 0


@pytest.fixture
def without_tiktoken(monkeypatch):
    monkeypatch.setattr(base, "_encoder", lambda: None)


def test_falls_back_to_whitespace_without_tiktoken(without_tiktoken):
    assert count_tokens("one two three four") == 4
    assert tokenizer_name() == "whitespace"


def test_compression_still_works_without_tiktoken(without_tiktoken):
    from grug.backends.rules import RulesBackend

    result = RulesBackend().compress(
        "It is important to note that the build did not pass on 3 runs.", rate=0.5
    )
    assert "not" in result.text.split()
    assert result.ratio < 1.0


def test_tokenizer_name_reports_cl100k_when_available():
    if base._encoder() is None:
        pytest.skip("tiktoken unavailable in this environment")
    assert tokenizer_name() == "cl100k_base"


# -- CompressionResult ------------------------------------------------------


def test_build_derives_counts_and_ratio():
    result = CompressionResult.build("one two three four", "one three", "test")
    assert result.original_tokens == count_tokens("one two three four")
    assert result.compressed_tokens == count_tokens("one three")
    assert result.ratio == result.compressed_tokens / result.original_tokens
    assert result.backend == "test"


def test_build_on_empty_input_reports_ratio_one():
    assert CompressionResult.build("", "", "test").ratio == 1.0


def test_saved_tokens():
    result = CompressionResult.build("one two three four five", "one two", "test")
    assert result.saved_tokens == result.original_tokens - result.compressed_tokens


def test_warnings_and_metadata_are_not_shared_between_results():
    first = CompressionResult.build("a", "a", "test")
    first.warnings.append("mine")
    assert CompressionResult.build("a", "a", "test").warnings == []


def test_to_dict_copies_the_warning_list():
    result = CompressionResult.build("a", "a", "test", warnings=["w"])
    payload = result.to_dict()
    payload["warnings"].append("extra")
    assert result.warnings == ["w"]


# -- CompressorBackend ------------------------------------------------------


class _Echo(CompressorBackend):
    name = "echo-test"

    def compress(self, text, rate=0.5, **kwargs):
        return CompressionResult.build(text, text, self.name)


def test_the_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        CompressorBackend()  # type: ignore[abstract]


def test_default_batch_is_sequential():
    results = _Echo().compress_batch(["a", "b", "c"], rate=0.5)
    assert [r.text for r in results] == ["a", "b", "c"]


def test_default_batch_of_nothing():
    assert _Echo().compress_batch([], rate=0.5) == []


@pytest.mark.parametrize("rate", [0.0, -1.0, 1.01, 2.0])
def test_validate_rate_rejects_out_of_range(rate):
    with pytest.raises(ValueError, match=r"rate must be in \(0.0, 1.0\]"):
        CompressorBackend._validate_rate(rate)


@pytest.mark.parametrize("rate", [0.01, 0.5, 1.0])
def test_validate_rate_accepts_the_documented_range(rate):
    assert CompressorBackend._validate_rate(rate) == rate


def test_backends_are_available_by_default():
    assert _Echo.is_available() is True
    _Echo.require_available()  # must not raise


def test_require_available_raises_for_an_unavailable_backend():
    class _Unavailable(_Echo):
        name = "unavailable-test"
        extra = "someextra"

        @classmethod
        def is_available(cls):
            return False

    with pytest.raises(MissingDependencyError, match=r"pip install 'grugify\[someextra\]'"):
        _Unavailable.require_available()


def test_repr_names_the_backend():
    assert "echo-test" in repr(_Echo())
