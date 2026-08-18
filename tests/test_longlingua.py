"""LongLLMLingua backend: question-aware compression on the causal-LM path.

The library's ``force_tokens`` argument is llmlingua2-only, so everything this
backend guarantees about negations, entities and placeholders comes from
:func:`grug.pinning.restore_forced` running on its output. Tests that would
load real weights are marked ``slow``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from importlib.machinery import ModuleSpec

import pytest

from grug.backends.longlingua import DEFAULT_MODEL, LongLinguaBackend
from grug.base import CompressorBackend, MissingDependencyError

HAS_LLMLINGUA = importlib.util.find_spec("llmlingua") is not None

QUESTION = "What was the p99 lag?"


# -- no model required ------------------------------------------------------


def test_default_model_is_a_small_causal_lm():
    assert DEFAULT_MODEL == "microsoft/phi-2"


def test_backend_declares_itself_question_aware():
    assert LongLinguaBackend.question_aware is True


def test_backends_are_not_question_aware_by_default():
    """The flag is what routing keys on, so the default must be conservative."""
    assert CompressorBackend.question_aware is False


def test_module_import_does_not_pull_in_torch():
    code = "import grug.backends.longlingua, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr


@pytest.mark.skipif(HAS_LLMLINGUA, reason="llmlingua is installed")
def test_construction_without_deps_names_the_extra():
    with pytest.raises(MissingDependencyError) as excinfo:
        LongLinguaBackend()
    assert "grug[longlingua]" in str(excinfo.value)


# -- with a stubbed PromptCompressor ---------------------------------------


def _fake_module(name: str):
    module = types.ModuleType(name)
    module.__spec__ = ModuleSpec(name, loader=None)
    return module


@pytest.fixture
def fake_llmlingua(monkeypatch):
    """Stub llmlingua/torch/transformers, record the call, replay a set output."""
    recorder: dict = {"init": None, "call": None, "output": "compressed text"}

    class FakePromptCompressor:
        def __init__(self, **kwargs):
            recorder["init"] = kwargs

        def compress_prompt(self, context, **kwargs):
            recorder["call"] = {"context": context, **kwargs}
            return {
                "compressed_prompt": recorder["output"],
                "origin_tokens": 20,
                "compressed_tokens": 10,
                "ratio": "2.0x",
                "rate": "50%",
            }

    llmlingua = _fake_module("llmlingua")
    llmlingua.PromptCompressor = FakePromptCompressor

    torch = _fake_module("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )

    monkeypatch.setitem(sys.modules, "llmlingua", llmlingua)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", _fake_module("transformers"))
    return recorder


def test_model_is_constructed_on_the_causal_lm_path(fake_llmlingua):
    LongLinguaBackend(device="cpu").compress("some text", rate=0.5)
    init = fake_llmlingua["init"]
    assert init["model_name"] == DEFAULT_MODEL
    assert init["use_llmlingua2"] is False
    assert init["device_map"] == "cpu"


def test_a_question_turns_on_contrastive_conditioning(fake_llmlingua):
    LongLinguaBackend(device="cpu").compress("some text", rate=0.5, question=QUESTION)
    call = fake_llmlingua["call"]
    assert call["question"] == QUESTION
    assert call["rank_method"] == "longllmlingua"
    assert call["condition_in_question"] == "after_condition"
    assert call["condition_compare"] is True


def test_the_question_is_not_concatenated_onto_the_output(fake_llmlingua):
    """The library defaults this True, which would copy the question into every chunk."""
    LongLinguaBackend(device="cpu").compress("some text", rate=0.5, question=QUESTION)
    assert fake_llmlingua["call"]["concate_question"] is False


def test_context_level_filtering_is_off_for_a_single_chunk(fake_llmlingua):
    """One context per call: there is nothing to rank or reallocate between."""
    LongLinguaBackend(device="cpu").compress("some text", rate=0.5, question=QUESTION)
    assert fake_llmlingua["call"]["use_context_level_filter"] is False


def test_without_a_question_it_falls_back_to_plain_llmlingua(fake_llmlingua):
    """The library asserts rank_method='longllmlingua' requires a question."""
    LongLinguaBackend(device="cpu").compress("some text", rate=0.5)
    call = fake_llmlingua["call"]
    assert call["rank_method"] == "llmlingua"
    assert not call.get("question")


def test_a_dropped_negation_is_pinned_back(fake_llmlingua):
    """force_tokens does nothing here, so the guarantee is post-hoc."""
    fake_llmlingua["output"] = "migration is automatic"
    result = LongLinguaBackend(device="cpu").compress(
        "the migration is not automatic", rate=0.5, question=QUESTION
    )
    assert result.text == "migration is not automatic"
    assert result.metadata["pinned_back"] == ["not"]


def test_a_dropped_entity_is_pinned_back(fake_llmlingua):
    fake_llmlingua["output"] = "vendor was billed twice"
    result = LongLinguaBackend(device="cpu").compress(
        "the vendor Acme was billed twice", rate=0.5, question=QUESTION
    )
    assert result.text == "vendor Acme was billed twice"


def test_entity_pinning_can_be_switched_off(fake_llmlingua):
    fake_llmlingua["output"] = "vendor was billed twice"
    result = LongLinguaBackend(device="cpu", preserve_entities=False).compress(
        "the vendor Acme was billed twice", rate=0.5, question=QUESTION
    )
    assert result.text == "vendor was billed twice"


def test_a_lone_sentence_initial_capital_is_not_pinned_as_an_entity(fake_llmlingua):
    """Deliberate: find_entities skips it, and so does the verifier that checks it.

    Pinning every sentence-opening capital would protect "The" and "Bills".
    """
    fake_llmlingua["output"] = "was billed twice"
    result = LongLinguaBackend(device="cpu").compress("Acme was billed twice", rate=0.5)
    assert result.text == "was billed twice"


def test_a_dropped_placeholder_is_pinned_back(fake_llmlingua):
    fake_llmlingua["output"] = "see for details"
    result = LongLinguaBackend(device="cpu").compress(
        "see GRUGSPANc0X for details", rate=0.5
    )
    assert result.text == "see GRUGSPANc0X for details"


def test_force_tokens_are_not_sent_to_a_model_that_ignores_them(fake_llmlingua):
    """Passing them would read as a guarantee the causal-LM path does not honour."""
    LongLinguaBackend(device="cpu").compress("the plan is not ready", rate=0.5)
    assert "force_tokens" not in fake_llmlingua["call"]


def test_output_is_detokenised(fake_llmlingua):
    fake_llmlingua["output"] = "we ran 3 - 5 cycles , it doesn ' t matter"
    result = LongLinguaBackend(device="cpu").compress("we ran 3-5 cycles", rate=0.5)
    assert result.text == "we ran 3-5 cycles, it doesn't matter"


def test_metadata_carries_the_model_and_counters(fake_llmlingua):
    result = LongLinguaBackend(device="cpu").compress("some text", rate=0.5)
    assert result.metadata["model"] == DEFAULT_MODEL
    assert result.metadata["llmlingua_origin_tokens"] == 20


def test_metadata_records_whether_conditioning_ran(fake_llmlingua):
    backend = LongLinguaBackend(device="cpu")
    assert backend.compress("t", rate=0.5, question=QUESTION).metadata["conditioned"] is True
    assert backend.compress("t", rate=0.5).metadata["conditioned"] is False


def test_blank_input_skips_the_model(fake_llmlingua):
    result = LongLinguaBackend(device="cpu").compress("   ", rate=0.5)
    assert fake_llmlingua["init"] is None
    assert result.metadata["skipped"] == "blank"


def test_invalid_rate_is_rejected_before_loading(fake_llmlingua):
    with pytest.raises(ValueError, match="rate must be in"):
        LongLinguaBackend(device="cpu").compress("text", rate=1.5)
    assert fake_llmlingua["init"] is None


def test_model_loads_only_once_across_calls(fake_llmlingua):
    backend = LongLinguaBackend(device="cpu")
    first = backend.model
    backend.compress("some text", rate=0.5)
    assert backend.model is first


def test_registry_creates_it_when_deps_are_present(fake_llmlingua):
    from grug.registry import create_backend

    assert isinstance(create_backend("longlingua"), LongLinguaBackend)


def test_per_call_kwargs_override_the_defaults(fake_llmlingua):
    LongLinguaBackend(device="cpu").compress(
        "some text", rate=0.5, question=QUESTION, dynamic_context_compression_ratio=0.4
    )
    assert fake_llmlingua["call"]["dynamic_context_compression_ratio"] == 0.4


def test_a_dropped_number_is_pinned_back(fake_llmlingua):
    """force_reserve_digit is llmlingua2-only, so digits are pinned post-hoc too."""
    fake_llmlingua["output"] = "p99 lag was seconds"
    result = LongLinguaBackend(device="cpu").compress("p99 lag was 9.6 seconds", rate=0.5)
    assert result.text == "p99 lag was 9.6 seconds"


def test_a_thousands_separated_number_is_pinned_back(fake_llmlingua):
    fake_llmlingua["output"] = "across accounts"
    result = LongLinguaBackend(device="cpu").compress("across 4,800 accounts", rate=0.5)
    assert result.text == "across 4,800 accounts"


def test_number_pinning_can_be_switched_off(fake_llmlingua):
    fake_llmlingua["output"] = "across accounts"
    result = LongLinguaBackend(device="cpu", preserve_numbers=False).compress(
        "across 4,800 accounts", rate=0.5
    )
    assert result.text == "across accounts"


def test_negations_are_forced_by_default():
    from grug.backends.longlingua import DEFAULT_FORCE_TOKENS
    from grug.verify import NEGATION_FORCE_TOKENS

    assert set(DEFAULT_FORCE_TOKENS) >= set(NEGATION_FORCE_TOKENS)
