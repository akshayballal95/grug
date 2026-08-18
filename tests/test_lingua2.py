"""LLMLingua-2 backend.

Everything that does not touch the model runs always. Anything that loads
weights is marked ``slow`` and skipped unless llmlingua is installed, so the
default test run never downloads a checkpoint.
"""

from __future__ import annotations

import pytest

from grug.backends.lingua2 import (
    DEFAULT_FORCE_TOKENS,
    DEFAULT_MODEL,
    Lingua2Backend,
    _repair_detokenization,
)
from grug.base import MissingDependencyError
from grug.chunking import contains_placeholder
from grug.verify import NEGATION_FORCE_TOKENS

HAS_LLMLINGUA = Lingua2Backend.is_available()

requires_model = pytest.mark.skipif(
    not HAS_LLMLINGUA, reason="llmlingua not installed (pip install 'grug[lingua2]')"
)


# -- no model required ------------------------------------------------------


def test_negations_are_forced_by_default():
    """The whole reason this backend is safe at aggressive rates."""
    for word in NEGATION_FORCE_TOKENS:
        assert word in DEFAULT_FORCE_TOKENS
    assert "\n" in DEFAULT_FORCE_TOKENS
    assert "?" in DEFAULT_FORCE_TOKENS


def test_negation_force_list_covers_the_documented_words():
    expected = {
        "not",
        "no",
        "never",
        "none",
        "neither",
        "nor",
        "n't",
        "except",
        "unless",
        "without",
    }
    # The README's list is the guaranteed minimum, not the whole of it.
    assert expected <= set(NEGATION_FORCE_TOKENS)


def test_force_list_and_verifier_vocabulary_cannot_drift():
    """Every word the verifier calls a negation is also protected by default."""
    from grug.verify import NEGATION_WORDS

    assert set(NEGATION_FORCE_TOKENS) >= NEGATION_WORDS


def test_default_model_is_the_multilingual_meetingbank_checkpoint():
    assert DEFAULT_MODEL == "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"


def test_module_import_does_not_pull_in_torch():
    import subprocess
    import sys

    code = (
        "import sys, grug.backends.lingua2; "
        "print([m for m in ('torch', 'transformers', 'llmlingua') if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


@pytest.mark.skipif(HAS_LLMLINGUA, reason="llmlingua is installed")
def test_construction_without_deps_names_the_extra():
    with pytest.raises(MissingDependencyError) as excinfo:
        Lingua2Backend()
    message = str(excinfo.value)
    assert "pip install 'grug[lingua2]'" in message
    assert "llmlingua" in message


def test_is_available_matches_the_environment():
    assert Lingua2Backend.is_available() is HAS_LLMLINGUA


# -- detokenisation repair (pure string work) -------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("we ran 3 - 5 cycles", "we ran 3-5 cycles"),
        ("it doesn ' t work", "it doesn't work"),
        ("that is n't right", "that isn't right"),
        ("a 12 . 5 % gain", "a 12.5% gain"),
        ("1 , 250 accounts", "1,250 accounts"),
        ("version 1 . 2 . 3", "version 1.2.3"),
        ("costs $ 40", "costs $40"),
        ("wait , then stop .", "wait, then stop."),
        ("spaced    out   words", "spaced out words"),
        ("the company ' s plan", "the company's plan"),
    ],
)
def test_repair_detokenization(raw, expected):
    assert _repair_detokenization(raw) == expected


def test_repair_preserves_paragraph_breaks():
    assert _repair_detokenization("first line\n\nsecond line") == "first line\n\nsecond line"


def test_repair_collapses_runaway_blank_lines():
    assert _repair_detokenization("a\n\n\n\n\nb") == "a\n\nb"


def test_repair_is_idempotent():
    once = _repair_detokenization("we ran 3 - 5 cycles and it doesn ' t matter")
    assert _repair_detokenization(once) == once


# -- model required ---------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_backend():
    return Lingua2Backend(device="cpu")


@pytest.mark.slow
@requires_model
def test_compresses_a_document(loaded_backend):
    text = (
        "It is important to note that the billing pipeline has been rewritten. "
        "The migration is not automatic, and accounts on the legacy plan must be "
        "moved by hand before the cutover date."
    )
    result = loaded_backend.compress(text, rate=0.5)
    assert result.backend == "lingua2"
    assert result.compressed_tokens < result.original_tokens
    assert result.metadata["model"] == DEFAULT_MODEL


@pytest.mark.slow
@requires_model
def test_negation_survives_an_aggressive_rate(loaded_backend):
    text = (
        "The finance team confirmed the headline result of the study: bills "
        "scale with volume, not price, across every plan we currently support."
    )
    result = loaded_backend.compress(text, rate=0.3)
    assert "not" in result.text.lower().split()


@pytest.mark.slow
@requires_model
def test_digits_are_reserved(loaded_backend):
    text = (
        "We ran a trial of 4,800 accounts and measured a median lag of 1.2 "
        "seconds, with a p99 lag of 9.6 seconds over the whole window."
    )
    result = loaded_backend.compress(text, rate=0.4)
    assert any(digit in result.text for digit in ("4,800", "4800", "1.2", "9.6"))


@pytest.mark.slow
@requires_model
def test_model_is_loaded_once(loaded_backend):
    first = loaded_backend.model
    assert loaded_backend.model is first


@pytest.mark.slow
@requires_model
def test_end_to_end_through_the_public_api(sample_markdown):
    import grug

    result = grug.compress(sample_markdown, rate=0.5, backend="lingua2")
    assert result.ratio < 1.0
    assert "```python" in result.text


# -- integration contract, exercised against a stubbed llmlingua -------------
#
# These cover the wiring -- constructor kwargs, call kwargs, device resolution,
# version tolerance, post-processing -- without downloading a checkpoint.


def _fake_module(name):
    import types
    from importlib.machinery import ModuleSpec

    module = types.ModuleType(name)
    # find_spec() consults __spec__, and require_available() calls find_spec.
    module.__spec__ = ModuleSpec(name, loader=None)
    return module


@pytest.fixture
def fake_llmlingua(monkeypatch):
    """Install stub llmlingua/torch/transformers and record what gets called."""
    import sys
    import types

    recorder: dict = {"init": None, "call": None, "cuda": False, "mps": False}

    class FakePromptCompressor:
        def __init__(self, **kwargs):
            recorder["init"] = kwargs

        def compress_prompt(
            self,
            context,
            rate=0.5,
            force_tokens=None,
            force_reserve_digit=False,
            drop_consecutive=False,
            **kwargs,
        ):
            recorder["call"] = {
                "context": context,
                "rate": rate,
                "force_tokens": force_tokens,
                "force_reserve_digit": force_reserve_digit,
                "drop_consecutive": drop_consecutive,
                **kwargs,
            }
            return {
                "compressed_prompt": "we ran 3 - 5 cycles , it doesn ' t matter",
                "origin_tokens": 20,
                "compressed_tokens": 10,
                "ratio": "2.0x",
                "rate": "50%",
            }

    llmlingua = _fake_module("llmlingua")
    llmlingua.PromptCompressor = FakePromptCompressor

    torch = _fake_module("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: recorder["cuda"])
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: recorder["mps"])
    )

    monkeypatch.setitem(sys.modules, "llmlingua", llmlingua)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", _fake_module("transformers"))
    recorder["PromptCompressor"] = FakePromptCompressor
    return recorder


def test_model_is_constructed_with_llmlingua2_enabled(fake_llmlingua):
    Lingua2Backend(device="cpu").compress("some text", rate=0.5)
    init = fake_llmlingua["init"]
    assert init["model_name"] == DEFAULT_MODEL
    assert init["use_llmlingua2"] is True
    assert init["device_map"] == "cpu"


def test_negations_and_digit_reservation_reach_the_model(fake_llmlingua):
    Lingua2Backend(device="cpu").compress("some text", rate=0.4)
    call = fake_llmlingua["call"]
    assert call["rate"] == 0.4
    assert call["force_reserve_digit"] is True
    assert call["drop_consecutive"] is True
    for word in NEGATION_FORCE_TOKENS:
        assert word in call["force_tokens"]


def test_output_is_detokenised(fake_llmlingua):
    result = Lingua2Backend(device="cpu").compress("some text", rate=0.5)
    assert result.text == "we ran 3-5 cycles, it doesn't matter"


def test_detokenisation_can_be_disabled(fake_llmlingua):
    backend = Lingua2Backend(device="cpu", repair_detokenization=False)
    assert " - " in backend.compress("some text", rate=0.5).text


def test_metadata_carries_the_model_and_llmlingua_counters(fake_llmlingua):
    result = Lingua2Backend(device="cpu").compress("some text", rate=0.5)
    assert result.metadata["model"] == DEFAULT_MODEL
    assert result.metadata["device"] == "cpu"
    assert result.metadata["llmlingua_origin_tokens"] == 20
    assert result.metadata["llmlingua_rate"] == "50%"


def test_force_tokens_are_overridable(fake_llmlingua):
    backend = Lingua2Backend(device="cpu", force_tokens=["\n"])
    backend.compress("some text", rate=0.5)
    assert fake_llmlingua["call"]["force_tokens"] == ["\n"]


def test_per_call_kwargs_override_the_instance(fake_llmlingua):
    backend = Lingua2Backend(device="cpu")
    backend.compress("some text", rate=0.5, drop_consecutive=False)
    assert fake_llmlingua["call"]["drop_consecutive"] is False


def test_model_loads_only_once_across_calls(fake_llmlingua):
    backend = Lingua2Backend(device="cpu")
    loads = []
    original = fake_llmlingua["PromptCompressor"].__init__

    def counting_init(self, **kwargs):
        loads.append(kwargs)
        original(self, **kwargs)

    fake_llmlingua["PromptCompressor"].__init__ = counting_init
    try:
        backend.compress("a", rate=0.5)
        backend.compress("b", rate=0.5)
        backend.compress_batch(["c", "d"], rate=0.5)
    finally:
        fake_llmlingua["PromptCompressor"].__init__ = original
    assert len(loads) == 1


def test_unsupported_kwargs_are_dropped_for_older_llmlingua(fake_llmlingua):
    """An older release without force_reserve_digit must not raise TypeError."""

    class Narrow:
        def __init__(self, **kwargs):
            pass

        def compress_prompt(self, context, rate=0.5, force_tokens=None):
            return {"compressed_prompt": "narrow output"}

    import sys

    sys.modules["llmlingua"].PromptCompressor = Narrow
    result = Lingua2Backend(device="cpu").compress("some text", rate=0.5)
    assert result.text == "narrow output"


def test_blank_input_skips_the_model(fake_llmlingua):
    result = Lingua2Backend(device="cpu").compress("   ", rate=0.5)
    assert result.metadata == {"skipped": "blank"}
    assert fake_llmlingua["init"] is None


def test_invalid_rate_is_rejected_before_loading(fake_llmlingua):
    with pytest.raises(ValueError, match="rate must be in"):
        Lingua2Backend(device="cpu").compress("text", rate=0.0)
    assert fake_llmlingua["init"] is None


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, False, "cuda"), (False, True, "mps"), (False, False, "cpu"), (True, True, "cuda")],
)
def test_auto_device_resolution(fake_llmlingua, cuda, mps, expected):
    fake_llmlingua["cuda"] = cuda
    fake_llmlingua["mps"] = mps
    backend = Lingua2Backend(device="auto")
    backend.compress("some text", rate=0.5)
    assert backend._resolved_device == expected
    assert fake_llmlingua["init"]["device_map"] == expected


def test_extra_model_kwargs_reach_the_constructor(fake_llmlingua):
    Lingua2Backend(device="cpu", model_config={"revision": "main"}).compress("t", rate=0.5)
    assert fake_llmlingua["init"]["model_config"] == {"revision": "main"}


def test_registry_creates_it_when_deps_are_present(fake_llmlingua):
    from grug.registry import create_backend

    backend = create_backend("lingua2", device="cpu")
    assert isinstance(backend, Lingua2Backend)


@pytest.mark.slow
@requires_model
def test_placeholders_survive_the_real_tokenizer(loaded_backend):
    """Regression: private-use sentinels came back as the garbage text 'c : 0'.

    A protected span must be restorable after a real round trip, at rates well
    below the default.
    """
    import grug

    doc = (
        "Use the `--offline-batch` flag to preview changes without applying "
        "them, and read https://example.com/runbooks/backfill before you start. "
        "The endpoint is rate limited to 100 requests per minute per token."
    )
    for rate in (0.5, 0.3, 0.2):
        result = grug.compress(doc, rate=rate, backend=loaded_backend, verify=False)
        assert "`--offline-batch`" in result.text, f"inline code lost at rate={rate}"
        assert "https://example.com/runbooks/backfill" in result.text, f"url lost at {rate}"
        assert not contains_placeholder(result.text), f"placeholder leaked at rate={rate}"


@pytest.mark.slow
@requires_model
def test_thousands_separator_is_rejoined(loaded_backend):
    """Regression: LLMLingua emits "5,000" as "5 000", splitting one number in two."""
    import grug

    doc = (
        "A configuration change raised the batch flush interval from 200 ms to "
        "5,000 ms in order to reduce write amplification on the primary. Do not "
        "raise the flush interval above 1,000 ms without running a load test."
    )
    result = grug.compress(doc, rate=0.4, backend=loaded_backend)
    assert "5,000" in result.text
    assert "1,000" in result.text
    assert not [w for w in result.warnings if w.startswith("numbers")], result.warnings


@pytest.mark.slow
@requires_model
def test_real_document_is_faithful_at_moderate_rates(loaded_backend, sample_markdown):
    """The verifier must stay quiet on a real document at a sane rate."""
    import grug

    result = grug.compress(sample_markdown, rate=0.5, backend=loaded_backend)
    assert result.ratio < 0.75
    assert result.warnings == [], result.warnings
    assert "```python" in result.text


def test_force_tokens_gain_case_variants(fake_llmlingua):
    """LLMLingua matches force_tokens case-sensitively; lowercase alone is a trap."""
    Lingua2Backend(device="cpu").compress("some text", rate=0.5)
    forced = fake_llmlingua["call"]["force_tokens"]
    for word in ("not", "no", "never", "without", "cannot"):
        assert word in forced
        assert word.capitalize() in forced
        assert word.upper() in forced
    # Tokens with no case are passed through untouched and not duplicated.
    assert forced.count("\n") == 1
    assert forced.count("?") == 1


def test_case_expansion_applies_to_custom_force_tokens(fake_llmlingua):
    backend = Lingua2Backend(device="cpu", force_tokens=["widget"])
    backend.compress("some text", rate=0.5)
    forced = fake_llmlingua["call"]["force_tokens"]
    assert {"widget", "Widget", "WIDGET"} <= set(forced)


def test_placeholders_are_pinned_when_present(fake_llmlingua):
    from grug.chunking import INLINE_CODE_RE, protect_spans

    text, _ = protect_spans("run the `--dry-run` flag", INLINE_CODE_RE, tag="c")
    Lingua2Backend(device="cpu").compress(text, rate=0.5)
    forced = fake_llmlingua["call"]["force_tokens"]
    assert any(contains_placeholder(token) for token in forced)


@pytest.mark.slow
@requires_model
def test_sentence_initial_negation_survives(loaded_backend):
    """Regression: a capitalised leading negation was unprotected."""
    import grug

    for text in (
        "No customer was billed twice and no invoice was lost.",
        "Never raise the interval without a load test.",
        "Not every tenant is enrolled in the rollout.",
    ):
        for rate in (0.5, 0.3):
            result = grug.compress(text, rate=rate, backend=loaded_backend)
            assert result.warnings == [], f"{text!r} at rate={rate}: {result.warnings}"


def test_entities_are_pinned_by_default(fake_llmlingua):
    Lingua2Backend(device="cpu").compress("Acme Corporation and Globex met.", rate=0.5)
    forced = fake_llmlingua["call"]["force_tokens"]
    assert {"Acme", "Corporation", "Globex"} <= set(forced)


def test_entity_pinning_can_be_disabled(fake_llmlingua):
    backend = Lingua2Backend(device="cpu", preserve_entities=False)
    backend.compress("Acme Corporation and Globex met.", rate=0.5)
    assert "Acme" not in fake_llmlingua["call"]["force_tokens"]


def test_entity_pinning_is_overridable_per_call(fake_llmlingua):
    backend = Lingua2Backend(device="cpu")
    backend.compress("Acme Corporation met.", rate=0.5, preserve_entities=False)
    assert "Acme" not in fake_llmlingua["call"]["force_tokens"]


def test_no_entities_leaves_the_force_list_alone(fake_llmlingua):
    from grug.backends.lingua2 import _force_tokens

    Lingua2Backend(device="cpu").compress("the quick brown fox jumps", rate=0.5)
    expected = _force_tokens("", DEFAULT_FORCE_TOKENS, entities=True)
    assert set(fake_llmlingua["call"]["force_tokens"]) == set(expected)


@pytest.mark.slow
@requires_model
def test_entity_pinning_removes_the_entity_warning_class(sample_markdown):
    """Prevention beats detection: pin what the verifier would complain about."""
    import grug

    loose = Lingua2Backend(device="cpu", preserve_entities=False)
    pinned = Lingua2Backend(device="cpu", preserve_entities=True)
    for rate in (0.4, 0.3):
        without = grug.compress(sample_markdown, rate=rate, backend=loose)
        with_ = grug.compress(sample_markdown, rate=rate, backend=pinned)
        assert not [w for w in with_.warnings if w.startswith("entities")], with_.warnings
        # The safety costs a little ratio, and must not cost a lot.
        assert with_.ratio - without.ratio < 0.08
