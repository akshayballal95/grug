"""The rules backend end to end, plus the public API that wraps it."""

from __future__ import annotations

import pytest

import grug
from grug.backends.rules import RulesBackend


@pytest.fixture
def backend() -> RulesBackend:
    return RulesBackend()


# -- core behaviour ---------------------------------------------------------


def test_output_is_a_subset_of_the_input(backend):
    """A deletion-only compressor can never invent a word."""
    text = "The report is available on the internal dashboard for every team."
    result = backend.compress(text, rate=0.5)
    original_words = set(text.lower().replace(".", "").split())
    for word in result.text.lower().replace(".", "").split():
        assert word in original_words


def test_compression_actually_shrinks(backend):
    text = (
        "It is important to note that the report is available on the internal "
        "dashboard, and it is updated on a regular basis for every team."
    )
    result = backend.compress(text, rate=0.5)
    assert result.compressed_tokens < result.original_tokens
    assert result.ratio < 1.0
    assert result.backend == "rules"


def test_lower_rate_removes_at_least_as_much(backend):
    text = " ".join(
        [
            "The system is designed so that the operator can review the results",
            "of the run and then decide whether the batch should be retried.",
        ]
    )
    loose = backend.compress(text, rate=0.9).compressed_tokens
    tight = backend.compress(text, rate=0.3).compressed_tokens
    assert tight <= loose


def test_rate_of_one_is_a_near_passthrough(backend):
    text = "The system is designed so that the operator can review the results."
    assert backend.compress(text, rate=1.0).text == text


def test_reports_the_ratio_it_actually_achieved(backend):
    """Rules cannot hit aggressive rates; it must not pretend otherwise."""
    text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda."
    result = backend.compress(text, rate=0.1)
    assert result.ratio > 0.1
    assert result.metadata["requested_rate"] == 0.1


@pytest.mark.parametrize("rate", [0.0, -0.5, 1.5])
def test_invalid_rates_are_rejected(backend, rate):
    with pytest.raises(ValueError, match="rate must be in"):
        backend.compress("some text", rate=rate)


def test_blank_input_round_trips(backend):
    assert backend.compress("", rate=0.5).text == ""
    assert backend.compress("   \n  ", rate=0.5).text == "   \n  "


# -- what must never be dropped ---------------------------------------------


def test_negations_survive_the_most_aggressive_rate(backend):
    text = "The invoice total does not include tax and no refund is issued without approval."
    result = backend.compress(text, rate=0.05)
    for word in ("not", "no", "without"):
        assert word in result.text.split()
    assert grug.verify(text, result.text) == []


def test_numbers_survive(backend):
    text = "We ran a study of 1,250 accounts over 3-5 billing cycles for a 12.5% gain."
    result = backend.compress(text, rate=0.1)
    for number in ("1,250", "3-5", "12.5%"):
        assert number in result.text


def test_urls_survive_verbatim(backend):
    text = "Please see the documentation at https://example.com/a/b?c=1 for the details."
    result = backend.compress(text, rate=0.2)
    assert "https://example.com/a/b?c=1" in result.text


def test_inline_code_survives_verbatim(backend):
    text = "You should pass the `--dry-run of the thing` flag to the command."
    result = backend.compress(text, rate=0.2)
    assert "`--dry-run of the thing`" in result.text


def test_fenced_code_survives_verbatim(backend):
    text = "Intro of the thing.\n\n```py\nx = the_value  #  a  comment\n```\n\nOutro of it."
    result = backend.compress(text, rate=0.2)
    assert "```py\nx = the_value  #  a  comment\n```" in result.text


def test_keep_words_are_honoured():
    text = "The quarterly report is available on the dashboard."
    assert "the" not in RulesBackend().compress(text, rate=0.3).text.lower().split()
    kept = RulesBackend(keep_words={"the"}).compress(text, rate=0.3)
    assert "the" in kept.text.lower().split()


# -- pleasantries -----------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "It is important to note that the",
        "Please note that the",
        "It should be noted that the",
        "Needless to say, the",
    ],
)
def test_pleasantry_phrases_are_stripped(backend, phrase):
    result = backend.compress(f"{phrase} build failed.", rate=0.9)
    assert "note" not in result.text.lower()
    assert "build failed" in result.text.lower()


def test_pleasantries_can_be_disabled():
    text = "Please note that the build failed."
    result = RulesBackend(drop_pleasantries=False).compress(text, rate=0.95)
    assert "note" in result.text.lower()


def test_wordy_phrases_get_terser_replacements(backend):
    result = backend.compress("We did it in order to verify the claim.", rate=0.95)
    assert "in order to" not in result.text
    assert "verify" in result.text


# -- structure --------------------------------------------------------------


def test_paragraph_breaks_survive(backend):
    text = "The first paragraph is here.\n\nThe second paragraph is over here."
    result = backend.compress(text, rate=0.5)
    assert "\n\n" in result.text


def test_markdown_heading_structure_survives(backend):
    """The marker and the line stay; the heading's own articles may go."""
    result = backend.compress("# The Big Heading\n\nSome of the body text.", rate=0.4)
    assert result.text.startswith("# ")
    assert "Big Heading" in result.text.splitlines()[0]


def test_batch_matches_individual_calls(backend):
    texts = ["The first document is here.", "The second document is over there."]
    batched = backend.compress_batch(texts, rate=0.5)
    assert [r.text for r in batched] == [backend.compress(t, rate=0.5).text for t in texts]


# -- public API -------------------------------------------------------------


def test_default_backend_selection_prefers_lingua2_when_installed():
    """Checked without compressing: the default may be a backend that loads a model."""
    from grug.registry import get_backend_class

    name = grug.default_backend_name()
    assert name in grug.list_backends()
    assert get_backend_class(name).is_available()
    expected = "lingua2" if get_backend_class("lingua2").is_available() else "rules"
    assert name == expected


@pytest.mark.slow
def test_module_level_compress_works_with_no_backend_argument():
    """Marked slow: with the lingua2 extra installed this loads a model."""
    result = grug.compress("The report is available on the dashboard.", rate=0.5)
    assert result.backend == grug.default_backend_name()


def test_module_level_compress_accepts_a_backend_name():
    result = grug.compress("The report is on the dashboard.", rate=0.5, backend="rules")
    assert result.backend == "rules"


def test_verify_is_on_by_default():
    text = "Bills scale with volume, not price, across every plan we support."
    result = grug.compress(text, rate=0.5, backend=RulesBackend(keep_words=set()))
    assert isinstance(result.warnings, list)


def test_verify_can_be_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(grug, "run_verify", lambda *a: calls.append(a) or [])
    grug.compress("Some of the text here.", backend="rules", verify=False)
    assert calls == []


def test_compressor_is_reusable():
    comp = grug.Compressor(backend="rules")
    assert comp.backend_name == "rules"
    results = comp.compress_batch(["The first one.", "The second one."], rate=0.5)
    assert len(results) == 2
    assert all(r.backend == "rules" for r in results)


def test_compressor_accepts_a_ready_made_backend():
    instance = RulesBackend()
    comp = grug.Compressor(instance)
    assert comp.backend is instance


def test_compressor_rejects_kwargs_with_an_instance():
    with pytest.raises(TypeError, match="already-constructed backend"):
        grug.Compressor(RulesBackend(), device="cuda")


def test_result_serialises_to_json():
    import json

    result = grug.compress("The report is on the dashboard.", rate=0.5, backend="rules")
    payload = json.loads(json.dumps(result.to_dict()))
    assert payload["backend"] == "rules"
    assert payload["text"] == result.text
    assert set(payload) == {
        "text",
        "original_tokens",
        "compressed_tokens",
        "ratio",
        "backend",
        "warnings",
        "metadata",
    }


def test_saved_tokens_helper():
    result = grug.compress(
        "It is important to note that the build failed on the second attempt.",
        rate=0.5,
        backend="rules",
    )
    assert result.saved_tokens == result.original_tokens - result.compressed_tokens
    assert result.saved_tokens > 0


def test_unknown_kwargs_are_rejected(backend):
    """A typo'd option must not silently do nothing."""
    with pytest.raises(TypeError, match="keep_word"):
        backend.compress("some text", rate=0.5, keep_word={"the"})


def test_unknown_kwargs_are_rejected_through_the_public_api():
    with pytest.raises(TypeError, match="devise"):
        grug.compress("some text", rate=0.5, backend="rules", devise="cuda")


def test_known_kwargs_still_pass_through(backend):
    result = backend.compress("The report is on the dashboard.", rate=0.3, keep_words={"the"})
    assert "the" in result.text.lower().split()
    assert backend.compress("Please note that it failed.", rate=0.9, drop_pleasantries=False)
