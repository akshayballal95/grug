"""Faithfulness checks: negation, numbers, entities."""

from __future__ import annotations

import pytest

from grug.verify import find_entities, find_negations, find_numbers, verify

# -- negation ---------------------------------------------------------------


def test_dropped_negation_is_flagged():
    original = "Bills scale with volume, not price."
    compressed = "bills scale volume price."
    warnings = verify(original, compressed)
    assert any("negation" in w for w in warnings)
    assert "'not'" in warnings[0]


def test_kept_negation_is_not_flagged():
    original = "Bills scale with volume, not price."
    compressed = "bills scale volume, not price"
    assert verify(original, compressed) == []


def test_negation_is_the_first_warning_reported():
    """Negation outranks the other checks: inverted meaning beats lost meaning."""
    original = "Acme Corporation did not ship 42 units."
    compressed = "shipped units"
    warnings = verify(original, compressed)
    assert len(warnings) == 3
    assert warnings[0].startswith("negation")


def test_contractions_count_as_negation():
    assert find_negations("it doesn't matter")["n't"] == 1
    warnings = verify("It doesn't matter.", "It matters.")
    assert any("n't" in w for w in warnings)


def test_partial_negation_loss_is_flagged():
    original = "Not now, not ever, and not tomorrow."
    compressed = "not now"
    warnings = verify(original, compressed)
    assert "3×" in warnings[0] and "1×" in warnings[0]


@pytest.mark.parametrize(
    "word", ["not", "no", "never", "none", "neither", "nor", "except", "unless", "without"]
)
def test_every_core_negation_word_is_detected(word):
    assert find_negations(f"this is {word} that") == {word: 1}


# -- numbers ----------------------------------------------------------------


def test_dropped_number_is_flagged():
    warnings = verify("We ran 1,250 tests.", "ran tests")
    assert any("numbers missing" in w for w in warnings)
    assert "'1250'" in warnings[0]


def test_thousands_separator_is_normalised():
    """1,250 and 1250 are the same number and must not be reported as lost."""
    assert verify("We ran 1,250 tests.", "ran 1250 tests") == []


def test_percentages_floats_and_versions_are_tracked():
    found = find_numbers("a 12.5% gain in v2 after 3-5 runs of 1.2.3")
    assert "12.5%" in found
    assert "2" in found
    assert "1.2.3" in found


def test_kept_numbers_pass():
    assert verify("Lag was 1.2 seconds at p99 of 9.6.", "lag 1.2 seconds p99 9.6") == []


# -- entities ---------------------------------------------------------------


def test_dropped_entity_is_flagged():
    warnings = verify("Acme Corporation reported a gain.", "reported gain")
    assert any("entities missing" in w for w in warnings)
    assert "Acme Corporation" in warnings[0]


def test_partially_kept_entity_is_not_flagged():
    """Extractive compression clips 'Acme Corporation' to 'Acme'; that is fine."""
    assert verify("Acme Corporation reported a gain.", "Acme reported gain") == []


def test_sentence_initial_capital_is_not_an_entity():
    """'The' and 'Bills' opening a sentence are not proper nouns."""
    assert find_entities("The system works. Bills arrive monthly.") == []


def test_acronyms_are_entities():
    assert "API" in find_entities("Call the API for details.")


def test_multiword_entity_drops_its_leading_article():
    assert "Platform Reliability" in find_entities("Ask the Platform Reliability team.")


# -- overall ----------------------------------------------------------------


def test_identical_text_produces_no_warnings(sample_markdown):
    assert verify(sample_markdown, sample_markdown) == []


def test_empty_compression_flags_everything():
    warnings = verify("Acme Corporation did not ship 42 units.", "")
    assert len(warnings) == 3


def test_lone_sentence_initial_name_is_not_reported():
    """'Acme' opening a sentence is indistinguishable from any other capital."""
    assert verify("Acme did not ship.", "did not ship") == []


def test_verify_never_raises_on_odd_input():
    assert verify("", "") == []
    assert verify("...", "!!!") == []


def test_long_warning_lists_are_capped():
    original = " ".join(f"Item {i} costs {i * 11} dollars." for i in range(1, 15))
    warnings = verify(original, "")
    numbers = next(w for w in warnings if w.startswith("numbers"))
    assert "more)" in numbers
    assert numbers.count("'") == 12  # six reported items, quoted


def test_warning_text_is_human_readable():
    warnings = verify("Bills scale with volume, not price.", "bills scale volume price")
    assert warnings == ["negation lost: 'not' (1× → 0×) — meaning may be inverted"]


def test_possessive_entity_counts_as_present():
    """ "LLMLingua-2's" surviving as "LLMLingua-2" is not a loss."""
    assert verify("The CLI's flag and LLMLingua-2's model.", "CLI flag LLMLingua-2 model") == []


def test_genuinely_missing_possessive_entity_is_still_flagged():
    warnings = verify("Acme Corporation's revenue fell.", "revenue fell")
    assert any("entities missing" in w for w in warnings)


def test_entity_does_not_span_a_sentence_boundary():
    """Regression: "14:32 UTC. Acme Corp" was extracted as one entity."""
    found = find_entities("reverted at 14:32 UTC. Acme Corporation ran it.")
    assert "UTC" in found
    assert "Acme Corporation" in found
    assert not any("UTC." in e for e in found)


def test_dotted_names_hold_together():
    assert "Node.js" in find_entities("The Node.js team shipped it.")


def test_single_entity_after_an_article_is_kept():
    assert "U.S.A" in find_entities("The U.S.A office opened.")
