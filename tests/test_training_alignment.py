"""Deriving keep/drop labels from teacher output. Pure Python, no downloads."""

from __future__ import annotations

import pytest

from grug.training.alignment import (
    align,
    alignment_gap,
    annotate,
    filter_examples,
    fuzzy_match,
    split_words,
    variation_rate,
)


def kept(words, labels):
    return [w for w, keep in zip(words, labels, strict=True) if keep]


# -- fuzzy matching ---------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("program", "program"),
        ("program.", "program"),  # edge punctuation
        ("Program", "program"),  # case
        ("consenting", "Consent"),  # the paper's "Variation" case
        ("programs", "program"),  # plural
        ("joined", "join"),
        ("résumé", "resume"),  # accents
    ],
)
def test_words_that_are_the_same_word(a, b):
    assert fuzzy_match(a, b)


@pytest.mark.parametrize(("a", "b"), [("cat", "dog"), ("invoice", "invoke"), ("no", "on")])
def test_words_that_are_not(a, b):
    assert not fuzzy_match(a, b)


def test_short_words_need_an_exact_match():
    """A 3-char prefix rule must not make every short word match every other."""
    assert not fuzzy_match("is", "in")


# -- alignment --------------------------------------------------------------


def test_exact_subsequence():
    words, labels = align("the cat sat on the mat", "cat sat mat")
    assert kept(words, labels) == ["cat", "sat", "mat"]


def test_ambiguity_picks_the_nearest_occurrence():
    """'program' appears three times; labels must land on the right two."""
    original = "join the Victory program. join the California program. the Hero program."
    words, labels = align(original, "join California program. Hero program.")
    assert sum(labels) == 5
    assert labels[words.index("Victory")] is False


def test_variation_is_matched():
    words, labels = align(
        "consenting to the inclusion of properties", "Consent inclusion properties"
    )
    assert kept(words, labels) == ["consenting", "inclusion", "properties"]


def test_reordering_is_tolerated():
    words, labels = align("properties within the jurisdiction", "jurisdiction properties")
    assert set(kept(words, labels)) == {"properties", "jurisdiction"}


def test_nothing_kept_when_compressed_is_empty():
    words, labels = align("some original text", "")
    assert not any(labels)
    assert len(labels) == len(words)


def test_labels_are_one_per_original_word():
    original = "a b c d e f g"
    words, labels = align(original, "b d f")
    assert len(labels) == len(words) == 7


def test_hallucinated_words_are_ignored():
    """A teacher word absent from the original cannot label anything."""
    words, labels = align("the invoice is late", "the invoice is unicorn")
    assert "unicorn" not in kept(words, labels)


# -- quality metrics --------------------------------------------------------


def test_variation_rate_is_zero_for_a_pure_subsequence():
    assert variation_rate("the cat sat on the mat", "cat sat mat") == 0.0


def test_variation_rate_counts_invented_words():
    assert variation_rate("the cat sat", "cat unicorn") == pytest.approx(0.5)


def test_alignment_gap_is_zero_for_a_clean_alignment():
    words, labels = align("the cat sat on the mat", "cat sat mat")
    assert alignment_gap(words, "cat sat mat", labels) == pytest.approx(0.0, abs=1e-9)


def test_alignment_gap_grows_when_labels_miss_words():
    original, compressed = "alpha beta gamma delta", "beta delta"
    words, _ = align(original, compressed)
    assert alignment_gap(words, compressed, [False] * len(words)) > 0


# -- filtering --------------------------------------------------------------


def test_filter_drops_the_worst_examples():
    good = [annotate("the cat sat on the mat", "cat sat mat") for _ in range(18)]
    bad = [annotate("the cat sat on the mat", "unicorn dragon phoenix") for _ in range(2)]
    kept_examples, thresholds = filter_examples(good + bad)
    assert len(kept_examples) < len(good) + len(bad)
    assert thresholds["dropped"] >= 1


def test_filter_on_empty_input():
    kept_examples, thresholds = filter_examples([])
    assert kept_examples == []
    assert thresholds["vr_threshold"] == float("inf")


# -- end to end -------------------------------------------------------------


def test_annotate_reports_a_sane_keep_ratio():
    stats = annotate("the quick brown fox jumps over the lazy dog", "quick brown fox jumps dog")
    assert 0.0 < stats.keep_ratio < 1.0
    assert stats.variation_rate == 0.0


def test_split_words_handles_whitespace_runs():
    assert split_words("  a\n\n b \t c  ") == ["a", "b", "c"]


def test_the_papers_worked_example():
    """Figure 5: ambiguity, variation and reordering in one passage."""
    original = (
        "Item 15, report from City Manager Recommendation to adopt three resolutions. "
        "First, to join the Victory Pace program. Second, to join the California first program. "
        "And number three, consenting to to inclusion of certain properties within the "
        "jurisdiction in the California Hero program."
    )
    compressed = (
        "City Manager Recommendation adopt three resolutions. Join California first program. "
        "Consent properties inclusion jurisdiction California Hero program."
    )
    stats = annotate(original, compressed)
    survivors = " ".join(kept(stats.words, stats.labels))
    for token in ("City", "Manager", "Recommendation", "consenting", "properties", "jurisdiction"):
        assert token in survivors
    assert "Victory" not in survivors
    assert 0.2 < stats.keep_ratio < 0.6


def test_negation_retention_counts_occurrences_not_vocabulary():
    """Keeping the word "not" once does not retain eight separate negations.

    A set-membership test scored this 1.00, which let a teacher that dropped
    most of a passage's negations look perfectly faithful.
    """
    from grug.training.distill import _retention

    original = "It is not fair, not right, not legal, and not wise."
    kept_one = "It is not fair, right, legal, wise."

    assert _retention(original, original, "negation") == 1.0
    assert _retention(original, kept_one, "negation") == 0.25


def test_number_retention_counts_repeats():
    from grug.training.distill import _retention

    original = "3 apples, 3 pears, 12 plums."
    assert _retention(original, "3 apples, pears, 12 plums.", "number") == 2 / 3


def test_retention_is_none_without_anything_to_retain():
    from grug.training.distill import _retention

    assert _retention("plain text here", "plain text", "negation") is None
    assert _retention("plain text here", "plain text", "number") is None
