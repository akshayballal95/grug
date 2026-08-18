"""Post-hoc force-token restoration.

``restore_forced`` is the pinning guarantee for backends whose compressor has
no ``force_tokens`` of its own: it aligns the compressed output against the
original and splices back any protected word that was dropped.
"""

from __future__ import annotations

from grug.pinning import restore_forced, snap_to_words


def test_dropped_negation_is_reinserted_in_place():
    text, restored = restore_forced(
        "accounts on the legacy plan must not be moved",
        "accounts legacy plan must be moved",
        ["not"],
    )
    assert text == "accounts legacy plan must not be moved"
    assert restored == ["not"]


def test_surviving_negation_is_not_duplicated():
    text, restored = restore_forced(
        "the migration is not automatic", "migration not automatic", ["not"]
    )
    assert text == "migration not automatic"
    assert restored == []


def test_only_the_missing_occurrence_is_restored():
    """Two 'no' in the original, one survived: restore exactly the other one."""
    text, restored = restore_forced(
        "no customer was billed twice and no invoice was lost",
        "no customer billed twice invoice lost",
        ["no"],
    )
    assert text == "no customer billed twice no invoice lost"
    assert restored == ["no"]


def test_restored_word_keeps_its_original_casing():
    text, restored = restore_forced("No invoice was lost", "invoice lost", ["no"])
    assert text == "No invoice lost"
    assert restored == ["No"]


def test_punctuation_does_not_hide_a_forced_word():
    text, _ = restore_forced(
        "bills scale with volume, not price.", "bills scale volume, price.", ["not"]
    )
    assert text == "bills scale volume, not price."


def test_contraction_is_restored_by_its_suffix_token():
    """'n't' is a suffix cue; the word that gets spliced back is the whole word."""
    text, restored = restore_forced(
        "the cutover doesn't need approval", "cutover need approval", ["n't"]
    )
    assert text == "cutover doesn't need approval"
    assert restored == ["doesn't"]


def test_placeholder_is_restored():
    text, _ = restore_forced("see GRUGSPANc0X for details", "see for details", ["GRUGSPANc0X"])
    assert text == "see GRUGSPANc0X for details"


def test_insertion_anchors_to_the_previous_word_not_the_next():
    """Anchoring to the next word would push 'not' across the line break."""
    text, _ = restore_forced(
        "first line has not\nsecond line", "first line has\nsecond line", ["not"]
    )
    assert text == "first line has not\nsecond line"


def test_forced_word_after_the_last_surviving_word_is_appended():
    text, _ = restore_forced("the plan is not", "plan is", ["not"])
    assert text == "plan is not"


def test_forced_word_before_the_first_surviving_word_is_prepended():
    text, _ = restore_forced("never mind the gap", "mind gap", ["never"])
    assert text == "never mind gap"


def test_empty_force_list_returns_the_input_untouched():
    assert restore_forced("a b c", "a c", []) == ("a c", [])


def test_everything_dropped_restores_in_original_order():
    text, restored = restore_forced("not this and never that", "", ["not", "never"])
    assert text == "not never"
    assert restored == ["not", "never"]


# -- snapping a token-level output back onto whole words --------------------


def test_a_mid_word_fragment_is_dropped():
    """LLMLingua-1 filters raw BPE tokens, so it can keep half a word."""
    text, dropped = snap_to_words("accounts on the legacy plan", "accounts the acy plan")
    assert text == "accounts the plan"
    assert dropped == ["acy"]


def test_whole_words_are_kept_in_order():
    text, dropped = snap_to_words("the quick brown fox jumps", "quick fox jumps")
    assert text == "quick fox jumps"
    assert dropped == []


def test_a_mangled_placeholder_is_dropped_not_left_as_garbage():
    text, _ = snap_to_words("see GRUGSPANc0X for details", "see GRUGSPigrating details")
    assert text == "see details"


def test_a_repeated_word_is_not_treated_as_a_fragment():
    text, dropped = snap_to_words("pay the fee and pay the tax", "pay fee pay tax")
    assert text == "pay fee pay tax"
    assert dropped == []


def test_line_structure_survives_a_dropped_fragment():
    text, _ = snap_to_words("first line here\nsecond line there", "first acy\nsecond there")
    assert text == "first\nsecond there"


def test_a_paragraph_break_is_not_downgraded_by_a_dropped_fragment():
    text, _ = snap_to_words("alpha beta\n\ngamma delta", "alpha acy\n\ngamma delta")
    assert text == "alpha\n\ngamma delta"


def test_punctuation_attached_to_a_real_word_is_kept():
    text, dropped = snap_to_words("bills scale with volume, not price.", "bills volume, price.")
    assert text == "bills volume, price."
    assert dropped == []


def test_snapping_an_empty_output_is_empty():
    assert snap_to_words("some original text", "") == ("", [])


def test_snap_then_restore_recovers_a_placeholder_intact():
    """The two halves of the guarantee: drop the wreckage, put the real one back."""
    snapped, _ = snap_to_words("see GRUGSPANc0X for details", "see GRUGSPigrating details")
    text, restored = restore_forced("see GRUGSPANc0X for details", snapped, ["GRUGSPANc0X"])
    assert text == "see GRUGSPANc0X details"
    assert restored == ["GRUGSPANc0X"]
