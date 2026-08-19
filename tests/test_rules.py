"""The rules backend end to end: engine, composable rules, language packs."""

from __future__ import annotations

import pytest

import grug
from grug.backends.rules import (
    ENGLISH,
    Language,
    PatternRule,
    PhraseRule,
    Rule,
    RulesBackend,
    RuleSet,
    WordClassRule,
    available_languages,
    get_language,
    register_language,
    unregister_language,
)


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
    assert result.metadata["language"] == "en"


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


def test_keep_words_add_instance_level_vetoes():
    text = "The quarterly report is available on the dashboard."
    assert "the" not in RulesBackend().compress(text, rate=0.3).text.lower().split()
    kept = RulesBackend(keep_words={"The"}).compress(text, rate=0.3)
    assert "the" in kept.text.lower().split()


def test_relations_between_numbers_survive(backend):
    """ "3 of 12" must never become "3 12": the relation is load-bearing."""
    result = backend.compress("The build failed on 3 of 12 runs today.", rate=0.05)
    assert "3 of 12" in result.text


def test_number_ranges_survive(backend):
    result = backend.compress("The batch takes 3 to 5 days of processing time.", rate=0.05)
    assert "3 to 5" in result.text


def test_multiword_number_relations_survive(backend):
    result = backend.compress("The rollout reached 2 out of 3 regions overnight.", rate=0.05)
    assert "2 out of 3" in result.text


def test_connectives_still_drop_away_from_numbers(backend):
    result = backend.compress("The report of the team is available to everyone here.", rate=0.05)
    assert "of" not in result.text.split()
    assert "to" not in result.text.split()


def test_engine_vetoes_beat_a_hostile_rule():
    """Rules only nominate; negations, numbers and URLs are never droppable."""

    class DropEverything(Rule):
        name = "hostile"

        def drop_candidates(self, cores):
            return ((0.0, i) for i in range(len(cores)))

    backend = RulesBackend(rules=RuleSet(DropEverything()))
    text = "Do not delete the 3 backups at https://example.com/x before Friday."
    result = backend.compress(text, rate=0.05)
    assert "not" in result.text.split()
    assert "3" in result.text
    assert "https://example.com/x" in result.text
    assert "delete" not in result.text.split()


# -- composing rule sets ------------------------------------------------------


def test_ruleset_lookup_and_ordering():
    rules = ENGLISH.rules
    assert rules.names[0] == "pleasantries"
    assert "articles" in rules
    assert rules["articles"].priority < rules["pronouns"].priority
    assert len(rules) == len(rules.names)


def test_remove_takes_a_whole_word_class_out():
    text = "They shipped it to them yesterday for a review."
    dropped = RulesBackend().compress(text, rate=0.2)
    kept = RulesBackend(rules=ENGLISH.rules.remove("pronouns")).compress(text, rate=0.2)
    assert "them" not in dropped.text.split()
    assert "them" in kept.text.split()


def test_remove_of_an_unknown_rule_is_rejected():
    with pytest.raises(KeyError, match="pronounz"):
        ENGLISH.rules.remove("pronounz")


def test_add_appends_a_new_word_class():
    rules = ENGLISH.rules.add(WordClassRule("corp-speak", {"synergy", "leverage"}, priority=5))
    text = "We leverage synergy to ship the product faster."
    default = RulesBackend().compress(text, rate=0.3)
    custom = RulesBackend(rules=rules).compress(text, rate=0.3)
    assert "synergy" in default.text.lower()
    assert "synergy" not in custom.text.lower()


def test_add_with_a_duplicate_name_is_rejected():
    with pytest.raises(ValueError, match="articles"):
        ENGLISH.rules.add(WordClassRule("articles", {"la"}, priority=5))


def test_replace_swaps_a_rule_in_place():
    swapped = ENGLISH.rules.replace(WordClassRule("articles", {"la"}, priority=20))
    assert swapped.names == ENGLISH.rules.names
    assert "la" in swapped["articles"].words
    assert "the" not in swapped["articles"].words


def test_replace_of_an_unknown_rule_is_rejected():
    with pytest.raises(KeyError, match="artikles"):
        ENGLISH.rules.replace(WordClassRule("artikles", {"la"}, priority=20))


def test_composition_returns_new_sets():
    before = ENGLISH.rules.names
    ENGLISH.rules.remove("pronouns")
    assert ENGLISH.rules.names == before


def test_including_and_excluding_words():
    articles = ENGLISH.rules["articles"]
    grown = articles.including("El", "la")
    assert {"el", "la"} <= grown.words
    shrunk = grown.excluding("THE")
    assert "the" not in shrunk.words
    assert "the" in articles.words


def test_excluding_a_word_stops_it_dropping():
    rules = ENGLISH.rules.replace(ENGLISH.rules["articles"].excluding("the"))
    text = "The report is available on the dashboard."
    result = RulesBackend(rules=rules).compress(text, rate=0.3)
    assert "the" in result.text.lower().split()


def test_lower_priority_word_classes_drop_first():
    rules = RuleSet(
        WordClassRule("cheap", {"alpha"}, priority=1),
        WordClassRule("dear", {"beta"}, priority=2),
    )
    pack = Language(code="xx-priority", rules=rules)
    text = "alpha beta cat dog bird fish tree rock lake hill"
    result = RulesBackend(language=pack).compress(text, rate=0.98)
    words = result.text.split()
    assert "alpha" not in words
    assert "beta" in words


def test_a_custom_phrase_rule_rewrites_text():
    rules = ENGLISH.rules.add(PhraseRule("corp-phrases", ((r"\bcircle back\b", "revisit"),)))
    result = RulesBackend(rules=rules).compress("We will circle back on the plan.", rate=0.95)
    assert "circle back" not in result.text
    assert "revisit" in result.text


def test_a_bare_rule_subclass_changes_nothing():
    class Noop(Rule):
        name = "noop"

    text = "Some words stay right here."
    result = RulesBackend(rules=RuleSet(Noop())).compress(text, rate=0.5)
    assert result.text == text


def test_a_rule_nominating_a_bad_index_is_rejected():
    class Broken(Rule):
        name = "broken"

        def drop_candidates(self, cores):
            return [(0.0, 999)]

    with pytest.raises(ValueError, match="broken"):
        RulesBackend(rules=RuleSet(Broken())).compress("a few plain words", rate=0.5)


def test_rules_must_be_named():
    class Anonymous(Rule):
        pass

    with pytest.raises(ValueError, match="name"):
        RuleSet(Anonymous())


# -- regex: dropping and keeping by pattern -----------------------------------


def test_pattern_rule_drops_matching_words():
    rules = ENGLISH.rules.add(PatternRule("hedges", r"(arguabl|probabl|possibl)\w*", priority=5))
    text = "Arguably the fix probably works everywhere."
    result = RulesBackend(rules=rules).compress(text, rate=0.4)
    lowered = result.text.lower()
    assert "arguably" not in lowered
    assert "probably" not in lowered
    assert "works" in lowered


def test_pattern_rule_matches_whole_cores_only():
    """``prob`` must not drop ``probably``: patterns describe the full word."""
    rules = RuleSet(PatternRule("partial", r"prob", priority=5))
    result = RulesBackend(rules=rules).compress("the fix probably works everywhere", rate=0.05)
    assert "probably" in result.text.split()


def test_pattern_rule_rejects_a_bad_regex_at_construction():
    import re

    with pytest.raises(re.error):
        PatternRule("broken", r"(", priority=5)


def test_keep_patterns_veto_nominated_words():
    pack = Language(
        code="greek-test",
        rules=RuleSet(WordClassRule("greek", {"alpha", "beta"}, priority=10)),
    )
    plain = RulesBackend(language=pack).compress("alpha beta gamma delta words", rate=0.05)
    kept = RulesBackend(language=pack, keep_patterns=(r"al\w+",)).compress(
        "alpha beta gamma delta words", rate=0.05
    )
    assert "alpha" not in plain.text.split()
    assert "alpha" in kept.text.split()
    assert "beta" not in kept.text.split()


def test_never_drop_patterns_see_the_raw_token():
    """Cores are lowercased, so an ALL-CAPS veto must match the token itself."""
    pack = Language(
        code="acronym-test",
        rules=RuleSet(WordClassRule("jargon", {"api", "beta"}, priority=10)),
        never_drop_patterns=(r"[A-Z]{2,}",),
    )
    result = RulesBackend(language=pack).compress("Ship the API beta build today", rate=0.05)
    words = result.text.split()
    assert "API" in words
    assert "beta" not in words


# -- pleasantries -------------------------------------------------------------


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


def test_pleasantries_can_be_removed_from_the_ruleset():
    backend = RulesBackend(rules=ENGLISH.rules.remove("pleasantries"))
    result = backend.compress("Please note that the build failed.", rate=0.95)
    assert "note" in result.text.lower()


def test_wordy_phrases_get_terser_replacements(backend):
    result = backend.compress("We did it in order to verify the claim.", rate=0.95)
    assert "in order to" not in result.text
    assert "verify" in result.text


# -- languages ----------------------------------------------------------------


def test_english_is_registered():
    assert "en" in available_languages()
    assert get_language("en") is ENGLISH


def test_unknown_languages_are_rejected_with_the_available_list():
    with pytest.raises(KeyError, match="Unknown language 'xx'"):
        get_language("xx")
    with pytest.raises(KeyError, match="Unknown language"):
        RulesBackend(language="xx")


def test_a_new_language_can_be_registered_and_used():
    pack = Language(
        code="de-test",
        rules=RuleSet(
            WordClassRule("artikel", {"der", "die", "das"}, priority=10),
            # Nominates the negation too, to prove never_drop wins.
            WordClassRule("kopula", {"ist", "nicht"}, priority=20),
        ),
        never_drop=frozenset({"nicht"}),
    )
    register_language(pack)
    try:
        backend = RulesBackend(language="de-test")
        result = backend.compress("Der Plan ist nicht die Antwort auf das Problem.", rate=0.05)
        words = [w.strip(".,").lower() for w in result.text.split()]
        assert "nicht" in words
        assert "ist" not in words
        assert "der" not in words
        assert result.metadata["language"] == "de-test"
    finally:
        unregister_language("de-test")


def test_an_unregistered_language_pack_can_be_passed_directly():
    pack = Language(
        code="uni-test",
        rules=RuleSet(WordClassRule("füllwörter", {"über", "während"}, priority=10)),
    )
    result = RulesBackend(language=pack).compress(
        "Wir sprechen über den Plan während der Nacht.", rate=0.05
    )
    assert "über" not in result.text
    assert "während" not in result.text


# -- structure ----------------------------------------------------------------


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


# -- public API ---------------------------------------------------------------


def test_the_default_backend_is_rules():
    """Zero-configuration means zero dependencies: the default never loads a model."""
    assert grug.default_backend_name() == "rules"


def test_module_level_compress_works_with_no_backend_argument():
    result = grug.compress("The report is available on the dashboard.", rate=0.5)
    assert result.backend == "rules"


def test_module_level_compress_accepts_a_backend_name():
    result = grug.compress("The report is on the dashboard.", rate=0.5, backend="rules")
    assert result.backend == "rules"


def test_verify_is_on_by_default():
    text = "Bills scale with volume, not price, across every plan we support."
    result = grug.compress(text, rate=0.5, backend=RulesBackend())
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


def test_compress_takes_no_per_call_options(backend):
    """Composition happens at construction; a per-call option is a typo."""
    with pytest.raises(TypeError, match="keep_words"):
        backend.compress("some text", rate=0.5, keep_words={"the"})


def test_unknown_kwargs_are_rejected_through_the_public_api():
    with pytest.raises(TypeError, match="devise"):
        grug.compress("some text", rate=0.5, backend="rules", devise="cuda")
