"""Rule-based compression: a composable engine plus per-language rule packs.

The engine (:class:`RulesBackend`) never drops negations, numbers, code or
document structure. Everything else is rules, and rules compose::

    from grug.backends.rules import ENGLISH, RulesBackend, WordClassRule

    rules = ENGLISH.rules.remove("pronouns")
    rules = rules.add(WordClassRule("corp-speak", {"synergy"}, priority=5))
    backend = RulesBackend(rules=rules)

New languages register a :class:`Language` pack and get the same engine::

    register_language(Language(code="de", rules=RuleSet(...), never_drop={...}))
    backend = RulesBackend(language="de")
"""

from .backend import RulesBackend
from .core import (
    Language,
    LanguageNotFoundError,
    PatternRule,
    PhraseRule,
    Rule,
    RuleSet,
    WordClassRule,
    available_languages,
    get_language,
    register_language,
    unregister_language,
)
from .english import ENGLISH

__all__ = [
    "ENGLISH",
    "Language",
    "LanguageNotFoundError",
    "PatternRule",
    "PhraseRule",
    "Rule",
    "RuleSet",
    "RulesBackend",
    "WordClassRule",
    "available_languages",
    "get_language",
    "register_language",
    "unregister_language",
]
