"""Composable rule primitives: word classes, phrase rewrites, language packs.

A rule *nominates* text to lose -- a phrase to rewrite, or words to drop at a
priority. Safety lives in the engine (:mod:`.backend`), which vetoes negations,
numbers, code and document structure no matter what a rule asks for.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from functools import cached_property

__all__ = [
    "Language",
    "LanguageNotFoundError",
    "PatternRule",
    "PhraseRule",
    "Rule",
    "RuleSet",
    "WordClassRule",
    "available_languages",
    "get_language",
    "register_language",
    "unregister_language",
]


class Rule:
    """One named, composable transformation the rules engine runs.

    Subclasses override either hook (or both); the defaults do nothing. Rules
    only nominate: the engine's vetoes -- negations, numbers, code, markdown
    structure, a language's ``never_drop`` list -- always win.
    """

    #: Handle used by :meth:`RuleSet.remove`/:meth:`RuleSet.replace` and errors.
    #: Annotation only: a value here would become an inherited dataclass
    #: default and force every subclass field after it to have one too.
    name: str

    def rewrite(self, text: str) -> str:
        """Rewrite phase, before any word is scored for deletion."""
        return text

    def drop_candidates(self, cores: Sequence[str]) -> Iterable[tuple[float, int]]:
        """Nominate words to drop, as ``(priority, index)`` pairs.

        ``cores[i]`` is the lowercase alphabetic core of the i-th token
        (``""`` when the token has no letters). Lower priorities drop first.
        """
        return ()


@dataclass(frozen=True)
class WordClassRule(Rule):
    """A named class of droppable words -- articles, fillers -- at one priority."""

    name: str
    words: frozenset[str] | set[str]
    priority: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "words", frozenset(w.lower() for w in self.words))

    def including(self, *words: str) -> WordClassRule:
        """A copy that also drops ``words``."""
        return replace(self, words=self.words | frozenset(words))

    def excluding(self, *words: str) -> WordClassRule:
        """A copy that no longer drops ``words``."""
        return replace(self, words=self.words - frozenset(w.lower() for w in words))

    def drop_candidates(self, cores: Sequence[str]) -> Iterable[tuple[float, int]]:
        return ((self.priority, i) for i, core in enumerate(cores) if core in self.words)


@dataclass(frozen=True)
class PatternRule(Rule):
    """Drops every word whose core matches a regex, at one priority.

    The pattern must describe the whole word (it is ``fullmatch``-ed against
    the lowercase core), so ``prob`` cannot accidentally take out "probably".
    """

    name: str
    pattern: str
    priority: float

    def __post_init__(self) -> None:
        self._compiled  # noqa: B018  (compile now, so a bad regex fails at construction)

    @cached_property
    def _compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern)

    def drop_candidates(self, cores: Sequence[str]) -> Iterable[tuple[float, int]]:
        return (
            (self.priority, i) for i, core in enumerate(cores) if self._compiled.fullmatch(core)
        )


@dataclass(frozen=True)
class PhraseRule(Rule):
    """Regex phrase rewrites, applied before any word is dropped.

    Rewrites are not budgeted: a matched phrase is always rewritten, even at a
    rate of 0.99. They also run before the engine's word vetoes, so a pattern
    that swallows a negation is on the rule's author; the verifier is the net.
    """

    name: str
    #: ``(pattern, replacement)`` pairs, applied in order, case-insensitively.
    patterns: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "patterns", tuple((p, r) for p, r in self.patterns))

    @cached_property
    def _compiled(self) -> tuple[tuple[re.Pattern[str], str], ...]:
        return tuple((re.compile(p, re.IGNORECASE), r) for p, r in self.patterns)

    @cached_property
    def _gate(self) -> re.Pattern[str]:
        # One scan decides whether any of the substitutions is worth running;
        # most chunks contain none of them.
        return re.compile("|".join(f"(?:{p})" for p, _ in self.patterns), re.IGNORECASE)

    def rewrite(self, text: str) -> str:
        if not self.patterns or not self._gate.search(text):
            return text
        for pattern, replacement in self._compiled:
            text = pattern.sub(replacement, text)
        return text


class RuleSet:
    """An immutable, ordered collection of rules, addressed by name.

    Composition returns new sets: :meth:`add` appends, :meth:`remove` deletes
    by name, :meth:`replace` swaps a same-named rule. Unknown and duplicate
    names raise, so a typo can never silently change nothing.
    """

    def __init__(self, *rules: Rule) -> None:
        by_name: dict[str, Rule] = {}
        for rule in rules:
            if not getattr(rule, "name", ""):
                raise ValueError(f"{type(rule).__name__} must have a non-empty name")
            if rule.name in by_name:
                raise ValueError(f"duplicate rule name {rule.name!r}")
            by_name[rule.name] = rule
        self._rules: tuple[Rule, ...] = tuple(rules)
        self._by_name = by_name

    @property
    def names(self) -> tuple[str, ...]:
        """Rule names, in application order."""
        return tuple(rule.name for rule in self._rules)

    def __iter__(self) -> Iterator[Rule]:
        return iter(self._rules)

    def __len__(self) -> int:
        return len(self._rules)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> Rule:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(self._unknown(name)) from None

    def add(self, *rules: Rule) -> RuleSet:
        """A new set with ``rules`` appended. A name already present raises."""
        return RuleSet(*self._rules, *rules)

    def remove(self, *names: str) -> RuleSet:
        """A new set without the named rules. An unknown name raises."""
        for name in names:
            if name not in self._by_name:
                raise KeyError(self._unknown(name))
        gone = set(names)
        return RuleSet(*(rule for rule in self._rules if rule.name not in gone))

    def replace(self, *rules: Rule) -> RuleSet:
        """A new set with each same-named rule swapped in place. Unknown names raise."""
        for rule in rules:
            if rule.name not in self._by_name:
                raise KeyError(self._unknown(rule.name))
        swap = {rule.name: rule for rule in rules}
        return RuleSet(*(swap.get(rule.name, rule) for rule in self._rules))

    def _unknown(self, name: str) -> str:
        listed = ", ".join(self._by_name) or "(none)"
        return f"no rule named {name!r}; have: {listed}"

    def __repr__(self) -> str:
        return f"RuleSet({', '.join(self.names)})"


@dataclass(frozen=True, eq=False)
class Language:
    """Everything the rules engine needs to compress one language."""

    #: Registry key, e.g. ``"en"``.
    code: str
    #: The rules run by default for this language.
    rules: RuleSet
    #: Words the engine must never drop, whatever a rule nominates. Negations
    #: belong here: losing one flips the meaning of what remains.
    never_drop: frozenset[str] | set[str] = frozenset()
    #: Regexes vetoing whole classes of words. Searched against the *raw*
    #: token, punctuation and case included, so ``[A-Z]{2,}`` can protect
    #: acronyms that lowercased cores erase. A veto that matches too much only
    #: keeps extra words, which is the safe direction to err in.
    never_drop_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "never_drop", frozenset(w.lower() for w in self.never_drop))
        patterns = tuple(self.never_drop_patterns)
        for pattern in patterns:
            re.compile(pattern)  # a bad regex should fail here, not mid-compress
        object.__setattr__(self, "never_drop_patterns", patterns)


class LanguageNotFoundError(KeyError):
    """Raised when a language code is not in the registry."""

    def __init__(self, code: str, available: list[str]) -> None:
        self.code = code
        self.available = available
        listed = ", ".join(available) if available else "(none)"
        super().__init__(f"Unknown language {code!r}. Registered languages: {listed}")

    def __str__(self) -> str:  # KeyError repr-quotes its message otherwise
        return self.args[0]


_LANGUAGES: dict[str, Language] = {}


def register_language(language: Language) -> Language:
    """Register a pack under its code, replacing any previous one.

    Returns the pack, so a module can register and export in one statement.
    """
    if not language.code:
        raise ValueError("Language must have a non-empty code")
    _LANGUAGES[language.code] = language
    return language


def unregister_language(code: str) -> None:
    """Remove a language from the registry. Mainly useful in tests."""
    _LANGUAGES.pop(code, None)


def get_language(code: str) -> Language:
    """Look up a registered language pack by code."""
    try:
        return _LANGUAGES[code]
    except KeyError:
        raise LanguageNotFoundError(code, available_languages()) from None


def available_languages() -> list[str]:
    """Codes of every registered language."""
    return sorted(_LANGUAGES)
