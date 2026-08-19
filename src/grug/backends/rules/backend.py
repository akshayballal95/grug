"""The rules engine: no model, no dependencies, no surprises.

Deletes low-information words, so every word in the output appeared in the
input. Rules nominate -- a phrase to rewrite, words to drop at a priority --
and the engine decides: it vetoes negations, numbers and the words relating
them, URLs, code and document structure outright, then drops the cheapest
nominations until the rate's token budget is met.
"""

from __future__ import annotations

import re
from typing import Any

from ...base import CompressionResult, CompressorBackend, count_tokens
from ...chunking import (
    FENCE_RE,
    INLINE_CODE_RE,
    URL_RE,
    contains_placeholder,
    protect_spans,
    restore_spans,
    word_cost,
)
from ...registry import register_backend
from .core import Language, RuleSet, get_language
from .english import ENGLISH  # noqa: F401  (importing registers the default language)

__all__ = ["RulesBackend"]

#: Protected when this backend runs standalone; the chunker does the same on the
#: pipeline path. A distinct tag lets the two passes nest.
_PROTECTED_PATTERNS = (FENCE_RE, INLINE_CODE_RE, URL_RE)
_STASH_TAG = "r"

_TOKEN_RE = re.compile(r"\s+|\S+")
#: A word's core: Unicode letters plus internal apostrophes, so packs beyond
#: English ("über", "während") match their word lists.
_CORE_RE = re.compile(r"[^\W\d_](?:[^\W\d_]|')*")
_DIGIT_RE = re.compile(r"\d")


def _core_word(token: str) -> str:
    """The alphabetic core of a token: ``"(the,"`` -> ``"the"``."""
    match = _CORE_RE.search(token)
    return match.group(0).lower() if match else ""


def _is_protected(token: str) -> bool:
    """Markdown markers, anything with a digit, and stashed spans are never dropped."""
    return (
        token.startswith(("#", ">", "|"))
        or _DIGIT_RE.search(token) is not None
        or contains_placeholder(token)
    )


#: How many words away a digit still protects a connective between numbers.
#: Two covers "3 of 12" and "2 out of 3" without freezing whole sentences.
_NUMBER_BRIDGE_WINDOW = 2


def _bridges_numbers(tokens: list[str]) -> list[bool]:
    """True for each token sitting between two nearby digit-bearing tokens.

    "3 of 12" must never become "3 12": the word between two quantities
    carries their relation, which is as load-bearing as the quantities
    themselves -- so the engine vetoes it the way it vetoes the numbers.
    """
    words = [i for i, token in enumerate(tokens) if token.strip()]
    digit = [_DIGIT_RE.search(tokens[i]) is not None for i in words]
    bridges = [False] * len(tokens)
    for position, index in enumerate(words):
        if digit[position]:
            continue
        before = digit[max(0, position - _NUMBER_BRIDGE_WINDOW) : position]
        after = digit[position + 1 : position + 1 + _NUMBER_BRIDGE_WINDOW]
        if any(before) and any(after):
            bridges[index] = True
    return bridges


def _tidy(text: str) -> str:
    """Collapse the whitespace and punctuation debris deletion leaves behind."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,;:])\s*(?=[,.;:!?])", "", text)
    text = re.sub(r"(?<=[(\[]) +", "", text)
    text = re.sub(r" +(?=[)\]])", "", text)
    text = re.sub(r"^[ \t]*[,;:]+[ \t]*", "", text, flags=re.MULTILINE)
    return text.strip()


@register_backend
class RulesBackend(CompressorBackend):
    """Deterministic, rule-driven compressor with zero ML dependencies."""

    name = "rules"
    description = (
        "Rule-based word/phrase removal: composable rules, language packs, "
        "no ML dependencies, milliseconds per document."
    )
    extra = None
    generative = False

    def __init__(
        self,
        *,
        language: str | Language = "en",
        rules: RuleSet | None = None,
        keep_words: frozenset[str] | set[str] = frozenset(),
        keep_patterns: tuple[str, ...] | list[str] = (),
    ) -> None:
        """
        Args:
            language: A registered language code, or a :class:`Language` pack
                passed directly (it does not need to be registered).
            rules: Ruleset to run instead of the language's own.
            keep_words: Extra words this instance must never drop, on top of
                the language's ``never_drop`` list. Case-insensitive.
            keep_patterns: Regexes vetoing whole classes of words, merged with
                the language's ``never_drop_patterns``. Searched against the
                raw token -- punctuation and case included -- so patterns can
                see what lowercased cores erase, e.g. ``[A-Z]{2,}`` for
                acronyms.
        """
        pack = language if isinstance(language, Language) else get_language(language)
        self.language = pack
        self.rules = pack.rules if rules is None else rules
        self.never_drop = (
            pack.negations | pack.never_drop | frozenset(w.lower() for w in keep_words)
        )
        self.never_drop_patterns = tuple(
            re.compile(p) for p in (*pack.never_drop_patterns, *keep_patterns)
        )

    def compress(self, text: str, rate: float = 0.5, **kwargs: Any) -> CompressionResult:
        self._validate_rate(rate)
        if kwargs:
            # Composition happens at construction; reject typos like "keep_word="
            # instead of silently ignoring them.
            raise TypeError(
                f"{self.name} backend got unexpected keyword argument(s): "
                + ", ".join(sorted(map(repr, kwargs)))
            )

        if not text.strip():
            return CompressionResult.build(text, text, self.name, metadata={"dropped_words": 0})

        original_tokens = count_tokens(text)
        working, stash = protect_spans(text, *_PROTECTED_PATTERNS, tag=_STASH_TAG)

        if rate < 1.0:
            for rule in self.rules:
                working = rule.rewrite(working)

        budget = original_tokens * rate
        tokens = _TOKEN_RE.findall(working)
        dropped = self._select_drops(tokens, budget, rate, count_tokens(working))

        kept = [tok for i, tok in enumerate(tokens) if i not in dropped]
        # Tidy before restoring, so collapsing never reaches a stashed code block.
        compressed = restore_spans(_tidy("".join(kept)), stash, tag=_STASH_TAG)

        return CompressionResult.build(
            text,
            compressed,
            self.name,
            metadata={
                "dropped_words": len(dropped),
                "requested_rate": rate,
                "language": self.language.code,
            },
        )

    def _select_drops(
        self,
        tokens: list[str],
        budget: float,
        rate: float,
        working_tokens: int,
    ) -> set[int]:
        """Pick token indices to delete, cheapest nomination first, until under budget."""
        if rate >= 1.0:
            return set()

        cores = [_core_word(token) for token in tokens]
        bridges = _bridges_numbers(tokens)
        droppable = [
            self._may_drop(token, core) and not bridge
            for token, core, bridge in zip(tokens, cores, bridges, strict=True)
        ]

        candidates: list[tuple[float, int]] = []
        for rule in self.rules:
            for priority, index in rule.drop_candidates(cores):
                if not 0 <= index < len(tokens):
                    raise ValueError(
                        f"rule {rule.name!r} nominated out-of-range word index {index}"
                    )
                if droppable[index]:
                    candidates.append((priority, index))
        candidates.sort()

        # Estimate as words come out rather than re-encoding after each deletion;
        # the ratio reported at the end is measured, not estimated.
        remaining = float(working_tokens)
        dropped: set[int] = set()
        for _priority, index in candidates:
            if remaining <= budget:
                break
            if index in dropped:
                continue
            dropped.add(index)
            remaining -= word_cost(tokens[index].strip())
        return dropped

    def _may_drop(self, token: str, core: str) -> bool:
        """The engine's veto: rules nominate, this decides."""
        if not token.strip() or _is_protected(token):
            return False
        if not core or core in self.never_drop or core.endswith("n't"):
            return False
        if any(pattern.search(token) for pattern in self.never_drop_patterns):
            return False
        # A token carrying sentence-ending punctuation anchors structure.
        return not token.rstrip("\"')]").endswith((".", "!", "?", ":"))
