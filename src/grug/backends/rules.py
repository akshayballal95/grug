"""Rule-based compressor: no model, no dependencies, no surprises.

Deletes low-information words, so every word in the output appeared in the
input. Deletion is budgeted: words are ranked into tiers from "safe to lose"
(fillers) to "risky" (pronouns) and only as many are dropped as the rate
demands. Numbers, negations, URLs and code are never candidates.
"""

from __future__ import annotations

import re
from typing import Any

from ..base import CompressionResult, CompressorBackend, count_tokens
from ..chunking import (
    FENCE_RE,
    INLINE_CODE_RE,
    URL_RE,
    contains_placeholder,
    protect_spans,
    restore_spans,
    word_cost,
)
from ..registry import register_backend
from ..verify import NEGATION_WORDS

__all__ = ["RulesBackend"]

# --------------------------------------------------------------------------
# Word tiers, cheapest to lose first.
# --------------------------------------------------------------------------

# fmt: off
FILLERS = frozenset({
    "basically", "actually", "really", "very", "quite", "simply", "just",
    "essentially", "literally", "obviously", "clearly", "indeed", "truly",
    "extremely", "totally", "utterly", "fairly", "somewhat", "kind", "sort",
    "well", "so", "then", "thus", "hence", "therefore", "moreover",
    "furthermore", "additionally", "please", "kindly", "thanks", "thank",
    "hello", "hi", "regards", "sincerely",
})

ARTICLES = frozenset({"a", "an", "the"})

COPULAS = frozenset({
    "is", "are", "was", "were", "am", "be", "been", "being",
    "has", "have", "had", "does", "do", "did", "get", "gets", "got",
})

FUNCTION_WORDS = frozenset({
    "of", "to", "in", "on", "at", "for", "with", "from", "by", "as",
    "that", "which", "who", "whom", "whose", "into", "onto", "upon",
    "and", "also", "about", "over", "under", "between", "through",
    "during", "while", "when", "where", "than", "such", "some", "any",
})

PRONOUNS = frozenset({
    "it", "its", "they", "them", "their", "theirs", "we", "our", "ours",
    "us", "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "i", "my", "me", "mine", "this", "these", "those", "there", "here",
})
# fmt: on

#: word -> tier. Earlier tiers are dropped first; a word keeps its lowest tier.
_WORD_TIER: dict[str, int] = {}
for _tier, _words in enumerate((FILLERS, ARTICLES, COPULAS, FUNCTION_WORDS, PRONOUNS)):
    for _word in _words:
        _WORD_TIER.setdefault(_word, _tier)

#: Phrases that carry no information at all, with optional terser replacements.
PLEASANTRIES: tuple[tuple[str, str], ...] = (
    (r"\bit is important to note that\b", ""),
    (r"\bit is important to (?:understand|remember|realize) that\b", ""),
    (r"\bit is important to\b", ""),
    (r"\bit should be noted that\b", ""),
    (r"\bit is worth noting that\b", ""),
    (r"\bplease note that\b", ""),
    (r"\bplease be aware that\b", ""),
    (r"\bplease note\b", ""),
    (r"\bas you (?:can see|may know|are aware)\b", ""),
    (r"\bneedless to say\b", ""),
    (r"\bas a matter of fact\b", ""),
    (r"\bkeep in mind that\b", ""),
    (r"\bbear in mind that\b", ""),
    (r"\bthe fact that\b", "that"),
    (r"\bdue to the fact that\b", "because"),
    (r"\bin spite of the fact that\b", "although"),
    (r"\bin order to\b", "to"),
    (r"\bin the event that\b", "if"),
    (r"\bat this point in time\b", "now"),
    (r"\bfor the purpose of\b", "for"),
    (r"\bwith (?:regard|respect) to\b", "about"),
    (r"\bin terms of\b", "in"),
    (r"\ba (?:large |great )?number of\b", "many"),
    (r"\bin the near future\b", "soon"),
    (r"\bon a regular basis\b", "regularly"),
    (r"\bhope this helps\b", ""),
    (r"\bfeel free to\b", ""),
    (r"\blet me know if\b", "if"),
)

_PLEASANTRY_RES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in PLEASANTRIES
)
#: Checked before running all of the above; most chunks contain none of them.
_PLEASANTRY_GATE = re.compile(
    "|".join(f"(?:{pattern})" for pattern, _ in PLEASANTRIES), re.IGNORECASE
)

#: Protected when this backend runs standalone; the chunker does the same on the
#: pipeline path. A distinct tag lets the two passes nest.
_PROTECTED_PATTERNS = (FENCE_RE, INLINE_CODE_RE, URL_RE)
_STASH_TAG = "r"

_TOKEN_RE = re.compile(r"\s+|\S+")
_CORE_RE = re.compile(r"[A-Za-z][A-Za-z']*")
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
    """Deterministic stopword-and-filler stripper with zero dependencies."""

    name = "rules"
    description = (
        "Rule-based stopword/filler removal. No ML dependencies, milliseconds per document."
    )
    extra = None
    generative = False

    def __init__(
        self,
        *,
        keep_words: frozenset[str] | set[str] | None = None,
        drop_pleasantries: bool = True,
    ) -> None:
        """
        Args:
            keep_words: Extra lowercase words this backend must never drop.
            drop_pleasantries: Whether to strip filler phrases like
                "it is important to note that".
        """
        self.keep_words = frozenset(keep_words or ())
        self.drop_pleasantries = drop_pleasantries

    def compress(self, text: str, rate: float = 0.5, **kwargs: Any) -> CompressionResult:
        self._validate_rate(rate)
        drop_pleasantries = kwargs.pop("drop_pleasantries", self.drop_pleasantries)
        keep_words = frozenset(kwargs.pop("keep_words", ())) | self.keep_words
        if kwargs:
            # Reject typos like "keep_word=" instead of silently ignoring them.
            raise TypeError(
                f"{self.name} backend got unexpected keyword argument(s): "
                + ", ".join(sorted(map(repr, kwargs)))
            )

        if not text.strip():
            return CompressionResult.build(text, text, self.name, metadata={"dropped_words": 0})

        original_tokens = count_tokens(text)
        working, stash = protect_spans(text, *_PROTECTED_PATTERNS, tag=_STASH_TAG)

        if drop_pleasantries and rate < 1.0 and _PLEASANTRY_GATE.search(working):
            for pattern, replacement in _PLEASANTRY_RES:
                working = pattern.sub(replacement, working)

        budget = original_tokens * rate
        tokens = _TOKEN_RE.findall(working)
        dropped = self._select_drops(tokens, budget, keep_words, rate, count_tokens(working))

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
                "pleasantries_removed": bool(drop_pleasantries and rate < 1.0),
            },
        )

    def _select_drops(
        self,
        tokens: list[str],
        budget: float,
        keep_words: frozenset[str],
        rate: float,
        working_tokens: int,
    ) -> set[int]:
        """Pick token indices to delete, cheapest tier first, until under budget."""
        if rate >= 1.0:
            return set()

        candidates: list[tuple[int, int]] = []  # (tier, index)
        for index, token in enumerate(tokens):
            if not token.strip() or _is_protected(token):
                continue
            core = _core_word(token)
            if not core or core in keep_words or core in NEGATION_WORDS or core.endswith("n't"):
                continue
            tier = _WORD_TIER.get(core)
            if tier is None:
                continue
            # A token carrying sentence-ending punctuation anchors structure.
            if token.rstrip("\"')]").endswith((".", "!", "?", ":")):
                continue
            candidates.append((tier, index))

        candidates.sort()

        # Estimate as words come out rather than re-encoding after each deletion;
        # the ratio reported at the end is measured, not estimated.
        remaining = float(working_tokens)
        dropped: set[int] = set()
        for _tier, index in candidates:
            if remaining <= budget:
                break
            dropped.add(index)
            remaining -= word_cost(tokens[index].strip())
        return dropped
