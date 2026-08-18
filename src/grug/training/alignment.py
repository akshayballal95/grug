"""Turn (original, compressed) text pairs into per-word keep/drop labels.

This is Algorithm 1 of the LLMLingua-2 paper plus its two quality metrics. The
teacher returns compressed *text*; training needs a binary label per word of the
*original*. Recovering that mapping is not a simple set membership because the
teacher introduces three problems the paper names:

* **Ambiguity** -- a compressed word occurs many times in the original.
* **Variation** -- the teacher silently changes tense or plural form.
* **Reordering** -- word order may differ.

Deliberately dependency-free: no torch, no datasets. The most bug-prone stage of
the pipeline is therefore testable in the fast suite.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

__all__ = [
    "AlignmentStats",
    "align",
    "alignment_gap",
    "filter_examples",
    "fuzzy_match",
    "split_words",
    "variation_rate",
]

#: How far either side of the last match to search for the next word.
DEFAULT_WINDOW = 400

#: Discard the examples whose teacher output looks most hallucinated.
DEFAULT_VR_QUANTILE = 0.05
#: Discard the examples whose labels aligned worst.
DEFAULT_AG_QUANTILE = 0.10

_WORD_SPLIT_RE = re.compile(r"\s+")
_STRIP_RE = re.compile(r"^\W+|\W+$", re.UNICODE)


def split_words(text: str) -> list[str]:
    """Whitespace tokenisation. Words are the unit the compressor scores."""
    return [w for w in _WORD_SPLIT_RE.split(text.strip()) if w]


def _normalise(word: str) -> str:
    """Casefold, strip edge punctuation, and drop accents for comparison."""
    stripped = _STRIP_RE.sub("", word).casefold()
    decomposed = unicodedata.normalize("NFKD", stripped)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def fuzzy_match(a: str, b: str, *, threshold: float = 0.8) -> bool:
    """Whether two words are the same word, allowing for teacher variation.

    Exact after normalisation, or a shared prefix (``program``/``programs``,
    ``consent``/``consenting``), or a high enough character-level ratio. The
    prefix rule is what handles the paper's "Variation" case.
    """
    x, y = _normalise(a), _normalise(b)
    if not x or not y:
        return x == y
    if x == y:
        return True
    shorter, longer = (x, y) if len(x) <= len(y) else (y, x)
    if len(shorter) >= 3 and longer.startswith(shorter):
        return True
    if abs(len(x) - len(y)) > 3:
        return False
    return SequenceMatcher(None, x, y).ratio() >= threshold


def align(
    original: str | list[str],
    compressed: str | list[str],
    *,
    window: int = DEFAULT_WINDOW,
) -> tuple[list[str], list[bool]]:
    """Label each word of ``original`` as kept (True) or dropped (False).

    Walks the compressed words in order, searching outward from the previous
    match -- right first, then left, which is what tolerates reordering while
    still preferring the monotonic reading.

    Returns:
        The original word list and a label per word, same length.
    """
    words = split_words(original) if isinstance(original, str) else list(original)
    targets = split_words(compressed) if isinstance(compressed, str) else list(compressed)

    labels = [False] * len(words)
    if not words or not targets:
        return words, labels

    previous = 0
    for target in targets:
        for offset in range(1, max(1, window // 2) + 1):
            right = min(len(words) - 1, previous + offset)
            if fuzzy_match(target, words[right]):
                labels[right] = True
                previous = right
                break
            left = max(0, previous - offset)
            if fuzzy_match(target, words[left]):
                labels[left] = True
                break
            if right == len(words) - 1 and left == 0:
                break
    return words, labels


def variation_rate(original: str | list[str], compressed: str | list[str]) -> float:
    """Fraction of compressed words absent from the original: a hallucination signal.

    ``VR = 1/|S_comp| * sum over w in S_comp of 1(w not in S_ori)``
    """
    words = split_words(original) if isinstance(original, str) else list(original)
    targets = split_words(compressed) if isinstance(compressed, str) else list(compressed)
    if not targets:
        return 0.0
    vocabulary = {_normalise(w) for w in words}
    missing = sum(1 for t in targets if _normalise(t) not in vocabulary)
    return missing / len(targets)


def alignment_gap(
    original: str | list[str],
    compressed: str | list[str],
    labels: list[bool],
) -> float:
    """``AG = HR - MR``: how much of the compressed text the labels failed to explain.

    A perfect annotation scores 0. A large gap means words were found in the
    original but not labelled, i.e. the alignment gave up somewhere.
    """
    words = split_words(original) if isinstance(original, str) else list(original)
    targets = split_words(compressed) if isinstance(compressed, str) else list(compressed)
    if not words:
        return 0.0
    vocabulary = {_normalise(w) for w in words}
    matching_rate = sum(labels) / len(words)
    hitting_rate = sum(1 for t in targets if _normalise(t) in vocabulary) / len(words)
    return hitting_rate - matching_rate


@dataclass
class AlignmentStats:
    """One aligned example plus the numbers used to decide whether to keep it."""

    words: list[str]
    labels: list[bool]
    variation_rate: float
    alignment_gap: float

    @property
    def keep_ratio(self) -> float:
        """Fraction of original words the teacher kept."""
        return sum(self.labels) / len(self.labels) if self.labels else 0.0


def annotate(
    original: str | list[str],
    compressed: str | list[str],
    *,
    window: int = DEFAULT_WINDOW,
) -> AlignmentStats:
    """Align one pair and compute its quality metrics."""
    words, labels = align(original, compressed, window=window)
    return AlignmentStats(
        words=words,
        labels=labels,
        variation_rate=variation_rate(words, compressed),
        alignment_gap=alignment_gap(words, compressed, labels),
    )


def _drop_worst(
    examples: list[AlignmentStats], key: str, fraction: float
) -> tuple[set[int], float]:
    """Indices of the worst ``fraction`` by ``key``, and the cutoff value.

    Count-based rather than threshold-based: "discard the top 5%" must drop
    examples even when many of them tie at the same score.
    """
    count = math.ceil(len(examples) * fraction)
    if count <= 0:
        return set(), float("inf")
    ranked = sorted(range(len(examples)), key=lambda i: getattr(examples[i], key), reverse=True)
    dropped = set(ranked[:count])
    cutoff = getattr(examples[ranked[count - 1]], key)
    return dropped, cutoff


def filter_examples(
    examples: list[AlignmentStats],
    *,
    vr_quantile: float = DEFAULT_VR_QUANTILE,
    ag_quantile: float = DEFAULT_AG_QUANTILE,
) -> tuple[list[AlignmentStats], dict[str, float]]:
    """Drop the worst-hallucinated and worst-aligned examples.

    The paper discards the top 5% by variation rate and the top 10% by alignment
    gap. Returns the survivors and the cutoffs used, so a run is reproducible
    from its log.
    """
    if not examples:
        return [], {"vr_threshold": float("inf"), "ag_threshold": float("inf")}

    by_vr, vr_cut = _drop_worst(examples, "variation_rate", vr_quantile)
    by_ag, ag_cut = _drop_worst(examples, "alignment_gap", ag_quantile)
    dropped = by_vr | by_ag
    kept = [e for i, e in enumerate(examples) if i not in dropped]
    return kept, {
        "vr_threshold": vr_cut,
        "ag_threshold": ag_cut,
        "kept": len(kept),
        "dropped": len(dropped),
    }
