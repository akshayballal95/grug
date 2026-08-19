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

import difflib
import math
import re
import unicodedata
from collections import Counter
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


def _matches(x: str, y: str, threshold: float) -> bool:
    """Compare two already-normalised words. The hot path of :func:`align`."""
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


def fuzzy_match(a: str, b: str, *, threshold: float = 0.8) -> bool:
    """Whether two words are the same word, allowing for teacher variation.

    Exact after normalisation, or a shared prefix (``program``/``programs``,
    ``consent``/``consenting``), or a high enough character-level ratio. The
    prefix rule is what handles the paper's "Variation" case.
    """
    return _matches(_normalise(a), _normalise(b), threshold)


def align(
    original: str | list[str],
    compressed: str | list[str],
    *,
    window: int = DEFAULT_WINDOW,
) -> tuple[list[str], list[bool]]:
    """Label each word of ``original`` as kept (True) or dropped (False).

    Finds the longest matching blocks between the two word sequences, then
    fills the gaps between those anchors with near-matches, so a teacher that
    wrote "monitored" for "monitoring" still labels the source word.

    A greedy outward scan was tried first and is a trap here. Taking the first
    fuzzy match in either direction lets a common word match spuriously far
    ahead: on a 48k-word transcript the cursor advanced a median of 37 words
    per match where the true stride was 3.6, ran ahead of the real position,
    and then 75% of the compressed words fell outside its window and matched
    nothing. It labelled 3% of a document the teacher had kept 30% of -- and
    reported no error, because stray backward matches look like progress.
    Anchoring on matching blocks first cannot drift, and is ~24x faster.

    Args:
        window: How far to hunt for a near-match inside a gap between anchors.
            Bounds the cost; gaps wider than this are left to exact matching.

    Returns:
        The original word list and a label per word, same length.
    """
    words = split_words(original) if isinstance(original, str) else list(original)
    targets = split_words(compressed) if isinstance(compressed, str) else list(compressed)

    labels = [False] * len(words)
    if not words or not targets:
        return words, labels

    # Normalise once. The comparison runs over these thousands of times, and
    # normalising inside it dominated the whole stage.
    normalised = [_normalise(w) for w in words]
    needles = [_normalise(t) for t in targets]

    blocks = difflib.SequenceMatcher(a=normalised, b=needles, autojunk=False).get_matching_blocks()
    for block in blocks:
        for offset in range(block.size):
            labels[block.a + offset] = True

    # Between two anchors sit the words the teacher altered rather than copied.
    # Search for them only inside the gap, so a bad guess cannot move the
    # alignment: the next anchor puts it back regardless.
    source_end = target_end = 0
    for block in blocks:
        gap = block.a - source_end
        if 0 < gap <= window:
            cursor = source_end
            for index in range(target_end, block.b):
                needle = needles[index]
                for position in range(cursor, block.a):
                    if not labels[position] and _matches(needle, normalised[position], 0.8):
                        labels[position] = True
                        cursor = position + 1
                        break
        source_end, target_end = block.a + block.size, block.b + block.size

    # A word the teacher moved is missing from every block, because blocks are
    # monotonic by construction. Give each still-unaccounted word the earliest
    # source position holding it. Counting rather than searching keeps this
    # order-free, so reordering is tolerated without any cursor to lead astray.
    wanted = Counter(needles)
    found = Counter(normalised[i] for i, on in enumerate(labels) if on)
    positions: dict[str, list[int]] = {}
    for position, word in enumerate(normalised):
        positions.setdefault(word, []).append(position)
    for word, count in wanted.items():
        missing = count - found.get(word, 0)
        for position in positions.get(word, ()):
            if missing <= 0:
                break
            if not labels[position]:
                labels[position] = True
                missing -= 1

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
