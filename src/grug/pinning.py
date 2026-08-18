"""Force-token machinery shared by every extractive backend.

LLMLingua-2 has a native ``force_tokens`` argument; the LongLLMLingua path has
nothing of the kind. This module supplies both halves of the guarantee after
the fact, using the one property every extractive backend has -- the output is
a subsequence of the input, in order -- so a single greedy pointer walk is
enough to say which original words survived and which did not.

:func:`snap_to_words` drops what is not a whole word of the input, and
:func:`restore_forced` puts back the protected words that were cut.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .chunking import PLACEHOLDER_RE
from .verify import find_entities

__all__ = [
    "NUMBER_RE",
    "collect_force_tokens",
    "normalise_word",
    "restore_forced",
    "snap_to_words",
]

_WORD_RE = re.compile(r"\S+")

#: Numeric literals as they are written, not normalised: "9.6", "4,800", "1.2.3",
#: "3-5", "99%". The lookbehind keeps "p99" and version tails out of it.
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,:/-]\d+)*%?(?![\w])")

#: Stripped before comparing two words, so "volume," matches "volume" and
#: "(9.6" matches "9.6". Internal punctuation is kept: "3-5" stays one token.
#: Stripped before comparing two words. Markdown wrappers are included: a
#: negation written as **no** or `not` must still match the force list.
_EDGE_PUNCT = "\"'“”‘’.,;:!?()[]{}<>—–…*_~`#"


def normalise_word(word: str) -> str:
    """Strip edge punctuation and case, so "not," and "not" are the same word."""
    return word.strip(_EDGE_PUNCT).lower()


_norm = normalise_word


def _align(originals: list[str], survivors: list[str]) -> list[int | None]:
    """For each survivor word, the index of the original word it matched.

    Greedy, because an extractive output is a subsequence: the first unconsumed
    occurrence ahead is the right one. ``None`` means the survivor matched
    nothing ahead of the cursor, which for a subsequence can only mean it is an
    artefact of decoding -- a fragment of a word whose other tokens were cut.
    """
    matched: list[int | None] = []
    cursor = 0
    for word in survivors:
        target = _norm(word)
        scan = cursor
        while scan < len(originals) and _norm(originals[scan]) != target:
            scan += 1
        if scan == len(originals):
            matched.append(None)
            continue
        matched.append(scan)
        cursor = scan + 1
    return matched


def _collapse(gap: str) -> str:
    """The strongest separator inside ``gap``.

    Dropping a word takes its whitespace with it, so the gaps on either side are
    merged and reduced to one separator. Keeping the strongest means a removed
    fragment cannot merge two paragraphs or pull a line onto the one above.
    """
    if not gap:
        return ""
    if gap.count("\n") >= 2:
        return "\n\n"
    if "\n" in gap:
        return "\n"
    return " "


def snap_to_words(original: str, compressed: str) -> tuple[str, list[str]]:
    """Drop anything in ``compressed`` that is not a whole word of ``original``.

    LLMLingua-2 merges token probabilities up to word boundaries before deciding,
    so its output is word-clean. The causal-LM path decides per raw token, so a
    BPE model routinely keeps "acy" out of "legacy" and grinds a protected-span
    placeholder into unrecoverable rubble. Snapping restores the property the
    rest of grug relies on: the output is a subsequence of the input word for
    word, and any placeholder in it is intact or absent -- never mangled.

    Returns the snapped text and the fragments removed, in document order.
    """
    originals = [m.group(0) for m in _WORD_RE.finditer(original)]
    survivors = [(m.start(), m.end(), m.group(0)) for m in _WORD_RE.finditer(compressed)]
    matched = _align(originals, [word for _, _, word in survivors])

    out: list[str] = []
    dropped: list[str] = []
    pending = ""
    last_end = 0
    for index, (start, end, word) in enumerate(survivors):
        pending += compressed[last_end:start]
        last_end = end
        if matched[index] is None:
            dropped.append(word)
            continue
        out.append(_collapse(pending) if out else "")
        out.append(word)
        pending = ""
    return "".join(out), dropped


def restore_forced(original: str, compressed: str, forced: Iterable[str]) -> tuple[str, list[str]]:
    """Splice protected words the compressor dropped back into ``compressed``.

    Args:
        original: The text that was compressed.
        compressed: What the backend returned. Words it does not recognise are
            skipped rather than trusted, so a decoding artefact cannot derail
            the alignment; run :func:`snap_to_words` first to remove them.
        forced: Words that must survive. Matched case-insensitively and ignoring
            edge punctuation. An entry containing an apostrophe also matches as
            a suffix, so ``"n't"`` protects ``"doesn't"``.

    Returns:
        The repaired text and the list of words that had to be put back, in
        document order. An empty list means the backend kept everything.

    Restored words are anchored to the end of the previous surviving word, not
    the start of the next one: anchoring forward would push a word across a line
    break and into the following paragraph.
    """
    whole = {_norm(t) for t in forced if t}
    suffixes = {n for n in whole if "'" in n}
    if not whole:
        return compressed, []

    def pinned(word: str) -> bool:
        norm = _norm(word)
        return norm in whole or any(norm.endswith(s) for s in suffixes)

    originals = [m.group(0) for m in _WORD_RE.finditer(original)]
    survivors = [(m.start(), m.end(), m.group(0)) for m in _WORD_RE.finditer(compressed)]

    # anchor index -> words to insert after that survivor; -1 means "at the front".
    inserts: dict[int, list[str]] = {}
    restored: list[str] = []

    def drop_range(start: int, stop: int, anchor: int) -> None:
        for word in originals[start:stop]:
            if pinned(word):
                inserts.setdefault(anchor, []).append(word)
                restored.append(word)

    cursor = 0
    for index, origin in enumerate(_align(originals, [word for _, _, word in survivors])):
        if origin is None:
            continue  # an artefact: it anchors nothing and consumes nothing
        drop_range(cursor, origin, index - 1)
        cursor = origin + 1
    drop_range(cursor, len(originals), len(survivors) - 1)

    if not inserts:
        return compressed, []

    edits: list[tuple[int, str]] = []
    for anchor, words in inserts.items():
        joined = " ".join(words)
        if anchor < 0:
            edits.append((0, joined + " " if survivors else joined))
        else:
            edits.append((survivors[anchor][1], " " + joined))
    edits.sort(key=lambda edit: edit[0])

    out: list[str] = []
    at = 0
    for offset, text in edits:
        out.append(compressed[at:offset])
        out.append(text)
        at = offset
    out.append(compressed[at:])
    return "".join(out), restored


def collect_force_tokens(
    text: str,
    base: Iterable[str],
    *,
    entities: bool,
    numbers: bool = False,
    limit: int | None = None,
) -> list[str]:
    """The list of tokens a compressor may not drop.

    Sources: the caller's vocabulary, protected-span placeholders in this chunk,
    and optionally its proper nouns and numeric literals. Words absent from the
    text cost nothing, so the list can be generous.

    ``numbers`` is off by default because LLMLingua-2 has ``force_reserve_digit``
    and does not need the help; the causal-LM path has no such option.

    ``limit`` truncates the result, dropping the least important sources first.
    LLMLingua-2 asserts on a force list longer than its ``max_force_token``, and
    a long document produces more proper nouns than that on its own. Priority is
    negations, then protected spans, then numbers, then entities -- so what
    survives truncation is what inverts meaning if lost.
    """
    forced: list[str] = []
    for token in base:
        if token not in forced:
            forced.append(token)
    # LLMLingua-2 matches force_tokens case-sensitively, so a lowercase list
    # leaves every sentence-initial "No"/"Not" unprotected.
    for token in list(forced):
        if not token[:1].isalpha():
            continue  # "\n", "?" and friends have no case
        for variant in (token[:1].upper() + token[1:], token.upper()):
            if variant not in forced:
                forced.append(variant)

    seen = set(forced)
    ranked: list[str] = list(forced)

    def add(candidates: set[str]) -> None:
        for token in sorted(candidates - seen):
            seen.add(token)
            ranked.append(token)

    # Ordered by what it costs to lose them.
    add({m.group(0) for m in PLACEHOLDER_RE.finditer(text)})
    if numbers:
        add({m.group(0) for m in NUMBER_RE.finditer(text)})
    if entities:
        add({word for entity in find_entities(text) for word in entity.split() if word})

    return ranked[:limit] if limit is not None else ranked
