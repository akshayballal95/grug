"""Force-token machinery shared by every extractive backend.

LLMLingua-2 has a native ``force_tokens`` argument; the LongLLMLingua path has
nothing of the kind. :func:`restore_forced` supplies the guarantee generically,
after the fact: it aligns the compressed output against the original and
splices back any protected word the compressor dropped.

The alignment leans on the one property every extractive backend has -- the
output is a subsequence of the input, in order -- so a single greedy pointer
walk is enough to say which original words survived and which did not.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .chunking import PLACEHOLDER_RE
from .verify import find_entities

__all__ = ["NUMBER_RE", "collect_force_tokens", "restore_forced"]

_WORD_RE = re.compile(r"\S+")

#: Numeric literals as they are written, not normalised: "9.6", "4,800", "1.2.3",
#: "3-5", "99%". The lookbehind keeps "p99" and version tails out of it.
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,:/-]\d+)*%?(?![\w])")

#: Stripped before comparing two words, so "volume," matches "volume" and
#: "(9.6" matches "9.6". Internal punctuation is kept: "3-5" stays one token.
_EDGE_PUNCT = "\"'“”‘’.,;:!?()[]{}<>—–…"


def _norm(word: str) -> str:
    return word.strip(_EDGE_PUNCT).lower()


def restore_forced(
    original: str, compressed: str, forced: Iterable[str]
) -> tuple[str, list[str]]:
    """Splice protected words the compressor dropped back into ``compressed``.

    Args:
        original: The text that was compressed.
        compressed: What the backend returned. Assumed to be a subsequence of
            ``original`` word-wise; words it does not recognise are skipped
            rather than trusted, so detokenisation artefacts cannot derail the
            alignment.
        forced: Words that must survive. Matched case-insensitively and
            ignoring edge punctuation. An entry containing an apostrophe also
            matches as a suffix, so ``"n't"`` protects ``"doesn't"``.

    Returns:
        The repaired text and the list of words that had to be put back, in
        document order. An empty list means the backend kept everything.

    Restored words are anchored to the end of the previous surviving word, not
    the start of the next one: anchoring forward would push a word across a
    line break and move it into the following paragraph.
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
    for index, (_, _, word) in enumerate(survivors):
        target = _norm(word)
        scan = cursor
        while scan < len(originals) and _norm(originals[scan]) != target:
            scan += 1
        if scan == len(originals):
            # Not a word from the original -- a detokenisation artefact, or a
            # rewrite. Skip it and keep the original-side cursor where it is.
            continue
        drop_range(cursor, scan, index - 1)
        cursor = scan + 1
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
    text: str, base: Iterable[str], *, entities: bool, numbers: bool = False
) -> list[str]:
    """The list of tokens a compressor may not drop.

    Sources: the caller's vocabulary, protected-span placeholders in this chunk,
    and optionally its proper nouns and numeric literals. Words absent from the
    text cost nothing, so the list can be generous.

    ``numbers`` is off by default because LLMLingua-2 has ``force_reserve_digit``
    and does not need the help; the causal-LM path has no such option.
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

    extra = {m.group(0) for m in PLACEHOLDER_RE.finditer(text)}
    if entities:
        extra |= {word for entity in find_entities(text) for word in entity.split() if word}
    if numbers:
        extra |= {m.group(0) for m in NUMBER_RE.finditer(text)}

    seen = set(forced)
    return forced + sorted(extra - seen)
