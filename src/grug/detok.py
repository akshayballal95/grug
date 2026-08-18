"""Undoing the spacing artefacts a tokenizer's decode step leaves behind.

Both model backends reassemble their output from token ids, and both come back
with punctuation floated off the words it belongs to -- "3-5" as "3 - 5",
"doesn't" as "doesn ' t". The repairs are shared because the artefacts are:
they come from decoding a subsequence of tokens, not from any one tokenizer.
"""

from __future__ import annotations

import re

__all__ = ["repair_detokenization"]

_FIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    # Contractions: doesn ' t -> doesn't, it ' s -> it's
    (re.compile(r"(\w)\s+'\s*(\w)"), r"\1'\2"),
    (re.compile(r"(\w)\s+n't\b"), r"\1n't"),
    # Numeric separators: "3 - 5" -> "3-5". Lookahead so chains collapse in one pass.
    (re.compile(r"(\d)\s*([-/:.,])\s*(?=\d)"), r"\1\2"),
    # Rejoin a dropped thousands separator: "5 000" -> "5,000".
    (re.compile(r"(?<![\d,.])(\d{1,3})[ \t]+(\d{3})(?![\d,.])"), r"\1,\2"),
    (re.compile(r"(\d)\s+%"), r"\1%"),
    (re.compile(r"([$€£¥])\s+(\d)"), r"\1\2"),
    # Space before closing punctuation.
    (re.compile(r"\s+([,.;:!?%)\]}])"), r"\1"),
    # Space after opening punctuation.
    (re.compile(r"([(\[{$])\s+"), r"\1"),
    # Stray space inside possessives left by the contraction rule.
    (re.compile(r"\s+'s\b"), r"'s"),
    # Whitespace hygiene, newlines preserved.
    (re.compile(r"[ \t]{2,}"), " "),
    (re.compile(r"[ \t]*\n[ \t]*"), "\n"),
    (re.compile(r"\n{3,}"), "\n\n"),
)


def repair_detokenization(text: str) -> str:
    """Undo the spacing artefacts a decode of kept-token ids leaves behind."""
    for pattern, replacement in _FIXES:
        text = pattern.sub(replacement, text)
    return text.strip()
