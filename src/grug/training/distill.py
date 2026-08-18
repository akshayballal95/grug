"""Re-distil the corpus with a modern teacher, and judge which teacher is best.

LLMLingua-2's labels came from GPT-4-32k in early 2024, and the paper documents
it *not following the instruction* -- which is why compression had to be done
chunk-wise and 15% of the data was discarded. Current models follow constraints
better, so the corpus is worth rebuilding.

"Better teacher" is not a single number. A teacher that compresses hard but
drops negations produces labels that teach the student to drop negations, which
is the one failure grug exists to prevent. The rubric here scores adherence,
fidelity and consistency separately.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from ..verify import find_numbers, is_negation
from .alignment import annotate, split_words

__all__ = [
    "INSTRUCTIONS",
    "TEACHER_INSTRUCTION",
    "TeacherScore",
    "compress_with_teacher",
    "score_teacher",
]

#: The paper's five conditions, verbatim. Kept because the constraints are what
#: make the output alignable: reordering or rewording breaks label derivation.
TEACHER_INSTRUCTION = """Compress the given text to short expressions, and such \
that you can reconstruct it as close as possible to the original. Unlike the usual \
text compression, I need you to comply with the 5 conditions below:
1. You can ONLY remove unimportant words.
2. Do not reorder the original words.
3. Do not change the original words.
4. Do not use abbreviations or emojis.
5. Do not add new words or symbols.
Compress the origin aggressively by removing words only. Compress the origin as \
short as you can, while retaining as much information as possible. If you \
understand, please compress the following text:

{text}

The compressed text is:"""

#: Adds a negation rule. The teacher's choices become the student's labels, so a
#: teacher that drops "not" trains a model to drop it -- and losing a negation
#: inverts a claim rather than merely blurring it.
_NEGATION_RULE = """6. NEVER remove a negation. Keep every one of: not, no, never, \
none, neither, nor, cannot, without, unless, except, and every -n't contraction \
(don't, doesn't, won't, isn't). Removing a negation reverses the meaning of the \
sentence, which is far worse than keeping a few extra words."""

#: Also pins the other two things the verifier polices.
_CRITICAL_RULE = """7. NEVER remove a number, date, quantity, or proper noun \
(names of people, organisations, and places). These carry facts that cannot be \
recovered from context."""


def _with_rules(*rules: str) -> str:
    """Insert extra numbered conditions into the base instruction."""
    marker = "5. Do not add new words or symbols."
    return TEACHER_INSTRUCTION.replace(marker, marker + "\n" + "\n".join(rules), 1).replace(
        "the 5 conditions below", f"the {5 + len(rules)} conditions below", 1
    )


#: Instruction variants, so a prompt change can be measured rather than assumed.
INSTRUCTIONS: dict[str, str] = {
    "baseline": TEACHER_INSTRUCTION,
    "negation": _with_rules(_NEGATION_RULE),
    "critical": _with_rules(_NEGATION_RULE, _CRITICAL_RULE),
}


@dataclass
class TeacherScore:
    """How good a teacher's output is as training data."""

    model: str
    samples: int
    ratio: float
    variation_rate: float
    alignment_gap: float
    negation_retention: float
    number_retention: float
    order_violation_rate: float
    self_consistency: float
    failures: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.model:<34} ratio={self.ratio:.2f} VR={self.variation_rate:.3f} "
            f"AG={self.alignment_gap:+.3f} neg={self.negation_retention:.2f} "
            f"num={self.number_retention:.2f} order={self.order_violation_rate:.2f} "
            f"consist={self.self_consistency:.2f} fail={self.failures}"
        )


def compress_with_teacher(text: str, client: Any) -> str:
    """Ask a teacher to compress one passage under the five conditions."""
    return client.one(TEACHER_INSTRUCTION.format(text=text), "")


def _out_of_order_fraction(original: list[str], compressed: list[str]) -> float:
    """Fraction of output words that could not be matched in input order.

    A fraction rather than a flag: one displaced word in four hundred is a
    different thing from a wholesale rewrite, and a boolean scored both as a
    total violation. Compared on normalised words so recapitalising a sentence
    start is not counted as reordering.
    """
    from ..pinning import normalise_word

    wanted = [w for w in (normalise_word(x) for x in compressed) if w]
    if not wanted:
        return 0.0
    haystack = [normalise_word(w) for w in original]
    cursor = matched = 0
    for word in wanted:
        # Scan forward from the last match. An unmatched word must not consume
        # the rest of the input, or a single miss fails everything after it.
        probe = cursor
        while probe < len(haystack) and haystack[probe] != word:
            probe += 1
        if probe < len(haystack):
            matched += 1
            cursor = probe + 1
    return 1.0 - matched / len(wanted)


def _retention(original: str, compressed: str, kind: str) -> float | None:
    """Fraction of the original's negations (or numbers) still present."""
    if kind == "negation":
        wanted = [w for w in split_words(original) if is_negation(w.strip(".,;:!?").lower())]
        kept = {w.strip(".,;:!?").lower() for w in split_words(compressed)}
        if not wanted:
            return None
        return sum(1 for w in wanted if w.strip(".,;:!?").lower() in kept) / len(wanted)

    wanted_numbers = find_numbers(original)
    if not wanted_numbers:
        return None
    present = find_numbers(compressed)
    return sum(1 for n in wanted_numbers if present.get(n, 0) > 0) / len(wanted_numbers)


def score_teacher(model: str, originals: list[str], samples: list[list[str]]) -> TeacherScore:
    """Judge a teacher from k compressions of each passage.

    Args:
        model: Display name.
        originals: The source passages.
        samples: ``samples[i]`` is the list of k compressions of ``originals[i]``.
    """
    ratios, variations, gaps, orders = [], [], [], []
    negations, numbers, consistencies = [], [], []
    failures = 0

    for original, outputs in zip(originals, samples, strict=True):
        usable = [o for o in outputs if o and o.strip()]
        failures += len(outputs) - len(usable)
        if not usable:
            continue

        label_sets = []
        for output in usable:
            stats = annotate(original, output)
            ratios.append(stats.keep_ratio)
            variations.append(stats.variation_rate)
            gaps.append(stats.alignment_gap)
            orders.append(_out_of_order_fraction(stats.words, split_words(output)))
            label_sets.append(stats.labels)

            for store, kind in ((negations, "negation"), (numbers, "number")):
                value = _retention(original, output, kind)
                if value is not None:
                    store.append(value)

        # Self-consistency: how often k runs agree on a word, averaged.
        if len(label_sets) > 1:
            per_word = [
                max(col.count(True), col.count(False)) / len(col)
                for col in zip(*label_sets, strict=True)
            ]
            consistencies.append(statistics.fmean(per_word) if per_word else 0.0)

    mean = lambda xs: statistics.fmean(xs) if xs else 0.0  # noqa: E731
    return TeacherScore(
        model=model,
        samples=len(originals),
        ratio=mean(ratios),
        variation_rate=mean(variations),
        alignment_gap=mean(gaps),
        negation_retention=mean(negations),
        number_retention=mean(numbers),
        order_violation_rate=mean(orders),
        self_consistency=mean(consistencies) if consistencies else 1.0,
        failures=failures,
    )
