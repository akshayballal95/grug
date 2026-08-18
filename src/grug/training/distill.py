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


def _is_subsequence(original: list[str], compressed: list[str]) -> bool:
    """Whether the output preserves input order -- condition 2 of the prompt."""
    it = iter(original)
    return all(any(w == o for o in it) for w in compressed)


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
            orders.append(0.0 if _is_subsequence(stats.words, split_words(output)) else 1.0)
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
