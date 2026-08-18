"""Question answering over compressed context: the end-to-end test.

Token accuracy says the classifier learned its labels. Ratio says the prompt
got shorter. Neither says the answer is still in there. This asks a model
questions about the compressed context and scores what comes back, which is
the only measurement that speaks to "did we lose information".

Scoring is the standard SQuAD pair -- exact match and token F1 -- so numbers
are comparable with published prompt-compression results.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DEFAULT_DATASET",
    "QAExample",
    "exact_match",
    "load_qa",
    "normalise_answer",
    "score_answers",
    "token_f1",
]

DEFAULT_DATASET = "microsoft/MeetingBank-QA-Summary"

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


@dataclass
class QAExample:
    """One context with the questions asked about it."""

    context: str
    questions: list[str]
    answers: list[str]
    idx: int = 0
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def normalise_answer(text: str) -> str:
    """SQuAD normalisation: lowercase, strip punctuation, articles, extra space."""
    lowered = text.lower()
    without_punct = lowered.translate(_PUNCT)
    without_articles = _ARTICLES.sub(" ", without_punct)
    return " ".join(without_articles.split())


def exact_match(prediction: str, gold: str) -> float:
    """1.0 when the normalised strings are identical."""
    return float(normalise_answer(prediction) == normalise_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    """Token-overlap F1 between prediction and gold, after normalisation.

    Partial credit matters here: a compressed prompt often yields an answer
    that is right but phrased differently, which exact match scores as zero.
    """
    predicted_tokens = normalise_answer(prediction).split()
    gold_tokens = normalise_answer(gold).split()
    if not predicted_tokens or not gold_tokens:
        return float(predicted_tokens == gold_tokens)

    shared = Counter(predicted_tokens) & Counter(gold_tokens)
    overlap = sum(shared.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answers(predictions: list[str], golds: list[str]) -> dict[str, float]:
    """Mean exact match and token F1 over a set of answers."""
    if not golds:
        return {"exact_match": 0.0, "f1": 0.0, "n": 0}
    pairs = list(zip(predictions, golds, strict=True))
    return {
        "exact_match": sum(exact_match(p, g) for p, g in pairs) / len(pairs),
        "f1": sum(token_f1(p, g) for p, g in pairs) / len(pairs),
        "n": len(pairs),
    }


def load_qa(
    limit: int | None = None, *, dataset: str = DEFAULT_DATASET, split: str = "test"
) -> list[QAExample]:
    """Load contexts and their question/answer pairs.

    Args:
        limit: Stop after this many contexts. Benchmarks cost API calls, so the
            default is to take a slice rather than the whole set.
        dataset: Hugging Face dataset id.
        split: Split to read.
    """
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - guarded by the extra
        raise ImportError(
            "grug.benchmark needs 'datasets'. Install with: pip install 'grug[bench]'"
        ) from exc

    rows = datasets.load_dataset(dataset, split=split)
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))

    examples: list[QAExample] = []
    for row in rows:
        pairs = row.get("QA_pairs") or []
        if isinstance(pairs, str):
            import ast

            pairs = ast.literal_eval(pairs)
        questions = [p["question"] for p in pairs if p.get("question")]
        answers = [p.get("answer", "") for p in pairs if p.get("question")]
        if not questions or not row.get("prompt"):
            continue
        examples.append(
            QAExample(
                context=row["prompt"],
                questions=questions,
                answers=answers,
                idx=row.get("idx", len(examples)),
                summary=row.get("summary", "") or "",
            )
        )
    return examples
