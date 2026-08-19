"""Rules first, then the classifier on what survives.

The rules engine deletes only what it can prove is safe to delete, and stops at
roughly 60% of the tokens because that is all it will claim. The classifier has
no such floor but has to rank every word, filler included, against a budget.

Running them in that order plays to both. The classifier inherits a document
that has already lost its padding, so its budget is spent deciding between
words that actually carry something, rather than re-discovering that "the" is
droppable.

Measured on MeetingBank QA, 200 contexts and 600 questions judged by Sonnet 4.6:

    backend                tokens kept   exact match   F1
    original                     100%          0.622   0.749
    rules                         62%          0.612   0.761
    cascade (rate 0.45)           50%          0.622   0.748
    classifier (rate 0.33)        37%          0.577   0.703

At half the tokens the cascade answers as well as the uncompressed document.
Against the classifier alone at a matched ratio it is worth about +0.03 exact
match, and the gap widens with sample size rather than shrinking.
"""

from __future__ import annotations

from typing import Any

from ..base import CompressionResult, CompressorBackend
from ..registry import register_backend
from .classifier import ClassifierBackend
from .rules import RulesBackend

__all__ = ["CascadeBackend"]

#: What the rules stage is asked for. It largely ignores this -- it deletes what
#: it can prove is safe and no more -- but the argument is required.
_RULES_RATE = 0.33


@register_backend
class CascadeBackend(CompressorBackend):
    """Compress with the rules engine, then rank what is left with a model."""

    name = "cascade"
    description = "Rules engine first, then a trained classifier. Needs torch and --model."
    extra = "classifier"
    generative = False
    #: Inherited from the classifier stage, which has no default checkpoint.
    requires_configuration = True

    def __init__(self, **kwargs: Any) -> None:
        self._rules = RulesBackend()
        self._classifier = ClassifierBackend(**kwargs)

    @classmethod
    def is_available(cls) -> bool:
        return ClassifierBackend.is_available()

    @classmethod
    def require_available(cls) -> None:
        ClassifierBackend.require_available()

    def compress(self, text: str, rate: float = 0.5, **kwargs: Any) -> CompressionResult:
        self._validate_rate(rate)
        if not text.strip():
            return CompressionResult.build(text, text, self.name, metadata={"skipped": "blank"})

        stripped = self._rules.compress(text, rate=_RULES_RATE).text

        # `rate` is a share of the original, but the second stage sees only what
        # the first left. Asking it for `rate` directly would compress twice.
        words_in = max(1, len(text.split()))
        survived = max(1e-6, len(stripped.split()) / words_in)
        inner = min(1.0, rate / survived)

        if inner >= 1.0:
            # The rules stage already cut past the target; nothing left to do,
            # and asking the classifier to keep everything would only cost time.
            result = CompressionResult.build(text, stripped, self.name, metadata={})
            stage_two = None
        else:
            stage_two = self._classifier.compress(stripped, rate=inner)
            result = CompressionResult.build(text, stage_two.text, self.name, metadata={})

        result.metadata.update(
            {
                "rules_ratio": round(survived, 4),
                "classifier_rate": round(inner, 4),
                "classifier_ran": stage_two is not None,
                "model": getattr(self._classifier, "model_name", None),
            }
        )
        return result
