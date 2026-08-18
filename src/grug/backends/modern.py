"""Token-classification backend for any modern encoder.

Same method as :mod:`~grug.backends.lingua2` -- score every word, keep the
top-``rate`` fraction in original order -- but without the ``llmlingua``
dependency, and without its tokenizer coupling: ``llmlingua`` dispatches
sub-word merging on a substring of the model name and raises
``NotImplementedError`` for anything but mBERT and XLM-R.

This backend groups sub-words with the tokenizer's own ``word_ids()``, so it
works with any fast tokenizer -- ModernBERT, mmBERT, EuroBERT, LFM2.5 -- and
therefore with the checkpoints :mod:`grug.training` produces.
"""

from __future__ import annotations

import importlib.util
import re
from typing import Any

from ..base import CompressionResult, CompressorBackend, MissingDependencyError
from ..pinning import NUMBER_RE, collect_force_tokens, normalise_word
from ..registry import register_backend
from ..verify import NEGATION_FORCE_TOKENS

__all__ = ["DEFAULT_MODEL", "ModernBackend"]

#: Trained by ``grug train``; see the reproduction guide in the README.
DEFAULT_MODEL = "answerdotai/ModernBERT-base"

DEFAULT_FORCE_TOKENS: tuple[str, ...] = ("\n", "?", *NEGATION_FORCE_TOKENS)

_REQUIRED_MODULES = ("torch", "transformers")

#: Words and newline runs. Newlines are scored as words so line structure can be
#: forced through instead of being flattened by the join.
_TOKEN_RE = re.compile(r"\n+|\S+")


def split_words(text: str) -> list[str]:
    """Split into the unit the classifier scores: words plus newline runs."""
    return _TOKEN_RE.findall(text)


def join_words(words: list[str]) -> str:
    """Reassemble scored words, keeping newline runs tight against their neighbours."""
    out: list[str] = []
    for word in words:
        if word.startswith("\n") or not out or out[-1].endswith("\n"):
            out.append(word)
        else:
            out.append(" " + word)
    return "".join(out)


@register_backend
class ModernBackend(CompressorBackend):
    """Token classification with a modern 8k-context encoder."""

    name = "modern"
    description = "Token classification on a modern encoder (8k context). Needs torch."
    extra = "modern"
    generative = False

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        force_tokens: list[str] | tuple[str, ...] | None = None,
        force_reserve_digit: bool = True,
        preserve_entities: bool = True,
        max_length: int | None = None,
        **model_kwargs: Any,
    ) -> None:
        """
        Args:
            model_name: A token-classification checkpoint with a preserve/discard head.
            device: ``cpu``, ``cuda``, ``mps``, or ``auto``.
            force_tokens: Words the compressor may never drop.
            force_reserve_digit: Never drop a word containing a digit.
            preserve_entities: Pin proper nouns found in the input.
            max_length: Override the model's own context window.
            **model_kwargs: Forwarded to ``from_pretrained``.
        """
        self.model_name = model_name
        self.device = device
        self.force_tokens = list(force_tokens if force_tokens is not None else DEFAULT_FORCE_TOKENS)
        self.force_reserve_digit = force_reserve_digit
        self.preserve_entities = preserve_entities
        self.max_length = max_length
        self.model_kwargs = model_kwargs
        self._model: Any = None
        self._tokenizer: Any = None
        self._resolved_device: str | None = None
        self._preserve_id = 1
        self.require_available()

    # -- dependencies ---------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        return all(importlib.util.find_spec(m) is not None for m in _REQUIRED_MODULES)

    @classmethod
    def require_available(cls) -> None:
        missing = [m for m in _REQUIRED_MODULES if importlib.util.find_spec(m) is None]
        if missing:
            raise MissingDependencyError(cls.name, cls.extra or cls.name, ", ".join(missing))

    # -- model ----------------------------------------------------------

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        return "mps" if mps is not None and mps.is_available() else "cpu"

    def _load(self) -> tuple[Any, Any]:
        """Load tokenizer and model once, and cache them on the instance."""
        if self._model is not None:
            return self._tokenizer, self._model

        self.require_available()
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self._resolved_device = self._resolve_device()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.model_kwargs)
        if not self._tokenizer.is_fast:
            raise RuntimeError(
                f"{self.model_name!r} has no fast tokenizer; word alignment needs word_ids()"
            )
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, **self.model_kwargs
        )
        model.eval()
        model.to(self._resolved_device)
        self._model = model
        self._preserve_id = _preserve_label_id(model.config)
        return self._tokenizer, self._model

    @property
    def model(self) -> Any:
        """The underlying model, loading it on first access."""
        return self._load()[1]

    @property
    def metadata_device(self) -> str | None:
        """The device in use, or ``None`` before the model is loaded."""
        return self._resolved_device

    # -- compression -----------------------------------------------------

    def compress(self, text: str, rate: float = 0.5, **kwargs: Any) -> CompressionResult:
        self._validate_rate(rate)
        if not text.strip():
            return CompressionResult.build(text, text, self.name, metadata={"skipped": "blank"})

        preserve_entities = kwargs.pop("preserve_entities", self.preserve_entities)
        reserve_digit = kwargs.pop("force_reserve_digit", self.force_reserve_digit)
        base = kwargs.pop("force_tokens", self.force_tokens)
        if kwargs:
            raise TypeError(
                f"{self.name} backend got unexpected keyword argument(s): "
                + ", ".join(sorted(map(repr, kwargs)))
            )

        words = split_words(text)
        probs = self._score(words)

        # Compare normalised forms: a raw token carries its punctuation, so
        # "not," would otherwise miss a force list containing "not" -- and the
        # words that escaped were exactly the negations this is meant to pin.
        forced = {
            normalise_word(w) for w in collect_force_tokens(text, base, entities=preserve_entities)
        }
        for i, word in enumerate(words):
            pinned = word.startswith("\n") or normalise_word(word) in forced
            if pinned or (reserve_digit and NUMBER_RE.search(word)):
                probs[i] = 1.0

        keep = round(rate * len(words))
        keep = max(1, min(len(words), keep))
        chosen = sorted(sorted(range(len(words)), key=lambda i: -probs[i])[:keep])
        compressed = join_words([words[i] for i in chosen])

        return CompressionResult.build(
            text,
            compressed,
            self.name,
            metadata={
                "model": self.model_name,
                "device": self._resolved_device,
                "requested_rate": rate,
                "words_in": len(words),
                "words_out": len(chosen),
            },
        )

    def _score(self, words: list[str]) -> list[float]:
        """Probability that each word should be preserved.

        Sub-word probabilities are averaged per word, which is what lets a
        multi-token word be kept or dropped atomically. Word lists longer than
        the context window are scored in successive windows.
        """
        import torch

        tokenizer, model = self._load()
        limit = self.max_length or getattr(model.config, "max_position_embeddings", 512)
        budget = max(8, limit - 2)  # leave room for the special tokens

        counts = _subword_counts(tokenizer, words)
        probs: list[float] = []
        for window in _windows(counts, budget):
            chunk = words[window[0] : window[1]]
            encoded = tokenizer(
                chunk,
                is_split_into_words=True,
                truncation=True,
                max_length=limit,
                return_tensors="pt",
            )
            word_ids = encoded.word_ids()
            encoded = {k: v.to(self._resolved_device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits[0]
            keep_prob = logits.softmax(-1)[:, self._preserve_id].tolist()

            totals = [0.0] * len(chunk)
            seen = [0] * len(chunk)
            for position, word_index in enumerate(word_ids):
                if word_index is None:
                    continue
                totals[word_index] += keep_prob[position]
                seen[word_index] += 1
            probs.extend(totals[i] / seen[i] if seen[i] else 0.0 for i in range(len(chunk)))
        return probs


def _preserve_label_id(config: Any) -> int:
    """Which logit index means "keep"; falls back to 1, the usual convention."""
    id2label = getattr(config, "id2label", None) or {}
    for index, label in id2label.items():
        if str(label).lower() in {"preserve", "keep", "label_1", "1"}:
            return int(index)
    return 1 if getattr(config, "num_labels", 2) > 1 else 0


def _subword_counts(tokenizer: Any, words: list[str]) -> list[int]:
    """How many sub-word tokens each word costs, so windows can be sized."""
    encoded = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
    counts = [0] * len(words)
    for word_index in encoded.word_ids():
        if word_index is not None:
            counts[word_index] += 1
    return [max(1, c) for c in counts]


def _windows(counts: list[int], budget: int) -> list[tuple[int, int]]:
    """Greedy [start, end) spans of words whose sub-words fit the budget."""
    spans: list[tuple[int, int]] = []
    start = running = 0
    for index, cost in enumerate(counts):
        if index > start and running + cost > budget:
            spans.append((start, index))
            start, running = index, 0
        running += cost
    if start < len(counts):
        spans.append((start, len(counts)))
    return spans or [(0, len(counts))]
