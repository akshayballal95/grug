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
import os
import re
from typing import Any

from ..base import CompressionResult, CompressorBackend, MissingDependencyError
from ..pinning import NUMBER_RE, collect_force_tokens, normalise_word
from ..registry import register_backend
from ..verify import NEGATION_FORCE_TOKENS, is_negation

__all__ = ["DEFAULT_MODEL", "ModernBackend"]

#: Trained by ``grug train``; see the reproduction guide in the README.
#: No default on purpose. A base encoder like ``answerdotai/ModernBERT-base``
#: loads happily with a randomly initialised classification head and then scores
#: tokens at random, which looks like a working compressor and is not one.
#: Train one with ``grug train`` or pass a checkpoint that has a trained head.
DEFAULT_MODEL: str | None = None

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
    description = "Token classification, any modern encoder. Needs torch and --model."
    extra = "modern"
    generative = False
    # No DEFAULT_MODEL: a base encoder's untrained head scores at random, so
    # there is nothing honest to fall back to. Callers must name a checkpoint.
    requires_configuration = True

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "auto",
        force_tokens: list[str] | tuple[str, ...] | None = None,
        force_reserve_digit: bool = True,
        preserve_entities: bool = True,
        max_length: int | None = None,
        token: str | None = None,
        **model_kwargs: Any,
    ) -> None:
        """
        Args:
            model_name: A token-classification checkpoint with a *trained*
                preserve/discard head. A bare encoder is rejected.
            device: ``cpu``, ``cuda``, ``mps``, or ``auto``.
            force_tokens: Words the compressor may never drop.
            force_reserve_digit: Never drop a word containing a digit.
            preserve_entities: Pin proper nouns found in the input.
            max_length: Override the model's own context window.
            token: Hugging Face token for a private checkpoint. Defaults to
                ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``) from the environment.
            **model_kwargs: Forwarded to ``from_pretrained``.
        """
        self.model_name = model_name
        self.device = device
        self.force_tokens = list(force_tokens if force_tokens is not None else DEFAULT_FORCE_TOKENS)
        self.force_reserve_digit = force_reserve_digit
        self.preserve_entities = preserve_entities
        self.max_length = max_length
        # Read the token here rather than relying on the hub library to find it:
        # a private checkpoint otherwise fails with "not a valid model
        # identifier", which reads like a typo rather than a missing token.
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
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
        auth = {"token": self.token} if self.token else {}
        self._check_token()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, **auth, **self.model_kwargs
        )
        if not self._tokenizer.is_fast:
            raise RuntimeError(
                f"{self.model_name!r} has no fast tokenizer; word alignment needs word_ids()"
            )
        model, loading_info = AutoModelForTokenClassification.from_pretrained(
            self.model_name, output_loading_info=True, **auth, **self.model_kwargs
        )
        untrained = [k for k in loading_info.get("missing_keys", []) if "classifier" in k]
        if untrained:
            raise ValueError(
                f"{self.model_name!r} has no trained classification head "
                f"(missing {', '.join(sorted(untrained))}). It is a base encoder, so its "
                "scores would be random. Train one with 'grug train run --model "
                f"{self.model_name}' and pass the resulting checkpoint."
            )
        model.eval()
        model.to(self._resolved_device)
        self._model = model
        self._preserve_id = _preserve_label_id(model.config)
        return self._tokenizer, self._model

    def _check_token(self) -> None:
        """Say plainly that a token was rejected.

        The hub reports a stale token as "Repository Not Found" plus "Invalid
        username or password", which reads like a wrong repo id rather than a
        credential that needs replacing.
        """
        if not self.token:
            return
        try:
            from huggingface_hub import HfApi

            HfApi(token=self.token).whoami()
        except Exception as exc:
            source = (
                "HF_TOKEN" if os.environ.get("HF_TOKEN") == self.token else "the token argument"
            )
            raise ValueError(
                f"the Hugging Face token from {source} was rejected ({type(exc).__name__}). "
                "It is probably expired or revoked; replace it and retry. A private "
                "checkpoint will otherwise look like a missing repository."
            ) from exc

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
        pinned: list[int] = []
        for i, word in enumerate(words):
            core = normalise_word(word)
            # is_negation catches -n't contractions, whose cue is a suffix and
            # so never matches a force list of whole words.
            if (
                word.startswith("\n")
                or core in forced
                or is_negation(core)
                or (reserve_digit and NUMBER_RE.search(word))
            ):
                probs[i] = 1.0
                pinned.append(i)

        # Pinned words are kept outright rather than competing for the budget.
        # Ranking everything by score and taking the top k lets a short,
        # pin-dense chunk cut a negation, which makes the guarantee no
        # guarantee at all.
        keep = max(1, min(len(words), round(rate * len(words))))
        pinned_set = set(pinned)
        spare = sorted(
            (i for i in range(len(words)) if i not in pinned_set), key=lambda i: -probs[i]
        )
        chosen = sorted(pinned + spare[: max(0, keep - len(pinned))])
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
