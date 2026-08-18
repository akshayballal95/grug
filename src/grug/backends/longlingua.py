"""LongLLMLingua backend: question-aware compression on the causal-LM path.

A causal LM scores every token twice -- once with the question as a prefix and
once without -- and keeps the tokens whose surprisal the question lowers most.
That is the load-bearing signal: a token survives not because it is rare, but
because knowing the question makes it predictable. Without a question the
backend degrades to plain perplexity filtering and still works.

``force_tokens`` does nothing here. The keyword exists on ``compress_prompt``
but only ``compress_prompt_llmlingua2`` reads it, so passing it would read as a
guarantee this path does not honour. Negations, entities, placeholders and
numbers are pinned afterwards instead, by :func:`grug.pinning.restore_forced`.

This path also filters raw tokens rather than whole words, so a BPE model will
happily keep "acy" out of "legacy". :func:`grug.pinning.snap_to_words` drops
those fragments first, which is what keeps the output a subsequence of the
input and keeps protected-span placeholders intact rather than mangled.

torch, transformers and llmlingua are imported inside :meth:`_load`, never at
module import time.
"""

from __future__ import annotations

from typing import Any

from ..base import CompressionResult, CompressorBackend, MissingDependencyError
from ..detok import repair_detokenization
from ..pinning import collect_force_tokens, restore_forced, snap_to_words
from ..registry import register_backend
from ..verify import NEGATION_FORCE_TOKENS
from ._llmlingua import filter_supported, missing_modules, resolve_device

__all__ = ["DEFAULT_FORCE_TOKENS", "DEFAULT_MODEL", "LongLinguaBackend"]

#: Small enough to run on a laptop, 2048-token window, no gated download.
DEFAULT_MODEL = "microsoft/phi-2"

#: Only the negation vocabulary. Unlike LLMLingua-2's list this one is matched
#: against whole words after the fact, so "\n" and "?" have no meaning in it.
DEFAULT_FORCE_TOKENS: tuple[str, ...] = NEGATION_FORCE_TOKENS


@register_backend
class LongLinguaBackend(CompressorBackend):
    """Question-aware compression via ``llmlingua.PromptCompressor``."""

    name = "longlingua"
    description = "LongLLMLingua question-aware perplexity (causal LM). Needs torch."
    extra = "longlingua"
    generative = False
    question_aware = True

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        force_tokens: list[str] | tuple[str, ...] | None = None,
        preserve_entities: bool = True,
        preserve_numbers: bool = True,
        repair_detokenization: bool = True,
        **model_kwargs: Any,
    ) -> None:
        """
        Args:
            model_name: Any causal LM transformers can load. Must have a
                context window comfortably above the chunk size plus question.
            device: ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"auto"``.
            force_tokens: Words that may not be dropped, restored after the
                fact. Defaults to :data:`DEFAULT_FORCE_TOKENS`.
            preserve_entities: Pin proper nouns as well.
            preserve_numbers: Pin numeric literals as well. On by default
                because this path has no ``force_reserve_digit``.
            repair_detokenization: Fix decode spacing artefacts in output.
            **model_kwargs: Forwarded to ``PromptCompressor``.
        """
        self.model_name = model_name
        self.device = device
        self.force_tokens = list(force_tokens if force_tokens is not None else DEFAULT_FORCE_TOKENS)
        self.preserve_entities = preserve_entities
        self.preserve_numbers = preserve_numbers
        self.repair_detokenization = repair_detokenization
        self.model_kwargs = model_kwargs
        self._model: Any | None = None
        self._resolved_device: str | None = None
        # Fail here, with the install hint, rather than deep inside a batch.
        self.require_available()

    # -- dependency handling -------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        return not missing_modules()

    @classmethod
    def require_available(cls) -> None:
        """Name the exact missing modules rather than "optional dependencies"."""
        missing = missing_modules()
        if missing:
            raise MissingDependencyError(cls.name, cls.extra or cls.name, ", ".join(missing))

    # -- model lifecycle -----------------------------------------------

    def _load(self) -> Any:
        """Load the model once and cache it on the instance."""
        if self._model is not None:
            return self._model

        from llmlingua import PromptCompressor

        self._resolved_device = resolve_device(self.device)
        self._model = PromptCompressor(
            model_name=self.model_name,
            use_llmlingua2=False,
            device_map=self._resolved_device,
            **self.model_kwargs,
        )
        return self._model

    @property
    def model(self) -> Any:
        """The underlying ``PromptCompressor``, loading it on first access."""
        return self._load()

    @property
    def metadata_device(self) -> str | None:
        """The device actually in use, or ``None`` before the model is loaded."""
        return self._resolved_device

    # -- compression ----------------------------------------------------

    def compress(self, text: str, rate: float = 0.5, **kwargs: Any) -> CompressionResult:
        self._validate_rate(rate)
        if not text.strip():
            return CompressionResult.build(text, text, self.name, metadata={"skipped": "blank"})

        question = str(kwargs.pop("question", "") or "").strip()
        repair = kwargs.pop("repair_detokenization", self.repair_detokenization)
        forced = collect_force_tokens(
            text,
            kwargs.pop("force_tokens", self.force_tokens),
            entities=kwargs.pop("preserve_entities", self.preserve_entities),
            numbers=kwargs.pop("preserve_numbers", self.preserve_numbers),
        )

        model = self._load()
        call_kwargs = self._build_call_kwargs(model, rate, text, question, kwargs)
        raw = self._call(model, text, call_kwargs)
        compressed = raw["compressed_prompt"] if isinstance(raw, dict) else str(raw)

        # Order matters. Repair first: the alignment compares words, and "3 - 5"
        # does not match the "3-5" it came from. Then snap, so the pinning step
        # sees no fragments. Then restore what the snap and the model dropped.
        if repair:
            compressed = repair_detokenization(compressed)
        compressed, fragments = snap_to_words(text, compressed)
        compressed, pinned_back = restore_forced(text, compressed, forced)

        metadata: dict[str, Any] = {
            "model": self.model_name,
            "device": self._resolved_device,
            "requested_rate": rate,
            "conditioned": bool(question),
            "pinned_back": pinned_back,
            "fragments_dropped": fragments,
        }
        if isinstance(raw, dict):
            for key in ("origin_tokens", "compressed_tokens", "ratio", "rate"):
                if key in raw:
                    metadata[f"llmlingua_{key}"] = raw[key]

        return CompressionResult.build(text, compressed, self.name, metadata=metadata)

    @staticmethod
    def _call(model: Any, text: str, call_kwargs: dict[str, Any]) -> Any:
        """Run the compressor, translating one upstream crash into a real diagnosis.

        llmlingua 0.2.x slices the KV cache as ``for k, v in past_key_values``,
        the tuple layout transformers dropped in 5.0. The bare unpacking error
        that produces says nothing about which two packages disagree.
        """
        try:
            return model.compress_prompt(text, **call_kwargs)
        except ValueError as exc:
            if "too many values to unpack" not in str(exc):
                raise
            raise RuntimeError(
                "llmlingua's multi-window path reads past_key_values as (key, value) "
                "tuples, a layout transformers removed in 5.0. grug normally avoids "
                "it by compressing each chunk in a single window, so reaching this "
                "means iterative_size was overridden below the chunk length. Drop "
                "the override, install transformers<5, or use the lingua2 backend, "
                "whose encoder keeps no cache and is unaffected."
            ) from exc

    def _build_call_kwargs(
        self, model: Any, rate: float, text: str, question: str, overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """Assemble ``compress_prompt`` kwargs, dropping any this version lacks."""
        call_kwargs: dict[str, Any] = {
            "rate": rate,
            # The library appends the question to its output by default, which
            # would stamp a copy of it onto every chunk of the document.
            "concate_question": False,
            # One context per call, so there is nothing to rank or reallocate
            # between -- and a context-level filter could drop the only chunk.
            "use_context_level_filter": False,
        }
        window = self._window(model, text)
        if window is not None:
            call_kwargs["iterative_size"] = window
        if question:
            call_kwargs.update(
                question=question,
                rank_method="longllmlingua",
                # Prefix the question so token surprisal is measured against it,
                # then subtract the unconditioned pass. The difference is the
                # question-relevance score.
                condition_in_question="after_condition",
                condition_compare=True,
            )
        else:
            # compress_prompt asserts that longllmlingua ranking needs a question.
            call_kwargs["rank_method"] = "llmlingua"
        call_kwargs.update(overrides)
        return filter_supported(model.compress_prompt, call_kwargs)

    @staticmethod
    def _window(model: Any, text: str) -> int | None:
        """One iteration window spanning the whole chunk, or ``None`` if unmeasurable.

        llmlingua compresses only what lies before the trailing ``iterative_size``
        tokens, so with the library default of 200 a chunk shorter than that comes
        back byte-for-byte and a 450-token chunk keeps its last 200 tokens intact.
        Sizing the window to the chunk makes the whole chunk eligible -- and, as a
        side effect, keeps the run to a single pass, which is the only path that
        does not touch the legacy KV-cache layout. Measured without special
        tokens: a window one token wider than the context compresses nothing.
        """
        measure = getattr(model, "get_token_length", None)
        if measure is None:  # pragma: no cover - every released version has it
            return None
        try:
            return max(1, int(measure(text, add_special_tokens=False)))
        except (TypeError, ValueError):  # pragma: no cover - exotic tokenizer
            return None
