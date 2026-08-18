"""LLMLingua-2 backend: extractive compression by token classification.

A BERT encoder scores every token "keep or drop" and the top ``rate`` fraction
survives in order, so the output is always a subsequence of the input. A scorer
trained on transcripts does not know that losing "not" inverts a sentence, so
negations ship in ``force_tokens`` by default.

torch, transformers and llmlingua are imported inside :meth:`_load`, never at
module import time. Force-token assembly and detokenisation repair live in
:mod:`grug.pinning` and :mod:`grug.detok`, shared with the longlingua backend.
"""

from __future__ import annotations

from typing import Any

from ..base import CompressionResult, CompressorBackend, MissingDependencyError
from ..detok import repair_detokenization as _repair_detokenization
from ..pinning import collect_force_tokens as _force_tokens
from ..registry import register_backend
from ..verify import NEGATION_FORCE_TOKENS
from ._llmlingua import filter_supported as _filter_supported
from ._llmlingua import missing_modules, resolve_device

__all__ = ["DEFAULT_MODEL", "Lingua2Backend"]

DEFAULT_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

#: Structural tokens plus the negation vocabulary. Overridable, but the
#: negations are the reason this backend is safe to use at aggressive rates.
DEFAULT_FORCE_TOKENS: tuple[str, ...] = ("\n", "?", *NEGATION_FORCE_TOKENS)


@register_backend
class Lingua2Backend(CompressorBackend):
    """Token-classification compression via ``llmlingua.PromptCompressor``."""

    name = "lingua2"
    description = "LLMLingua-2 token classification (BERT). Highest quality, needs torch."
    extra = "lingua2"
    generative = False

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        force_tokens: list[str] | tuple[str, ...] | None = None,
        force_reserve_digit: bool = True,
        drop_consecutive: bool = True,
        preserve_entities: bool = True,
        repair_detokenization: bool = True,
        **model_kwargs: Any,
    ) -> None:
        """
        Args:
            model_name: Any LLMLingua-2 token-classification checkpoint.
            device: ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"auto"`` to pick the
                best available at load time.
            force_tokens: Tokens the compressor may never drop. Defaults to
                :data:`DEFAULT_FORCE_TOKENS` (newline, ``?``, and negations).
                Capitalised and upper-case variants are added automatically.
            force_reserve_digit: Keep digits regardless of their score.
            preserve_entities: Pin proper nouns so the compressor cannot drop
                them. Costs 1-3% of ratio; set ``False`` for maximum compression.
            drop_consecutive: Collapse repeated forced tokens.
            repair_detokenization: Fix word-piece spacing artefacts in output.
            **model_kwargs: Forwarded to ``PromptCompressor``.
        """
        self.model_name = model_name
        self.device = device
        self.force_tokens = list(force_tokens if force_tokens is not None else DEFAULT_FORCE_TOKENS)
        self.force_reserve_digit = force_reserve_digit
        self.drop_consecutive = drop_consecutive
        self.preserve_entities = preserve_entities
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
            use_llmlingua2=True,
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

        repair = kwargs.pop("repair_detokenization", self.repair_detokenization)
        model = self._load()
        call_kwargs = self._build_call_kwargs(model, rate, text, kwargs)

        raw = model.compress_prompt(text, **call_kwargs)
        compressed = raw["compressed_prompt"] if isinstance(raw, dict) else str(raw)

        if repair:
            compressed = _repair_detokenization(compressed)

        metadata: dict[str, Any] = {
            "model": self.model_name,
            "device": self._resolved_device,
            "requested_rate": rate,
        }
        if isinstance(raw, dict):
            for key in ("origin_tokens", "compressed_tokens", "ratio", "rate"):
                if key in raw:
                    metadata[f"llmlingua_{key}"] = raw[key]

        return CompressionResult.build(text, compressed, self.name, metadata=metadata)

    def _build_call_kwargs(
        self, model: Any, rate: float, text: str, overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """Assemble ``compress_prompt`` kwargs, dropping any this version lacks."""
        call_kwargs: dict[str, Any] = {
            "rate": rate,
            "force_tokens": _force_tokens(
                text,
                overrides.pop("force_tokens", self.force_tokens),
                entities=overrides.pop("preserve_entities", self.preserve_entities),
            ),
            "force_reserve_digit": overrides.pop("force_reserve_digit", self.force_reserve_digit),
            "drop_consecutive": overrides.pop("drop_consecutive", self.drop_consecutive),
        }
        call_kwargs.update(overrides)
        return _filter_supported(model.compress_prompt, call_kwargs)
