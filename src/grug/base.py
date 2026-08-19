"""Core abstractions: the backend contract and the result it returns.

Nothing in this module imports a machine-learning library. Backends pull their
heavy dependencies in lazily so that ``import grug`` stays cheap.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CompressionResult",
    "CompressorBackend",
    "MissingDependencyError",
    "count_tokens",
    "tokenizer_name",
]


class MissingDependencyError(ImportError):
    """Raised when a backend's optional dependencies are not installed."""

    def __init__(self, backend: str, extra: str, missing: str) -> None:
        self.backend = backend
        self.extra = extra
        self.missing = missing
        super().__init__(
            f"The {backend!r} backend requires {missing!r}, which is not installed. "
            f"Install it with:  pip install 'grugify[{extra}]'"
        )


@functools.lru_cache(maxsize=1)
def _encoder() -> Any | None:
    """Return a cached ``cl100k_base`` encoder, or ``None`` if tiktoken is absent."""
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Network-less environments cannot fetch the BPE ranks on first use.
        return None


def count_tokens(text: str) -> int:
    """Count tokens with ``cl100k_base``, or whitespace-split if tiktoken is absent.

    One tokenizer for every backend keeps the reported ratios comparable.
    """
    if not text:
        return 0
    enc = _encoder()
    if enc is None:
        return len(text.split())
    return len(enc.encode(text, disallowed_special=()))


def tokenizer_name() -> str:
    """Name of the tokenizer :func:`count_tokens` is currently using."""
    return "cl100k_base" if _encoder() is not None else "whitespace"


@dataclass
class CompressionResult:
    """The outcome of compressing a single document."""

    text: str
    original_tokens: int
    compressed_tokens: int
    ratio: float
    backend: str
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def saved_tokens(self) -> int:
        """Tokens removed relative to the original."""
        return self.original_tokens - self.compressed_tokens

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view, used by ``grug compress --json``."""
        return {
            "text": self.text,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "ratio": self.ratio,
            "backend": self.backend,
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }

    @classmethod
    def build(
        cls,
        text: str,
        compressed: str,
        backend: str,
        warnings: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CompressionResult:
        """Construct a result, deriving token counts and ratio from the text pair."""
        original_tokens = count_tokens(text)
        compressed_tokens = count_tokens(compressed)
        ratio = compressed_tokens / original_tokens if original_tokens else 1.0
        return cls(
            text=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            ratio=ratio,
            backend=backend,
            warnings=list(warnings or []),
            metadata=dict(metadata or {}),
        )


class CompressorBackend(ABC):
    """Base class every compression backend implements: text + rate -> shorter text.

    Extractive or generative, the contract only asks that the reported ``ratio``
    reflect what actually came out.
    """

    #: Registry key, e.g. ``"rules"``. Subclasses must set this.
    name: str = ""

    #: Human-readable one-liner, surfaced by ``grug backends``.
    description: str = ""

    #: Extra to install for this backend's dependencies, if any.
    extra: str | None = None

    #: ``True`` when the backend rewrites text rather than selecting from it.
    generative: bool = False

    #: Constructor arguments that cannot be varied per call, because they decide
    #: how the model is loaded. Passing one to :func:`grug.compress` as a
    #: per-call keyword would forward it to :meth:`compress`, where a backend is
    #: free to ignore it -- so they are rejected there and routed through
    #: ``backend_kwargs`` instead. Extend this in a subclass that adds more.
    construction_only: tuple[str, ...] = ("device", "model_name")

    @abstractmethod
    def compress(self, text: str, rate: float = 0.5, **kwargs: Any) -> CompressionResult:
        """Compress ``text`` keeping roughly ``rate`` of its tokens.

        Args:
            text: The document to compress.
            rate: Fraction of tokens to keep, in ``(0.0, 1.0]``. Backends that
                cannot hit an exact rate should approximate and report the
                achieved value in :attr:`CompressionResult.ratio`.
            **kwargs: Backend-specific options.
        """

    def compress_batch(
        self, texts: list[str], rate: float = 0.5, **kwargs: Any
    ) -> list[CompressionResult]:
        """Compress several documents. Sequential by default; override to batch."""
        return [self.compress(text, rate=rate, **kwargs) for text in texts]

    @classmethod
    def is_available(cls) -> bool:
        """Whether this backend's dependencies are importable right now."""
        return True

    @classmethod
    def require_available(cls) -> None:
        """Raise :class:`MissingDependencyError` if dependencies are missing.

        Called from ``__init__`` so a missing extra is reported at construction.
        """
        if not cls.is_available():
            raise MissingDependencyError(
                cls.name, cls.extra or cls.name, "its optional dependencies"
            )

    @staticmethod
    def _validate_rate(rate: float) -> float:
        if not 0.0 < rate <= 1.0:
            raise ValueError(f"rate must be in (0.0, 1.0], got {rate!r}")
        return rate

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
