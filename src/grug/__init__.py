"""grug -- shrink documents into terse caveman text so LLM input costs less.

    >>> import grug
    >>> result = grug.compress(long_document, rate=0.4)
    >>> result.ratio
    0.41
    >>> result.warnings
    []

Importing this module is cheap: no torch, no transformers, no model weights.
Backends pull their dependencies in when you instantiate one.
"""

from __future__ import annotations

from typing import Any

from .base import (
    CompressionResult,
    CompressorBackend,
    MissingDependencyError,
    count_tokens,
    tokenizer_name,
)
from .chunking import DEFAULT_CHUNK_TOKENS, Chunk, chunk_document, compress_document
from .registry import (
    BackendNotFoundError,
    backend_info,
    create_backend,
    default_backend_name,
    get_backend_class,
    list_backends,
    register_backend,
)
from .verify import verify as run_verify

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_CHUNK_TOKENS",
    "BackendNotFoundError",
    "Chunk",
    "CompressionResult",
    "Compressor",
    "CompressorBackend",
    "MissingDependencyError",
    "__version__",
    "backend_info",
    "chunk_document",
    "compress",
    "compress_document",
    "count_tokens",
    "create_backend",
    "default_backend_name",
    "get_backend_class",
    "list_backends",
    "register_backend",
    "tokenizer_name",
    "verify",
]

#: Public re-export of the faithfulness checker (``grug.verify(a, b)``).
verify = run_verify

_BACKEND_CACHE: dict[tuple[str, str], CompressorBackend] = {}


def _shared_backend(name: str, backend_kwargs: dict[str, Any]) -> CompressorBackend:
    """Reuse a backend instance across calls so models load exactly once.

    Keyed on the construction arguments as well as the name: two calls asking
    for different devices are asking for different backends, and handing the
    second one the first one's model would put it on the wrong device.

    The key is the *rendering* of those arguments rather than the arguments
    themselves, because they are not all hashable -- ``force_tokens`` is a list
    on every backend that has it, and a tuple key would raise instead of
    caching. Two configurations that render alike share an instance, which is
    what sharing an instance means.
    """
    key = (name, repr(sorted(backend_kwargs.items())))
    if key not in _BACKEND_CACHE:
        _BACKEND_CACHE[key] = create_backend(name, **backend_kwargs)
    return _BACKEND_CACHE[key]


def _reject_construction_kwargs(backend: CompressorBackend, kwargs: dict[str, Any]) -> None:
    """Refuse per-call kwargs that only mean something at construction time.

    Forwarding one to ``compress`` is worse than an error: a backend that does
    not recognise the keyword drops it, so ``device="cuda"`` runs happily on
    the CPU and reports nothing.
    """
    misplaced = sorted(k for k in kwargs if k in getattr(backend, "construction_only", ()))
    if misplaced:
        named = ", ".join(repr(k) for k in misplaced)
        raise TypeError(
            f"{named} configures how backend {backend.name!r} is constructed, not how one "
            f"document is compressed. Pass it as backend_kwargs={{{misplaced[0]!r}: ...}}, "
            f"or construct Compressor({backend.name!r}, {misplaced[0]}=...) once and reuse it."
        )


class Compressor:
    """A reusable compressor bound to one backend.

    Prefer this over :func:`compress` in a loop: the backend (and its model)
    is constructed once and held.

        >>> comp = Compressor(backend="rules")
        >>> results = comp.compress_batch(documents, rate=0.5)
    """

    def __init__(
        self,
        backend: str | CompressorBackend | None = None,
        *,
        verify: bool = True,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        preserve_code: bool = True,
        preserve_inline_code: bool = True,
        preserve_markdown: bool = True,
        preserve_numbers: bool = True,
        preserve_identifiers: bool = True,
        **backend_kwargs: Any,
    ) -> None:
        """
        Args:
            backend: Registry name, a ready-made backend instance, or ``None``
                for the default (``rules``).
            verify: Run faithfulness checks and populate ``warnings``.
            chunk_tokens: Chunk size ceiling handed to the chunker.
            preserve_code: Pass fenced code blocks through untouched.
            preserve_inline_code: Shield inline code spans and URLs.
            preserve_markdown: Keep tables verbatim and markdown markers intact.
            preserve_numbers: Shield numbers with internal separators.
            preserve_identifiers: Shield names whose internal separator is
                load-bearing, e.g. ``us-east-1`` or ``text/plain``.
            **backend_kwargs: Forwarded to the backend constructor
                (e.g. ``device="cuda"``, ``model_name=...``).
        """
        if isinstance(backend, CompressorBackend):
            if backend_kwargs:
                raise TypeError(
                    "backend_kwargs cannot be combined with an already-constructed backend"
                )
            self._backend = backend
        else:
            self._backend = create_backend(backend or default_backend_name(), **backend_kwargs)

        self.verify = verify
        self.chunk_tokens = chunk_tokens
        self.preserve_code = preserve_code
        self.preserve_inline_code = preserve_inline_code
        self.preserve_markdown = preserve_markdown
        self.preserve_numbers = preserve_numbers
        self.preserve_identifiers = preserve_identifiers

    @property
    def backend(self) -> CompressorBackend:
        """The backend in use."""
        return self._backend

    @property
    def backend_name(self) -> str:
        """Registry name of the backend in use."""
        return self._backend.name

    def compress(
        self,
        text: str,
        rate: float = 0.5,
        *,
        verify: bool | None = None,
        **kwargs: Any,
    ) -> CompressionResult:
        """Compress one document, chunking it first if it is long.

        Args:
            text: The document.
            rate: Fraction of tokens to keep.
            verify: Override the instance's faithfulness setting.
            **kwargs: Forwarded to the backend's ``compress``.
        """
        result = compress_document(
            text,
            self._backend,
            rate=rate,
            max_tokens=self.chunk_tokens,
            preserve_code=self.preserve_code,
            preserve_inline_code=self.preserve_inline_code,
            preserve_markdown=self.preserve_markdown,
            preserve_numbers=self.preserve_numbers,
            preserve_identifiers=self.preserve_identifiers,
            **kwargs,
        )
        should_verify = self.verify if verify is None else verify
        if should_verify:
            # A backend that carries a language pack (the rules backend does)
            # is verified with that language's vocabulary; anything else gets
            # the English default.
            language = getattr(self._backend, "language", "en")
            result.warnings.extend(run_verify(text, result.text, language=language))
        return result

    def compress_batch(
        self,
        texts: list[str],
        rate: float = 0.5,
        *,
        verify: bool | None = None,
        **kwargs: Any,
    ) -> list[CompressionResult]:
        """Compress several documents, reusing the loaded backend for all of them."""
        return [self.compress(t, rate=rate, verify=verify, **kwargs) for t in texts]

    def __repr__(self) -> str:
        return f"Compressor(backend={self.backend_name!r})"


def compress(
    text: str,
    rate: float = 0.5,
    *,
    backend: str | CompressorBackend | None = None,
    verify: bool = True,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    preserve_code: bool = True,
    preserve_inline_code: bool = True,
    preserve_markdown: bool = True,
    preserve_numbers: bool = True,
    preserve_identifiers: bool = True,
    backend_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CompressionResult:
    """Compress ``text``, keeping roughly ``rate`` of its tokens.

    A one-shot :class:`Compressor`; see its docstring for the arguments. The
    ``backend`` may be a registry name or an instance, and defaults to
    ``rules``. Backend construction is cached, so repeated calls do not reload
    a model.

    Args:
        text: The document to compress.
        rate: Fraction of tokens to keep, in ``(0.0, 1.0]``. The achieved value
            is reported as :attr:`CompressionResult.ratio`.
        backend_kwargs: Forwarded to the backend's *constructor*, the way
            ``Compressor(backend, **kwargs)`` does -- ``{"device": "cuda"}``,
            ``{"model_name": ...}``. Instances are cached per distinct set, so
            asking for a second device loads a second model rather than reusing
            the first. Ignored when ``backend`` is already an instance.
        **kwargs: Forwarded to the backend's ``compress``, per call.
    """
    construction = dict(backend_kwargs or {})
    if isinstance(backend, CompressorBackend):
        if construction:
            raise TypeError("backend_kwargs cannot be combined with an already-constructed backend")
        instance = backend
    else:
        instance = _shared_backend(backend or default_backend_name(), construction)
    _reject_construction_kwargs(instance, kwargs)
    return Compressor(
        instance,
        verify=verify,
        chunk_tokens=chunk_tokens,
        preserve_code=preserve_code,
        preserve_inline_code=preserve_inline_code,
        preserve_markdown=preserve_markdown,
        preserve_numbers=preserve_numbers,
        preserve_identifiers=preserve_identifiers,
    ).compress(text, rate=rate, **kwargs)
