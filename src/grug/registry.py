"""Backend registry and third-party entry-point discovery.

Classes register eagerly; their dependencies load only on instantiation, which
is what keeps ``import grug`` free of torch.
"""

from __future__ import annotations

import warnings
from typing import Any, TypeVar

from .base import CompressorBackend

__all__ = [
    "ENTRY_POINT_GROUP",
    "BackendNotFoundError",
    "backend_info",
    "create_backend",
    "default_backend_name",
    "get_backend_class",
    "list_backends",
    "register_backend",
    "unregister_backend",
]

#: Entry-point group external packages advertise their backends under.
ENTRY_POINT_GROUP = "grug.backends"

#: Preference order when the caller does not name a backend. ``classifier`` is
#: deliberately absent: it cannot be constructed without a checkpoint, so the
#: zero-configuration default is always ``rules``.
_PREFERRED_ORDER = ("rules",)

_REGISTRY: dict[str, type[CompressorBackend]] = {}
_builtins_loaded = False
_entry_points_loaded = False

B = TypeVar("B", bound=type[CompressorBackend])


class BackendNotFoundError(KeyError):
    """Raised when a backend name is not in the registry."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        listed = ", ".join(available) if available else "(none)"
        super().__init__(f"Unknown backend {name!r}. Available backends: {listed}")

    def __str__(self) -> str:  # KeyError repr-quotes its message otherwise
        return self.args[0]


def register_backend(cls: B) -> B:
    """Class decorator registering a :class:`CompressorBackend` under its ``name``."""
    name = getattr(cls, "name", "")
    if not name:
        raise ValueError(f"{cls.__name__} must set a non-empty class attribute 'name'")
    if not issubclass(cls, CompressorBackend):
        raise TypeError(f"{cls.__name__} must subclass CompressorBackend")
    _REGISTRY[name] = cls
    return cls


def unregister_backend(name: str) -> None:
    """Remove a backend from the registry. Mainly useful in tests."""
    _REGISTRY.pop(name, None)


def _load_builtins() -> None:
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True  # set first: the import below re-enters this module
    from . import backends  # noqa: F401  (importing registers the built-ins)


def _load_entry_points() -> None:
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True

    from importlib.metadata import entry_points

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            obj = ep.load()
        except Exception as exc:  # pragma: no cover - depends on installed plugins
            warnings.warn(
                f"Failed to load grug backend entry point {ep.name!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        try:
            register_backend(obj)
        except (TypeError, ValueError) as exc:  # pragma: no cover - plugin bug
            warnings.warn(
                f"Entry point {ep.name!r} did not provide a valid backend: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def _ensure_loaded() -> None:
    _load_builtins()
    _load_entry_points()


def list_backends() -> list[str]:
    """Names of every registered backend, built-in and third-party."""
    _ensure_loaded()
    known = [n for n in _PREFERRED_ORDER if n in _REGISTRY]
    extra = sorted(n for n in _REGISTRY if n not in _PREFERRED_ORDER)
    return known + extra


def get_backend_class(name: str) -> type[CompressorBackend]:
    """Look up a backend class by registry key."""
    _ensure_loaded()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise BackendNotFoundError(name, list_backends()) from None


def create_backend(name: str, **kwargs: Any) -> CompressorBackend:
    """Instantiate a backend by name, passing ``kwargs`` to its constructor.

    Dependency errors surface here, not at import time, and name the extra to
    install.
    """
    return get_backend_class(name)(**kwargs)


def backend_info() -> list[dict[str, Any]]:
    """Describe every backend: name, availability, extra, and description."""
    _ensure_loaded()
    info = []
    for name in list_backends():
        cls = _REGISTRY[name]
        info.append(
            {
                "name": name,
                "available": _safe_is_available(cls),
                "extra": getattr(cls, "extra", None),
                "generative": getattr(cls, "generative", False),
                "description": (getattr(cls, "description", "") or "").strip(),
            }
        )
    return info


def _safe_is_available(cls: type[CompressorBackend]) -> bool:
    try:
        return bool(cls.is_available())
    except Exception:  # pragma: no cover - defensive against plugin bugs
        return False


def default_backend_name() -> str:
    """The backend a plain call gets: ``rules``, the zero-dependency default.

    Falls back to the first available backend of any kind, and finally to
    ``rules`` so that callers always get a usable name.
    """
    _ensure_loaded()
    for name in _PREFERRED_ORDER:
        cls = _REGISTRY.get(name)
        if cls is not None and _safe_is_available(cls):
            return name
    for name in list_backends():
        if _safe_is_available(_REGISTRY[name]):
            return name
    return "rules"
