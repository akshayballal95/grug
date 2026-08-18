"""Plumbing common to the two llmlingua-backed backends.

Both need the same optional dependencies, the same device resolution, and the
same tolerance for older llmlingua releases that lack a keyword. None of it
imports torch at module scope.
"""

from __future__ import annotations

import importlib.util
import inspect
from typing import Any

__all__ = ["REQUIRED_MODULES", "filter_supported", "missing_modules", "resolve_device"]

REQUIRED_MODULES = ("llmlingua", "torch", "transformers")


def missing_modules() -> list[str]:
    """Which of :data:`REQUIRED_MODULES` are not importable, in declared order."""
    return [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]


def resolve_device(device: str) -> str:
    """Turn ``"auto"`` into the best device actually available."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:  # pragma: no cover - guarded by require_available
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def filter_supported(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs ``func`` accepts, so older llmlingua releases still work."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - C-implemented callables
        return kwargs
    params = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}
