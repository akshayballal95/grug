"""Built-in backends.

Importing this package registers every built-in backend class. The imports are
cheap by construction: each module keeps its machine-learning dependencies
inside method bodies, so ``import grug`` never pulls in torch.
"""

from __future__ import annotations

from . import classifier, rules  # noqa: F401  (registration side effects)
from .classifier import ClassifierBackend
from .rules import RulesBackend

__all__ = ["ClassifierBackend", "RulesBackend"]
