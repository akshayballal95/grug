"""Built-in backends.

Importing this package registers every built-in backend class. The imports are
cheap by construction: each module keeps its machine-learning dependencies
inside method bodies, so ``import grug`` never pulls in torch.
"""

from __future__ import annotations

from . import lingua2, longlingua, modern, rules  # noqa: F401  (registration side effects)
from .lingua2 import Lingua2Backend
from .longlingua import LongLinguaBackend
from .modern import ModernBackend
from .rules import RulesBackend

__all__ = ["Lingua2Backend", "LongLinguaBackend", "ModernBackend", "RulesBackend"]
