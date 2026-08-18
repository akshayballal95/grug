"""Built-in backends.

Importing this package registers every built-in backend class. The imports are
cheap by construction: each module keeps its machine-learning dependencies
inside method bodies, so ``import grug`` never pulls in torch.
"""

from __future__ import annotations

from . import lingua2, longlingua, rules  # noqa: F401  (imported for registration side effects)
from .lingua2 import Lingua2Backend
from .longlingua import LongLinguaBackend
from .rules import RulesBackend

__all__ = ["Lingua2Backend", "LongLinguaBackend", "RulesBackend"]
