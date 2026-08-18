"""Reproduce the compressor: corpus -> labels -> fine-tuned encoder.

Everything here is behind ``pip install 'grug[train]'`` and is never imported
by ``grug`` itself, so the library stays free of torch. The one exception is
:mod:`grug.training.alignment`, which is pure Python and importable anywhere.

    grug train prepare --out data/
    grug train run --data data/ --model answerdotai/ModernBERT-base
    grug train evaluate --model out/checkpoint
"""

from __future__ import annotations

from .alignment import align, annotate, filter_examples

__all__ = ["align", "annotate", "filter_examples"]
