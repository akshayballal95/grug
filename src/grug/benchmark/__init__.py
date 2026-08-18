"""End-to-end benchmarking: does the compressed prompt still answer questions?

    grug benchmark qa --model bedrock/<model-id> --limit 25

Behind ``pip install 'grug[bench]'``. Never imported by ``grug`` itself.
"""

from __future__ import annotations

from .qa import QAExample, exact_match, load_qa, score_answers, token_f1
from .report import to_csv, to_svg
from .runner import BenchmarkRow, run_benchmark, save

__all__ = [
    "BenchmarkRow",
    "QAExample",
    "exact_match",
    "load_qa",
    "run_benchmark",
    "save",
    "score_answers",
    "to_csv",
    "to_svg",
    "token_f1",
]
