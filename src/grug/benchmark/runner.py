"""Run the QA benchmark across backends and rates, and collect chart-ready rows."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..base import count_tokens
from ..verify import verify
from .llm import ANSWER_PROMPT
from .qa import QAExample, score_answers

__all__ = ["BenchmarkRow", "run_benchmark"]

#: The uncompressed prompt: the ceiling every compressor is measured against.
ORIGINAL = "original"


@dataclass
class BenchmarkRow:
    """One (backend, rate) cell of the benchmark."""

    backend: str
    rate: float
    ratio: float
    original_tokens: int
    compressed_tokens: int
    exact_match: float
    f1: float
    questions: int
    compress_seconds: float
    answer_seconds: float
    warnings_per_doc: float = 0.0
    negation_loss_rate: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens


def _compress_all(
    examples: list[QAExample], backend: Any, rate: float
) -> tuple[list[str], dict[str, float]]:
    """Compress every context, and record what it cost in tokens and fidelity."""
    import grug

    started = time.perf_counter()
    contexts: list[str] = []
    original_tokens = compressed_tokens = warnings = negation_losses = 0

    for example in examples:
        if backend is None:  # the uncompressed ceiling
            text = example.context
        else:
            text = grug.compress(example.context, rate=rate, backend=backend, verify=False).text
        contexts.append(text)
        original_tokens += count_tokens(example.context)
        compressed_tokens += count_tokens(text)
        found = verify(example.context, text)
        warnings += len(found)
        negation_losses += any(w.startswith("negation lost") for w in found)

    return contexts, {
        "compress_seconds": time.perf_counter() - started,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "warnings_per_doc": warnings / max(1, len(examples)),
        "negation_loss_rate": negation_losses / max(1, len(examples)),
    }


def run_benchmark(
    examples: list[QAExample],
    backends: dict[str, Any],
    rates: list[float],
    client: Any,
    *,
    include_original: bool = True,
    progress: bool = True,
) -> list[BenchmarkRow]:
    """Compress, answer, and score across every backend and rate.

    Args:
        examples: Contexts with their questions.
        backends: Display name -> backend instance (or registry name).
        rates: Compression rates to sweep.
        client: Anything with ``many(pairs) -> list[str]``.
        include_original: Also measure the uncompressed prompt, as the ceiling.
        progress: Print each row as it completes.
    """
    plan: list[tuple[str, Any, float]] = []
    if include_original:
        plan.append((ORIGINAL, None, 1.0))
    plan += [(name, backend, rate) for name, backend in backends.items() for rate in rates]

    rows: list[BenchmarkRow] = []
    for name, backend, rate in plan:
        contexts, stats = _compress_all(examples, backend, rate)

        pairs: list[tuple[str, str]] = []
        golds: list[str] = []
        for context, example in zip(contexts, examples, strict=True):
            for question, answer in zip(example.questions, example.answers, strict=True):
                pairs.append((context, question))
                golds.append(answer)

        started = time.perf_counter()
        predictions = client.many(pairs)
        answer_seconds = time.perf_counter() - started
        scores = score_answers(predictions, golds)

        row = BenchmarkRow(
            backend=name,
            rate=rate,
            ratio=stats["compressed_tokens"] / max(1, stats["original_tokens"]),
            original_tokens=stats["original_tokens"],
            compressed_tokens=stats["compressed_tokens"],
            exact_match=scores["exact_match"],
            f1=scores["f1"],
            questions=scores["n"],
            compress_seconds=stats["compress_seconds"],
            answer_seconds=answer_seconds,
            warnings_per_doc=stats["warnings_per_doc"],
            negation_loss_rate=stats["negation_loss_rate"],
        )
        rows.append(row)
        if progress:
            print(
                f"  {name:<18}rate={rate:<5.2f} ratio={row.ratio:.2f} "
                f"EM={row.exact_match:.3f} F1={row.f1:.3f} "
                f"neg_loss={row.negation_loss_rate:.2f}",
                flush=True,
            )
    return rows


def save(rows: list[BenchmarkRow], path: str | Path, *, prompt: str = ANSWER_PROMPT) -> Path:
    """Write results as JSON, including the prompt so a run is reproducible."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"prompt": prompt, "rows": [asdict(r) for r in rows]}, indent=2),
        encoding="utf-8",
    )
    return target
