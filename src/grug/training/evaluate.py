"""Score a trained checkpoint the way grug actually uses it.

Token-level accuracy says whether the classifier learned the labels. It does
not say whether the compressed prompt is still *true*, which is the property
grug exists to protect -- so faithfulness is measured here as a first-class
metric, using the same verifier that runs in production.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..verify import verify

__all__ = ["FAITHFULNESS_CHECKS", "compare", "evaluate_checkpoint"]

#: Warning prefixes emitted by :func:`grug.verify`, one per check.
FAITHFULNESS_CHECKS = ("negation lost", "negation kept", "numbers missing", "entities missing")

DEFAULT_RATES = (0.7, 0.5, 0.33, 0.2)


def _rate_row(backend: Any, documents: list[str], rate: float) -> dict[str, Any]:
    """Compress every document at one rate and summarise ratio and faithfulness."""
    import grug

    ratios: list[float] = []
    counts = dict.fromkeys(FAITHFULNESS_CHECKS, 0)
    clean = 0
    started = time.perf_counter()

    for document in documents:
        result = grug.compress(document, rate=rate, backend=backend, verify=False)
        ratios.append(result.ratio)
        warnings = verify(document, result.text)
        if not warnings:
            clean += 1
        for warning in warnings:
            for check in FAITHFULNESS_CHECKS:
                if warning.startswith(check):
                    counts[check] += 1
                    break

    total = max(1, len(documents))
    return {
        "rate": rate,
        "ratio": sum(ratios) / total,
        "clean_fraction": clean / total,
        "seconds_per_doc": (time.perf_counter() - started) / total,
        **{k.replace(" ", "_"): v / total for k, v in counts.items()},
    }


def evaluate_checkpoint(
    model_path: str | Path,
    documents: list[str],
    *,
    rates: tuple[float, ...] = DEFAULT_RATES,
    device: str = "auto",
    out_file: str | Path | None = None,
) -> dict[str, Any]:
    """Ratio and faithfulness for one checkpoint across several rates."""
    from ..backends.modern import ModernBackend

    backend = ModernBackend(model_name=str(model_path), device=device)
    report = {
        "model": str(model_path),
        "documents": len(documents),
        "rows": [_rate_row(backend, documents, rate) for rate in rates],
    }
    if out_file:
        Path(out_file).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def compare(
    documents: list[str],
    backends: dict[str, Any],
    *,
    rates: tuple[float, ...] = DEFAULT_RATES,
) -> list[dict[str, Any]]:
    """Same documents, same rates, several backends -- the isolation experiment.

    ``backends`` maps a display name to a backend instance or registry name.
    """
    rows = []
    for name, backend in backends.items():
        for rate in rates:
            rows.append({"backend": name, **_rate_row(backend, documents, rate)})
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    """Render :func:`compare` output as a fixed-width table."""
    if not rows:
        return "(no rows)"
    header = f"{'backend':<14}{'rate':>6}{'ratio':>8}{'clean':>8}{'sec/doc':>9}"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.get('backend', '-'):<14}{row['rate']:>6.2f}{row['ratio']:>8.2f}"
            f"{row['clean_fraction']:>8.2f}{row['seconds_per_doc']:>9.2f}"
        )
    return "\n".join(lines)
