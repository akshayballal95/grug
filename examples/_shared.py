"""Small helpers shared by the example scripts. Not part of the grug API."""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent


def sample_doc() -> str:
    """The incident report used by every example."""
    return (HERE / "sample_doc.md").read_text(encoding="utf-8")


def rule(title: str = "", width: int = 78) -> None:
    if title:
        print(f"\n{title}\n" + "-" * width)
    else:
        print("-" * width)


def show(result, label: str = "") -> None:
    """Print a CompressionResult: the stats line, then the text."""
    head = f"{label}  " if label else ""
    print(
        f"{head}{result.original_tokens} -> {result.compressed_tokens} tokens "
        f"(ratio {result.ratio:.2f}, backend={result.backend})"
    )
    for warning in result.warnings:
        print(f"  WARN {warning}")
    print()
    print(result.text)
