from __future__ import annotations

import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_markdown() -> str:
    """A ~500-word markdown doc with a code block, numbers, and negations."""
    return (FIXTURES / "sample.md").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def cli_command() -> list[str]:
    """How to invoke the CLI as a subprocess."""
    return [sys.executable, "-m", "grug.cli"]


#: Warning prefixes that mean information left the document.
#:
#: "negation kept without its scope" is deliberately absent. It reports a cue
#: that survived while the word it applied to was cut -- a real degradation, but
#: an expected consequence of extractive compression at aggressive rates, and one
#: the backends cannot currently avoid: force-pinning keeps the cue, and nothing
#: pins its target. Tests that assert a document survived intact use this list so
#: that a scope warning does not mask the loss classes they were written to catch.
LOSS_PREFIXES = (
    "negation lost",
    "numbers missing",
    "entities missing",
    "entities now indistinguishable",
)


def losses(warnings: list[str]) -> list[str]:
    """The subset of ``warnings`` that report information actually gone."""
    return [w for w in warnings if w.startswith(LOSS_PREFIXES)]
