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
