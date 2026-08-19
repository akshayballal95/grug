"""The examples must keep running. They are documentation that can rot."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"
SCRIPTS = sorted(p.name for p in EXAMPLES.glob("*.py") if not p.name.startswith("_"))


def test_the_examples_directory_is_populated():
    assert SCRIPTS, "no example scripts found"
    assert "rules_backend.py" in SCRIPTS
    assert "compare_backends.py" in SCRIPTS


@pytest.mark.parametrize("script", SCRIPTS)
def test_example_runs_cleanly(script):
    """Every example exits 0, including when an optional backend is missing."""
    result = subprocess.run(
        [sys.executable, script],
        cwd=EXAMPLES,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"{script} failed:\n{result.stdout}\n{result.stderr}"
    assert result.stdout.strip(), f"{script} printed nothing"


def test_rules_example_needs_no_optional_dependency():
    """It must run on a bare `pip install grugify`, so it may not import torch."""
    source = (EXAMPLES / "rules_backend.py").read_text(encoding="utf-8")
    assert "import torch" not in source
