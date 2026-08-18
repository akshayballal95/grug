"""The examples must keep running. They are documentation that can rot."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from grug.backends.lingua2 import Lingua2Backend

EXAMPLES = Path(__file__).parent.parent / "examples"
SCRIPTS = sorted(p.name for p in EXAMPLES.glob("*.py") if not p.name.startswith("_"))

#: Scripts that use whatever backend is installed, so they load a model when the
#: lingua2 extra is present. Marked slow to keep `-m "not slow"` model-free.
MODEL_SCRIPTS = {"lingua2_backend.py", "compare_backends.py", "faithfulness.py"}

SCRIPT_PARAMS = [
    pytest.param(name, marks=pytest.mark.slow) if name in MODEL_SCRIPTS else name
    for name in SCRIPTS
]


def test_the_examples_directory_is_populated():
    assert SCRIPTS, "no example scripts found"
    assert "rules_backend.py" in SCRIPTS
    assert "lingua2_backend.py" in SCRIPTS


@pytest.mark.parametrize("script", SCRIPT_PARAMS)
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
    """It must run on a bare `pip install grug`, so it may not import torch."""
    source = (EXAMPLES / "rules_backend.py").read_text(encoding="utf-8")
    for forbidden in ("import torch", "llmlingua", "lingua2"):
        assert forbidden not in source


def test_every_model_script_is_a_real_example():
    assert set(SCRIPTS) >= MODEL_SCRIPTS


def test_lingua2_example_degrades_without_the_extra():
    """With the extra missing it prints the install hint instead of crashing."""
    if Lingua2Backend.is_available():
        pytest.skip("llmlingua is installed; the missing-extra path cannot be exercised")
    result = subprocess.run(
        [sys.executable, "lingua2_backend.py"],
        cwd=EXAMPLES,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    assert "pip install 'grug[lingua2]'" in result.stdout
