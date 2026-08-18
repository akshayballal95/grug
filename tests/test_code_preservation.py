"""Code must never be compressed: a shorter program is a different program."""

from __future__ import annotations

import pytest

import grug
from grug.chunking import CODE_EXTENSIONS, code_regions, looks_like_code

RAW_PY = 'import os\n\n\ndef total(items, rate=0.5):\n    """Return the total."""\n    return sum(i.price for i in items) * rate\n'
SQL = "SELECT id, name\nFROM accounts\nWHERE region = 'us-east-1'\nORDER BY created_at;\n"
DOCKERFILE = "FROM python:3.12\nWORKDIR /app\nCOPY . .\nRUN pip install -e .\n"
PROSE = "The billing pipeline was rewritten. No data is deleted during the migration."
INDENTED_PROSE = "Intro line.\n\n    an indented paragraph of ordinary prose that\n    continues onto a second line with no code in it\n\nOutro line."


def verbatim(text: str, rate: float = 0.3) -> bool:
    return grug.compress(text, rate=rate, backend="rules").text.strip() == text.strip()


# -- whole-document detection ----------------------------------------------


@pytest.mark.parametrize("text", [RAW_PY, SQL, DOCKERFILE, "#!/bin/bash\nset -e\ncd /tmp\n"])
def test_source_is_detected_by_content(text):
    assert looks_like_code(text)


@pytest.mark.parametrize("text", [PROSE, INDENTED_PROSE])
def test_prose_is_not_detected_as_code(text):
    assert not looks_like_code(text)


def test_indented_prose_is_not_code():
    """The weak indentation signal alone must not convict a list continuation."""
    assert not looks_like_code(INDENTED_PROSE)
    assert code_regions(INDENTED_PROSE) == []


@pytest.mark.parametrize("name", ["a.py", "a.rs", "a.sql", "a.toml", "Dockerfile", "Makefile"])
def test_filename_alone_is_enough(name):
    assert looks_like_code("anything at all", name)


def test_prose_filename_does_not_convict():
    assert not looks_like_code(PROSE, "notes.md")


def test_docstring_heavy_source_needs_the_filename():
    """Most lines are prose, so only the extension gives it away."""
    text = '"""Helpers.\n\nA paragraph of ordinary prose describing the module at length.\n"""\n\n\ndef f(x):\n    return x\n'
    assert not looks_like_code(text)
    assert looks_like_code(text, "helpers.py")


def test_code_extensions_cover_the_common_languages():
    for ext in (".py", ".js", ".ts", ".rs", ".go", ".java", ".sql", ".yaml", ".json"):
        assert ext in CODE_EXTENSIONS


# -- compression leaves code alone -----------------------------------------


@pytest.mark.parametrize("text", [RAW_PY, SQL, DOCKERFILE])
def test_source_survives_compression_verbatim(text):
    assert verbatim(text)


def test_fenced_code_survives():
    text = "Intro text here.\n\n```python\ndef f(x):\n    return x * 2\n```\n\nOutro text here."
    assert "def f(x):\n    return x * 2" in grug.compress(text, rate=0.3, backend="rules").text


def test_unfenced_code_in_prose_survives():
    """No fence, no indent marker -- just code sitting in a document."""
    text = (
        "Here is the helper we use.\n\n"
        "def compute_total(items, rate=0.5):\n"
        "    return sum(i.price for i in items) * rate\n\n"
        "It is called once per invoice and it is not cached."
    )
    out = grug.compress(text, rate=0.3, backend="rules").text
    assert "sum(i.price for i in items) * rate" in out
    assert "It is called once per invoice" not in out  # prose still compresses


def test_prose_still_compresses():
    assert not verbatim(
        "It is important to note that the pipeline was rewritten and that "
        "the totals are computed once per invoice for every account."
    )


def test_two_line_code_block_is_enough():
    text = "Intro.\n\nx = compute(1)\ny = compute(2)\n\nOutro."
    assert "x = compute(1)\ny = compute(2)" in grug.compress(text, rate=0.3, backend="rules").text


def test_code_regions_do_not_swallow_the_document():
    text = "Prose paragraph one here.\n\ndef f():\n    return 1\n\nProse paragraph two here."
    covered = sum(e - b for b, e in code_regions(text))
    assert 0 < covered < len(text)


def test_preserve_code_can_be_disabled():
    from grug.chunking import chunk_document

    chunks = chunk_document(RAW_PY, preserve_code=False)
    assert not any(not c.compressible and c.text.strip() for c in chunks)
