"""End-to-end over a realistic markdown document.

The fixture is deliberately awkward: it is longer than one chunk, it contains a
fenced code block, several number formats, and negations that carry the meaning
of the paragraphs they sit in.
"""

from __future__ import annotations

import pathlib

import pytest

import grug
from conftest import losses
from grug.backends.lingua2 import Lingua2Backend
from grug.chunking import chunk_document

CODE_BLOCK = '''```python
def wave_for(tenant_id: str) -> int:
    """Return the rollout wave for a tenant, or 0 if it is not enrolled."""
    response = client.get(f"/v2/tenants/{tenant_id}/rollout")
    if response.status_code == 404:
        return 0
    return response.json()["wave"]
```'''


@pytest.fixture(scope="module")
def compressed(sample_markdown):
    return grug.compress(sample_markdown, rate=0.5, backend="rules")


def test_the_fixture_is_long_enough_to_need_chunking(sample_markdown):
    assert grug.count_tokens(sample_markdown) > grug.DEFAULT_CHUNK_TOKENS
    assert len(chunk_document(sample_markdown)) > 1


def test_document_gets_smaller(compressed):
    assert compressed.compressed_tokens < compressed.original_tokens
    assert compressed.ratio < 1.0
    assert compressed.metadata["chunks"] > 1


def test_code_block_survives_verbatim(compressed):
    assert CODE_BLOCK in compressed.text


def test_code_block_is_never_handed_to_the_backend(sample_markdown):
    for chunk in chunk_document(sample_markdown):
        if "def wave_for" in chunk.text:
            assert chunk.compressible is False


def test_verify_passes(compressed):
    assert compressed.warnings == []


def test_verify_passes_at_an_aggressive_rate(sample_markdown):
    result = grug.compress(sample_markdown, rate=0.2, backend="rules")
    assert result.warnings == []


@pytest.mark.parametrize(
    "negation", ["not automatic", "no data is deleted", "not exposed", "no refunds are needed"]
)
def test_negations_survive(compressed, negation):
    """Not the exact phrasing -- the negation word itself, in context."""
    head = negation.split()[0]
    tail = negation.split()[-1]
    assert head in compressed.text.lower()
    assert tail in compressed.text.lower()


@pytest.mark.parametrize(
    "number", ["2026-03-15", "1.2", "4,800", "9.6", "40%", "72", "0.02", "15%", "100"]
)
def test_numbers_survive(compressed, number):
    assert number in compressed.text


@pytest.mark.parametrize("entity", ["Acme Corporation", "Globex", "Platform Reliability"])
def test_named_entities_survive(compressed, entity):
    assert entity in compressed.text


def test_markdown_headings_survive(compressed):
    headings = [line for line in compressed.text.splitlines() if line.startswith("#")]
    assert len(headings) == 5
    assert headings[0].startswith("# ")


def test_inline_code_survives(compressed):
    assert "`--dry-run`" in compressed.text
    assert "`UNRESOLVED`" in compressed.text


def test_paragraph_structure_survives(compressed, sample_markdown):
    original_paragraphs = sample_markdown.count("\n\n")
    assert compressed.text.count("\n\n") == original_paragraphs


def test_the_pipeline_is_deterministic(sample_markdown):
    first = grug.compress(sample_markdown, rate=0.5, backend="rules")
    second = grug.compress(sample_markdown, rate=0.5, backend="rules")
    assert first.text == second.text


def test_compressor_matches_the_module_level_function(sample_markdown, compressed):
    comp = grug.Compressor(backend="rules")
    assert comp.compress(sample_markdown, rate=0.5).text == compressed.text


def test_round_trip_through_the_cli(cli_command, sample_markdown, tmp_path):
    import subprocess

    source = tmp_path / "doc.md"
    source.write_text(sample_markdown, encoding="utf-8")
    result = subprocess.run(
        [*cli_command, "compress", str(source), "--rate", "0.5", "--backend", "rules"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    written = (tmp_path / "doc.grug.md").read_text(encoding="utf-8")
    assert CODE_BLOCK in written


# -- the project's own README, the document that surfaced these bugs ---------

README = pathlib.Path(__file__).parent.parent / "README.md"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def _structure(text: str) -> dict[str, int]:
    import re

    return {
        "headings": len(re.findall(r"^#{1,6} ", text, re.M)),
        "table_rows": len(re.findall(r"^\|", text, re.M)),
        "list_items": len(re.findall(r"^- ", text, re.M)),
        "blockquotes": len(re.findall(r"^> ", text, re.M)),
        "fences": len(re.findall(r"^```", text, re.M)),
        "blank_lines": len(re.findall(r"^$", text, re.M)),
    }


def test_readme_structure_is_preserved_exactly(readme):
    """Regression: compressing the README used to destroy every table and heading."""
    result = grug.compress(readme, rate=0.5, backend="rules")
    assert _structure(result.text) == _structure(readme)


def test_readme_compresses_without_warnings(readme):
    result = grug.compress(readme, rate=0.5, backend="rules")
    assert result.warnings == []
    assert result.compressed_tokens < result.original_tokens


@pytest.mark.slow
def test_readme_structure_survives_the_classifier(readme):
    if not Lingua2Backend.is_available():
        pytest.skip("llmlingua not installed")
    result = grug.compress(readme, rate=0.5, backend="lingua2")
    assert _structure(result.text) == _structure(readme)
    assert losses(result.warnings) == [], result.warnings
