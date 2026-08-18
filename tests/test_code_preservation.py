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


# -- language coverage ------------------------------------------------------

LANGUAGES = {
    "python": "def total(items):\n    return sum(i.price for i in items)\n",
    "javascript": "function total(items) {\n  return items.reduce((a, b) => a + b.price, 0);\n}\n",
    "typescript": "export const total = (items: Item[]): number => {\n  return items.length;\n};\n",
    "rust": "pub fn total(items: &[Item]) -> f64 {\n    items.iter().map(|i| i.price).sum()\n}\n",
    "go": "func Total(items []Item) float64 {\n\tvar t float64\n\treturn t\n}\n",
    "java": "public class Billing {\n    private int total(List<Item> items) {\n        return 0;\n    }\n}\n",
    "c": '#include <stdio.h>\n\nint main(void) {\n    printf("hi");\n    return 0;\n}\n',
    "cpp": "template <typename T>\nT total(const std::vector<T>& v) {\n    return T{};\n}\n",
    "csharp": "public class Billing {\n    public int Total(List<Item> items) => items.Count;\n}\n",
    "ruby": "def total(items)\n  items.sum { |i| i.price }\nend\n",
    "php": "<?php\nfunction total(array $items): float {\n    return array_sum($items);\n}\n",
    "swift": "func total(_ items: [Item]) -> Double {\n    return items.reduce(0) { $0 + $1.price }\n}\n",
    "kotlin": "fun total(items: List<Item>): Double {\n    return items.sumOf { it.price }\n}\n",
    "scala": "def total(items: Seq[Item]): Double =\n  items.map(_.price).sum\n",
    "bash": '#!/bin/bash\nset -euo pipefail\nfor f in *.txt; do\n  echo "$f"\ndone\n',
    "sql": "SELECT id FROM accounts\nWHERE region = 'us-east-1';\n",
    "html": '<div class="card">\n  <p>Hello there</p>\n</div>\n',
    "css": ".card {\n  color: #333;\n  margin: 0 auto;\n}\n",
    "yaml": "name: grug\nversion: 0.1.0\ndeps:\n  - typer\n",
    "json": '{\n  "name": "grug",\n  "version": "0.1.0"\n}\n',
    "toml": '[project]\nname = "grug"\nrequires-python = ">=3.10"\n',
    "haskell": "total :: [Item] -> Double\ntotal items = sum (map price items)\n",
    "clojure": "(defn total [items]\n  (reduce + (map :price items)))\n",
    "r": "total <- function(items) {\n  sum(items$price)\n}\n",
    "lua": "function total(items)\n  return #items\nend\n",
    "perl": "sub total {\n    my @items = @_;\n    return scalar @items;\n}\n",
    "elixir": "def total(items) do\n  Enum.sum(items)\nend\n",
    "dart": "double total(List<Item> items) {\n  return items.length.toDouble();\n}\n",
    "terraform": 'resource "aws_s3_bucket" "b" {\n  bucket = "my-bucket"\n}\n',
    "makefile": "build:\n\tpython -m build\n\ntest:\n\tpytest -q\n",
}


@pytest.mark.parametrize("language", sorted(LANGUAGES))
def test_every_language_survives_verbatim(language):
    """The property that matters: no language comes back rewritten."""
    assert verbatim(LANGUAGES[language]), f"{language} was modified"


@pytest.mark.parametrize("language", sorted(set(LANGUAGES) - {"yaml"}))
def test_content_detection_covers_the_languages(language):
    """YAML is excluded: 'name: grug' is genuinely prose-shaped, so only the
    filename can identify it."""
    assert looks_like_code(LANGUAGES[language])


def test_yaml_needs_its_filename():
    assert not looks_like_code(LANGUAGES["yaml"])
    assert looks_like_code(LANGUAGES["yaml"], "config.yaml")


def test_prose_aside_in_parentheses_is_not_an_s_expression():
    """Regression: indented prose starting with '(' read as a Lisp form."""
    text = (
        "The verifier reports three things:\n"
        "   - Number loss, where a figure vanishes.\n"
        '     ("1,250" and "1250" count as one number.)\n'
        "   - Entity loss, where a name vanishes.\n"
        '     ("Bank of America" clipped to "Bank".)\n'
    )
    assert code_regions(text) == []
