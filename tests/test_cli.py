"""CLI smoke tests, driven as a real subprocess so exit codes are real."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from grug.backends.lingua2 import Lingua2Backend

OK, ERROR, WARNINGS = 0, 1, 2

WORDY = (
    "It is important to note that the invoice totals will not include tax.\n"
    "We ran 3 tests on 1,250 accounts and saw a 12.5% improvement.\n"
)

# CLI mechanics are backend-agnostic. Pin the dependency-free backend so the
# suite never downloads a model and does not change behaviour when extras are
# installed. Tests that care about backend selection pass their own -b.
RULES = ("--backend", "rules")


def run(cli_command, *args, stdin=None, cwd=None):
    return subprocess.run(
        [*cli_command, *args],
        input=stdin,
        capture_output=True,
        text=True,
        cwd=cwd,
    )


@pytest.fixture
def doc(tmp_path):
    path = tmp_path / "input.md"
    path.write_text(WORDY, encoding="utf-8")
    return path


# -- compress ---------------------------------------------------------------


def test_writes_a_sibling_file_by_default(cli_command, doc):
    result = run(cli_command, "compress", *RULES, str(doc), "--rate", "0.5")
    assert result.returncode == OK
    output = doc.with_name("input.grug.md")
    assert output.is_file()
    assert len(output.read_text()) < len(WORDY)


def test_stats_go_to_stderr_not_stdout(cli_command, doc):
    result = run(cli_command, "compress", *RULES, str(doc))
    assert "tokens" in result.stderr
    assert "backend=" in result.stderr
    assert "→" in result.stderr
    assert result.stdout == ""


def test_quiet_suppresses_the_stats_line(cli_command, doc):
    result = run(cli_command, "compress", *RULES, str(doc), "-q")
    assert "tokens (" not in result.stderr


def test_explicit_output_path(cli_command, doc, tmp_path):
    target = tmp_path / "out.txt"
    assert run(cli_command, "compress", *RULES, str(doc), "-o", str(target)).returncode == OK
    assert target.is_file()


def test_stdin_to_stdout(cli_command):
    result = run(cli_command, "compress", *RULES, "-", "--rate", "0.5", stdin=WORDY)
    assert result.returncode == OK
    assert result.stdout.strip()
    assert "invoice totals" in result.stdout
    assert "It is important to note" not in result.stdout


def test_multiple_files(cli_command, tmp_path):
    first = tmp_path / "a.md"
    second = tmp_path / "b.md"
    first.write_text(WORDY, encoding="utf-8")
    second.write_text(WORDY, encoding="utf-8")

    result = run(cli_command, "compress", *RULES, str(first), str(second), "--rate", "0.5")
    assert result.returncode == OK
    assert (tmp_path / "a.grug.md").is_file()
    assert (tmp_path / "b.grug.md").is_file()
    assert "a.md:" in result.stderr and "b.md:" in result.stderr


def test_output_flag_rejects_multiple_inputs(cli_command, tmp_path):
    first = tmp_path / "a.md"
    second = tmp_path / "b.md"
    first.write_text(WORDY)
    second.write_text(WORDY)
    result = run(
        cli_command, "compress", *RULES, str(first), str(second), "-o", str(tmp_path / "o.md")
    )
    assert result.returncode == ERROR
    assert "single input file" in result.stderr


def test_json_emits_a_full_result(cli_command, doc):
    result = run(cli_command, "compress", *RULES, str(doc), "--json", "-q")
    assert result.returncode == OK
    payload = json.loads(result.stdout)
    assert payload["backend"]
    assert payload["original_tokens"] > payload["compressed_tokens"]
    assert payload["source"] == str(doc)
    assert 0 < payload["ratio"] <= 1.0
    assert not doc.with_name("input.grug.md").exists()


def test_json_emits_an_array_for_multiple_files(cli_command, tmp_path):
    for name in ("a.md", "b.md"):
        (tmp_path / name).write_text(WORDY, encoding="utf-8")
    result = run(
        cli_command, "compress", str(tmp_path / "a.md"), str(tmp_path / "b.md"), "--json", "-q"
    )
    payload = json.loads(result.stdout)
    assert isinstance(payload, list) and len(payload) == 2


def test_backend_flag_is_honoured(cli_command, doc):
    result = run(cli_command, "compress", str(doc), "--backend", "rules")
    assert result.returncode == OK
    assert "backend=rules" in result.stderr


def test_unknown_backend_exits_one(cli_command, doc):
    result = run(cli_command, "compress", str(doc), "--backend", "nope")
    assert result.returncode == ERROR
    assert "Unknown backend" in result.stderr


def test_missing_file_exits_one(cli_command, tmp_path):
    result = run(cli_command, "compress", *RULES, str(tmp_path / "ghost.md"))
    assert result.returncode == ERROR
    assert "no such file" in result.stderr


def test_out_of_range_rate_is_rejected(cli_command, doc):
    assert run(cli_command, "compress", *RULES, str(doc), "--rate", "2.0").returncode != OK


# A backend that deliberately eats negations, to exercise the exit-2 path.
LOSSY_BACKEND = """
from grug.base import CompressionResult, CompressorBackend
from grug.registry import register_backend


@register_backend
class Lossy(CompressorBackend):
    name = "lossy-test"

    def compress(self, text, rate=0.5, **kwargs):
        return CompressionResult.build(text, text.replace(" not", ""), self.name)


from grug.cli import app

app()
"""


def test_faithfulness_warning_exits_two(tmp_path):
    """Exit 2 lets CI gate on 'compressed, but the meaning may have shifted'."""
    doc = tmp_path / "n.txt"
    doc.write_text("The result was not reproducible.\n")
    result = run([sys.executable, "-c", LOSSY_BACKEND], "compress", str(doc), "-b", "lossy-test")
    assert result.returncode == WARNINGS
    assert "⚠" in result.stderr
    assert "negation" in result.stderr


def test_no_verify_suppresses_the_exit_two(tmp_path):
    doc = tmp_path / "n.txt"
    doc.write_text("The result was not reproducible.\n")
    result = run(
        [sys.executable, "-c", LOSSY_BACKEND],
        "compress",
        str(doc),
        "-b",
        "lossy-test",
        "--no-verify",
    )
    assert result.returncode == OK
    assert "⚠" not in result.stderr


def test_rules_backend_never_trips_the_verifier(cli_command, tmp_path):
    """By construction rules drops only function words, so it stays faithful."""
    doc = tmp_path / "n.txt"
    doc.write_text(
        "Acme Corporation did not ship 1,250 units, and no refund was issued "
        "without approval from the Platform Reliability team.\n"
    )
    result = run(cli_command, "compress", str(doc), "--rate", "0.1", "--backend", "rules")
    assert result.returncode == OK
    assert "⚠" not in result.stderr


def test_no_verify_skips_the_checks(cli_command, tmp_path):
    doc = tmp_path / "n.txt"
    doc.write_text("The result was not reproducible.\n")
    result = run(cli_command, "compress", *RULES, str(doc), "--no-verify", "--rate", "0.3")
    assert result.returncode == OK
    assert "⚠" not in result.stderr


def test_device_flag_on_a_backend_that_lacks_it(cli_command, doc):
    result = run(cli_command, "compress", str(doc), "--device", "cuda", "--backend", "rules")
    assert result.returncode == ERROR
    assert "does not accept --device" in result.stderr


# -- verify -----------------------------------------------------------------


def test_verify_clean_exits_zero(cli_command, tmp_path):
    original = tmp_path / "a.txt"
    original.write_text("Bills scale with volume, not price.\n")
    result = run(cli_command, "verify", str(original), str(original))
    assert result.returncode == OK
    assert "no faithfulness issues" in result.stderr


def test_verify_dropped_negation_exits_two(cli_command, tmp_path):
    original = tmp_path / "a.txt"
    compressed = tmp_path / "b.txt"
    original.write_text("Bills scale with volume, not price.\n")
    compressed.write_text("bills scale volume price\n")
    result = run(cli_command, "verify", str(original), str(compressed))
    assert result.returncode == WARNINGS
    assert "negation" in result.stderr
    assert "⚠" in result.stderr


def test_verify_json(cli_command, tmp_path):
    original = tmp_path / "a.txt"
    compressed = tmp_path / "b.txt"
    original.write_text("We shipped 42 units.\n")
    compressed.write_text("shipped units\n")
    result = run(cli_command, "verify", str(original), str(compressed), "--json")
    payload = json.loads(result.stdout)
    assert payload["warnings"]
    assert payload["original_tokens"] > payload["compressed_tokens"]


def test_verify_missing_file_exits_one(cli_command, tmp_path):
    result = run(cli_command, "verify", str(tmp_path / "x"), str(tmp_path / "y"))
    assert result.returncode == ERROR


# -- backends ---------------------------------------------------------------


def test_backends_lists_availability(cli_command):
    result = run(cli_command, "backends")
    assert result.returncode == OK
    assert "rules" in result.stdout
    assert "lingua2" in result.stdout
    assert "*" in result.stdout


def test_backends_names_the_extra_for_uninstalled_deps(cli_command):
    if Lingua2Backend.is_available():
        pytest.skip("llmlingua is installed")
    result = run(cli_command, "backends")
    assert "pip install 'grug[lingua2]'" in result.stdout


def test_backends_json(cli_command):
    result = run(cli_command, "backends", "--json")
    rows = {row["name"]: row for row in json.loads(result.stdout)}
    assert rows["rules"]["available"] is True
    assert sum(row["default"] for row in rows.values()) == 1


# -- misc -------------------------------------------------------------------


def test_version(cli_command):
    result = run(cli_command, "--version")
    assert result.returncode == OK
    assert result.stdout.startswith("grug ")


def test_bare_invocation_shows_help(cli_command):
    result = run(cli_command)
    assert "compress" in result.stdout + result.stderr


def test_console_script_is_installed():
    """The packaged `grug` entry point resolves, not just `python -m`."""
    import shutil
    from pathlib import Path

    candidate = Path(sys.executable).parent / "grug"
    executable = str(candidate) if candidate.exists() else shutil.which("grug")
    if executable is None:
        pytest.skip("console script not installed in this environment")
    result = subprocess.run([executable, "--version"], capture_output=True, text=True)
    assert result.returncode == OK
    assert result.stdout.startswith("grug ")


# A backend that blows up, to exercise the per-file error path.
RAISING_BACKEND = """
from grug.base import CompressorBackend
from grug.registry import register_backend


@register_backend
class Boom(CompressorBackend):
    name = "boom-test"

    def compress(self, text, rate=0.5, **kwargs):
        raise RuntimeError("model exploded")


from grug.cli import app

app()
"""


def test_backend_failure_is_reported_per_file_and_exits_one(tmp_path):
    good = tmp_path / "a.txt"
    good.write_text("some text\n")
    result = run([sys.executable, "-c", RAISING_BACKEND], "compress", str(good), "-b", "boom-test")
    assert result.returncode == ERROR
    assert "model exploded" in result.stderr
    assert str(good) in result.stderr


def test_one_bad_file_does_not_abort_the_others(cli_command, tmp_path):
    """A missing file is reported, but the readable ones still get compressed."""
    good = tmp_path / "good.md"
    good.write_text(WORDY, encoding="utf-8")
    result = run(cli_command, "compress", *RULES, str(tmp_path / "ghost.md"), str(good))
    assert result.returncode == ERROR
    assert "no such file" in result.stderr
    assert (tmp_path / "good.grug.md").is_file()


def test_directory_input_says_so(cli_command, tmp_path):
    result = run(cli_command, "compress", *RULES, str(tmp_path))
    assert result.returncode == ERROR
    assert "is a directory" in result.stderr


# -- question conditioning --------------------------------------------------


def test_compress_help_documents_the_question_flag(cli_command):
    out = run(cli_command, "compress", "--help")
    assert "--question" in out.stdout


def test_backends_lists_the_question_aware_backend(cli_command):
    out = run(cli_command, "backends")
    assert "longlingua" in out.stdout


def test_a_question_on_a_plain_backend_warns_and_exits_two(cli_command, doc):
    """Silently dropping the question would be the worst of the options."""
    out = run(cli_command, "compress", str(doc), *RULES, "--question", "does it include tax?")
    assert "question ignored" in out.stderr
    assert "rules" in out.stderr
    assert out.returncode == WARNINGS


def test_the_short_question_flag_works(cli_command, doc):
    out = run(cli_command, "compress", str(doc), *RULES, "-Q", "does it include tax?")
    assert "question ignored" in out.stderr


def test_no_question_produces_no_question_warning(cli_command, doc):
    out = run(cli_command, "compress", str(doc), *RULES)
    assert "question ignored" not in out.stderr


def test_source_files_pass_through_unchanged(cli_command, tmp_path):
    """Compressing code rewrites the program, so grug refuses by default."""
    src = tmp_path / "calc.py"
    src.write_text("import os\n\n\ndef total(items):\n    return sum(i.price for i in items)\n")
    result = run(cli_command, "compress", str(src), *RULES, "--rate", "0.3")
    assert result.returncode == OK
    assert "passed through unchanged" in result.stderr
    assert (tmp_path / "calc.grug.py").read_text() == src.read_text()


def test_compress_code_overrides_the_guard(cli_command, tmp_path):
    src = tmp_path / "calc.py"
    src.write_text("import os\n\n\ndef total(items):\n    return sum(i.price for i in items)\n")
    result = run(cli_command, "compress", str(src), *RULES, "--rate", "0.3", "--compress-code")
    assert result.returncode == OK
    assert "passed through unchanged" not in result.stderr


def test_prose_is_still_compressed_alongside_code(cli_command, tmp_path):
    code = tmp_path / "a.py"
    prose = tmp_path / "b.md"
    code.write_text("def f(x):\n    return x\n")
    prose.write_text("It is important to note that the build did not pass on 3 of 12 runs.\n")
    result = run(cli_command, "compress", str(code), str(prose), *RULES, "--rate", "0.4")
    assert result.returncode == OK
    assert (tmp_path / "a.grug.py").read_text() == code.read_text()
    assert len((tmp_path / "b.grug.md").read_text()) < len(prose.read_text())
