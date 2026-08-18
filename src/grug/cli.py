"""Command-line interface.

    grug compress notes.md --rate 0.4
    grug compress - < in.txt > out.txt
    grug verify original.txt compressed.txt
    grug backends

Exit codes are meant to be gated on in CI:

* ``0`` -- success, no faithfulness warnings
* ``1`` -- error (bad file, unknown backend, missing dependency)
* ``2`` -- success, but the verifier flagged something
"""

from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer

from . import __version__
from .base import CompressionResult, MissingDependencyError, tokenizer_name
from .chunking import DEFAULT_CHUNK_TOKENS
from .registry import (
    BackendNotFoundError,
    backend_info,
    default_backend_name,
    get_backend_class,
)
from .verify import verify as run_verify

__all__ = ["app", "main"]

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_WARNINGS = 2

STDIO = "-"

app = typer.Typer(
    name="grug",
    help="Shrink documents into terse caveman text so prompts cost fewer tokens.",
    no_args_is_help=True,
    add_completion=False,
)


def _err(message: str) -> None:
    print(message, file=sys.stderr)


def _read(source: str) -> str:
    if source == STDIO:
        return sys.stdin.read()
    path = Path(source)
    if path.is_dir():
        raise IsADirectoryError(f"{source} is a directory, not a file")
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {source}")
    return path.read_text(encoding="utf-8")


def _default_output(source: str) -> Path:
    """``docs/note.md`` -> ``docs/note.grug.md``; extensionless keeps the suffix."""
    path = Path(source)
    suffix = path.suffix or ".txt"
    return path.with_name(f"{path.stem}.grug{suffix}")


def _stats_line(result: CompressionResult, elapsed: float, label: str | None = None) -> str:
    prefix = f"{label}: " if label else ""
    return (
        f"{prefix}{result.original_tokens} → {result.compressed_tokens} tokens "
        f"({result.ratio:.2f}, backend={result.backend}, {elapsed:.1f}s)"
    )


def _print_warnings(warnings: list[str], label: str | None = None) -> None:
    prefix = f"{label}: " if label else ""
    for warning in warnings:
        _err(f"⚠ {prefix}{warning}")


def _version_callback(value: bool) -> None:
    if value:
        print(f"grug {__version__}")
        raise typer.Exit(EXIT_OK)


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """grug: caveman-style text compression for LLM prompts."""


@app.command()
def compress(
    files: Annotated[
        list[str],
        typer.Argument(
            metavar="FILES...",
            help="Input files. Use '-' to read stdin and write stdout.",
        ),
    ],
    output: Annotated[
        str | None,
        typer.Option(
            "-o", "--output", help="Write here instead of <name>.grug.<ext>. '-' is stdout."
        ),
    ] = None,
    rate: Annotated[
        float,
        typer.Option("--rate", "-r", min=0.01, max=1.0, help="Fraction of tokens to keep."),
    ] = 0.5,
    backend: Annotated[
        str | None,
        typer.Option("--backend", "-b", help="Backend name. Default: best installed."),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Backend device hint, e.g. cpu, cuda, mps, auto."),
    ] = None,
    verify: Annotated[
        bool, typer.Option("--verify/--no-verify", help="Run faithfulness checks.")
    ] = True,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit full results as JSON on stdout; write no output files."),
    ] = False,
    chunk_tokens: Annotated[
        int, typer.Option("--chunk-tokens", min=16, help="Max tokens per chunk.")
    ] = DEFAULT_CHUNK_TOKENS,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress the stats line.")] = False,
) -> None:
    """Compress one or more documents."""
    if output is not None and output != STDIO and len(files) > 1:
        _err("error: -o/--output takes a single input file; omit it to write <name>.grug.<ext>")
        raise typer.Exit(EXIT_ERROR)

    try:
        compressor = _build_compressor(backend, device, verify, chunk_tokens)
    except (BackendNotFoundError, MissingDependencyError, TypeError) as exc:
        _err(f"error: {exc}")
        raise typer.Exit(EXIT_ERROR) from exc

    label_results = len(files) > 1
    payloads: list[dict[str, Any]] = []
    had_error = False
    had_warnings = False

    for source in files:
        try:
            text = _read(source)
        except (OSError, UnicodeDecodeError) as exc:
            _err(f"error: {exc}")
            had_error = True
            continue

        started = time.perf_counter()
        try:
            result = compressor.compress(text, rate=rate)
        except Exception as exc:  # a backend blowing up is a per-file failure
            _err(f"error: {source}: {exc}")
            had_error = True
            continue
        elapsed = time.perf_counter() - started

        label = source if label_results else None
        if not quiet:
            _err(_stats_line(result, elapsed, label))
        _print_warnings(result.warnings, label)
        had_warnings = had_warnings or bool(result.warnings)

        if as_json:
            payload = result.to_dict()
            payload["source"] = source
            payloads.append(payload)
            continue

        try:
            _write_output(result.text, source, output)
        except OSError as exc:
            _err(f"error: {exc}")
            had_error = True

    if as_json and payloads:
        json.dump(payloads[0] if len(payloads) == 1 else payloads, sys.stdout, indent=2)
        sys.stdout.write("\n")

    raise typer.Exit(EXIT_ERROR if had_error else EXIT_WARNINGS if had_warnings else EXIT_OK)


def _build_compressor(
    backend: str | None, device: str | None, verify: bool, chunk_tokens: int
) -> Any:
    from . import Compressor

    name = backend or default_backend_name()
    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    if kwargs and not _accepts(get_backend_class(name), *kwargs):
        raise TypeError(f"backend {name!r} does not accept --device")
    return Compressor(name, verify=verify, chunk_tokens=chunk_tokens, **kwargs)


def _accepts(cls: type, *names: str) -> bool:
    """Whether ``cls.__init__`` takes all of ``names`` as keyword arguments."""
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return all(name in params for name in names)


def _write_output(text: str, source: str, output: str | None) -> None:
    """Route compressed text to stdout or to a file, defaulting beside the input."""
    destination = output if output is not None else (STDIO if source == STDIO else None)
    if destination == STDIO:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        return
    path = Path(destination) if destination is not None else _default_output(source)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    _err(f"wrote {path}")


@app.command()
def verify(
    original: Annotated[str, typer.Argument(help="Path to the original document, or '-'.")],
    compressed: Annotated[str, typer.Argument(help="Path to the compressed document, or '-'.")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit warnings as JSON on stdout.")
    ] = False,
) -> None:
    """Run faithfulness checks on an already-compressed document."""
    try:
        original_text = _read(original)
        compressed_text = _read(compressed)
    except (OSError, UnicodeDecodeError) as exc:
        _err(f"error: {exc}")
        raise typer.Exit(EXIT_ERROR) from exc

    result = CompressionResult.build(
        original_text,
        compressed_text,
        "verify",
        warnings=run_verify(original_text, compressed_text),
    )

    if as_json:
        payload = result.to_dict()
        del payload["text"], payload["backend"], payload["metadata"]
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _err(f"{result.original_tokens} → {result.compressed_tokens} tokens ({result.ratio:.2f})")
        _print_warnings(result.warnings)
        if not result.warnings:
            _err("✓ no faithfulness issues found")

    raise typer.Exit(EXIT_WARNINGS if result.warnings else EXIT_OK)


@app.command()
def backends(
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the table as JSON on stdout.")
    ] = False,
) -> None:
    """List registered backends and whether their dependencies are installed."""
    info = backend_info()
    default = default_backend_name()

    if as_json:
        for row in info:
            row["default"] = row["name"] == default
        json.dump(info, sys.stdout, indent=2)
        sys.stdout.write("\n")
        raise typer.Exit(EXIT_OK)

    width = max((len(row["name"]) for row in info), default=4)
    for row in info:
        status = "ready" if row["available"] else f"needs: pip install 'grug[{row['extra']}]'"
        marker = "*" if row["name"] == default else " "
        print(f"{marker} {row['name']:<{width}}  {status:<38}  {row['description']}")
    print(f"\n* = default backend. Token counts use {_tokenizer_note()}.")
    raise typer.Exit(EXIT_OK)


def _tokenizer_note() -> str:
    name = tokenizer_name()
    return (
        name if name != "whitespace" else "whitespace splitting (install tiktoken for cl100k_base)"
    )


def main() -> None:
    """Console-script shim, for ``python -m grug.cli``."""
    app()


if __name__ == "__main__":
    main()
