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
import os
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer

from . import __version__
from .base import CompressionResult, MissingDependencyError, tokenizer_name
from .chunking import DEFAULT_CHUNK_TOKENS, looks_like_code
from .registry import (
    BackendNotFoundError,
    backend_info,
    create_backend,
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
    question: Annotated[
        str | None,
        typer.Option(
            "--question",
            "-Q",
            help="What the output must stay sufficient to answer. Selects a "
            "question-aware backend unless -b names one.",
        ),
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
    skip_code: Annotated[
        bool,
        typer.Option(
            "--skip-code/--compress-code",
            help="Pass source files through unchanged rather than compressing them.",
        ),
    ] = True,
) -> None:
    """Compress one or more documents.

    Source files are passed through unchanged, because compressing code
    rewrites the program. ``--compress-code`` overrides that.
    """
    if output is not None and output != STDIO and len(files) > 1:
        _err("error: -o/--output takes a single input file; omit it to write <name>.grug.<ext>")
        raise typer.Exit(EXIT_ERROR)

    try:
        compressor = _build_compressor(backend, device, verify, chunk_tokens, question)
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

        # The filename is the strongest signal available: a .sql file or a
        # docstring-heavy .py does not read as code by content alone, and
        # compressing either rewrites the program itself.
        if skip_code and looks_like_code(text, None if source == STDIO else source):
            _err(f"{source}: source code, passed through unchanged")
            try:
                _write_output(text, source, output)
            except OSError as exc:
                _err(f"error: {exc}")
                had_error = True
            continue

        started = time.perf_counter()
        try:
            result = compressor.compress(text, rate=rate, question=question)
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
    backend: str | None,
    device: str | None,
    verify: bool,
    chunk_tokens: int,
    question: str | None = None,
) -> Any:
    """Resolve the backend and wrap it. A question only steers an unnamed one."""
    from . import Compressor

    name = backend or default_backend_name(question=bool(question))
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


train_app = typer.Typer(
    name="train",
    help="Reproduce the compressor: corpus, labels, fine-tuned encoder. Needs the train extra.",
    no_args_is_help=True,
)
app.add_typer(train_app)


def _training(module: str) -> Any:
    """Import a training module, or explain which extra installs it."""
    import importlib

    try:
        return importlib.import_module(f"grug.training.{module}")
    except ImportError as exc:
        _err(f"error: grug train needs the training extra: pip install 'grug[train]'  ({exc})")
        raise typer.Exit(EXIT_ERROR) from exc


@train_app.command("prepare")
def train_prepare(
    out: Annotated[
        str, typer.Option("--out", "-o", help="Directory for the JSONL shards.")
    ] = "data",
    dataset: Annotated[
        str, typer.Option("--dataset", help="Hugging Face dataset id.")
    ] = "microsoft/MeetingBank-LLMCompressed",
    limit: Annotated[
        int | None, typer.Option("--limit", help="Stop after N rows (smoke runs).")
    ] = None,
    val_fraction: Annotated[float, typer.Option("--val-fraction", min=0.0, max=0.5)] = 0.05,
    from_hub: Annotated[
        str | None,
        typer.Option("--from-hub", help="Download cached labels instead of deriving them."),
    ] = None,
    push_to_hub: Annotated[
        str | None,
        typer.Option("--push-to-hub", help="Publish the derived labels as a dataset."),
    ] = None,
) -> None:
    """Get per-word keep/drop labels, from the cache or by deriving them.

    Deriving costs about eight minutes of single-threaded alignment, which on a
    rented GPU is eight minutes of paying for an idle card. ``--from-hub`` skips
    it, and falls back to deriving if the cache is missing.
    """
    data = _training("data")

    if from_hub:
        try:
            summary = data.pull_prepared(from_hub, out)
            _err(f"labels from cache {from_hub}: {summary['train']} train, {summary['val']} val")
            json.dump(summary, sys.stdout, indent=2)
            sys.stdout.write("\n")
            raise typer.Exit(EXIT_OK)
        except typer.Exit:
            raise
        except Exception as exc:
            _err(f"cache {from_hub} unavailable ({type(exc).__name__}); deriving instead")

    summary = data.prepare(out, dataset=dataset, limit=limit, val_fraction=val_fraction)
    _err(
        f"{summary['pairs_kept']}/{summary['pairs_aligned']} pairs kept, "
        f"{summary['words']} words, keep rate {summary['keep_rate']:.2f} -> {out}"
    )
    if push_to_hub:
        _err(f"published: {data.push_prepared(out, push_to_hub)}")
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")


@train_app.command("run")
def train_run(
    data: Annotated[
        str, typer.Option("--data", "-d", help="Directory written by 'train prepare'.")
    ] = "data",
    out: Annotated[str, typer.Option("--out", "-o", help="Where to save the checkpoint.")] = "out",
    model: Annotated[
        str, typer.Option("--model", "-m", help="Base encoder to fine-tune.")
    ] = "answerdotai/ModernBERT-base",
    epochs: Annotated[int, typer.Option("--epochs", min=1)] = 10,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 10,
    learning_rate: Annotated[float, typer.Option("--lr")] = 1e-5,
    max_length: Annotated[int, typer.Option("--max-length", min=16)] = 512,
    device: Annotated[str, typer.Option("--device")] = "auto",
    cs_weight: Annotated[
        float,
        typer.Option(
            "--cs-weight", help="MOOSComp inter-class cosine loss weight. 0 reproduces LLMLingua-2."
        ),
    ] = 0.0,
    push_to: Annotated[
        str | None, typer.Option("--push-to", help="Hub repo to stream per-epoch metrics to.")
    ] = None,
    select_on: Annotated[
        str, typer.Option("--select-on", help="Metric the saved checkpoint is chosen by.")
    ] = "val_f1",
    patience: Annotated[
        int, typer.Option("--patience", min=0, help="Stop after N epochs without gain. 0 = never.")
    ] = 0,
) -> None:
    """Fine-tune an encoder into a preserve/discard token classifier."""
    trainer = _training("trainer")
    config = trainer.TrainConfig(
        model_name=model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        device=device,
        cs_weight=cs_weight,
        push_to=push_to,
        select_on=select_on,
        patience=patience,
    )
    metrics = trainer.train(data, out, config)
    _err(f"saved checkpoint to {out}")
    json.dump(metrics["history"][-1] if metrics["history"] else {}, sys.stdout, indent=2)
    sys.stdout.write("\n")


@train_app.command("watch")
def train_watch(
    repo: Annotated[str, typer.Argument(help="Hub repo the run is streaming metrics to.")],
    interval: Annotated[int, typer.Option("--interval", min=5, help="Seconds between polls.")] = 60,
    once: Annotated[
        bool, typer.Option("--once", help="Print the current metrics and exit.")
    ] = False,
) -> None:
    """Follow a training run's per-epoch metrics from the Hub.

    Remote training is otherwise a black box: the checkpoint only appears at the
    end. The trainer uploads metrics.json each epoch, and this tails it.
    """
    import time

    from huggingface_hub import hf_hub_download

    seen = 0
    while True:
        try:
            path = hf_hub_download(
                repo_id=repo,
                filename="metrics.json",
                repo_type="model",
                token=os.environ.get("HF_TOKEN"),
                force_download=True,
            )
            metrics = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            _err(f"waiting for metrics ({type(exc).__name__})")
            metrics = None

        if metrics:
            total = metrics.get("config", {}).get("epochs", "?")
            for entry in metrics.get("history", [])[seen:]:
                fields = "  ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in entry.items()
                    if k != "epoch"
                )
                _err(f"epoch {entry.get('epoch')}/{total}  {fields}")
            seen = len(metrics.get("history", []))
            if seen and seen >= (total if isinstance(total, int) else 0):
                _err("training complete")
                json.dump(metrics, sys.stdout, indent=2)
                sys.stdout.write("\n")
                raise typer.Exit(EXIT_OK)
        if once:
            raise typer.Exit(EXIT_OK)
        time.sleep(interval)


@train_app.command("evaluate")
def train_evaluate(
    model: Annotated[str, typer.Option("--model", "-m", help="Checkpoint directory or hub id.")],
    files: Annotated[list[str] | None, typer.Argument(help="Documents to evaluate on.")] = None,
    device: Annotated[str, typer.Option("--device")] = "auto",
    out: Annotated[
        str | None, typer.Option("--out", "-o", help="Write the JSON report here.")
    ] = None,
) -> None:
    """Measure achieved ratio and faithfulness for a checkpoint."""
    evaluate = _training("evaluate")
    documents = [_read(f) for f in (files or [])]
    if not documents:
        _err("error: pass at least one document to evaluate on")
        raise typer.Exit(EXIT_ERROR)
    report = evaluate.evaluate_checkpoint(model, documents, device=device, out_file=out)
    for row in report["rows"]:
        _err(
            f"rate={row['rate']:.2f}  ratio={row['ratio']:.2f}  "
            f"clean={row['clean_fraction']:.2f}  {row['seconds_per_doc']:.2f}s/doc"
        )
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")


bench_app = typer.Typer(
    name="benchmark",
    help="Measure whether a compressed prompt still answers questions. Needs the bench extra.",
    no_args_is_help=True,
)
app.add_typer(bench_app)


def _benchmark(module: str) -> Any:
    import importlib

    try:
        return importlib.import_module(f"grug.benchmark.{module}")
    except ImportError as exc:
        _err(f"error: grug benchmark needs: pip install 'grug[bench]'  ({exc})")
        raise typer.Exit(EXIT_ERROR) from exc


def _make_backend(spec: str) -> tuple[str, Any]:
    """``rules`` or ``modern:hub/model-id`` -> (display name, backend)."""
    name, _, model_id = spec.partition(":")
    if not model_id:
        return name, create_backend(name)
    from .backends.modern import ModernBackend

    label = model_id.split("/")[-1]
    for prefix in ("grug-",):
        label = label.removeprefix(prefix)
    for suffix in ("-meetingbank",):
        label = label.removesuffix(suffix)
    return label, ModernBackend(model_name=model_id, device="cpu")


@bench_app.command("qa")
def benchmark_qa(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="litellm model id, e.g. bedrock/moonshotai.kimi-k2.5"),
    ],
    backends: Annotated[
        str,
        typer.Option("--backends", "-b", help="Comma list. 'rules', 'lingua2', 'modern:hub/id'."),
    ] = "rules,lingua2",
    rates: Annotated[str, typer.Option("--rates", help="Comma list of rates.")] = "0.5,0.33",
    limit: Annotated[int, typer.Option("--limit", min=1, help="Contexts to evaluate.")] = 25,
    out: Annotated[str, typer.Option("--out", "-o", help="Directory for results.")] = "benchmarks",
    workers: Annotated[int, typer.Option("--workers", min=1, help="Concurrent LLM calls.")] = 8,
    metric: Annotated[
        str, typer.Option("--metric", help="Metric plotted on the chart.")
    ] = "exact_match",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Use a stub LLM; make no API calls.")
    ] = False,
) -> None:
    """Compress contexts, answer questions about them, and score the answers."""
    qa = _benchmark("qa")
    runner = _benchmark("runner")
    report = _benchmark("report")
    llm = _benchmark("llm")

    examples = qa.load_qa(limit=limit)
    _err(f"loaded {len(examples)} contexts, {sum(len(e.questions) for e in examples)} questions")

    built: dict[str, Any] = {}
    for spec in [s for s in backends.split(",") if s.strip()]:
        try:
            label, backend = _make_backend(spec.strip())
            built[label] = backend
        except Exception as exc:
            _err(f"error: backend {spec!r}: {exc}")
            raise typer.Exit(EXIT_ERROR) from exc

    client = llm.StubClient() if dry_run else llm.LLMClient(model=model, workers=workers)
    rate_list = [float(r) for r in rates.split(",") if r.strip()]

    rows = runner.run_benchmark(examples, built, rate_list, client)

    directory = Path(out)
    runner.save(rows, directory / "results.json")
    report.to_csv(rows, directory / "results.csv")
    report.to_svg(rows, directory / "results.svg", metric=metric)
    _err(f"wrote {directory}/results.json, results.csv, results.svg")

    json.dump([r.__dict__ for r in rows], sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def main() -> None:
    """Console-script shim, for ``python -m grug.cli``."""
    app()


if __name__ == "__main__":
    main()
