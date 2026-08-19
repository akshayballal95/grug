#!/usr/bin/env python3
"""Every installed backend on the same document, side by side.

    python examples/compare_backends.py
    python examples/compare_backends.py --rates 0.7 0.5 0.3

Backends whose dependencies are missing, or that need a checkpoint, are listed
and skipped, so this runs on a bare `pip install grugify` and gets richer as
you add extras. Pass --model to include the classifier backend.
"""

from __future__ import annotations

import argparse
import time

from _shared import rule, sample_doc

import grug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.8, 0.5, 0.3])
    parser.add_argument("--device", default="auto", help="device for backends that use one")
    parser.add_argument(
        "--model",
        metavar="CHECKPOINT",
        help="checkpoint for the classifier backend; without it the classifier sits out",
    )
    return parser.parse_args()


def build(name: str, device: str, model: str | None) -> grug.Compressor | None:
    """Construct a Compressor, or return None if the backend cannot run here."""
    for kwargs in ({"device": device, "model_name": model}, {"device": device}, {}):
        try:
            return grug.Compressor(backend=name, **{k: v for k, v in kwargs.items() if v})
        except grug.MissingDependencyError:
            return None
        except TypeError:
            continue  # backend does not take these options; try fewer
    return None


def main() -> int:
    args = parse_args()
    doc = sample_doc()
    original = grug.count_tokens(doc)

    rule("Backends")
    for row in grug.backend_info():
        status = (
            "available" if row["available"] else f"missing -> pip install 'grugify[{row['extra']}]'"
        )
        print(f"  {row['name']:<10} {status}")
    print(f"\n  Document: {original} tokens, tokenizer={grug.tokenizer_name()}")

    rule("Comparison")
    header = f"  {'backend':<10} {'rate':>5} {'tokens':>8} {'ratio':>7} {'secs':>7}  warnings"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name in grug.list_backends():
        comp = build(name, args.device, args.model)
        if comp is None:
            note = "missing dependencies, or needs a checkpoint (pass --model)"
            print(f"  {name:<10} {'-':>5} {'-':>8} {'-':>7} {'-':>7}  {note}")
            continue
        comp.compress("warm up", rate=0.5)  # keep model-load time out of the timings
        for rate in args.rates:
            started = time.perf_counter()
            result = comp.compress(doc, rate=rate)
            elapsed = time.perf_counter() - started
            note = "clean" if not result.warnings else f"{len(result.warnings)} WARN"
            print(
                f"  {name:<10} {rate:>5.2f} {result.compressed_tokens:>8} "
                f"{result.ratio:>7.2f} {elapsed:>7.2f}  {note}"
            )
            for warning in result.warnings:
                print(f"  {'':<10} {'':>5} -> {warning}")

    rule("Reading the table")
    print(
        "  ratio is what was ACHIEVED, not requested. 'clean' means the verifier\n"
        "  found no dropped negation, number, or name -- a smoke alarm, not a proof."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
