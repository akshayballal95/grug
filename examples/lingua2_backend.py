#!/usr/bin/env python3
"""LLMLingua-2 backend: compression by token classification.

    pip install 'grug[lingua2]'
    python examples/lingua2_backend.py
    python examples/lingua2_backend.py --device cpu --rate 0.33

A BERT encoder scores every token for "keep or drop" and the top `rate`
fraction survives in original order. The first run downloads the checkpoint
(a few hundred MB) and takes a while; later runs load from the HF cache.

If the extra is not installed this script explains what to install and exits 0,
so it stays safe to run in CI.
"""

from __future__ import annotations

import argparse
import time

from _shared import rule, sample_doc, show

import grug
from grug.backends.lingua2 import DEFAULT_FORCE_TOKENS, DEFAULT_MODEL
from grug.base import MissingDependencyError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--rate", type=float, default=0.5, help="fraction of tokens to keep")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="any LLMLingua-2 checkpoint")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    doc = sample_doc()

    rule("0. Defaults that make this backend safe")
    print(f"  model        : {args.model}")
    print(f"  force_tokens : {list(DEFAULT_FORCE_TOKENS)}")
    print(
        "\n  Everything after '?' is a negation cue. A scorer trained on meeting\n"
        "  transcripts sees 'not' as a droppable three-letter function word; drop\n"
        "  it and the sentence asserts the opposite. These are pinned by default."
    )

    rule("1. Load the model")
    try:
        comp = grug.Compressor(backend="lingua2", device=args.device, model_name=args.model)
    except MissingDependencyError as exc:
        print(f"  skipped: {exc}")
        print("\n  Install the extra and re-run:\n    pip install 'grug[lingua2]'")
        return 0

    started = time.perf_counter()
    comp.compress("warm up", rate=0.5)  # the first call is what loads the model
    device = comp.backend.metadata_device
    print(f"  loaded in {time.perf_counter() - started:.1f}s on device={device}")

    rule(f"2. Compress at rate={args.rate}")
    started = time.perf_counter()
    result = comp.compress(doc, rate=args.rate)
    elapsed = time.perf_counter() - started
    show(result, f"[{elapsed:.1f}s]")
    print(
        f"\n  chunks: {result.metadata['chunks']}, "
        f"code blocks passed through: {result.metadata['code_blocks_preserved']}"
    )

    rule("3. Sweep the rate")
    for rate in (0.7, 0.5, 0.33, 0.2):
        started = time.perf_counter()
        swept = comp.compress(doc, rate=rate)
        print(
            f"  rate={rate:<5} -> {swept.compressed_tokens:>4} tokens "
            f"(ratio {swept.ratio:.2f}, {len(swept.warnings)} warnings, "
            f"{time.perf_counter() - started:.1f}s)"
        )

    rule("4. Negation survival under pressure")
    line = (
        "The finance team confirmed the headline result of the study: bills "
        "scale with volume, not price, across every plan we currently support."
    )
    for rate in (0.5, 0.3, 0.2):
        out = comp.compress(line, rate=rate)
        kept = "not" in out.text.lower().split()
        print(f"  rate={rate:<5} 'not' kept: {'yes' if kept else 'NO'} | {out.text}")

    rule("5. Same document, both backends")
    for name in ("rules", "lingua2"):
        started = time.perf_counter()
        out = grug.compress(doc, rate=args.rate, backend=name)
        print(
            f"  {name:<8} {out.compressed_tokens:>4} tokens (ratio {out.ratio:.2f}) "
            f"in {time.perf_counter() - started:.1f}s, {len(out.warnings)} warnings"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
