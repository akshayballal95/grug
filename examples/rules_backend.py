#!/usr/bin/env python3
"""Rules backend: deterministic, dependency-free compression.

    python examples/rules_backend.py

Runs everywhere -- no model, no download, no torch. This is the backend you get
from a plain `pip install grug`.
"""

from __future__ import annotations

from _shared import rule, sample_doc, show

import grug
from grug.chunking import FENCE_RE


def main() -> int:
    doc = sample_doc()

    rule("1. Compress at three rates")
    for rate in (0.9, 0.6, 0.3):
        result = grug.compress(doc, rate=rate, backend="rules")
        print(
            f"  rate={rate:<4} -> {result.compressed_tokens:>4} tokens "
            f"(ratio {result.ratio:.2f}, {len(result.warnings)} warnings)"
        )
    print(
        "\n  The ratio bottoms out above the requested rate: rules only deletes\n"
        "  words from curated lists, and the code block (70 of 406 tokens) is\n"
        "  never compressed. It reports what it achieved, not what you asked for."
    )

    rule("2. Full output at rate=0.5")
    show(grug.compress(doc, rate=0.5, backend="rules"))

    rule("3. What it refuses to drop, even at rate=0.05")
    result = grug.compress(doc, rate=0.05, backend="rules")
    checks = {
        "negations": ["not", "no", "none", "without"],
        "numbers": ["2026-02-11", "5,000", "200", "1,000", "18", "47"],
        "entities": ["Acme Corporation", "Globex", "Platform Reliability"],
        "inline code": ["`--offline-batch`"],
        "urls": ["https://example.com/runbooks/backfill"],
    }
    for kind, needles in checks.items():
        kept = [n for n in needles if n in result.text or n in result.text.split()]
        status = "OK  " if len(kept) == len(needles) else "MISS"
        print(f"  {status} {kind:<12} {len(kept)}/{len(needles)} preserved")

    code = FENCE_RE.search(doc).group(0)
    print(f"  {'OK  ' if code in result.text else 'MISS'} code block   passed through verbatim")
    print(
        f"  {'OK  ' if not result.warnings else 'MISS'} verifier     {len(result.warnings)} warnings"
    )

    rule("4. Reusable Compressor and batching")
    comp = grug.Compressor(backend="rules")
    docs = [
        "It is important to note that the build did not pass on 3 of 12 runs.",
        "Please note that the migration is not automatic for legacy accounts.",
    ]
    for source, result in zip(docs, comp.compress_batch(docs, rate=0.5), strict=True):
        print(f"  in : {source}")
        print(f"  out: {result.text}   (ratio {result.ratio:.2f})\n")

    rule("5. Tuning: keep_words and drop_pleasantries")
    from grug.backends.rules import RulesBackend

    line = "Please note that the report is available on the dashboard."
    print(f"  default        : {RulesBackend().compress(line, rate=0.4).text}")
    print(f"  keep 'the'     : {RulesBackend(keep_words={'the'}).compress(line, rate=0.4).text}")
    print(
        f"  keep filler    : {RulesBackend(drop_pleasantries=False).compress(line, rate=0.95).text}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
