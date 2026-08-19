"""Do the labels agree with the compression they were derived from?

If a teacher keeps 30% of a document, roughly 30% of source words should carry
a keep label. A large gap means the aligner lost the thread, not that the
teacher compressed hard.
"""

import json
import pathlib
import sys

d = pathlib.Path(sys.argv[1])
pairs = [json.loads(x) for x in (d / "pairs.jsonl").read_text().splitlines() if x.strip()]
labels = [json.loads(x) for x in (d / "labels.jsonl").read_text().splitlines() if x.strip()]

rows = []
for p, x in zip(pairs, labels, strict=True):
    n = len(x["labels"])
    out_words = len(p["compressed"].split())
    kept = sum(x["labels"])
    rows.append((n, out_words / max(1, n), kept / max(1, n), out_words, kept))

print(
    f"{'source words':>13} {'docs':>5} {'teacher ratio':>14} {'label ratio':>12} {'shortfall':>10}"
)
print("-" * 60)
buckets = [(0, 1000), (1000, 2500), (2500, 5000), (5000, 10000), (10000, 20000), (20000, 10**9)]
for lo, hi in buckets:
    sel = [r for r in rows if lo <= r[0] < hi]
    if not sel:
        continue
    tr = sum(r[3] for r in sel) / sum(r[0] for r in sel)
    lr = sum(r[4] for r in sel) / sum(r[0] for r in sel)
    name = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
    print(f"{name:>13} {len(sel):>5} {tr:>14.3f} {lr:>12.3f} {tr - lr:>10.3f}")

bad = [r for r in rows if r[1] - r[2] > 0.10]
print(f"\ndocuments where labels keep >10 points less than the teacher did: {len(bad)}/{len(rows)}")
print(f"they hold {sum(r[0] for r in bad) / sum(r[0] for r in rows):.0%} of all source words")

from grug.training.alignment import DEFAULT_WINDOW  # noqa: E402

print(f"\nalignment window: {DEFAULT_WINDOW}")
