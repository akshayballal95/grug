"""Is this corpus good enough to train on?"""

import json
import pathlib
import statistics
import sys

from grug.training.alignment import filter_examples
from grug.training.distill import _retention

d = pathlib.Path(sys.argv[1])
pairs = [json.loads(x) for x in (d / "pairs.jsonl").read_text().splitlines() if x.strip()]
labels = [json.loads(x) for x in (d / "labels.jsonl").read_text().splitlines() if x.strip()]
print(f"{len(pairs)} pairs, {len(labels)} labelled\n")


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


ratios = [sum(x["labels"]) / max(1, len(x["labels"])) for x in labels]
vrs = [x["variation_rate"] for x in labels]
gaps = [x["alignment_gap"] for x in labels]
lens = [len(x["labels"]) for x in labels]

print("keep ratio (fraction of source words labelled keep)")
print(
    f"   mean {statistics.fmean(ratios):.3f}   p10 {pct(ratios, 0.1):.2f}  "
    f"median {pct(ratios, 0.5):.2f}  p90 {pct(ratios, 0.9):.2f}"
)
print("variation rate (invented words)")
print(
    f"   mean {statistics.fmean(vrs):.4f}   p50 {pct(vrs, 0.5):.4f}  "
    f"p95 {pct(vrs, 0.95):.4f}  max {max(vrs):.4f}"
)
print("alignment gap")
print(
    f"   mean {statistics.fmean(gaps):+.4f}  p50 {pct(gaps, 0.5):+.4f}  "
    f"p90 {pct(gaps, 0.9):+.4f}  max {max(gaps):+.4f}"
)
print(f"document length: median {pct(lens, 0.5)} words, max {max(lens)}\n")

neg = [
    r
    for r in (_retention(p["original"], p["compressed"], "negation") for p in pairs)
    if r is not None
]
num = [
    r
    for r in (_retention(p["original"], p["compressed"], "number") for p in pairs)
    if r is not None
]
print(
    f"negation retention  mean {statistics.fmean(neg):.3f}  "
    f"perfect on {sum(1 for x in neg if x == 1) / len(neg):.0%} of docs  (n={len(neg)})"
)
print(
    f"number retention    mean {statistics.fmean(num):.3f}  "
    f"perfect on {sum(1 for x in num if x == 1) / len(num):.0%} of docs  (n={len(num)})\n"
)

degenerate = [
    i
    for i, (p, r) in enumerate(zip(pairs, ratios))
    if r < 0.05 or r > 0.95 or not p["compressed"].strip()
]
print(f"degenerate documents (kept <5% or >95%): {len(degenerate)}")

stats = filter_examples(labels) if callable(filter_examples) else None
print()
print("after the paper's quality filters (drop worst 5% variation, worst 10% gap):")
try:
    kept = filter_examples(
        [
            type(
                "S",
                (),
                dict(
                    variation_rate=x["variation_rate"],
                    alignment_gap=x["alignment_gap"],
                    words=[],
                    labels=x["labels"],
                ),
            )()
            for x in labels
        ]
    )
    print(f"   {len(kept)} of {len(labels)} survive ({len(kept) / len(labels):.0%})")
except Exception as e:
    print("   (filter_examples signature differs:", str(e)[:70], ")")
