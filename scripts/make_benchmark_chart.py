"""Chart the trade-off with one classifier rather than four.

Four near-identical points crowd the plot and imply a comparison between them
that 600 questions cannot support. mbert-control is the one shown: best exact
match and best F1 of the four.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from grug.benchmark.report import to_svg

SRC = Path("benchmarks/sonnet46/results.json")
KEEP = {"original", "rules", "lingua2", "mbert-control"}

rows = json.loads(SRC.read_text(encoding="utf-8"))["rows"]
chosen = [r for r in rows if r["backend"] in KEEP]
missing = KEEP - {r["backend"] for r in chosen}
if missing:
    raise SystemExit(f"missing rows: {sorted(missing)}")

# Name it for what it is in the chart, not for the ablation it started as.
for r in chosen:
    if r["backend"] == "mbert-control":
        r["backend"] = "grug (mBERT)"

order = {"original": 0, "rules": 1, "grug (mBERT)": 2, "lingua2": 3}
chosen.sort(key=lambda r: order[r["backend"]])

for metric, name in (("exact_match", "em"), ("f1", "f1")):
    out = to_svg(
        chosen,
        SRC.parent / f"chart-{name}.svg",
        metric=metric,
        title=f"{'Exact match' if metric == 'exact_match' else 'F1'} vs compression"
        " (Sonnet 4.6, 600 questions)",
    )
    print("wrote", out)

for r in chosen:
    print(
        f"   {r['backend']:<14} ratio={r['ratio']:.3f} EM={r['exact_match']:.3f} "
        f"F1={r['f1']:.3f} neg_loss={r['negation_loss_rate']:.2f}"
    )
