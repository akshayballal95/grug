"""Build the comparison data using the real aligner the training pipeline uses."""

import json
import pathlib
import re

from grug.training.alignment import annotate, split_words
from grug.training.distill import _retention, strip_envelope

root = pathlib.Path(
    "/Users/akshayballal/Developer/Projects/Starlight/grug"
    "/.claude/worktrees/negation-metric-fix/benchmarks"
)
main = json.load(open(root / "teachers-corrected-metric.json"))
son = json.load(open(root / "teachers-corrected-metric-sonnet.json"))

TOP = {}
for t in main["teachers"]:
    for key, tag in (
        ("gemini", "Gemini 3.7 Flash"),
        ("haiku", "Haiku 4.5"),
        ("kimi", "Kimi K2.5"),
    ):
        if key in t["model"]:
            TOP[tag] = t["metadata"]["outputs"]
for t in son["teachers"]:
    if t["model"].endswith("sonnet-4-6") and t["metadata"]["outputs"]:
        TOP["Sonnet 4.6"] = t["metadata"]["outputs"]

ORDER = ["Gemini 3.7 Flash", "Sonnet 4.6", "Haiku 4.5", "Kimi K2.5"]
PICK = [9, 2, 1]

# One tokenizer for both sides, or "novel" just measures tokenizer disagreement.
TOKEN = re.compile(r"[a-z0-9']+")


def toks(s):
    return TOKEN.findall(s.lower())


out = {"passages": [], "totals": []}

for name in ORDER:
    novel_all, kept_all, src_all = 0, 0, 0
    for i, src in enumerate(main["passages"]):
        clean = strip_envelope(TOP[name][i][0])
        vocab = set(toks(src))
        novel_all += sum(1 for w in toks(clean) if w not in vocab)
        kept_all += len(clean.split())
        src_all += len(src.split())
    out["totals"].append({"name": name, "novel": novel_all, "ratio": kept_all / src_all})

for idx in PICK:
    src = main["passages"][idx]
    vocab = set(toks(src))
    entry = {"index": idx, "words": split_words(src), "models": []}
    for name in ORDER:
        raw = TOP[name][idx][0]
        clean = strip_envelope(raw)
        stats = annotate(src, clean)
        novel = [w for w in toks(clean) if w not in vocab]
        entry["models"].append(
            {
                "name": name,
                "labels": stats.labels,
                "compressed": clean,
                "envelope": raw.strip()[: raw.strip().find("\n")] if clean != raw.strip() else "",
                "ratio": len(clean.split()) / len(src.split()),
                "variation": stats.variation_rate,
                "gap": stats.alignment_gap,
                "negation": _retention(src, clean, "negation"),
                "novel": sorted(set(novel))[:10],
                "novel_count": len(novel),
            }
        )
    out["passages"].append(entry)

dest = pathlib.Path("/Users/akshayballal/.claude/jobs/c7968b33/tmp/examples.json")
dest.write_text(json.dumps(out))

print("corpus-wide (12 passages):")
for t in out["totals"]:
    print(f"   {t['name']:<18} ratio={t['ratio']:.2f}  novel word tokens={t['novel']}")
print()
for p in out["passages"]:
    print(f"passage {p['index']}: {len(p['words'])} words")
    for m in p["models"]:
        neg = "n/a" if m["negation"] is None else f"{m['negation']:.2f}"
        print(
            f"   {m['name']:<18} kept {sum(m['labels']):4d}  ratio={m['ratio']:.2f} "
            f"VR={m['variation']:.3f} neg={neg} novel={m['novel_count']} {m['novel'][:5]}"
        )
