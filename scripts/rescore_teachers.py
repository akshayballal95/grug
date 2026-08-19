"""Re-score a saved teacher sweep with the current metric code. No API calls."""

import json
import sys
from pathlib import Path

from grug.training.distill import score_teacher


def read_json(path):
    """Read one JSON file. A helper so each call site is a single expression."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


for path in sys.argv[1:]:
    d = read_json(path)
    if "passages" not in d:
        print(f"{path}: pre-fix file, no raw outputs saved -- cannot re-score")
        continue
    print(f"== {path}")
    for t in d["teachers"]:
        outs = t.get("metadata", {}).get("outputs")
        if not outs:
            print(f"   {t['model']}: no outputs stored")
            continue
        print("   " + score_teacher(t["model"], d["passages"], outs).summary())
