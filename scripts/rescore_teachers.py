"""Re-score a saved teacher sweep with the current metric code. No API calls."""
import json, sys
from grug.training.distill import score_teacher

for path in sys.argv[1:]:
    d = json.load(open(path))
    if "passages" not in d:
        print(f"{path}: pre-fix file, no raw outputs saved -- cannot re-score")
        continue
    print(f"== {path}")
    for t in d["teachers"]:
        outs = t.get("metadata", {}).get("outputs")
        if not outs:
            print(f"   {t['model']}: no outputs stored"); continue
        print("   " + score_teacher(t["model"], d["passages"], outs).summary())
