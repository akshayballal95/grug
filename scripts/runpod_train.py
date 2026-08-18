#!/usr/bin/env python3
"""Dispatch a grug training run to a RunPod GPU and push the result to the Hub.

    export RUNPOD_API_KEY=...   HF_TOKEN=...
    python scripts/runpod_train.py --repo <user>/grug-modernbert --dry-run
    python scripts/runpod_train.py --repo <user>/grug-modernbert --go

The pod clones this repository, installs ``grug[train]``, prepares the corpus,
trains, pushes the checkpoint, and then terminates itself -- so a crashed run
costs minutes, not the rest of the day. Nothing is created without ``--go``.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import textwrap
import time

REPO_URL = "https://github.com/akshayballal95/grug.git"
#: Needs a torch new enough for the transformers major we install; a stale torch
#: makes transformers report "PyTorch not found" even though it is installed.
DEFAULT_IMAGE = "runpod/pytorch:1.1.0-rc.154-cu1290-torch280-ubuntu2204"
#: ModernBERT-base at seq 512 needs well under this; more VRAM buys bigger batches.
MIN_VRAM_GB = 16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--repo", required=True, help="Hugging Face repo id to push to, e.g. you/grug-modernbert"
    )
    p.add_argument("--branch", default="main", help="Git branch the pod should clone")
    p.add_argument("--model", default="answerdotai/ModernBERT-base")
    p.add_argument(
        "--labels",
        default="akshayballal/grug-meetingbank-labels",
        help="Cached label dataset; the pod falls back to deriving if it is missing.",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--cs-weight", type=float, default=0.0)
    p.add_argument(
        "--gpu", default=None, help="Exact GPU type id. Default: cheapest with enough VRAM."
    )
    p.add_argument("--max-price", type=float, default=0.60, help="Refuse GPUs above this $/hr.")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--private", action="store_true", help="Create the Hub repo private.")
    p.add_argument(
        "--hold",
        type=int,
        default=600,
        help="On failure, keep the pod alive this long (s) for inspection.",
    )
    p.add_argument(
        "--go", action="store_true", help="Actually create the pod. Without this, dry run."
    )
    p.add_argument("--poll", type=int, default=60, help="Seconds between status polls.")
    return p.parse_args()


def pick_gpu(runpod, args) -> list[dict]:
    """Affordable community GPUs with enough VRAM, cheapest first.

    ``get_gpus()`` carries no pricing, so each candidate needs its own lookup.
    """
    candidates = [g for g in runpod.get_gpus() if (g.get("memoryInGb") or 0) >= MIN_VRAM_GB]
    priced = []
    for gpu in candidates:
        try:
            detail = runpod.get_gpu(gpu["id"])
        except Exception:
            continue
        price = detail.get("communityPrice") or detail.get("securePrice")
        if price and detail.get("communityCloud"):
            priced.append(
                {
                    "id": gpu["id"],
                    "name": detail.get("displayName"),
                    "vram": detail.get("memoryInGb"),
                    "price": price,
                }
            )
    priced.sort(key=lambda g: (g["price"], -g["vram"]))

    if args.gpu:
        priced = [g for g in priced if g["id"] == args.gpu]
        if not priced:
            sys.exit(f"error: --gpu {args.gpu!r} not available with >={MIN_VRAM_GB}GB")
    priced = [g for g in priced if g["price"] <= args.max_price]
    if not priced:
        sys.exit(f"error: nothing with >={MIN_VRAM_GB}GB under ${args.max_price}/hr")

    print(f"  affordable with >={MIN_VRAM_GB}GB:")
    for gpu in priced[:8]:
        print(f"    {gpu['id']:<32} {gpu['vram']:>3}GB  ${gpu['price']:.3f}/hr")
    return priced


_TERMINATE = """
import json, os, urllib.request

# The API authenticates with a Bearer header; the ?api_key= form returns 403,
# which would leave the pod billing after the run finished.
pod_id = os.environ["RUNPOD_POD_ID"]
query = 'mutation { podTerminate(input: {podId: "' + pod_id + '"}) }'
req = urllib.request.Request(
    "https://api.runpod.io/graphql",
    data=json.dumps({"query": query}).encode(),
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"],
        # Required: the edge rejects the default "Python-urllib/x.y" agent with
        # a 403, which silently left pods billing.
        "User-Agent": "grug-train/0.1",
    },
)
print(urllib.request.urlopen(req, timeout=30).read().decode()[:200])
"""

# Placeholders rather than an f-string: this text is full of shell ${} and JSON
# braces, and escaping them all is how bugs get in.
# Placeholders rather than an f-string: this text is full of shell ${} and JSON
# braces, and escaping them all is how bugs get in.
#
# Written to be safely re-run. RunPod restarts a container whose command exits,
# and /workspace survives the restart, so every stage must tolerate having
# already happened -- a bare `git clone` fails the second time and takes the
# whole run down with it.
_BOOTSTRAP = """
set -euxo pipefail
mkdir -p /workspace
exec > >(tee -a /workspace/bootstrap.log) 2>&1
export DEBIAN_FRONTEND=noninteractive

cat > /tmp/terminate.py <<'TERMINATE_EOF'
@TERMINATE@
TERMINATE_EOF

cat > /tmp/upload_log.py <<'LOG_EOF'
import os, pathlib, subprocess, sys

# This runs on any exit path, including one before `pip install` finished, so
# the dependency it needs may not be there yet.
try:
    from huggingface_hub import HfApi
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)
    from huggingface_hub import HfApi
try:
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo("@HFREPO@", repo_type="model", private=@PRIVATE@, exist_ok=True)
    api.upload_file(
        path_or_fileobj="/workspace/bootstrap.log",
        path_in_repo="bootstrap.log", repo_id="@HFREPO@", repo_type="model",
        commit_message="bootstrap log",
    )
    print("LOG UPLOADED")
except Exception as exc:
    print("log upload failed:", exc)
LOG_EOF

# Always ship the log somewhere retrievable, then stop billing.
cleanup() {
  code=$?
  echo "GRUG_EXIT status=$code"
  if [ $code -ne 0 ] && [ -n "${GRUG_HOLD:-}" ]; then
    echo "holding $GRUG_HOLD s for inspection"; sleep "$GRUG_HOLD"
  fi
  python /tmp/upload_log.py || true
  python /tmp/terminate.py || true
}
trap cleanup EXIT

# A restart must not re-run a stage that already succeeded.
if [ ! -f /workspace/.installed ]; then
  apt-get update -qq && apt-get install -y -qq git
  rm -rf /workspace/grug
  git clone --depth 1 --branch @BRANCH@ @REPO_URL@ /workspace/grug
  pip install --no-cache-dir -e /workspace/grug'[train]'
  touch /workspace/.installed
fi
cd /workspace/grug

# Fail in a minute, not after the eight-minute prepare. Importing the model
# class is what trips a torch/transformers mismatch.
python - <<'PREFLIGHT_EOF'
import torch, transformers
from transformers import AutoModelForTokenClassification  # noqa: F401

print("PREFLIGHT torch", torch.__version__, "transformers", transformers.__version__)
assert torch.cuda.is_available(), "CUDA is not available in this container"
print("PREFLIGHT gpu", torch.cuda.get_device_name(0))
PREFLIGHT_EOF

if [ ! -f /workspace/data/train.jsonl ]; then
  grug train prepare --out /workspace/data --from-hub @LABELREPO@
fi

grug train run \
  --data /workspace/data --out /workspace/ckpt \
  --model @MODEL@ --epochs @EPOCHS@ \
  --batch-size @BATCH@ --lr @LR@ \
  --max-length @MAXLEN@ --cs-weight @CSW@ \
  --push-to @HFREPO@ \
  --device cuda

cat > /tmp/push.py <<'PUSH_EOF'
import json, os, pathlib
from huggingface_hub import HfApi

repo = "@HFREPO@"
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="model", private=@PRIVATE@, exist_ok=True)
metrics = json.loads(pathlib.Path("/workspace/ckpt/metrics.json").read_text())
last = metrics["history"][-1] if metrics["history"] else {}
card = [
    "---", "license: cc-by-nc-sa-4.0", "library_name: transformers",
    "pipeline_tag: token-classification", "base_model: @MODEL@",
    "tags:", "- prompt-compression", "- grug", "---", "",
    "# grug prompt compressor", "",
    "Binary preserve/discard token classifier for prompt compression, trained with",
    "`grug train` on `microsoft/MeetingBank-LLMCompressed`.", "",
    "```python", "import grug",
    "from grug.backends.modern import ModernBackend", "",
    "comp = grug.Compressor(ModernBackend(model_name=" + repr(repo) + "))",
    "result = comp.compress(text, rate=0.4)", "```", "",
    "## Training", "", "```json", json.dumps(metrics["config"], indent=2), "```", "",
    "## Final epoch", "", "```json", json.dumps(last, indent=2), "```", "",
    "The training corpus is CC-BY-NC-SA-4.0, so this model inherits a",
    "non-commercial constraint.", "",
]
pathlib.Path("/workspace/ckpt/README.md").write_text(chr(10).join(card))
api.upload_folder(folder_path="/workspace/ckpt", repo_id=repo, repo_type="model")
print("GRUG_PUSHED", repo)
PUSH_EOF

python /tmp/push.py
echo GRUG_TRAINING_COMPLETE
"""


def bootstrap(args) -> str:
    """The single command the pod runs. Self-terminating, fail-fast."""
    fields = {
        "@TERMINATE@": _TERMINATE.strip(),
        "@BRANCH@": args.branch,
        "@REPO_URL@": REPO_URL,
        "@MODEL@": args.model,
        "@EPOCHS@": str(args.epochs),
        "@BATCH@": str(args.batch_size),
        "@LR@": str(args.lr),
        "@MAXLEN@": str(args.max_length),
        "@CSW@": str(args.cs_weight),
        "@HFREPO@": args.repo,
        "@LABELREPO@": args.labels,
        "@PRIVATE@": str(bool(args.private)),
    }
    script = _BOOTSTRAP
    for key, value in fields.items():
        script = script.replace(key, value)
    return script.strip()


def _docker_args(script: str) -> str:
    """Wrap the bootstrap so it survives GraphQL and shell quoting.

    The script is passed to RunPod inside a GraphQL string; backslashes and
    quotes in it are not valid GraphQL escapes. Base64 is alphanumeric, so it
    passes through both layers untouched.
    """
    encoded = base64.b64encode(script.encode()).decode()
    # Single quotes: the SDK interpolates this into a GraphQL string literal,
    # so a double quote here would terminate it early. Base64 has neither.
    return f"bash -lc 'echo {encoded} | base64 -d | bash'"


def main() -> int:
    args = parse_args()
    for key in ("RUNPOD_API_KEY", "HF_TOKEN"):
        if not os.environ.get(key):
            sys.exit(f"error: {key} is not set")

    import runpod

    runpod.api_key = os.environ["RUNPOD_API_KEY"]

    print("Selecting GPU...")
    candidates = pick_gpu(runpod, args)
    gpu = candidates[0]
    examples = 31775
    est_hours = args.epochs * examples / 70 / 3600  # ~70 ex/s on a 24GB card
    print(
        f"\n  chosen : {gpu['id']} ({gpu['vram']}GB) at ${gpu['price']:.3f}/hr"
        f"\n  job    : {args.model}, {args.epochs} epochs, batch {args.batch_size}"
        f"\n  push to: https://huggingface.co/{args.repo}"
        f"\n  rough  : ~{est_hours:.1f}h  ->  ~${est_hours * gpu['price']:.2f}\n"
    )

    if not args.go:
        print("DRY RUN - nothing created. Re-run with --go to launch.\n")
        print(textwrap.indent(bootstrap(args), "  "))
        return 0

    pod = None
    for candidate in candidates:
        try:
            pod = runpod.create_pod(
                name=f"grug-train-{args.model.split('/')[-1]}",
                image_name=args.image,
                gpu_type_id=candidate["id"],
                cloud_type="COMMUNITY",
                container_disk_in_gb=40,
                volume_in_gb=0,
                min_memory_in_gb=16,
                docker_args=_docker_args(bootstrap(args)),
                env={
                    "HF_TOKEN": os.environ["HF_TOKEN"],
                    "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"],
                    "GRUG_HOLD": str(args.hold),
                },
            )
            gpu = candidate
            break
        except Exception as exc:
            message = str(exc)
            if "no longer any instances" not in message and "not available" not in message:
                raise  # a bug here must not masquerade as a full GPU
            print(f"  {candidate['id']}: out of capacity")
    if pod is None:
        sys.exit("error: every affordable GPU type is out of capacity right now")
    print(f"  launched on {gpu['id']} at ${gpu['price']:.3f}/hr")

    pod_id = pod["id"]
    print(f"  pod {pod_id} created; polling every {args.poll}s. Ctrl-C stops polling, not the pod.")

    started = time.time()
    while True:
        time.sleep(args.poll)
        try:
            status = runpod.get_pod(pod_id)
        except Exception as exc:  # pod gone == it terminated itself
            print(f"  pod no longer queryable ({exc}); assuming it finished")
            break
        if status is None:
            print("  pod terminated")
            break
        runtime = status.get("runtime") or {}
        print(
            f"  [{(time.time() - started) / 60:5.1f}m] {status.get('desiredStatus')} {runtime.get('uptimeInSeconds', 0)}s"
        )
    print(f"\nDone. Check https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
