"""Turn the public distillation corpus into per-word training labels.

``microsoft/MeetingBank-LLMCompressed`` ships (original, compressed) *text*
pairs, not labels, so the labels are derived here with the paper's alignment
algorithm. The dataset is CC-BY-NC-SA-4.0: anything trained on it inherits a
non-commercial constraint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .alignment import AlignmentStats, annotate, filter_examples

__all__ = [
    "DEFAULT_DATASET",
    "DEFAULT_LABEL_REPO",
    "Example",
    "prepare",
    "pull_prepared",
    "push_prepared",
    "read_jsonl",
]

DEFAULT_DATASET = "microsoft/MeetingBank-LLMCompressed"

#: Where derived labels are cached. Deriving them costs ~8 minutes of CPU, which
#: on a rented GPU is 8 minutes of paying for an idle card.
#: Rederived after the alignment fix. The labels the greedy scan produced
#: disagreed with the compressions they came from on 6.2% of words, and on
#: 13.6% of negations; retraining on these gained ~0.013 f1 on every encoder.
DEFAULT_LABEL_REPO = "akshayballal/grug-meetingbank-labels"

_SHARDS = ("train.jsonl", "val.jsonl", "summary.json")

#: Chunk-level columns. The whole-document columns exist too, but the teacher
#: compressed chunk by chunk, so these pair up far more reliably.
_ORIGINAL_COLUMN = "prompt_list"
_COMPRESSED_COLUMN = "compressed_prompt_list"


@dataclass
class Example:
    """One training example: words of the original, and whether each survived."""

    words: list[str]
    labels: list[int]

    def to_json(self) -> str:
        return json.dumps({"words": self.words, "labels": self.labels})


def _require_datasets() -> Any:
    try:
        import datasets
    except ImportError as exc:  # pragma: no cover - guarded by the extra
        raise ImportError(
            "grug.training.data requires 'datasets'. Install it with: pip install 'grugify[train]'"
        ) from exc
    return datasets


def _pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
    """Chunk pairs from one row, falling back to the whole-document columns."""
    originals = row.get(_ORIGINAL_COLUMN)
    compressed = row.get(_COMPRESSED_COLUMN)
    if isinstance(originals, str):
        originals = json.loads(originals)
    if isinstance(compressed, str):
        compressed = json.loads(compressed)
    if originals and compressed and len(originals) == len(compressed):
        return list(zip(originals, compressed, strict=True))
    whole, whole_compressed = row.get("prompt"), row.get("compressed_prompt")
    return [(whole, whole_compressed)] if whole and whole_compressed else []


def prepare(
    out_dir: str | Path,
    *,
    dataset: str = DEFAULT_DATASET,
    split: str = "train",
    limit: int | None = None,
    val_fraction: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Download the corpus, derive labels, filter, and write JSONL shards.

    Args:
        out_dir: Directory to write ``train.jsonl`` and ``val.jsonl`` into.
        dataset: Hugging Face dataset id.
        split: Split to read.
        limit: Stop after this many rows. Useful for a smoke run.
        val_fraction: Portion held out for validation.
        seed: Shuffle seed for the split.

    Returns:
        A summary dict, also written to ``summary.json`` so a run is auditable.
    """
    datasets = _require_datasets()
    rows = datasets.load_dataset(dataset, split=split)
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))

    stats: list[AlignmentStats] = []
    for row in rows:
        for original, compressed in _pairs(row):
            if original and compressed:
                stats.append(annotate(original, compressed))

    kept, thresholds = filter_examples(stats)
    return _write(out_dir, kept, thresholds, dataset, split, len(stats), val_fraction, seed)


def _write(
    out_dir: str | Path,
    kept: list[AlignmentStats],
    thresholds: dict[str, float],
    dataset: str,
    split: str,
    total: int,
    val_fraction: float,
    seed: int,
) -> dict[str, Any]:
    import random

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    examples = [Example(e.words, [int(x) for x in e.labels]) for e in kept]
    random.Random(seed).shuffle(examples)
    cut = max(1, int(len(examples) * val_fraction)) if examples else 0
    shards = {"val": examples[:cut], "train": examples[cut:]}

    for name, rowset in shards.items():
        with (directory / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for example in rowset:
                handle.write(example.to_json() + "\n")

    total_words = sum(len(e.words) for e in examples)
    summary = {
        "dataset": dataset,
        "split": split,
        "pairs_aligned": total,
        "pairs_kept": len(examples),
        "train": len(shards["train"]),
        "val": len(shards["val"]),
        "words": total_words,
        "keep_rate": (sum(sum(e.labels) for e in examples) / total_words if total_words else 0.0),
        **thresholds,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def read_jsonl(path: str | Path) -> list[Example]:
    """Read a shard written by :func:`prepare`."""
    examples = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                examples.append(Example(payload["words"], payload["labels"]))
    return examples


def push_prepared(data_dir: str | Path, repo: str, *, private: bool = False) -> str:
    """Publish derived labels so other runs can skip the alignment stage."""
    import os

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    for name in _SHARDS:
        path = Path(data_dir) / name
        if path.exists():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=name,
                repo_id=repo,
                repo_type="dataset",
            )
    return f"https://huggingface.co/datasets/{repo}"


def pull_prepared(repo: str, out_dir: str | Path) -> dict[str, Any]:
    """Fetch labels published by :func:`push_prepared`.

    Returns the summary of what was downloaded, or raises if the cache is
    missing so the caller can fall back to deriving them.
    """
    import os

    from huggingface_hub import hf_hub_download

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    for name in _SHARDS:
        local = hf_hub_download(repo_id=repo, filename=name, repo_type="dataset", token=token)
        (directory / name).write_bytes(Path(local).read_bytes())
    return json.loads((directory / "summary.json").read_text(encoding="utf-8"))
