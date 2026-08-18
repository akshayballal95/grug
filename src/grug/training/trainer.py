"""Fine-tune an encoder into a preserve/discard token classifier.

A plain PyTorch loop rather than ``Trainer``: it is short enough to read, and
the extra loss terms worth experimenting with (see ``cs_weight``) are easier to
add here than through a callback.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .data import Example, read_jsonl

__all__ = ["TrainConfig", "train"]

#: Sub-word positions with this label are ignored by the loss.
IGNORE_INDEX = -100


@dataclass
class TrainConfig:
    """Everything a run needs. Serialised next to the checkpoint."""

    model_name: str = "answerdotai/ModernBERT-base"
    epochs: int = 10
    learning_rate: float = 1e-5
    batch_size: int = 10
    max_length: int = 512
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    device: str = "auto"
    seed: int = 0
    #: Hub repo to stream per-epoch metrics to. Training on a remote GPU is
    #: otherwise a black box until it finishes.
    push_to: str | None = None
    #: Weight on the MOOSComp inter-class cosine similarity term. 0 reproduces
    #: the LLMLingua-2 objective exactly; raise it to separate the two classes
    #: in the final layer, which is where over-smoothing bites.
    cs_weight: float = 0.0


def _resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps is not None and mps.is_available() else "cpu"


def _encode(tokenizer: Any, batch: list[Example], max_length: int) -> dict[str, Any]:
    """Tokenise a batch and project word labels onto every sub-word.

    Every sub-word of a word carries that word's label, which matches inference:
    the backend averages sub-word probabilities to score the word.
    """
    import torch

    encoded = tokenizer(
        [e.words for e in batch],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    labels = torch.full(encoded["input_ids"].shape, IGNORE_INDEX, dtype=torch.long)
    for row, example in enumerate(batch):
        for position, word_index in enumerate(encoded.word_ids(row)):
            if word_index is not None and word_index < len(example.labels):
                labels[row, position] = example.labels[word_index]
    encoded["labels"] = labels
    return encoded


def _inter_class_cosine(hidden: Any, labels: Any) -> Any:
    """Mean cosine similarity between preserve and discard representations.

    Minimising this counteracts over-smoothing, where a deep encoder's token
    representations converge and the two classes become hard to separate in the
    very layer the classifier reads.
    """
    import torch

    mask = labels != IGNORE_INDEX
    if not mask.any():
        return torch.zeros((), device=hidden.device)
    flat = torch.nn.functional.normalize(hidden[mask], dim=-1)
    flat_labels = labels[mask]
    keep, drop = flat[flat_labels == 1], flat[flat_labels == 0]
    if keep.numel() == 0 or drop.numel() == 0:
        return torch.zeros((), device=hidden.device)
    return (keep @ drop.T).mean()


def train(
    data_dir: str | Path,
    out_dir: str | Path,
    config: TrainConfig | None = None,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    """Fine-tune ``config.model_name`` on the shards in ``data_dir``.

    Returns a metrics dict, also written to ``out_dir/metrics.json``.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = read_jsonl(data_dir / "train.jsonl")
    val_path = data_dir / "val.jsonl"
    val_set = read_jsonl(val_path) if val_path.exists() else []
    if not train_set:
        raise ValueError(f"no training examples in {data_dir}")

    device = _resolve_device(config.device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if not tokenizer.is_fast:
        raise RuntimeError(f"{config.model_name!r} needs a fast tokenizer for word_ids()")
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=2,
        id2label={0: "discard", 1: "preserve"},
        label2id={"discard": 0, "preserve": 1},
    )
    model.to(device)

    collate = lambda batch: _encode(tokenizer, batch, config.max_length)  # noqa: E731
    loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    steps = max(1, len(loader) * config.epochs)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=config.learning_rate,
        total_steps=steps,
        pct_start=config.warmup_ratio,
        anneal_strategy="linear",
    )

    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, output_hidden_states=config.cs_weight > 0)
            loss = outputs.loss
            if config.cs_weight > 0:
                loss = loss + config.cs_weight * _inter_class_cosine(
                    outputs.hidden_states[-1], batch["labels"]
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()
            running += loss.item()
            if progress and step % 20 == 0:
                print(
                    f"  epoch {epoch + 1}/{config.epochs}  step {step + 1}/{len(loader)}"
                    f"  loss {loss.item():.4f}",
                    flush=True,
                )
        entry = {"epoch": epoch + 1, "train_loss": running / max(1, len(loader))}
        if val_set:
            entry.update(evaluate_tokens(model, tokenizer, val_set, config, device))
        history.append(entry)
        if progress:
            print(f"  -> {entry}", flush=True)
        _write_metrics(out_dir, config, device, len(train_set), len(val_set), history)
        _publish(config.push_to, out_dir, epoch + 1, config.epochs)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return _write_metrics(out_dir, config, device, len(train_set), len(val_set), history)


def _write_metrics(
    out_dir: Path,
    config: TrainConfig,
    device: str,
    train_examples: int,
    val_examples: int,
    history: list[dict[str, float]],
) -> dict[str, Any]:
    """Write metrics.json after every epoch, not just at the end."""
    metrics = {
        "config": asdict(config),
        "device": device,
        "train_examples": train_examples,
        "val_examples": val_examples,
        "epochs_done": len(history),
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _publish(repo: str | None, out_dir: Path, epoch: int, total: int) -> None:
    """Upload the running metrics to the Hub so progress is visible remotely.

    Never fatal: losing a progress ping must not kill a training run.
    """
    if not repo:
        return
    try:
        import os

        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(out_dir / "metrics.json"),
            path_in_repo="metrics.json",
            repo_id=repo,
            repo_type="model",
            commit_message=f"epoch {epoch}/{total}",
        )
    except Exception as exc:  # pragma: no cover - network
        print(f"  (metrics push failed: {exc})", flush=True)


def evaluate_tokens(
    model: Any, tokenizer: Any, examples: list[Example], config: TrainConfig, device: str
) -> dict[str, float]:
    """Token-level loss, accuracy and preserve-class F1 on a held-out shard."""
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    collate = lambda batch: _encode(tokenizer, batch, config.max_length)  # noqa: E731
    loader = DataLoader(examples, batch_size=config.batch_size, collate_fn=collate)

    loss_total = correct = counted = tp = fp = fn = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss_total += outputs.loss.item()
            predicted = outputs.logits.argmax(-1)
            labels = batch["labels"]
            mask = labels != IGNORE_INDEX
            correct += int((predicted[mask] == labels[mask]).sum())
            counted += int(mask.sum())
            tp += int(((predicted == 1) & (labels == 1) & mask).sum())
            fp += int(((predicted == 1) & (labels == 0) & mask).sum())
            fn += int(((predicted == 0) & (labels == 1) & mask).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "val_loss": loss_total / max(1, len(loader)),
        "val_accuracy": correct / counted if counted else 0.0,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": (2 * precision * recall / (precision + recall) if precision + recall else 0.0),
        "val_perplexity": math.exp(min(20.0, loss_total / max(1, len(loader)))),
    }
