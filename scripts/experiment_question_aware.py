"""Does prefixing the question help the classifier decide what to keep?

Answer, measured: not reliably. Kept so the experiment is not repeated.

    ratio  n     agnostic EM/F1   question-aware EM/F1   delta
    0.335  300   0.570 / 0.701    0.573 / 0.707          +0.003 / +0.006
    0.211  300   0.433 / 0.539    0.457 / 0.559          +0.024 / +0.020
    0.209  750   0.429 / 0.547    0.441 / 0.548          +0.012 / +0.001

The middle row looks like a real gain at tight budgets. It is not: at 750
questions the exact-match gain halves and the F1 gain disappears, and the
standard error there is about 0.018, so +0.012 sits inside it.

The mechanism explains the null. The encoder is bidirectional, so a prefixed
question is visible to every document token -- but the model never saw a
question during training, so nothing taught it to raise keep-probabilities for
question-relevant words. Conditioning on the question would have to be trained
in, not prefixed at inference.

The cost is real either way: one compression per question rather than per
document, so a compressed document can no longer be cached across queries.


The encoder is bidirectional, so document tokens can attend to a question
placed in front of them, and question-relevant words should score higher. The
model was trained without questions, though, so this is out of distribution --
it may do nothing at all.

Compressing per question also changes the use case: the document can no longer
be compressed once and reused across queries.
"""

import os
import pathlib
import statistics
import sys

for line in (pathlib.Path.home() / ".config/grug/env").read_text().splitlines():
    line = line.replace("export ", "").strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k] = v.strip().strip("\"'")

import grug  # noqa: E402
from grug.backends.classifier import ClassifierBackend, _subword_counts, _windows  # noqa: E402
from grug.benchmark import llm, qa  # noqa: E402

CLF = "akshayballal/grug-mbert-control-meetingbank"
RATE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.33
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 100


class QuestionAware(ClassifierBackend):
    """Scores each window with the question in front of it, then drops it."""

    name = "question-aware"
    question: str = ""

    def _score(self, words: list[str]) -> list[float]:
        import torch

        if not self.question.strip():
            return super()._score(words)

        tokenizer, model = self._load()
        limit = self.max_length or getattr(model.config, "max_position_embeddings", 512)
        prefix = [*self.question.split(), "|"]
        prefix_cost = sum(_subword_counts(tokenizer, prefix))
        budget = max(8, limit - 2 - prefix_cost)

        counts = _subword_counts(tokenizer, words)
        probs: list[float] = []
        for window in _windows(counts, budget):
            chunk = words[window[0] : window[1]]
            encoded = tokenizer(
                prefix + chunk,
                is_split_into_words=True,
                truncation=True,
                max_length=limit,
                return_tensors="pt",
            )
            word_ids = encoded.word_ids()
            encoded = {k: v.to(self._resolved_device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits[0]
            keep = logits.softmax(-1)[:, self._preserve_id].tolist()

            totals = [0.0] * len(chunk)
            seen = [0] * len(chunk)
            for position, word_index in enumerate(word_ids):
                if word_index is None or word_index < len(prefix):
                    continue  # the question itself is not part of the output
                idx = word_index - len(prefix)
                totals[idx] += keep[position]
                seen[idx] += 1
            probs.extend(totals[i] / seen[i] if seen[i] else 0.0 for i in range(len(chunk)))
        return probs


examples = qa.load_qa(limit=LIMIT)
client = llm.LLMClient(
    model="bedrock/global.anthropic.claude-sonnet-4-6", workers=16, max_tokens=64
)


def score(pairs, golds):
    answers = client.many(pairs)
    em = statistics.fmean(qa.exact_match(g, w) for g, w in zip(answers, golds, strict=True))
    f1 = statistics.fmean(qa.token_f1(g, w) for g, w in zip(answers, golds, strict=True))
    return em, f1


# --- question-agnostic: compress once, answer every question from it ---------
pairs, golds, ratios = [], [], []
for e in examples:
    r = grug.compress(
        e.context, rate=RATE, backend="classifier", backend_kwargs={"model_name": CLF}
    )
    ratios.append(len(r.text.split()) / max(1, len(e.context.split())))
    for q, a in zip(e.questions, e.answers, strict=True):
        pairs.append((r.text, q))
        golds.append(a)
em, f1 = score(pairs, golds)
print(
    f"question-agnostic  ratio={statistics.fmean(ratios):.3f}  EM={em:.3f}  F1={f1:.3f}  "
    f"({len(examples)} compressions)"
)

# --- question-aware: one compression per question ----------------------------
backend = QuestionAware(model_name=CLF)
pairs, golds, ratios = [], [], []
for e in examples:
    for q, a in zip(e.questions, e.answers, strict=True):
        backend.question = q
        r = grug.compress(e.context, rate=RATE, backend=backend)
        ratios.append(len(r.text.split()) / max(1, len(e.context.split())))
        pairs.append((r.text, q))
        golds.append(a)
em, f1 = score(pairs, golds)
print(
    f"question-aware     ratio={statistics.fmean(ratios):.3f}  EM={em:.3f}  F1={f1:.3f}  "
    f"({len(pairs)} compressions)"
)
