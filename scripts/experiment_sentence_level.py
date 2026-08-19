"""At a tight budget, is it better to keep whole sentences than scattered words?

No, and not close: EM 0.320 / F1 0.406 against word-level's 0.429 / 0.547 at
the same 0.21 ratio. Keeping a fifth of the words as whole sentences keeps only
a fifth of the sentences, so most facts are simply absent; word-level keeps a
trace of every sentence. For extractive QA, coverage beats coherence.

Word-level top-k at ratio 0.21 leaves a fifth of the words with none of the
syntax that connected them. Keeping fewer, complete sentences might answer
questions better even though it keeps the same number of tokens.
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
from grug.backends.classifier import ClassifierBackend  # noqa: E402
from grug.base import CompressionResult  # noqa: E402
from grug.benchmark import llm, qa  # noqa: E402
from grug.chunking import split_sentences  # noqa: E402
from grug.pinning import NUMBER_RE, collect_force_tokens, normalise_word  # noqa: E402
from grug.training.alignment import split_words  # noqa: E402
from grug.verify import is_negation  # noqa: E402

CLF = "akshayballal/grug-mbert-control-meetingbank"
RATE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.21
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 150


class SentenceLevel(ClassifierBackend):
    """Keep whole sentences, ranked by their mean word score."""

    name = "sentence-level"

    def compress(self, text: str, rate: float = 0.5, **kwargs) -> CompressionResult:
        words = split_words(text)
        if not words:
            return CompressionResult.build(text, text, self.name, metadata={})
        probs = self._score(words)

        forced = {
            normalise_word(w)
            for w in collect_force_tokens(text, self.force_tokens, entities=self.preserve_entities)
        }

        def is_pinned(w: str) -> bool:
            core = normalise_word(w)
            return core in forced or is_negation(core) or bool(NUMBER_RE.search(w))

        # Map words onto sentences by consuming the word list in order.
        spans, cursor = [], 0
        for sentence in split_sentences(text):
            n = len(split_words(sentence))
            if n:
                spans.append((cursor, min(cursor + n, len(words))))
                cursor += n
        if not spans or cursor != len(words):
            return super().compress(text, rate, **kwargs)

        budget = max(1, round(rate * len(words)))
        scored = []
        for index, (a, b) in enumerate(spans):
            block = probs[a:b]
            pins = sum(1 for w in words[a:b] if is_pinned(w))
            scored.append((pins > 0, statistics.fmean(block) if block else 0.0, index))
        scored.sort(key=lambda t: (not t[0], -t[1]))

        taken, used = [], 0
        for _, _, index in scored:
            a, b = spans[index]
            if used + (b - a) > budget and taken:
                continue
            taken.append(index)
            used += b - a
            if used >= budget:
                break

        keep = sorted(taken)
        chosen = [i for index in keep for i in range(*spans[index])]
        compressed = " ".join(words[i] for i in chosen)
        return CompressionResult.build(text, compressed, self.name, metadata={})


examples = qa.load_qa(limit=LIMIT)
client = llm.LLMClient(
    model="bedrock/global.anthropic.claude-sonnet-4-6", workers=16, max_tokens=64
)


def run(label, compress):
    pairs, golds, ratios = [], [], []
    for e in examples:
        r = compress(e.context)
        ratios.append(len(r.text.split()) / max(1, len(e.context.split())))
        for q, a in zip(e.questions, e.answers, strict=True):
            pairs.append((r.text, q))
            golds.append(a)
    answers = client.many(pairs)
    em = statistics.fmean(qa.exact_match(g, w) for g, w in zip(answers, golds, strict=True))
    f1 = statistics.fmean(qa.token_f1(g, w) for g, w in zip(answers, golds, strict=True))
    print(f"{label:<16} ratio={statistics.fmean(ratios):.3f}  EM={em:.3f}  F1={f1:.3f}")


run(
    "word-level",
    lambda t: grug.compress(t, rate=RATE, backend="classifier", backend_kwargs={"model_name": CLF}),
)
sent = SentenceLevel(model_name=CLF)
run("sentence-level", lambda t: grug.compress(t, rate=RATE, backend=sent))
