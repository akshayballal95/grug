"""How far below rules' floor can the cascade go while holding F1?

rules stops at ~62% of the tokens and scores F1 ~0.75. The question is how much
further the classifier can take it before answer quality gives way.
"""

import os
import pathlib
import statistics

for line in (pathlib.Path.home() / ".config/grug/env").read_text().splitlines():
    line = line.replace("export ", "").strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k] = v.strip().strip("\"'")

import grug  # noqa: E402
from grug.benchmark import llm, qa  # noqa: E402

CLF = "akshayballal/grug-mbert-control-meetingbank"
LIMIT = 150

examples = qa.load_qa(limit=LIMIT)
client = llm.LLMClient(
    model="bedrock/global.anthropic.claude-sonnet-4-6", workers=16, max_tokens=64
)

# rules output is deterministic, so compress once and reuse for every target.
stripped = [grug.compress(e.context, rate=0.33, backend="rules").text for e in examples]


def score(label, texts):
    pairs, golds, ratios = [], [], []
    for text, e in zip(texts, examples, strict=True):
        ratios.append(len(text.split()) / max(1, len(e.context.split())))
        for q, a in zip(e.questions, e.answers, strict=True):
            pairs.append((text, q))
            golds.append(a)
    answers = client.many(pairs)
    em = statistics.fmean(qa.exact_match(g, w) for g, w in zip(answers, golds, strict=True))
    f1 = statistics.fmean(qa.token_f1(g, w) for g, w in zip(answers, golds, strict=True))
    print(f"{label:<28} ratio={statistics.fmean(ratios):.3f}  EM={em:.3f}  F1={f1:.3f}", flush=True)
    return f1


print(f"{LIMIT} contexts, {LIMIT * 3} questions, judged by Sonnet 4.6\n")
score("rules alone (the floor)", stripped)
score("original", [e.context for e in examples])

for target in (0.55, 0.50, 0.45, 0.40):
    texts = []
    for text, e in zip(stripped, examples, strict=True):
        kept = max(1e-6, len(text.split()) / max(1, len(e.context.split())))
        inner = min(0.99, target / kept)
        texts.append(
            grug.compress(
                text, rate=inner, backend="classifier", backend_kwargs={"model_name": CLF}
            ).text
        )
    score(f"rules -> classifier @{target:.2f}", texts)
