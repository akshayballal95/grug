"""Let rules strip the safe filler first, then spend the classifier's budget
on what is actually contested.

Measured on MeetingBank QA at ~0.22, answers judged by Sonnet 4.6:

    n(questions)  classifier only      rules -> classifier   delta
    450           0.484 / 0.603        0.500 / 0.617         +0.016 / +0.014
    900           0.477 / 0.595        0.508 / 0.612         +0.031 / +0.017

Better on both metrics, at a slightly tighter ratio, and the gap widens with
sample size rather than shrinking -- which is what separates this from the
question-prefix experiment next door.

rules reaches 0.62 deterministically and never touches content. Reaching 0.21
from there asks the classifier to keep a third of a cleaner input rather than a
fifth of a raw one.
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
from grug.benchmark import llm, qa  # noqa: E402

CLF = "akshayballal/grug-mbert-control-meetingbank"
TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 0.21
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 150

examples = qa.load_qa(limit=LIMIT)
client = llm.LLMClient(
    model="bedrock/global.anthropic.claude-sonnet-4-6", workers=16, max_tokens=64
)


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
    print(f"{label:<26} ratio={statistics.fmean(ratios):.3f}  EM={em:.3f}  F1={f1:.3f}")


direct = [
    grug.compress(
        e.context, rate=TARGET, backend="classifier", backend_kwargs={"model_name": CLF}
    ).text
    for e in examples
]
score("classifier only", direct)

cascaded = []
for e in examples:
    stripped = grug.compress(e.context, rate=TARGET, backend="rules").text
    # Re-express the target as a share of what rules left.
    kept = max(1e-6, len(stripped.split()) / max(1, len(e.context.split())))
    inner = min(0.95, TARGET / kept)
    cascaded.append(
        grug.compress(
            stripped, rate=inner, backend="classifier", backend_kwargs={"model_name": CLF}
        ).text
    )
score("rules -> classifier", cascaded)
