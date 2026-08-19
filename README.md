<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/akshayballal95/grug/main/docs/assets/grug-banner-dark.svg">
  <img alt="grug. grug make text small. grug keep meaning." src="https://raw.githubusercontent.com/akshayballal95/grug/main/docs/assets/grug-banner-light.svg" width="540">
</picture>

Shrink anything you feed an LLM. Keep the words that change the answer.

[![PyPI](https://img.shields.io/pypi/v/grugify?cacheSeconds=300)](https://pypi.org/project/grugify/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/akshayballal95/grug)
[![CI](https://github.com/akshayballal95/grug/actions/workflows/ci.yml/badge.svg)](https://github.com/akshayballal95/grug/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/akshayballal95/grug/blob/main/LICENSE)

</div>

---

Most of the text that ends up in a context window is padding. Documents you stuff
into RAG, tool results an agent loops back to itself, transcripts, tickets, logs:
models answer just as well without the articles, the copulas, and the "it is
important to note that". grug deletes the padding and refuses to touch what
actually carries meaning: negations, numbers, names, code, URLs, markdown
structure. Then it checks its own output and warns you if anything load-bearing
went missing anyway.

```text
Before (94 tokens)                          After `grug compress --rate 0.5` (65 tokens)

It is important to note that the billing    billing pipeline rewritten run streaming
pipeline has been rewritten to run on the   ingest service. migration not automatic:
streaming ingest service. The migration     accounts legacy monthly plan must moved
is not automatic: accounts on the legacy    hand before cutover date. practice
monthly plan must be moved by hand before   measured median lag 1.2 seconds across
the cutover date. In practice we measured   4,800 accounts, and p99 lag of 9.6
a median lag of 1.2 seconds across 4,800    seconds. key economic point easy wrong:
accounts, and a p99 lag of 9.6 seconds.     bills scale volume, not price.
The key economic point is easy to get
wrong: bills scale with volume, not price.
```

Every number made it through, and so did both negations. The second one
(`not price`) is the whole point of the paragraph. Zero warnings from the verifier.

## Why this exists

Compressing LLM input is not new. The problem with existing compressors is that
they score words by how much information they seem to carry, and a word like "not"
is three characters of function word that looks eminently droppable. Drop it and
the compressed text asserts the opposite of the source, in fluent prose that
nothing downstream will question.

That risk is the same whether the text is a prompt, a retrieved document, or the
output of a tool call an agent is about to reason over. grug is built around not
taking it:

- Negations, digits, code, URLs, and the words relating two numbers can never be
  dropped. `3 of 12` never comes back as `3 12`. This lives in the engine, not in
  a setting you could forget to turn on.
- Every result gets checked afterwards for lost negations, numbers, and names,
  and the warnings land in `result.warnings`. The CLI exit codes are made for CI.
- The default backend is pure Python. No torch, no downloads, milliseconds per
  document. `import grug` never imports torch even when the extras are installed.
- Word lists, regex rules, phrase rewrites, and whole languages are data you can
  add and remove, not code you have to fork.

## The numbers

600 questions over 694k tokens of [MeetingBank](https://huggingface.co/datasets/microsoft/MeetingBank-LLMCompressed)
transcripts, compressed at rate 0.33, answered by Claude Sonnet 4.6, scored
against reference answers:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/akshayballal95/grug/main/docs/assets/qa-quality-dark.svg">
  <img alt="Answer quality vs tokens sent: grug rules beats the uncompressed baseline with 62% of the tokens; the grug classifier keeps F1 0.70 with 37%." src="https://raw.githubusercontent.com/akshayballal95/grug/main/docs/assets/qa-quality-light.svg">
</picture>

Compression only helps if the meaning survives it. Scored for dropped
negations on the same run, neither grug backend lost a single one:

| Backend | Tokens kept | Exact match | F1 | Negations lost |
| --- | ---: | ---: | ---: | ---: |
| original (no compression) | 100% | 0.62 | 0.75 | n/a |
| **grug rules** | 62% | **0.61** | **0.76** | **0%** |
| **grug classifier** (mbert-control) | 37% | 0.58 | 0.70 | **0%** |
| LLMLingua-2 | 30% | 0.56 | 0.69 | 43% |

`grug rules` keeps 62% of the tokens and answers about as well as the full
document: 0.76 F1 against 0.75, 0.61 exact match against 0.62. Both gaps are
0.01, inside the standard error on 600 questions.

The classifier spends some of that quality on a smaller prompt: 37% of the
tokens for 0.70 F1. At a comparable size it answers slightly better than
LLMLingua-2, and keeps every negation where LLMLingua-2 drops 43% of them.

Reproduce it with `grug benchmark qa`. Raw results are in
[`benchmarks/`](https://github.com/akshayballal95/grug/tree/main/benchmarks/sonnet46/).

## Quick start

```bash
pip install grugify
```

```python
import grug

result = grug.compress(document, rate=0.5)
result.text  # the compressed document
result.ratio  # what was achieved, not what was asked for
result.warnings  # faithfulness report; [] means nothing suspicious
```

Or from the shell:

```bash
grug compress notes.md                  # writes notes.grug.md, stats to stderr
grug compress - < in.txt > out.txt      # stdin to stdout
grug verify original.txt compressed.txt # faithfulness checks on their own
```

If you want deeper compression than the rule lists can reach, install the trained
classifier with `pip install 'grugify[classifier]'`:

```python
result = grug.compress(
    document,
    rate=0.33,
    backend="classifier",
    backend_kwargs={"model_name": "akshayballal/grug-mbert-control-meetingbank"},
)
```

## Two backends

| Backend | Install | Method | Rates it reaches | Speed |
| --- | --- | --- | --- | --- |
| `rules` | included | Deletes words its rule set nominates; the engine vetoes everything load-bearing | floors out around 0.6, when it runs out of safe words to drop | milliseconds |
| `classifier` | `grugify[classifier]` | A fine-tuned encoder scores every word and the top-`rate` fraction survives, in order | 0.2 to 0.5 | about 0.2s per 400-token chunk on CPU, after a one-off model load |

Both are extractive: the output is a subsequence of the input, so neither can
invent a fact. `rate` means the same thing everywhere, the fraction of tokens to
*keep*. A backend that cannot hit it exactly reports what it actually achieved in
`result.ratio`.

The classifier takes any Hugging Face token-classification checkpoint with a
trained preserve/discard head and a fast tokenizer (ModernBERT, mmBERT, EuroBERT
all work). Training your own is three commands; see [TRAINING.md](https://github.com/akshayballal95/grug/blob/main/TRAINING.md).

## The rules engine

The default backend splits responsibility in two. Rules nominate words to drop.
The engine decides, and its vetoes always win. A token budget derived from `rate`
controls how deep the cutting goes. Everything on the rules side is composable:

```python
from grug.backends.rules import ENGLISH, PatternRule, RulesBackend, WordClassRule

rules = (
    ENGLISH.rules.remove("pronouns")  # subtract a word class
    .add(WordClassRule("corp-speak", {"synergy", "leverage"}, priority=5))
    .add(PatternRule("hedges", r"(arguabl|probabl|possibl)\w*", priority=15))
)

backend = RulesBackend(
    rules=rules,
    keep_words={"pipeline"},  # exact vetoes
    keep_patterns=(r"[A-Z]{2,}",),  # regex vetoes, e.g. keep acronyms like "IT"
)
```

A new language is a data pack, not a fork:

```python
from grug.backends.rules import Language, RuleSet, WordClassRule, register_language

register_language(
    Language(
        code="de",
        rules=RuleSet(WordClassRule("artikel", {"der", "die", "das"}, priority=10)),
        negations=frozenset({"nicht", "kein", "keine", "ohne"}),
    )
)
backend = RulesBackend(language="de")
```

Because the vetoes belong to the engine, a badly written custom rule cannot break
the guarantees. There is a test where a hostile rule nominates every single word
in the document, and the negations, numbers, and code spans still come through.
[`examples/rules_backend.py`](https://github.com/akshayballal95/grug/blob/main/examples/rules_backend.py) walks through all of it.

## Faithfulness

```python
>>> grug.verify("bills scale with volume, not price", "bills scale volume price")
["negation lost: 'not' (1× → 0×) — meaning may be inverted"]
```

Prevention first: the rules engine will not drop negations, digits, or the
connectives between numbers at any rate, and the classifier pins negations,
digits, detected entities, and markdown structure before it ranks anything.

Verification second: every compression is checked for negation loss, negation
scope loss, number loss, lost relations between numbers (`3 of 12` collapsing
to `3 12`), and entity loss. The verifier speaks the backend's language: give
the rules backend a German pack and its negations are what get checked, while
the capitalisation-based entity heuristics stand down for languages that
capitalise every noun. There is no NER model and no ML in the
verifier, just regex and exact matching. That costs some precision on entities
and buys a checker that works on exactly the kind of terse, ungrammatical text a
model-based checker would choke on, in microseconds.

A clean run means nothing suspicious was found, not that the compression is
provably faithful. Treat it like a smoke alarm.

### CI gating

| Code | Meaning |
| --- | --- |
| `0` | Compressed, no faithfulness warnings |
| `1` | Error: bad file, unknown backend, missing dependency |
| `2` | Compressed, but the verifier flagged something |

Exit code 2 means "a human should read this diff", not "this failed".

## Markdown awareness

Documents are parsed with a real CommonMark parser
([markdown-it-py](https://github.com/executablebooks/markdown-it-py)) rather than
pattern-matched, so structure survives:

| Element | Treatment |
| --- | --- |
| Fenced / indented code, tables | Bypass the compressor, re-emitted byte for byte |
| Whole source files | Passed through unchanged (`--compress-code` overrides) |
| Headings, bullets, blockquote markers | Marker preserved, the text after it compresses |
| Inline code spans and URLs | Swapped for opaque placeholders, restored verbatim |
| Numbers and identifiers with separators (`1,250`, `us-east-1`, `v2.1.0-rc3`) | Protected as spans |
| Blank lines | Hard chunk boundaries, so no backend can collapse a paragraph break |

Long documents are chunked to roughly 450 tokens on sentence boundaries and
rejoined with the original layout.

## CLI reference

```bash
grug compress docs/*.md --rate 0.4        # many files, each to <name>.grug.md
grug compress notes.md --json -q          # full CompressionResult as JSON
grug backends                             # what is registered and installed
grug train --help                         # reproduce the classifier (TRAINING.md)
grug benchmark qa --help                  # reproduce the numbers above
```

| Flag | Effect |
| --- | --- |
| `--rate`, `-r` | Fraction of tokens to keep. Default `0.5`. |
| `--backend`, `-b` | Backend name. Default `rules`. |
| `--model` | Checkpoint for `--backend classifier`. |
| `--device` | `cpu`, `cuda`, `mps`, or `auto`. |
| `--no-verify` | Skip the faithfulness checks. |
| `--json` | Emit the full result as JSON on stdout. |
| `--quiet`, `-q` | Suppress the stats line. |

## Write your own backend

Subclass one ABC and register it. It then shows up in `grug.compress()`,
`grug.Compressor`, and the CLI without any changes to grug:

```python
from grug.base import CompressionResult, CompressorBackend


class ShoutyBackend(CompressorBackend):
    name = "shouty"
    description = "Keeps only the long words."

    def compress(self, text: str, rate: float = 0.5, **kwargs) -> CompressionResult:
        self._validate_rate(rate)
        cutoff = 3 + int((1 - rate) * 5)
        kept = " ".join(w for w in text.split() if len(w) > cutoff)
        return CompressionResult.build(text, kept, self.name)
```

To ship it as a package, advertise it through an entry point and `pip install`
is all a user needs:

```toml
[project.entry-points."grug.backends"]
shouty = "my_package.backend:ShoutyBackend"
```

## Contributing

```bash
git clone https://github.com/akshayballal95/grug && cd grug
uv sync --all-extras          # or: pip install -e '.[dev,tokens]'
uv run pytest -m "not slow"   # fast suite, no model downloads
uv run ruff check src tests examples scripts
```

The codebase is small on purpose: two backends, one verifier, one chunker. If you
are looking for somewhere to start, a language pack for the rules engine, a
benchmark on your own data, or a trained checkpoint for a new encoder would all
be welcome PRs.

## License

[MIT](https://github.com/akshayballal95/grug/blob/main/LICENSE). The bundled benchmark uses the
[MeetingBank-LLMCompressed](https://huggingface.co/datasets/microsoft/MeetingBank-LLMCompressed)
dataset; check its license before shipping a checkpoint trained on it.

---

<div align="center">

*grug not need many words. grug need right words.*

This README, [compressed by grug itself](https://github.com/akshayballal95/grug/blob/main/README.grug.md).

</div>
