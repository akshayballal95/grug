# grug

**grug make text small. grug keep meaning.**

grug compresses documents into terse, caveman-style text so you spend fewer tokens
getting context into an LLM. It drops the words that carry no information —
articles, copulas, filler, pleasantries — and keeps the ones that do: numbers,
names, and above all **negations**. Compression runs through a pluggable backend:
the default ships a rule-based stripper with zero dependencies, and
`pip install grug[lingua2]` swaps in LLMLingua-2's token-classification model for
much more aggressive rates. Every result comes back with a faithfulness report,
because a prompt that is 60% smaller and 5% wrong is not a win.

### Before, 94 tokens

> It is important to note that the billing pipeline has been rewritten to run on
> the streaming ingest service. The migration is not automatic: accounts on the
> legacy monthly plan must be moved by hand before the cutover date. In practice
> we measured a median lag of 1.2 seconds across 4,800 accounts, and a p99 lag of
> 9.6 seconds. The key economic point is easy to get wrong: bills scale with
> volume, not price.

### After, 63 tokens — `rate=0.5`, `backend="rules"`

> billing pipeline rewritten run streaming ingest service. migration not
> automatic: accounts legacy monthly plan must moved hand before cutover date.
> practice measured median lag 1.2 seconds across 4,800 accounts, p99 lag 9.6
> seconds. key economic point easy wrong: bills scale volume, not price.

Ratio 0.67, no faithfulness warnings. Every number survived, and so did both
negations — the second one (`not price`) is the entire point of the paragraph.

## Install

```bash
pip install grug              # core + rules backend + CLI. No torch.
pip install 'grug[lingua2]'   # adds llmlingua, torch, transformers.
```

`import grug` never imports torch, even with the extra installed. Backends load
their dependencies when you instantiate one, not when you import the package.
The core install is `typer` and `markdown-it-py`, both pure Python.

## CLI

```bash
grug compress notes.md                        # writes notes.grug.md, stats to stderr
grug compress notes.md -o out.txt --rate 0.4 --backend lingua2
grug compress - < in.txt > out.txt            # stdin to stdout
grug compress docs/*.md --rate 0.5            # many files, each to <name>.grug.md
grug verify original.txt compressed.txt       # faithfulness checks, standalone
grug backends                                 # what is registered, what is installed
```

Stats go to stderr so stdout stays pipeable:

```
132 → 78 tokens (0.59, backend=lingua2, 0.4s)
⚠ negation lost: 'not' (2× → 1×) — meaning may be inverted
```

| Flag | Effect |
| --- | --- |
| `--rate`, `-r` | Fraction of tokens to keep. Default `0.5`. |
| `--backend`, `-b` | Backend name. Default: best installed. |
| `--device` | `cpu`, `cuda`, `mps`, or `auto`, for backends that use one. |
| `--no-verify` | Skip the faithfulness checks. |
| `--json` | Emit the full `CompressionResult` as JSON on stdout; write no files. |
| `--chunk-tokens` | Max tokens per chunk. Default `450`. |
| `--quiet`, `-q` | Suppress the stats line. |

Exit codes are built for CI gating:

| Code | Meaning |
| --- | --- |
| `0` | Compressed, no faithfulness warnings. |
| `1` | Error — missing file, unknown backend, missing dependency. |
| `2` | Compressed, but the verifier flagged something. |

```bash
grug compress prompt.md --rate 0.4
case $? in
  0) ;;                                    # clean
  2) echo "compressed, but check the warnings" ;;
  *) echo "failed"; exit 1 ;;
esac
```

## Library

```python
import grug

result = grug.compress(text, rate=0.5)  # best installed backend
result = grug.compress(text, rate=0.4, backend="rules")

result.text  # the compressed document
result.original_tokens  # 94
result.compressed_tokens  # 63
result.ratio  # 0.67 — what was achieved, not what was asked for
result.warnings  # ["negation lost: 'not' (1× → 0×) — ..."]
result.metadata  # backend-specific extras

comp = grug.Compressor(backend="lingua2", device="cuda")  # model loads once
results = comp.compress_batch(documents, rate=0.5)

grug.list_backends()  # ['lingua2', 'rules']
grug.verify(original, compressed)  # -> list[str], standalone
```

## Markdown awareness

Documents are parsed with [markdown-it-py](https://github.com/executablebooks/markdown-it-py)
(a CommonMark parser, already a transitive dependency of `typer`) rather than
pattern-matched, so structure survives compression intact:

| Element | Treatment |
| --- | --- |
| Fenced and indented code | Bypasses the compressor; re-emitted byte-for-byte |
| Tables | Bypassed whole — a table is already dense, and compressing cells destroys the column-to-value mapping |
| Headings, list bullets, blockquote markers, rules | Marker preserved verbatim, the text after it compresses |
| Blank lines | Hard chunk boundaries, so no backend can collapse a paragraph break |
| Inline code spans and URLs | Swapped for opaque placeholders |
| Numbers with internal separators (`9.6`, `1,250`, `3-5`) | Protected — backends keep the digits but drop the punctuation between them |

Using a real parser matters: a pipe inside a fence is not a table row, an
indented block is still code, and a code span may wrap across a line. The
`lingua2` backend additionally pins every placeholder in `force_tokens`, so a
protected span comes back verbatim or not at all — never rewritten.

Long documents are chunked to ~450 tokens on sentence boundaries before they
reach the backend, then rejoined with the original layout.

## Examples

Runnable scripts live in [`examples/`](examples/):

```bash
python examples/rules_backend.py       # dependency-free backend, end to end
python examples/lingua2_backend.py     # the classifier backend
python examples/compare_backends.py    # every installed backend, side by side
python examples/faithfulness.py        # the negation problem and the verifier
```

Each exits 0 whether or not the optional extra is installed, so they are safe to
run in CI. See [`examples/README.md`](examples/README.md) for what each covers.

## Backends

| Backend | Install | Method | Rates it reaches | Speed |
| --- | --- | --- | --- | --- |
| `rules` | included | Deletes stopwords, fillers, and pleasantry phrases from curated lists | ~0.6 floor — it runs out of safe words to drop | milliseconds |
| `lingua2` | `grug[lingua2]` | LLMLingua-2 token classification (BERT), keeps the top-scoring `rate` fraction | 0.2–0.5 | ~0.2 s for a 400-token document on a laptop CPU, after a one-off model load |

Both are **extractive**: the output is a subsequence of the input, so neither can
invent a fact. The backend interface does not require this — a generative
seq2seq backend that rewrites text is a legitimate implementation of the same
ABC, and the `generative` class attribute marks one so callers can treat its
output with appropriate suspicion. None ships yet.

`rate` means the same thing everywhere: the fraction of tokens to **keep**. A
backend that cannot hit the number exactly approximates it and reports what it
actually achieved in `result.ratio`.

## Faithfulness

Aggressive compression has one failure mode that matters more than the rest.
Losing detail makes a prompt vaguer. Losing a negation makes it *wrong*.

> bills scale with volume, not price

A token-classification compressor scores each word for how much it contributes,
and `not` is a three-character function word that looks eminently droppable. Drop
it and the sentence now asserts the exact opposite of what the document says, in
fluent, confident prose that no downstream model will question.

grug handles this in two places:

1. **Negation-preserving defaults.** The `lingua2` backend ships every negation
   the verifier polices — `not, no, never, none, neither, nor, n't, except,
   unless, without, cannot, nothing, nobody, nowhere, lack(s/ing), absent` — in
   its `force_tokens` list, alongside newline and `?`, with digit preservation
   on. The force list is *derived* from the verifier's vocabulary so the two
   cannot drift apart, and each word is also sent in capitalised and upper-case
   form, because LLMLingua-2 matches `force_tokens` case-sensitively and a
   lowercase list leaves every sentence-initial `No…`/`Not…` unprotected.
   The `rules` backend refuses to consider those words as deletion candidates at
   any rate. Both are configurable; the defaults are the safe ones.
2. **A verifier that checks after the fact.** `grug.verify(original, compressed)`
   returns human-readable warnings, most severe first:
   - **Negation loss** — a negation cue that appears fewer times than it did.
   - **Number loss** — an integer, decimal, percentage, or version that vanished
     (`1,250` and `1250` count as the same number).
   - **Entity loss** — a capitalised multiword name or acronym with none of its
     words left standing.

   There is **no NER model** here, and no ML anywhere outside the `lingua2`
   backend. Entity detection is regex over capitalised runs, name connectors
   ("Bank *of* America"), and acronyms, with a stoplist for words that merely
   start a sentence. That buys zero dependencies and microseconds per check, and
   costs precision: it misses lowercase brands and a lone name opening a
   sentence. Treat the entity check as the weakest of the three.

   An NER model is deliberately not used. The verifier's job is to be
   trustworthy about *compressed* output — terse, ungrammatical caveman text
   that is exactly out-of-distribution for models trained on well-formed prose.
   A model there would make the safety net heavier, slower, and less reliable
   than the thing it checks.

3. **Prevention, where it is cheap.** What the verifier polices, the compressor
   protects. Negations are pinned in `force_tokens`, and so are the proper nouns
   the extractor finds in the input (`preserve_entities=True`, on by default).
   Measured on a mixed markdown document, pinning entities costs 1–3% of the
   compression ratio and no extra time — words absent from the text are free —
   and removes the entity-loss warning class outright. Numbers need no model
   either: they are matched exactly, protected as spans, and backed by
   `force_reserve_digit`.

Verification is on by default in both `compress()` and the CLI, and populates
`result.warnings`. It is a smoke alarm, not a proof: a clean run means nothing
suspicious was detected, not that the compression was faithful.

```python
>>> grug.verify("bills scale with volume, not price", "bills scale volume price")
["negation lost: 'not' (1× → 0×) — meaning may be inverted"]
```

## Writing your own backend

Subclass the ABC, register it, and it appears everywhere — `grug.compress()`,
`grug.Compressor`, and the CLI's `--backend` — with no changes to either.

```python
# grug_shouty/backend.py
from grug.base import CompressionResult, CompressorBackend


class ShoutyBackend(CompressorBackend):
    name = "shouty"  # the registry key and --backend value
    description = "Keeps only the long words."
    extra = "shouty"  # named in the missing-dependency error
    generative = False  # True if you rewrite rather than select

    def compress(self, text: str, rate: float = 0.5, **kwargs) -> CompressionResult:
        self._validate_rate(rate)
        cutoff = 3 + int((1 - rate) * 5)
        kept = " ".join(w for w in text.split() if len(w) > cutoff)
        return CompressionResult.build(text, kept, self.name)
```

Advertise it from your `pyproject.toml` and `pip install grug-shouty` is all a
user needs:

```toml
[project.entry-points."grug.backends"]
shouty = "grug_shouty.backend:ShoutyBackend"
```

Notes for real backends:

- Import heavy dependencies inside your methods, not at module scope, and
  override `is_available()` so `grug backends` can report honestly.
  `CompressorBackend.require_available()` raises the standard
  `MissingDependencyError` naming your extra — call it from `__init__`.
- Override `compress_batch()` if your model batches; the chunker calls it with
  every compressible chunk of a document at once. The default is a loop.
- `CompressionResult.build()` derives token counts and the achieved ratio for
  you, using `cl100k_base` so numbers stay comparable across backends.
- In-process registration works too: `@grug.register_backend` on the class.

## Not in this release

No seq2seq backend, no API server, no output-side compression, no training code.

## License

MIT.
