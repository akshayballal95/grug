# grug examples

Runnable scripts. Each one is standalone — run it from anywhere:

```bash
python examples/rules_backend.py
```

| Script | Needs | What it shows |
| --- | --- | --- |
| [`rules_backend.py`](rules_backend.py) | `pip install grug` | The dependency-free backend: rate sweep, what it refuses to drop, batching, tuning knobs. |
| [`lingua2_backend.py`](lingua2_backend.py) | `pip install 'grug[lingua2]'` | The token-classification backend: model load, rate sweep, negation survival under pressure. Prints an install hint and exits 0 if the extra is missing. |
| [`compare_backends.py`](compare_backends.py) | either | Every installed backend on the same document, side by side — tokens, ratio, seconds, warnings. |
| [`faithfulness.py`](faithfulness.py) | either | Why the verifier exists: the negation problem, the three checks, and load-bearing sentences run through every backend. |

`sample_doc.md` is the shared input — a short incident report with a fenced code
block, several number formats, inline code, a URL, and negations that carry the
meaning of their paragraphs. `_shared.py` is a tiny printing helper, not part of
the grug API.

## Flags

`lingua2_backend.py` and `compare_backends.py` take options:

```bash
python examples/lingua2_backend.py --device cpu --rate 0.33
python examples/compare_backends.py --rates 0.7 0.5 0.3
```

The first `lingua2` run downloads the checkpoint (a few hundred MB) and takes a
minute; later runs load from the Hugging Face cache in a second or two.

## The same thing from the CLI

```bash
grug backends                                        # what is installed
grug compress examples/sample_doc.md --rate 0.5      # writes sample_doc.grug.md
grug compress examples/sample_doc.md --rate 0.3 -b lingua2 -o - | head -20
grug compress examples/sample_doc.md --json -q | jq '{ratio, warnings}'
grug verify examples/sample_doc.md examples/sample_doc.grug.md
```

## Expected shape of the results

On `sample_doc.md` (406 tokens, of which 70 are an incompressible code block):

| backend | rate 0.8 | rate 0.5 | rate 0.3 |
| --- | --- | --- | --- |
| `rules` | 0.83 | 0.77 | 0.77 |
| `lingua2` | 0.85 | 0.60 | 0.44 |

`rules` floors out because it only deletes words from curated lists. `lingua2`
keeps going because it scores every token. Both report the ratio they actually
achieved. Numbers will shift with your document; the shape will not.
